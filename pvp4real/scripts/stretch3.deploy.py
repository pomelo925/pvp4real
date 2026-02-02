#!/usr/bin/env python3
"""
stretch3.deploy.py

Deployment-only (inference) node for Stretch3 using a trained PVPTD3 policy.
- Subscribes to RGB-D and teleop topics.
- Builds stacked 84x84 RGB-D observation (paper setting).
- Computes policy action a_n and, if teleop is active, overrides with human cmd_vel.
- Publishes final /stretch/cmd_vel.

This script does NOT update the model (no training).

Usage:
  python3 pvp4real/scripts/stretch3.deploy.py \
    --model_path /pvp4real/models/stretch3_hitl/pvp4real_stretch3_final.zip \
    --hz 5 --max_lin 0.4 --max_ang 1.2
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import gym
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

try:
    from cv_bridge import CvBridge  # type: ignore
except Exception as e:  # pragma: no cover
    CvBridge = None
    _cvbridge_import_error = e

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from pvp.pvp_td3 import PVPTD3


def _twist_to_np(t: Twist) -> np.ndarray:
    return np.array([float(t.linear.x), float(t.angular.z)], dtype=np.float32)


def _np_to_twist(vw: np.ndarray) -> Twist:
    msg = Twist()
    msg.linear.x = float(vw[0])
    msg.angular.z = float(vw[1])
    return msg


@dataclass
class ObsConfig:
    resize_hw: Tuple[int, int] = (84, 84)
    stack_n: int = 5
    depth_max_m: float = 5.0
    output_uint8: bool = True


class DeployNode(Node):
    def __init__(self, obs_cfg: ObsConfig, hz: float, model_path: str, max_lin: float, max_ang: float):
        super().__init__("stretch3_pvp4real_deploy")

        if CvBridge is None:
            raise RuntimeError(f"cv_bridge is required but not importable: {_cvbridge_import_error}")

        self.obs_cfg = obs_cfg
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.bridge = CvBridge()
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)

        # Stacks
        self._rgb_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)

        self._teleop: bool = False
        self._teleop_twist: Twist = Twist()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, sensor_qos)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self._on_depth, sensor_qos)
        self.create_subscription(Bool, "/stretch/is_teleop", self._on_teleop, default_qos)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop", self._on_teleop_twist, default_qos)

        self.cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", default_qos)

        # Load model
        self.get_logger().info(f"Loading model: {model_path}")
        # We need an env to load policy; create a dummy image env with matching spaces
        h, w = obs_cfg.resize_hw
        c = obs_cfg.stack_n * (3 + 1)
        dummy_env = gym.make("CartPole-v1")  # placeholder; we'll override spaces below
        dummy_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        dummy_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.model = PVPTD3.load(model_path, env=dummy_env)

        self._last = time.monotonic()
        self.timer = self.create_timer(self.dt, self._on_tick)

    def _on_rgb(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv2 is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_AREA)
        else:
            cv_img = cv_img[:, :, ::-1].copy()
            cv_img = cv_img[: self.obs_cfg.resize_hw[0], : self.obs_cfg.resize_hw[1], :]
        self._rgb_stack.append(cv_img)

    def _on_depth(self, msg: Image) -> None:
        cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_depth = np.array(cv_depth)
        if cv_depth.dtype == np.uint16:
            depth_m = cv_depth.astype(np.float32) / 1000.0
        else:
            depth_m = cv_depth.astype(np.float32)

        if cv2 is not None:
            depth_m = cv2.resize(depth_m, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_NEAREST)
        else:
            depth_m = depth_m[: self.obs_cfg.resize_hw[0], : self.obs_cfg.resize_hw[1]]
        self._depth_stack.append(depth_m)

    def _on_teleop(self, msg: Bool) -> None:
        self._teleop = bool(msg.data)

    def _on_teleop_twist(self, msg: Twist) -> None:
        self._teleop_twist = msg

    def _stacks_ready(self) -> bool:
        return (len(self._rgb_stack) == self.obs_cfg.stack_n) and (len(self._depth_stack) == self.obs_cfg.stack_n)

    def _build_obs(self) -> np.ndarray:
        rgb_list = list(self._rgb_stack)
        depth_list = list(self._depth_stack)

        depth_max = float(self.obs_cfg.depth_max_m)
        depth_u8_list = []
        for d in depth_list:
            d_clip = np.clip(d, 0.0, depth_max)
            d_u8 = (d_clip / depth_max * 255.0).astype(np.uint8)
            depth_u8_list.append(d_u8)

        rgb_cat = np.concatenate(rgb_list, axis=2)
        depth_cat = np.stack(depth_u8_list, axis=2)
        obs = np.concatenate([rgb_cat, depth_cat], axis=2).astype(np.uint8, copy=False)
        return obs

    def _action_to_vw(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a.astype(np.float32), -1.0, 1.0)
        return np.array([a[0] * self.max_lin, a[1] * self.max_ang], dtype=np.float32)

    def _on_tick(self) -> None:
        if not self._stacks_ready():
            return

        obs = self._build_obs()

        # SB3 expects batch or single obs; predict returns action in [-1,1]
        action, _ = self.model.predict(obs, deterministic=True)
        vw = self._action_to_vw(np.array(action, dtype=np.float32))

        if self._teleop:
            vw = _twist_to_np(self._teleop_twist)

        self.cmd_pub.publish(_np_to_twist(vw))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--max_lin", type=float, default=0.4)
    p.add_argument("--max_ang", type=float, default=1.2)
    p.add_argument("--depth_max_m", type=float, default=5.0)
    p.add_argument("--stack_n", type=int, default=5)
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84], metavar=("H", "W"))
    return p


def main() -> None:
    args = build_argparser().parse_args()

    rclpy.init()
    node = DeployNode(
        obs_cfg=ObsConfig(resize_hw=(int(args.resize[0]), int(args.resize[1])), stack_n=int(args.stack_n), depth_max_m=float(args.depth_max_m)),
        hz=float(args.hz),
        model_path=args.model_path,
        max_lin=float(args.max_lin),
        max_ang=float(args.max_ang),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
