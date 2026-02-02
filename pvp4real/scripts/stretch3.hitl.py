#!/usr/bin/env python3
"""
stretch3.hitl.py

Real-robot Human-In-The-Loop (HITL) online training loop for Stretch3 using the PVP4Real codebase.

This script:
- Subscribes to RGB + aligned depth topics.
- Subscribes to teleop toggle and teleop cmd_vel.
- Runs a wall-clock synced control loop (default 5 Hz).
- Wraps the ROS2 stream as a Gym environment and trains PVPTD3 (PVP4Real's TD3 variant).

ROS2 interfaces (given by user):
  RGB:   /camera/color/image_raw (sensor_msgs/Image)
  Depth: /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image)
  I(s):  /stretch/is_teleop (std_msgs/Bool)                 True => human takeover
  a_h:   /stretch/cmd_vel_teleop (geometry_msgs/Twist)      teleop velocity command
  a:     /stretch/cmd_vel (geometry_msgs/Twist)             final command (published by this script)

Notes:
- This script does NOT assume any "warmup". Takeover is controlled purely by /stretch/is_teleop.
- In non-takeover steps, behavior/raw_action is set equal to the novice action to match HACOReplayBuffer expectations.
- Depth scaling is configurable; default clips to [0, 5] meters and maps to uint8 [0, 255].

Usage example:
  python3 pvp4real/scripts/stretch3.hitl.py \
    --model_dir /pvp4real/models/stretch3_hitl \
    --hz 5 \
    --max_lin 0.4 --max_ang 1.2
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

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# Conversions
try:
    from cv_bridge import CvBridge  # type: ignore
except Exception as e:  # pragma: no cover
    CvBridge = None
    _cvbridge_import_error = e

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# PVP4Real algorithm (from the repo)
from pvp.pvp_td3 import PVPTD3


def _now_monotonic() -> float:
    return time.monotonic()


def _twist_to_np(t: Twist) -> np.ndarray:
    return np.array([float(t.linear.x), float(t.angular.z)], dtype=np.float32)


def _np_to_twist(vw: np.ndarray) -> Twist:
    msg = Twist()
    msg.linear.x = float(vw[0])
    msg.angular.z = float(vw[1])
    return msg


@dataclass
class ObsConfig:
    resize_hw: Tuple[int, int] = (84, 84)   # (H, W)
    stack_n: int = 5
    # Depth mapping
    depth_max_m: float = 5.0
    # If True, output dtype uint8 in [0,255] and let SB3 normalize by /255.
    output_uint8: bool = True


class Stretch3TopicCache(Node):
    """
    Caches the latest ROS2 messages and maintains stacked RGB-D frames.
    """
    def __init__(self, obs_cfg: ObsConfig, hz: float):
        super().__init__("stretch3_pvp4real_cache")

        if CvBridge is None:
            raise RuntimeError(f"cv_bridge is required but not importable: {_cvbridge_import_error}")

        self.obs_cfg = obs_cfg
        self.hz = hz
        self.bridge = CvBridge()

        # Latest state
        self._latest_rgb: Optional[np.ndarray] = None  # HxWx3 uint8
        self._latest_depth: Optional[np.ndarray] = None  # HxW float32 meters or uint16
        self._latest_teleop: bool = False
        self._latest_teleop_twist: Twist = Twist()

        # For takeover rising edge detection (takeover_start)
        self._prev_takeover: bool = False

        # Stacks (each entry is already resized)
        self._rgb_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)    # each: HxWx3
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)  # each: HxW

        # QoS: sensor data often uses best-effort
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

        # Subscribers
        self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, sensor_qos)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self._on_depth, sensor_qos)
        self.create_subscription(Bool, "/stretch/is_teleop", self._on_teleop, default_qos)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop", self._on_teleop_twist, default_qos)

        # Publisher (final command)
        self.cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", default_qos)

        self.get_logger().info(
            f"Subscribed RGB/Depth + teleop topics. Will build stacked obs: {obs_cfg.stack_n} frames @ {obs_cfg.resize_hw}."
        )

    def _on_rgb(self, msg: Image) -> None:
        # Expect encoding: rgb8 or bgr8
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # HxWx3 uint8
        if cv2 is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_AREA)
        else:
            # Fallback: numpy nearest resize (less ideal). User container should have cv2.
            cv_img = cv_img[:, :, ::-1].copy()
            cv_img = cv_img[:: max(1, cv_img.shape[0] // self.obs_cfg.resize_hw[0]),
                            :: max(1, cv_img.shape[1] // self.obs_cfg.resize_hw[1]), :]
            cv_img = cv_img[: self.obs_cfg.resize_hw[0], : self.obs_cfg.resize_hw[1], :]
        self._latest_rgb = cv_img
        self._rgb_stack.append(cv_img)

    def _on_depth(self, msg: Image) -> None:
        # Depth often: 16UC1 (mm) or 32FC1 (m)
        cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_depth = np.array(cv_depth)

        # Convert to meters float32
        if cv_depth.dtype == np.uint16:
            depth_m = cv_depth.astype(np.float32) / 1000.0
        else:
            depth_m = cv_depth.astype(np.float32)

        # Resize
        if cv2 is not None:
            depth_m = cv2.resize(depth_m, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_NEAREST)
        else:
            depth_m = depth_m[:: max(1, depth_m.shape[0] // self.obs_cfg.resize_hw[0]),
                              :: max(1, depth_m.shape[1] // self.obs_cfg.resize_hw[1])]
            depth_m = depth_m[: self.obs_cfg.resize_hw[0], : self.obs_cfg.resize_hw[1]]

        self._latest_depth = depth_m
        self._depth_stack.append(depth_m)

    def _on_teleop(self, msg: Bool) -> None:
        self._latest_teleop = bool(msg.data)

    def _on_teleop_twist(self, msg: Twist) -> None:
        self._latest_teleop_twist = msg

    def stacks_ready(self) -> bool:
        return (len(self._rgb_stack) == self.obs_cfg.stack_n) and (len(self._depth_stack) == self.obs_cfg.stack_n)

    def get_takeover_and_start(self) -> Tuple[bool, bool]:
        takeover = self._latest_teleop
        takeover_start = takeover and (not self._prev_takeover)
        self._prev_takeover = takeover
        return takeover, takeover_start

    def get_human_vw(self) -> np.ndarray:
        return _twist_to_np(self._latest_teleop_twist)

    def build_obs(self) -> np.ndarray:
        """
        Returns observation in HxWxC (channel-last) uint8 by default:
          C = stack_n*(3 + 1) = 20 (for stack_n=5)
        """
        if not self.stacks_ready():
            raise RuntimeError("Stacks not ready yet")

        rgb_list = list(self._rgb_stack)     # each HxWx3 uint8
        depth_list = list(self._depth_stack) # each HxW float32 (meters)

        # depth -> uint8 [0,255]
        depth_max = float(self.obs_cfg.depth_max_m)
        depth_u8_list = []
        for d in depth_list:
            d_clip = np.clip(d, 0.0, depth_max)
            d_u8 = (d_clip / depth_max * 255.0).astype(np.uint8)
            depth_u8_list.append(d_u8)

        # Stack along channel dimension
        # RGB: (H,W,3*stack_n) ; Depth: (H,W,stack_n)
        rgb_cat = np.concatenate(rgb_list, axis=2)  # HxWx(3N)
        depth_cat = np.stack(depth_u8_list, axis=2) # HxWxN
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)  # HxWx(4N)

        if self.obs_cfg.output_uint8:
            return obs.astype(np.uint8, copy=False)
        # float32 [0,1]
        return (obs.astype(np.float32) / 255.0)

    def publish_final_cmd(self, vw: np.ndarray) -> None:
        self.cmd_pub.publish(_np_to_twist(vw))


class Stretch3HITLGymEnv(gym.Env):
    """
    Gym env that steps at wall-clock dt and interfaces with Stretch3 via ROS topics.
    It provides HACOReplayBuffer-compatible info dict fields:
      takeover, takeover_start, takeover_cost, raw_action

    action: normalized in [-1, 1]^2 (v, w)
    observation: image-like uint8 HxWxC
    reward: 0 (reward-free setting)
    done: False (infinite-horizon)
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        node: Stretch3TopicCache,
        dt: float,
        max_lin: float,
        max_ang: float,
    ):
        super().__init__()
        self.node = node
        self.dt = float(dt)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)

        h, w = node.obs_cfg.resize_hw
        c = node.obs_cfg.stack_n * (3 + 1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._last_step_time = _now_monotonic()

    def _action_to_vw(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a.astype(np.float32), -1.0, 1.0)
        return np.array([a[0] * self.max_lin, a[1] * self.max_ang], dtype=np.float32)

    def _vw_to_action(self, vw: np.ndarray) -> np.ndarray:
        return np.array([vw[0] / self.max_lin, vw[1] / self.max_ang], dtype=np.float32).clip(-1.0, 1.0)

    def reset(self):
        # Wait until stacks ready
        start = _now_monotonic()
        while rclpy.ok() and (not self.node.stacks_ready()):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if _now_monotonic() - start > 10.0:
                raise TimeoutError("Timed out waiting for RGB-D stacks to become ready.")
        obs = self.node.build_obs()
        return obs

    def step(self, action: np.ndarray):
        # Ensure we step at wall-clock dt
        now = _now_monotonic()
        elapsed = now - self._last_step_time
        if elapsed < self.dt:
            # Use ROS spin during wait to keep callbacks flowing
            remaining = self.dt - elapsed
            end = _now_monotonic() + remaining
            while rclpy.ok() and _now_monotonic() < end:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        self._last_step_time = _now_monotonic()

        # Latch current obs s_t (already built from buffered stacks)
        obs = self.node.build_obs()

        # Determine takeover and human action
        takeover, takeover_start = self.node.get_takeover_and_start()
        human_vw = self.node.get_human_vw()
        human_action = self._vw_to_action(human_vw)

        # Novice action (policy output), normalized
        novice_action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        # Arbitration: choose which velocity to execute
        if takeover:
            exec_vw = human_vw
            raw_action = human_action  # behavior action
        else:
            exec_vw = self._action_to_vw(novice_action)
            raw_action = novice_action  # behavior == novice when no takeover

        # Publish final cmd_vel
        self.node.publish_final_cmd(exec_vw)

        # Let the world evolve and capture next obs s_{t+1}
        # We spin a bit to process incoming sensor messages
        t_end = _now_monotonic() + 0.02
        while rclpy.ok() and _now_monotonic() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        next_obs = self.node.build_obs() if self.node.stacks_ready() else obs

        # Reward-free
        reward = 0.0
        done = False

        info = {
            "takeover": float(takeover),
            "takeover_start": float(takeover_start),
            "takeover_cost": 0.0,
            "raw_action": raw_action.astype(np.float32),
        }
        return next_obs, reward, done, info


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    p.add_argument("--hz", type=float, default=5.0, help="Wall-clock control frequency (Hz). Default 5Hz (dt=0.2s).")
    p.add_argument("--max_lin", type=float, default=0.4, help="Max linear velocity (m/s) mapped from action=+1.")
    p.add_argument("--max_ang", type=float, default=1.2, help="Max angular velocity (rad/s) mapped from action=+1.")
    p.add_argument("--depth_max_m", type=float, default=5.0, help="Depth clip max in meters, mapped to 255.")
    p.add_argument("--stack_n", type=int, default=5, help="Number of frames to stack. Paper setting: 5.")
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84], metavar=("H", "W"), help="Resize H W. Paper: 84 84.")
    p.add_argument("--total_steps", type=int, default=200000, help="Total environment steps (wall-clock ticks) to train.")
    p.add_argument("--learning_starts", type=int, default=500, help="Steps before training starts collecting buffer.")
    p.add_argument("--buffer_size", type=int, default=200000, help="Replay buffer size.")
    p.add_argument("--batch_size", type=int, default=1024, help="Batch size for TD3/PVP updates.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="Torch device for SB3. e.g., cuda, cpu, auto")
    p.add_argument("--save_every", type=int, default=5000, help="Checkpoint every N steps.")
    p.add_argument("--log_interval", type=int, default=10, help="SB3 log interval.")
    # PVP-specific extras wired through PVPTD3 kwargs (see pvp/pvp_td3.py)
    p.add_argument("--q_value_bound", type=float, default=1.0)
    p.add_argument("--bc_loss_weight", type=float, default=1.0)
    p.add_argument("--with_human_proxy_value_loss", type=str, default="True", choices=["True", "False"])
    p.add_argument("--with_agent_proxy_value_loss", type=str, default="True", choices=["True", "False"])
    p.add_argument("--only_bc_loss", type=str, default="False", choices=["True", "False"])
    p.add_argument("--add_bc_loss", type=str, default="True", choices=["True", "False"])
    return p


def main() -> None:
    args = build_argparser().parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    rclpy.init()
    obs_cfg = ObsConfig(
        resize_hw=(int(args.resize[0]), int(args.resize[1])),
        stack_n=int(args.stack_n),
        depth_max_m=float(args.depth_max_m),
        output_uint8=True,
    )
    node = Stretch3TopicCache(obs_cfg=obs_cfg, hz=float(args.hz))

    dt = 1.0 / float(args.hz)
    env = Stretch3HITLGymEnv(node=node, dt=dt, max_lin=args.max_lin, max_ang=args.max_ang)

    # SB3/PVP model
    # Use CnnPolicy as observation is image-like
    model = PVPTD3(
        "CnnPolicy",
        env,
        seed=args.seed,
        verbose=1,
        device=args.device,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=(1, "step"),
        gradient_steps=1,
        gamma=0.99,
        tau=0.05,
        learning_rate=1e-4,
        q_value_bound=args.q_value_bound,
        # PVP extras:
        bc_loss_weight=args.bc_loss_weight,
        with_human_proxy_value_loss=args.with_human_proxy_value_loss,
        with_agent_proxy_value_loss=args.with_agent_proxy_value_loss,
        only_bc_loss=args.only_bc_loss,
        add_bc_loss=args.add_bc_loss,
    )

    # Ensure env is ready
    _ = env.reset()

    node.get_logger().info(f"Starting HITL training for {args.total_steps} steps @ {args.hz}Hz (dt={dt:.3f}s).")
    node.get_logger().info(f"Saving to: {args.model_dir}")

    # Training loop: we use SB3 learn with total_timesteps = total_steps
    # We implement a custom callback-like checkpoint by periodic save in a simple loop.
    # SB3's learn() can do it too, but keeping explicit control is useful for real-robot runs.
    remaining = int(args.total_steps)
    trained = 0
    while rclpy.ok() and remaining > 0:
        chunk = min(args.save_every, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, log_interval=args.log_interval)
        trained += chunk
        remaining -= chunk
        ckpt_path = os.path.join(args.model_dir, f"pvp4real_stretch3_step{trained}.zip")
        model.save(ckpt_path)
        node.get_logger().info(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.model_dir, "pvp4real_stretch3_final.zip")
    model.save(final_path)
    node.get_logger().info(f"Final model saved: {final_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
