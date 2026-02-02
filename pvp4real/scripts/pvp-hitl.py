#!/usr/bin/env python3
"""
pvp-hitl.py

Primary Human-In-The-Loop (HITL) training entry for PVP4Real on Stretch3.

This script:
- Runs entirely inside the PVP4Real container
- Acts as the wall-clock master (default 5 Hz)
- Latches RGB-D observations
- Latches human intervention signal and teleop action
- Performs arbitration and publishes /stretch/cmd_vel
- Trains PVPTD3 online using (s, a_n, a_h, I, s')

Assumed ROS2 interfaces (as specified by user):
  RGB:     /camera/color/image_raw                      sensor_msgs/Image
  Depth:   /camera/aligned_depth_to_color/image_raw     sensor_msgs/Image
  I(s):    /stretch/is_teleop                           std_msgs/Bool
  a_h:     /stretch/cmd_vel_teleop                      geometry_msgs/Twist
  a_final: /stretch/cmd_vel                             geometry_msgs/Twist

Paper-aligned vision setup:
  - Resize RGB + Depth to 84x84
  - Stack last 5 RGB-D frames
  - Channel-last: (84, 84, 20)

This script intentionally contains *no* Stretch3 bringup logic.
"""

import os
import time
from collections import deque
from typing import Deque, Tuple

import numpy as np
import gym

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import cv2

# PVP4Real
from pvp.pvp_td3 import PVPTD3


# ------------------------
# Utility
# ------------------------

def twist_to_np(msg: Twist) -> np.ndarray:
    return np.array([msg.linear.x, msg.angular.z], dtype=np.float32)


def np_to_twist(vw: np.ndarray) -> Twist:
    t = Twist()
    t.linear.x = float(vw[0])
    t.angular.z = float(vw[1])
    return t


# ------------------------
# ROS2 Cache Node
# ------------------------

class HITLCache(Node):
    def __init__(self):
        super().__init__("pvp_hitl_cache")

        self.bridge = CvBridge()

        # Latest signals
        self.rgb = None
        self.depth = None
        self.is_teleop = False
        self.teleop_twist = Twist()

        # Frame stacks
        self.rgb_stack: Deque[np.ndarray] = deque(maxlen=5)
        self.depth_stack: Deque[np.ndarray] = deque(maxlen=5)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Image, "/camera/color/image_raw", self.on_rgb, qos_sensor
        )
        self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self.on_depth, qos_sensor
        )
        self.create_subscription(
            Bool, "/stretch/is_teleop", self.on_is_teleop, qos_default
        )
        self.create_subscription(
            Twist, "/stretch/cmd_vel_teleop", self.on_teleop_twist, qos_default
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/stretch/cmd_vel", qos_default
        )

    # ---------- callbacks ----------

    def on_rgb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        self.rgb = img
        self.rgb_stack.append(img)

    def on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32)
        if depth.max() > 20.0:  # likely mm
            depth /= 1000.0
        depth = np.clip(depth, 0.0, 5.0)
        depth = cv2.resize(depth, (84, 84), interpolation=cv2.INTER_NEAREST)
        self.depth = depth
        self.depth_stack.append(depth)

    def on_is_teleop(self, msg: Bool):
        self.is_teleop = bool(msg.data)

    def on_teleop_twist(self, msg: Twist):
        self.teleop_twist = msg

    # ---------- helpers ----------

    def ready(self) -> bool:
        return len(self.rgb_stack) == 5 and len(self.depth_stack) == 5

    def build_obs(self) -> np.ndarray:
        """
        Returns uint8 observation with shape (84, 84, 20)
        """
        rgb_cat = np.concatenate(list(self.rgb_stack), axis=2)  # 15 ch
        depth_u8 = [
            (d / 5.0 * 255.0).astype(np.uint8) for d in self.depth_stack
        ]
        depth_cat = np.stack(depth_u8, axis=2)  # 5 ch
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)
        return obs.astype(np.uint8)

    def publish_cmd(self, vw: np.ndarray):
        self.cmd_pub.publish(np_to_twist(vw))


# ------------------------
# Gym Env Wrapper
# ------------------------

class Stretch3HITLEnv(gym.Env):
    def __init__(self, node: HITLCache, hz=5.0):
        super().__init__()
        self.node = node
        self.dt = 1.0 / hz
        self.last_t = time.monotonic()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 20), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_lin = 0.4
        self.max_ang = 1.2

    def reset(self):
        while rclpy.ok() and not self.node.ready():
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.node.build_obs()

    def step(self, action):
        # wall-clock sync
        now = time.monotonic()
        if now - self.last_t < self.dt:
            time.sleep(self.dt - (now - self.last_t))
        self.last_t = time.monotonic()

        obs = self.node.build_obs()

        a_n = np.clip(action, -1.0, 1.0)
        a_n_vw = np.array(
            [a_n[0] * self.max_lin, a_n[1] * self.max_ang], dtype=np.float32
        )

        I = self.node.is_teleop
        a_h_vw = twist_to_np(self.node.teleop_twist)
        a_h = np.array(
            [a_h_vw[0] / self.max_lin, a_h_vw[1] / self.max_ang],
            dtype=np.float32,
        ).clip(-1.0, 1.0)

        if I:
            exec_vw = a_h_vw
            raw_action = a_h
        else:
            exec_vw = a_n_vw
            raw_action = a_n

        self.node.publish_cmd(exec_vw)

        # spin briefly to get s'
        t_end = time.monotonic() + 0.02
        while rclpy.ok() and time.monotonic() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        next_obs = self.node.build_obs()

        info = {
            "takeover": float(I),
            "takeover_start": 0.0,   # edge not strictly needed here
            "takeover_cost": 0.0,
            "raw_action": raw_action,
        }

        return next_obs, 0.0, False, info


# ------------------------
# Main
# ------------------------

def main():
    rclpy.init()

    node = HITLCache()
    env = Stretch3HITLEnv(node, hz=5.0)

    model = PVPTD3(
        "CnnPolicy",
        env,
        buffer_size=200_000,
        learning_starts=500,
        batch_size=1024,
        train_freq=(1, "step"),
        gradient_steps=1,
        gamma=0.99,
        tau=0.05,
        learning_rate=1e-4,
        q_value_bound=1.0,
        add_bc_loss=True,
        with_human_proxy_value_loss=True,
        with_agent_proxy_value_loss=True,
        verbose=1,
    )

    obs = env.reset()
    model.learn(total_timesteps=200_000)

    model.save("pvp4real_stretch3_hitl_final")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
