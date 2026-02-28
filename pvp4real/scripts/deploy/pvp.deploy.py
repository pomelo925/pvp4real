#!/usr/bin/env python3
"""pvp.deploy.py — PVP4Real deployment / inference (pvp4real container).

Loads a checkpoint (.zip) and runs pure inference.
Publishes policy commands to /pvp/novice_cmd_vel.
No training, no buffer writes.

GUI:
  Displays current step count and a Quit button.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple
import tkinter as tk
from tkinter import ttk

import numpy as np
import yaml
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import gymnasium as gym
from pvp.pvp_td3 import PVPTD3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PVP_ROOT = Path(__file__).parent.parent.parent  # pvp4real/pvp4real/


def _now() -> float:
    return time.monotonic()


def _vw_to_twist(vw: np.ndarray) -> Twist:
    t = Twist()
    t.linear.x  = float(vw[0])
    t.angular.z = float(vw[1])
    return t


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 cache node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObsCfg:
    resize_hw:   Tuple[int, int] = (84, 84)
    stack_n:     int             = 5
    depth_max_m: float           = 5.0


class DeployCache(Node):
    """Caches latest RGB-D for observation building."""

    def __init__(self, obs_cfg: ObsCfg):
        super().__init__("pvp_deploy_cache")
        self.obs_cfg = obs_cfg

        self._rgb_stack:   Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, "/camera/camera/color/image_raw",                     self._on_rgb,   qos_sensor)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",    self._on_depth, qos_sensor)

        self.novice_pub = self.create_publisher(Twist, "/pvp/novice_cmd_vel", qos_default)

    # ── Image callbacks ───────────────────────────────────────────────────────

    def _on_rgb(self, msg: Image) -> None:
        from cv_bridge import CvBridge
        if not hasattr(self, "_bridge"):
            self._bridge = CvBridge()
        img_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb,
                             (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]),
                             interpolation=cv2.INTER_AREA)
        self._rgb_stack.append(img_rgb)

    def _on_depth(self, msg: Image) -> None:
        from cv_bridge import CvBridge
        if not hasattr(self, "_bridge"):
            self._bridge = CvBridge()
        d = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        d = np.array(d)
        if d.dtype == np.uint16:
            depth_m = d.astype(np.float32) / 1000.0
        else:
            depth_m = d.astype(np.float32)
        depth_m = cv2.resize(depth_m,
                             (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]),
                             interpolation=cv2.INTER_NEAREST)
        self._depth_stack.append(depth_m)

    # ── State access ──────────────────────────────────────────────────────────

    def stacks_ready(self) -> bool:
        return (len(self._rgb_stack)   == self.obs_cfg.stack_n and
                len(self._depth_stack) == self.obs_cfg.stack_n)

    def build_obs_uint8(self) -> np.ndarray:
        if not self.stacks_ready():
            raise RuntimeError("Stacks not ready")
        depth_max = float(self.obs_cfg.depth_max_m)
        depth_u8  = [
            (np.clip(d, 0.0, depth_max) / depth_max * 255.0).astype(np.uint8)
            for d in self._depth_stack
        ]
        rgb_cat   = np.concatenate(list(self._rgb_stack),  axis=2)
        depth_cat = np.stack(depth_u8, axis=2)
        return np.concatenate([rgb_cat, depth_cat], axis=2).astype(np.uint8, copy=False)

    def publish_novice_vw(self, vw: np.ndarray) -> None:
        self.novice_pub.publish(_vw_to_twist(vw))


# ─────────────────────────────────────────────────────────────────────────────
# tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────

class DeployGUI:

    def __init__(self, checkpoint_path: str):
        self.quit_requested = False
        self._step = 0

        self.root = tk.Tk()
        self.root.title("PVP4Real — Deploy")
        self.root.geometry("380x200")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._build_ui(checkpoint_path)

    def _build_ui(self, checkpoint_path: str) -> None:
        r = self.root
        ttk.Label(r, text="PVP4Real Deployment", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(14, 8)
        )

        sf = ttk.LabelFrame(r, text="Status", padding=10)
        sf.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        ttk.Label(sf, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        ttk.Label(sf, text=Path(checkpoint_path).name, foreground="gray", font=("Arial", 9)).grid(
            row=0, column=1, sticky="w", padx=6
        )

        ttk.Label(sf, text="Steps run:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._step_lbl = ttk.Label(sf, text="0", font=("Arial", 12, "bold"))
        self._step_lbl.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        sf.columnconfigure(1, weight=1)

        ttk.Button(r, text="Quit", width=18, command=self._on_quit).grid(
            row=2, column=0, columnspan=2, pady=14
        )
        r.columnconfigure(0, weight=1)

    def update_step(self, step: int) -> None:
        self._step = step
        self._step_lbl.config(text=str(step))

    def _on_quit(self) -> None:
        self.quit_requested = True
        self.root.quit()

    def start_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.root.mainloop, daemon=True)
        t.start()
        return t


# ─────────────────────────────────────────────────────────────────────────────
# Deploy inference loop
# ─────────────────────────────────────────────────────────────────────────────

def _inference_loop(
    node:     DeployCache,
    model:    PVPTD3,
    gui:      DeployGUI,
    dt:       float,
    max_lin:  float,
    max_ang:  float,
) -> None:
    """Runs in the main thread: waits for stacks then loops inference."""

    # ── Wait for first observation ────────────────────────────────────────────
    print("Waiting for RGB-D stacks…")
    start = _now()
    while rclpy.ok() and not node.stacks_ready():
        rclpy.spin_once(node, timeout_sec=0.05)
        if _now() - start > 15.0:
            print("[ERROR] Timed out waiting for camera frames.")
            return

    # ── Inference loop ────────────────────────────────────────────────────────
    last_t = _now()
    step   = 0

    while rclpy.ok() and not gui.quit_requested:
        # Wall-clock pacing
        elapsed = _now() - last_t
        if elapsed < dt:
            end_t = _now() + (dt - elapsed)
            while rclpy.ok() and _now() < end_t:
                rclpy.spin_once(node, timeout_sec=0.01)
        last_t = _now()

        if not node.stacks_ready():
            rclpy.spin_once(node, timeout_sec=0.05)
            continue

        obs = node.build_obs_uint8()
        obs_batch = obs[np.newaxis]  # (1, H, W, C)

        action, _ = model.predict(obs_batch, deterministic=True)
        action = action[0]  # (2,)

        vw = np.array([action[0] * max_lin, action[1] * max_ang], dtype=np.float32)
        node.publish_novice_vw(vw)

        # Brief spin for callbacks
        t_end = _now() + 0.02
        while rclpy.ok() and _now() < t_end:
            rclpy.spin_once(node, timeout_sec=0.0)

        step += 1
        gui.update_step(step)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PVP4Real deployment (inference only).")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .zip checkpoint (overrides config).")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    deploy = cfg["deployment"]

    hz          = float(common["hz"])
    max_lin     = float(common["max_lin"])
    max_ang     = float(common["max_ang"])
    depth_max_m = float(common["depth_max_m"])
    stack_n     = int(common["stack_n"])
    resize_hw   = (int(common["resize"]["height"]), int(common["resize"]["width"]))
    device      = str(common["device"])

    checkpoint_path = args.checkpoint or deploy.get("checkpoint_path")
    if not checkpoint_path:
        print("[ERROR] No checkpoint_path specified in config or --checkpoint argument.")
        sys.exit(1)

    chkpt_file = Path(checkpoint_path)
    if not chkpt_file.exists():
        # Try relative to PVP_ROOT
        chkpt_file = PVP_ROOT / checkpoint_path
    if not chkpt_file.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {chkpt_file}")

    # ── Build dummy env for space definition ─────────────────────────────────
    obs_shape = (resize_hw[0], resize_hw[1], stack_n * 4)

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
            self.action_space      = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        def reset(self, **kw): return np.zeros(obs_shape, np.uint8), {}
        def step(self, a):     return np.zeros(obs_shape, np.uint8), 0.0, False, False, {}
        def seed(self, seed=None): return [seed]

    dummy_env = _DummyEnv()
    model     = PVPTD3.load(str(chkpt_file), env=dummy_env, device=device)
    model.policy.set_training_mode(False)

    # ── ROS2 setup ────────────────────────────────────────────────────────────
    rclpy.init()
    node: Optional[DeployCache] = None
    gui:  Optional[DeployGUI]   = None

    try:
        obs_cfg = ObsCfg(resize_hw=resize_hw, stack_n=stack_n, depth_max_m=depth_max_m)
        node    = DeployCache(obs_cfg)

        gui = DeployGUI(checkpoint_path=str(chkpt_file))
        gui.start_thread()

        _inference_loop(
            node=node, model=model, gui=gui,
            dt=1.0 / hz, max_lin=max_lin, max_ang=max_ang,
        )

    except KeyboardInterrupt:
        print("[Interrupted] Shutting down…")

    finally:
        if gui is not None:
            try:
                gui.root.quit()
                gui.root.destroy()
            except Exception:
                pass
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
