#!/usr/bin/env python3
"""pvp.hitl.py — PVP4Real online HITL training (pvp4real container).

Publishes novice/policy command to /pvp/novice_cmd_vel.
Observes executed behaviour via /stretch/cmd_vel.
Uses /stretch/is_teleop + /stretch/cmd_vel_teleop for HITL signals.

Checkpoint / buffer naming convention
--------------------------------------
  models/online/0001/chkpt-250.zip
  models/online/0001/buffer_human-250.pkl
  models/online/0001/buffer_replay-250.pkl
  models/online/0001/chkpt-1560f.zip        ← final / interrupted
  models/online/0001/buffer_human-1560f.pkl
  models/online/0001/buffer_replay-1560f.pkl
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
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

PVP_ROOT = Path(__file__).parent.parent.parent.parent  # pvp4real/pvp4real/


def _now() -> float:
    return time.monotonic()


def _twist_to_vw(msg: Twist) -> np.ndarray:
    return np.array([float(msg.linear.x), float(msg.angular.z)], dtype=np.float32)


def _vw_to_twist(vw: np.ndarray) -> Twist:
    t = Twist()
    t.linear.x = float(vw[0])
    t.angular.z = float(vw[1])
    return t


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_next_run_dir(base_path: Path) -> Path:
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(d.name) for d in base_path.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4
    )
    next_num = (existing[-1] + 1) if existing else 1
    return base_path / f"{next_num:04d}"


def save_checkpoint(model: PVPTD3, run_dir: Path, step: int, final: bool = False, custom_path: str = None) -> None:
    if custom_path:
        path = Path(custom_path)
    else:
        suffix = f"{step}f" if final else str(step)
        path = run_dir / f"chkpt-{suffix}.zip"
    model.save(str(path))
    return path


def save_buffers(model: PVPTD3, run_dir: Path, step: int, final: bool = False, custom_human_path: str = None, custom_replay_path: str = None) -> None:
    if custom_human_path and custom_replay_path:
        h_path = Path(custom_human_path)
        r_path = Path(custom_replay_path)
    else:
        suffix = f"{step}f" if final else str(step)
        h_path = run_dir / f"buffer_human-{suffix}.pkl"
        r_path = run_dir / f"buffer_replay-{suffix}.pkl"
    model.save_replay_buffer(str(h_path), str(r_path))


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 cache node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObsCfg:
    resize_hw: Tuple[int, int] = (84, 84)
    stack_n: int = 5
    depth_max_m: float = 5.0


class HITLCache(Node):
    """Caches latest ROS messages and maintains stacked RGB-D frames."""

    def __init__(self, obs_cfg: ObsCfg):
        super().__init__("pvp_hitl_cache")
        self.obs_cfg = obs_cfg

        self._is_teleop: bool = False
        self._teleop_twist: Twist = Twist()
        self._exec_twist: Twist = Twist()

        self._rgb_stack:   Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._prev_takeover: bool = False

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

        self.create_subscription(Image, "/camera/camera/color/image_raw",                      self._on_rgb,          qos_sensor)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",     self._on_depth,        qos_sensor)
        self.create_subscription(Bool,  "/stretch/is_teleop",                                  self._on_is_teleop,    qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop",                             self._on_teleop_twist,  qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel",                                    self._on_exec_twist,   qos_default)

        self.novice_pub = self.create_publisher(Twist, "/pvp/novice_cmd_vel", qos_default)
        self.is_teleop_pub = self.create_publisher(Bool, "/stretch/is_teleop", qos_default)

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

    def _on_is_teleop(self, msg: Bool) -> None:
        self._is_teleop = bool(msg.data)

    def _on_teleop_twist(self, msg: Twist) -> None:
        self._teleop_twist = msg

    def _on_exec_twist(self, msg: Twist) -> None:
        self._exec_twist = msg

    # ── State access ──────────────────────────────────────────────────────────

    def stacks_ready(self) -> bool:
        return (len(self._rgb_stack) == self.obs_cfg.stack_n and
                len(self._depth_stack) == self.obs_cfg.stack_n)

    def get_takeover_and_start(self) -> Tuple[bool, bool]:
        takeover = self._is_teleop
        start = takeover and (not self._prev_takeover)
        self._prev_takeover = takeover
        return takeover, start

    def get_human_vw(self) -> np.ndarray:
        return _twist_to_vw(self._teleop_twist)

    def get_exec_vw(self) -> np.ndarray:
        return _twist_to_vw(self._exec_twist)

    def publish_novice_vw(self, vw: np.ndarray) -> None:
        self.novice_pub.publish(_vw_to_twist(vw))

    def publish_is_teleop(self, is_teleop: bool) -> None:
        msg = Bool()
        msg.data = is_teleop
        self.is_teleop_pub.publish(msg)

    def build_obs_uint8(self) -> np.ndarray:
        if not self.stacks_ready():
            raise RuntimeError("Stacks not ready")
        rgb_list   = list(self._rgb_stack)
        depth_list = list(self._depth_stack)
        depth_max  = float(self.obs_cfg.depth_max_m)
        depth_u8 = [
            (np.clip(d, 0.0, depth_max) / depth_max * 255.0).astype(np.uint8)
            for d in depth_list
        ]
        rgb_cat   = np.concatenate(rgb_list,  axis=2)   # H,W,(3N)
        depth_cat = np.stack(depth_u8,         axis=2)   # H,W,N
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)
        return obs.astype(np.uint8, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium env (HITL)
# ─────────────────────────────────────────────────────────────────────────────

class StretchHITLEnv(gym.Env):

    def __init__(self, node: HITLCache, dt: float, max_lin: float, max_ang: float):
        super().__init__()
        self.node    = node
        self.dt      = float(dt)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self._last_step_t = _now()

        h, w = node.obs_cfg.resize_hw
        c    = node.obs_cfg.stack_n * 4
        self.observation_space = gym.spaces.Box(0, 255, shape=(h, w, c), dtype=np.uint8)
        self.action_space      = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _action_to_vw(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a.astype(np.float32), -1.0, 1.0)
        return np.array([a[0] * self.max_lin, a[1] * self.max_ang], dtype=np.float32)

    def _vw_to_action(self, vw: np.ndarray) -> np.ndarray:
        return np.array([vw[0] / self.max_lin, vw[1] / self.max_ang], dtype=np.float32).clip(-1.0, 1.0)

    def seed(self, seed=None):
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        start = _now()
        while rclpy.ok() and (not self.node.stacks_ready()):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if _now() - start > 10.0:
                raise TimeoutError("Waiting for RGB-D stacks timed out.")
        return self.node.build_obs_uint8(), {}

    def step(self, action: np.ndarray):
        elapsed = _now() - self._last_step_t
        if elapsed < self.dt:
            end_t = _now() + (self.dt - elapsed)
            while rclpy.ok() and _now() < end_t:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        self._last_step_t = _now()

        obs = self.node.build_obs_uint8()

        takeover, takeover_start = self.node.get_takeover_and_start()
        exec_vw = self.node.get_exec_vw()

        novice_action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        self.node.publish_novice_vw(self._action_to_vw(novice_action))

        t_end = _now() + 0.02
        while rclpy.ok() and _now() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        next_obs   = self.node.build_obs_uint8() if self.node.stacks_ready() else obs
        raw_action = self._vw_to_action(exec_vw)

        info = {
            "takeover":       float(takeover),
            "takeover_start": float(takeover_start),
            "takeover_cost":  0.0,
            "raw_action":     raw_action.astype(np.float32),
        }
        return next_obs, 0.0, False, False, info


# ─────────────────────────────────────────────────────────────────────────────
# tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────

class HITLControlGUI:

    def __init__(self, node: HITLCache, total_steps: int, run_dir: Path, save_every: int):
        self.node         = node
        self.total_steps  = total_steps
        self.run_dir      = run_dir
        self.save_every   = save_every
        self.current_step = 0
        self.quit_requested = False
        self.is_teleop    = False
        self.training_started = False
        self.training_stopped = False

        self.root = tk.Tk()
        self.root.title("PVP4Real — Online HITL Training")
        self.root.geometry("520x370")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}
        r = self.root

        ttk.Label(r, text="PVP4Real Online HITL Training", font=("Arial", 15, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(14, 6)
        )

        # Status frame
        sf = ttk.LabelFrame(r, text="Training Status", padding=10)
        sf.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        ttk.Label(sf, text="Run dir:").grid(row=0, column=0, sticky="w")
        ttk.Label(sf, text=str(self.run_dir), font=("Arial", 9), foreground="gray").grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(sf, text="Steps:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._steps_lbl = ttk.Label(sf, text="0", font=("Arial", 12, "bold"))
        self._steps_lbl.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(sf, text="Total:").grid(row=2, column=0, sticky="w")
        ttk.Label(sf, text=str(self.total_steps)).grid(row=2, column=1, sticky="w", padx=6)

        self._pbar = ttk.Progressbar(sf, length=460, mode="determinate", maximum=self.total_steps)
        self._pbar.grid(row=3, column=0, columnspan=2, pady=(8, 2))
        self._pct_lbl = ttk.Label(sf, text="0.0%")
        self._pct_lbl.grid(row=4, column=0, columnspan=2)

        sf.columnconfigure(1, weight=1)

        # Mode frame
        mf = ttk.LabelFrame(r, text="Mode Control", padding=10)
        mf.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        self._mode_btn = ttk.Button(mf, text="Switch to Gamepad Mode", width=26, command=self._on_mode_switch)
        self._mode_btn.grid(row=0, column=0, padx=8)
        self._mode_lbl = ttk.Label(mf, text="Navigation (Policy)", font=("Arial", 10, "bold"), foreground="green")
        self._mode_lbl.grid(row=0, column=1, padx=8)


        # Start/Stop/Save & Quit buttons
        btn_frame = ttk.Frame(r)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=14)
        self._start_btn = ttk.Button(btn_frame, text="Start Training", width=16, command=self._on_start)
        self._start_btn.grid(row=0, column=0, padx=6)
        self._stop_btn = ttk.Button(btn_frame, text="Stop Training", width=16, command=self._on_stop, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text="Save & Quit", width=16, command=self._on_quit).grid(row=0, column=2, padx=6)

        self._status_lbl = ttk.Label(r, text="Waiting to start...", font=("Arial", 10, "italic"), foreground="blue")
        self._status_lbl.grid(row=4, column=0, columnspan=2)
    def _on_start(self) -> None:
        self.training_started = True
        self.training_stopped = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_lbl.config(text="Training in progress...", foreground="green")

    def _on_stop(self) -> None:
        self.training_stopped = True
        self._stop_btn.config(state="disabled")
        self._status_lbl.config(text="Training stopped. You may Save & Quit.", foreground="red")

        r.columnconfigure(0, weight=1)

    def _on_mode_switch(self) -> None:
        import subprocess
        self.is_teleop = not self.is_teleop
        self.node.publish_is_teleop(self.is_teleop)
        if self.is_teleop:
            self._mode_btn.config(text="Switch to Navigation Mode")
            self._mode_lbl.config(text="Gamepad (Teleop)", foreground="orange")
            # 切到 gamepad mode
            try:
                subprocess.Popen([
                    "ros2", "service", "call", "/switch_to_gamepad_mode", "std_srvs/srv/Trigger", "{}"
                ])
            except Exception as e:
                print(f"[WARN] Failed to call /switch_to_gamepad_mode: {e}")
        else:
            self._mode_btn.config(text="Switch to Gamepad Mode")
            self._mode_lbl.config(text="Navigation (Policy)", foreground="green")
            # 切到 navigation mode
            try:
                subprocess.Popen([
                    "ros2", "service", "call", "/switch_to_navigation_mode", "std_srvs/srv/Trigger", "{}"
                ])
            except Exception as e:
                print(f"[WARN] Failed to call /switch_to_navigation_mode: {e}")

    def _on_quit(self) -> None:
        self.quit_requested = True
        self.root.quit()

    def update_steps(self, steps: int) -> None:
        """Thread-safe: schedules the UI update on the main (Tk) thread."""
        self.root.after(0, self._do_update_steps, steps)

    def _do_update_steps(self, steps: int) -> None:
        self.current_step = steps
        self._steps_lbl.config(text=str(steps))
        self._pbar["value"] = steps
        pct = (steps / self.total_steps * 100) if self.total_steps > 0 else 0
        self._pct_lbl.config(text=f"{pct:.1f}%")

    def wait_for_start(self):
        # 阻塞直到按下 Start
        while not self.training_started and not self.quit_requested:
            self.root.update()
            self.root.after(100)

    def start_training_thread(self, target, *args, **kwargs) -> threading.Thread:
        """Run *target* in a background thread; mainloop stays on the calling (main) thread."""
        t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PVP4Real online HITL training.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    common  = cfg["common"]
    train_c = cfg["training"]
    pvp_c   = train_c["pvp"]
    ckpt_c  = train_c["checkpoint"]
    buf_c   = train_c["buffer"]

    hz           = float(common["hz"])
    max_lin      = float(common["max_lin"])
    max_ang      = float(common["max_ang"])
    depth_max_m  = float(common["depth_max_m"])
    stack_n      = int(common["stack_n"])
    resize_hw    = (int(common["resize"]["height"]), int(common["resize"]["width"]))
    seed         = int(common["seed"])
    device       = str(common["device"])

    is_resume       = bool(train_c["is_resume_training"])
    total_steps     = int(train_c["total_steps"])
    learning_starts = int(train_c["learning_starts"])
    batch_size      = int(train_c["batch_size"])
    log_interval    = int(train_c["log_interval"])

    chkpt_save_every = int(ckpt_c["save_every"])
    buf_save_every   = int(buf_c["save_every"])
    buf_size         = int(buf_c["size"])

    resume_chkpt = ckpt_c.get("resume_from")
    resume_buf   = buf_c.get("resume_from")

    # ── Run dir ───────────────────────────────────────────────────────────────
    model_base = PVP_ROOT / ckpt_c["saved_model_path"]
    run_dir    = get_next_run_dir(model_base)
    run_dir.mkdir(parents=True, exist_ok=True)
    # 新增：讀取 config.yaml 的自訂路徑欄位
    save_chkpt_path = ckpt_c.get("save_chkpt_path")
    save_buffer_human_path = buf_c.get("save_buffer_human_path")
    save_buffer_replay_path = buf_c.get("save_buffer_replay_path")

    rclpy.init()
    node: Optional[HITLCache] = None
    gui:  Optional[HITLControlGUI] = None

    try:
        obs_cfg = ObsCfg(resize_hw=resize_hw, stack_n=stack_n, depth_max_m=depth_max_m)
        node    = HITLCache(obs_cfg)

        dt  = 1.0 / hz
        env = StretchHITLEnv(node=node, dt=dt, max_lin=max_lin, max_ang=max_ang)

        trained = 0

        # ── Create / resume model ──────────────────────────────────────────────
        if is_resume and resume_chkpt:
            chkpt_path = Path(resume_chkpt)
            if not chkpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {chkpt_path}")
            node.get_logger().info(f"Resuming from checkpoint: {chkpt_path}")
            model = PVPTD3.load(
                str(chkpt_path), env=env, verbose=1, device=device,
                learning_rate=float(pvp_c["learning_rate"]),
            )
            m = re.search(r"chkpt-(\d+)", chkpt_path.name)
            if m:
                trained = int(m.group(1))
            learning_starts = 0
        else:
            model = PVPTD3(
                True,
                float(pvp_c["q_value_bound"]),
                "CnnPolicy",
                env,
                seed=seed,
                verbose=1,
                device=device,
                buffer_size=buf_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                train_freq=(int(pvp_c["train_freq"]), "step"),
                gradient_steps=int(pvp_c["gradient_steps"]),
                gamma=float(pvp_c["gamma"]),
                tau=float(pvp_c["tau"]),
                learning_rate=float(pvp_c["learning_rate"]),
                bc_loss_weight=float(pvp_c["bc_loss_weight"]),
                with_human_proxy_value_loss=str(pvp_c["with_human_proxy_value_loss"]),
                with_agent_proxy_value_loss=str(pvp_c["with_agent_proxy_value_loss"]),
                only_bc_loss=str(pvp_c["only_bc_loss"]),
                add_bc_loss=str(pvp_c["add_bc_loss"]),
                adaptive_batch_size="False",
            )

        # ── Optionally load buffers ────────────────────────────────────────────
        if is_resume and resume_buf:
            buf_run = Path(resume_buf)
            h_pkls  = sorted(buf_run.glob("buffer_human-*.pkl"),  reverse=True)
            r_pkls  = sorted(buf_run.glob("buffer_replay-*.pkl"), reverse=True)
            if h_pkls and r_pkls:
                node.get_logger().info(f"Loading buffers: {h_pkls[0].name}, {r_pkls[0].name}")
                model.load_replay_buffer(str(h_pkls[0]), str(r_pkls[0]))

        _ = env.reset(seed=seed)

        node.get_logger().info(
            f"HITL training: {total_steps} steps @ {hz}Hz  |  run_dir: {run_dir}"
        )

        # ── GUI ────────────────────────────────────────────────────────────────

        gui = HITLControlGUI(node=node, total_steps=total_steps,
                             run_dir=run_dir, save_every=chkpt_save_every)
        gui.update_steps(trained)

        # 等待使用者按下 Start
        gui.wait_for_start()

        # ── Training loop (background thread; mainloop runs on main thread) ───
        save_interval = min(chkpt_save_every, buf_save_every)
        remaining = total_steps - trained

        def _training_loop() -> None:
            nonlocal trained, remaining
            try:
                while rclpy.ok() and remaining > 0 and not gui.quit_requested and not gui.training_stopped:
                    chunk = min(save_interval, remaining)
                    model.learn(
                        total_timesteps=chunk,
                        reset_num_timesteps=False,
                        log_interval=log_interval,
                    )
                    trained   += chunk
                    remaining -= chunk
                    gui.update_steps(trained)

                    if ckpt_c["is_saved"] and trained % chkpt_save_every == 0:
                        save_checkpoint(model, run_dir, trained, custom_path=save_chkpt_path)
                    if trained % buf_save_every == 0:
                        save_buffers(model, run_dir, trained, custom_human_path=save_buffer_human_path, custom_replay_path=save_buffer_replay_path)

                    node.get_logger().info(f"Step {trained}/{total_steps}")

                # Final save
                save_checkpoint(model, run_dir, trained, final=True)
                save_buffers(model, run_dir, trained, final=True, custom_human_path=save_buffer_human_path, custom_replay_path=save_buffer_replay_path)
                node.get_logger().info(f"Training complete. Saved in: {run_dir}")
            finally:
                # Tell the GUI mainloop to exit when training finishes
                gui.root.after(0, gui.root.quit)

        gui.start_training_thread(_training_loop)
        gui.root.mainloop()  # must run on the main thread

    except KeyboardInterrupt:
        node.get_logger().info("[Interrupted] Saving checkpoint…")
        if "model" in dir() and "trained" in dir() and trained > 0:
            save_checkpoint(model, run_dir, trained, final=True, custom_path=save_chkpt_path)
            save_buffers(model, run_dir, trained, final=True, custom_human_path=save_buffer_human_path, custom_replay_path=save_buffer_replay_path)

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
