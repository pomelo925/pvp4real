#!/usr/bin/env python3
"""offline/train.py — Offline training from recorded ROS2 bag (.mcap).

Modes (controlled by config training.is_resume_training):
  False — Read bag → fill buffers → train from scratch.
  True  — Must provide checkpoint.resume_from (.zip).
           Optionally provide buffer.resume_from (run dir path).
           Then fill buffers from bag and continue training.

Checkpoint/buffer naming convention:
  models/offline/0001/chkpt-250.zip          # saved every save_every steps
  models/offline/0001/buffer_human-250.pkl   # human buffer snapshot
  models/offline/0001/buffer_replay-250.pkl  # replay buffer snapshot
  models/offline/0001/chkpt-1560f.zip        # final / interrupted checkpoint
  models/offline/0001/buffer_human-1560f.pkl
  models/offline/0001/buffer_replay-1560f.pkl

Usage:
  python train.py                         # uses config.yaml in same directory
  python train.py --bag path/to/bag_dir   # override bag path
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import numpy as np
import yaml
import cv2

# ── ROS2 serialization (no context needed) ────────────────────────────────────
from rclpy.serialization import deserialize_message
import rosbag2_py

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

# ── ML / gym ──────────────────────────────────────────────────────────────────
import gymnasium as gym
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.utils import configure_logger


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_CLASS_MAP: Dict[str, Any] = {
    "/camera/camera/color/image_raw":                   Image,
    "/camera/camera/aligned_depth_to_color/image_raw":  Image,
    "/camera/camera/camera_info":                       CameraInfo,
    "/stretch/cmd_vel_teleop":                          Twist,
    "/stretch/cmd_vel":                                 Twist,
    "/stretch/is_teleop":                               Bool,
    "/tf":                                              TFMessage,
    "/tf_static":                                       TFMessage,
}

PVP_ROOT = Path(__file__).parent.parent.parent.parent  # pvp4real/pvp4real/


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_next_run_dir(base_path: Path) -> Path:
    """Return next auto-incremented 4-digit subdirectory under base_path."""
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(d.name) for d in base_path.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4
    )
    next_num = (existing[-1] + 1) if existing else 1
    return base_path / f"{next_num:04d}"


# ─────────────────────────────────────────────────────────────────────────────
# Bag reading
# ─────────────────────────────────────────────────────────────────────────────

def read_bag(bag_dir: Path) -> Dict[str, List[Tuple[int, Any]]]:
    """Read a .mcap bag directory and return {topic: [(ts_ns, msg), ...]}."""
    reader = rosbag2_py.SequentialReader()
    storage_opts = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    conv_opts = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_opts, conv_opts)

    # Build topic→type map from bag metadata
    topic_type_map: Dict[str, str] = {}
    for meta in reader.get_all_topics_and_types():
        topic_type_map[meta.name] = meta.type

    # ── Compressed image guard ────────────────────────────────────────────────
    # Train expects raw Image topics. If bag_c (compressed) is passed directly,
    # fail early with a clear message rather than silently producing empty frames.
    compressed_topics = [t for t in topic_type_map if "compress" in t.lower()]
    if compressed_topics:
        raise ValueError(
            f"\n[ERROR] Bag contains compressed image topics:\n"
            + "".join(f"  {t}\n" for t in sorted(compressed_topics))
            + "\nPlease decompress first:\n"
            + "  python3 pvp4real/scripts/data/decompress.py\n"
            + "Then point --bag at the resulting bag/ directory (not bag_c/)."
        )

    messages: Dict[str, List[Tuple[int, Any]]] = {}
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        msg_cls = TOPIC_CLASS_MAP.get(topic)
        if msg_cls is None:
            continue
        try:
            msg = deserialize_message(data, msg_cls)
        except Exception:
            continue
        messages.setdefault(topic, []).append((ts_ns, msg))

    # Sort each topic by timestamp
    for topic in messages:
        messages[topic].sort(key=lambda x: x[0])

    return messages


def _bgr8_to_rgb_uint8(msg: Image, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Convert Image msg (bgr8 or rgb8) to resized RGB uint8 (H, W, 3)."""
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ("bgr8", "bgr"):
        arr = arr[:, :, ::-1].copy()  # BGR → RGB
    elif msg.encoding in ("rgba8",):
        arr = arr[:, :, :3]
    arr = cv2.resize(arr, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
    return arr.astype(np.uint8)


def _depth_to_uint8(msg: Image, depth_max_m: float, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Convert depth Image msg to resized uint8 (H, W, 1)."""
    if msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        depth_m = arr.astype(np.float32) / 1000.0
    else:
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        depth_m = arr
    depth_m = cv2.resize(depth_m, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_NEAREST)
    d_clip = np.clip(depth_m, 0.0, depth_max_m)
    d_u8 = (d_clip / depth_max_m * 255.0).astype(np.uint8)
    return d_u8    # (H, W)


def _closest_before(series: List[Tuple[int, Any]], ts_ns: int) -> Optional[Any]:
    """Return the message in series with the largest ts_ns <= ts_ns."""
    result = None
    for t, msg in series:
        if t <= ts_ns:
            result = msg
        else:
            break
    return result


def build_transitions(
    messages: Dict[str, List[Tuple[int, Any]]],
    stack_n: int,
    resize_hw: Tuple[int, int],
    depth_max_m: float,
    max_lin: float,
    max_ang: float,
) -> List[Dict]:
    """
    Build a list of transition dicts from the bag messages.

    Each dict:  obs, next_obs, action, reward, done, info
    """
    rgb_topic   = "/camera/camera/color/image_raw"
    depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
    # Use cmd_vel_teleop as the timestamp anchor: cmd_vel is only published
    # when the Authority node is running (online-only), while cmd_vel_teleop
    # is always published by the joystick driver during recording.
    cmd_topic   = "/stretch/cmd_vel_teleop"
    tel_topic   = "/stretch/is_teleop"

    cmd_series  = messages.get(cmd_topic, [])
    rgb_series  = messages.get(rgb_topic, [])
    depth_series = messages.get(depth_topic, [])
    tel_series  = messages.get(tel_topic, [])

    if not cmd_series:
        raise ValueError(f"No {cmd_topic} messages found in bag.")

    # Build per-tick frames (rgb, depth, cmd_vel, is_teleop)
    frames: List[Dict] = []
    for ts, cmd_msg in cmd_series:
        rgb   = _closest_before(rgb_series, ts)
        depth = _closest_before(depth_series, ts)
        tel   = _closest_before(tel_series, ts)

        if rgb is None or depth is None:
            continue

        rgb_u8   = _bgr8_to_rgb_uint8(rgb, resize_hw)           # H,W,3
        depth_u8 = _depth_to_uint8(depth, depth_max_m, resize_hw)  # H,W

        is_teleop = bool(tel.data) if tel is not None else False

        frames.append({
            "rgb":       rgb_u8,
            "depth":     depth_u8,
            "cmd_vel":   np.array([cmd_msg.linear.x, cmd_msg.angular.z], dtype=np.float32),
            "is_teleop": is_teleop,
        })

    print(f"Bag frames extracted: {len(frames)}")
    if len(frames) < stack_n + 1:
        raise ValueError(f"Not enough frames ({len(frames)}) to build stacks of {stack_n}.")

    # Build stacked observations from rolling window
    def build_obs(frame_window: List[Dict]) -> np.ndarray:
        rgb_list   = [f["rgb"]   for f in frame_window]
        depth_list = [f["depth"] for f in frame_window]
        rgb_cat   = np.concatenate(rgb_list, axis=2)          # H,W,(3*N)
        depth_cat = np.stack(depth_list, axis=2)               # H,W,N
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)     # H,W,(4*N)
        return obs.astype(np.uint8)

    def _normalize_action(vw: np.ndarray) -> np.ndarray:
        return np.array([vw[0] / max_lin, vw[1] / max_ang], dtype=np.float32).clip(-1.0, 1.0)

    transitions = []
    prev_is_teleop = False

    for i in range(stack_n - 1, len(frames) - 1):
        window_t   = frames[i - stack_n + 1 : i + 1]       # stack_n frames ending at i
        window_t1  = frames[i - stack_n + 2 : i + 2]       # stack_n frames ending at i+1

        obs      = build_obs(window_t)
        next_obs = build_obs(window_t1)

        f = frames[i]
        action     = _normalize_action(f["cmd_vel"])
        is_teleop  = f["is_teleop"]
        takeover_start = is_teleop and (not prev_is_teleop)
        prev_is_teleop = is_teleop

        info = {
            "takeover":       float(is_teleop),
            "takeover_start": float(takeover_start),
            "takeover_cost":  0.0,
            "raw_action":     action.astype(np.float32),
        }

        transitions.append({
            "obs":      obs,
            "next_obs": next_obs,
            "action":   action,
            "reward":   0.0,
            "done":     False,
            "info":     info,
        })

    print(f"Transitions built: {len(transitions)}")
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Dummy env for space definition
# ─────────────────────────────────────────────────────────────────────────────

class OfflineDummyEnv(gym.Env):
    """Minimal env to define obs / action spaces for offline PVPTD3 init."""

    def __init__(self, obs_shape: Tuple, action_dim: int = 2):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 0.0, False, False, {}

    def seed(self, seed=None):
        return [seed]


# ─────────────────────────────────────────────────────────────────────────────
# Buffer population
# ─────────────────────────────────────────────────────────────────────────────

def fill_buffers(model: PVPTD3, transitions: List[Dict]) -> None:
    """Add offline transitions into model's human_data_buffer / replay_buffer."""
    skipped = 0
    for t in transitions:
        obs      = t["obs"].transpose(2, 0, 1)[np.newaxis]       # (H,W,C) -> (1,C,H,W)
        next_obs = t["next_obs"].transpose(2, 0, 1)[np.newaxis]  # (H,W,C) -> (1,C,H,W)
        action   = t["action"][np.newaxis]     # (1, 2)
        reward   = np.array([t["reward"]])
        done     = np.array([float(t["done"])])
        info     = t["info"]

        if info["takeover"] or info["takeover_start"]:
            buf = model.human_data_buffer
        else:
            buf = model.replay_buffer

        try:
            buf.add(obs, next_obs, action, reward, done, [info])
        except Exception as e:
            if skipped == 0:
                import traceback
                print(f"[fill_buffers] First skip error: {e}")
                traceback.print_exc()
            skipped += 1

    human_n  = model.human_data_buffer.pos
    replay_n = model.replay_buffer.pos
    print(f"Buffers filled — human: {human_n}, replay: {replay_n}, skipped: {skipped}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / buffer save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: PVPTD3, run_dir: Path, step: int, final: bool = False) -> None:
    suffix = f"{step}f" if final else str(step)
    chkpt_path = run_dir / f"chkpt-{suffix}.zip"
    model.save(str(chkpt_path))
    print(f"Checkpoint saved: {chkpt_path}")


def save_buffers(model: PVPTD3, run_dir: Path, step: int, final: bool = False) -> None:
    suffix = f"{step}f" if final else str(step)
    human_path  = run_dir / f"buffer_human-{suffix}.pkl"
    replay_path = run_dir / f"buffer_replay-{suffix}.pkl"
    model.save_replay_buffer(str(human_path), str(replay_path))
    print(f"Buffers saved: {human_path.name}, {replay_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PVP4Real offline training from ROS2 bag.")
    parser.add_argument("--bag",    type=str, default=None, help="Path to .mcap bag directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to override config.yaml.")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    common   = cfg["common"]
    train_c  = cfg["training"]
    pvp_c    = train_c["pvp"]
    ckpt_c   = train_c["checkpoint"]
    buf_c    = train_c["buffer"]

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

    resume_chkpt = ckpt_c.get("resume_from")  # path to .zip
    resume_buf   = buf_c.get("resume_from")   # path to run dir

    # ── Resolve bag path ──────────────────────────────────────────────────────
    # Priority: --bag CLI > config.bag_path > auto-detect latest <run_dir>/bag
    cfg_bag = train_c.get("bag_path")
    if args.bag:
        bag_dir = Path(args.bag)
    elif cfg_bag:
        p = Path(cfg_bag)
        bag_dir = p if p.is_absolute() else PVP_ROOT / p
    else:
        # Auto-detect: find most recent run dir under dataset_base_path that has a bag/ subdir.
        dataset_base = train_c.get("dataset_base_path", "datasets/offline/")
        rec_base = PVP_ROOT / dataset_base
        candidates = sorted(
            [d / "bag" for d in rec_base.iterdir()
             if d.is_dir() and d.name.isdigit() and (d / "bag").exists()],
            reverse=True,
        )
        if not candidates:
            print(f"[ERROR] No decompressed bag/ found under {rec_base}.")
            print("        Run decompress.py first, or set training.bag_path in config.")
            sys.exit(1)
        bag_dir = candidates[0]
        print(f"[INFO] bag_path not set, auto-detected: {bag_dir}")

    if not bag_dir.exists():
        print(f"[ERROR] Bag directory not found: {bag_dir}")
        sys.exit(1)
    print(f"Reading bag: {bag_dir}")

    # ── Create run directory ──────────────────────────────────────────────────
    model_base = PVP_ROOT / ckpt_c["saved_model_path"]
    run_dir    = get_next_run_dir(model_base)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Read bag and build transitions ────────────────────────────────────────
    messages    = read_bag(bag_dir)
    transitions = build_transitions(
        messages, stack_n, resize_hw, depth_max_m, max_lin, max_ang
    )

    # ── Build env (space definition only) ────────────────────────────────────
    obs_shape = (resize_hw[0], resize_hw[1], stack_n * 4)
    env = OfflineDummyEnv(obs_shape=obs_shape)

    # ── Create / load model ───────────────────────────────────────────────────
    trained_steps = 0

    if is_resume and resume_chkpt:
        chkpt_path = Path(resume_chkpt)
        if not chkpt_path.exists():
            print(f"[ERROR] Checkpoint not found: {chkpt_path}")
            sys.exit(1)
        print(f"Resuming from checkpoint: {chkpt_path}")
        model = PVPTD3.load(
            str(chkpt_path),
            env=env,
            verbose=1,
            device=device,
            learning_rate=float(pvp_c["learning_rate"]),
        )
        # Try to extract step number from filename
        import re
        m = re.search(r"chkpt-(\d+)", chkpt_path.name)
        if m:
            trained_steps = int(m.group(1))
            print(f"Resuming from step {trained_steps}.")
    else:
        print("Creating new model from scratch.")
        model = PVPTD3(
            True,  # use_balance_sample
            float(pvp_c["q_value_bound"]),
            "CnnPolicy",
            env,
            seed=seed,
            verbose=1,
            device=device,
            buffer_size=buf_size,
            learning_starts=0,          # offline: all data already in buffer
            batch_size=batch_size,
            train_freq=(1, "step"),
            gradient_steps=1,
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

    # ── Optionally load buffers ───────────────────────────────────────────────
    if is_resume and resume_buf:
        buf_run_dir = Path(resume_buf)
        human_pkls  = sorted(buf_run_dir.glob("buffer_human-*.pkl"), reverse=True)
        replay_pkls = sorted(buf_run_dir.glob("buffer_replay-*.pkl"), reverse=True)
        if human_pkls and replay_pkls:
            print(f"Loading buffers from {buf_run_dir.name}: {human_pkls[0].name}, {replay_pkls[0].name}")
            model.load_replay_buffer(str(human_pkls[0]), str(replay_pkls[0]))
        else:
            print(f"[WARN] No buffer .pkl files found in {buf_run_dir}. Starting with empty buffers.")

    # ── Fill buffers from bag ─────────────────────────────────────────────────
    print("Filling buffers from bag data…")
    fill_buffers(model, transitions)

    # Make sure model won't skip training (learning_starts threshold)
    model.learning_starts = 0
    model.num_timesteps   = trained_steps + len(transitions)

    # ── Initialize logger (required before calling model.train() directly) ────
    model.set_logger(configure_logger(verbose=1, tensorboard_log=str(run_dir), tb_log_name="offline_pvptd3", reset_num_timesteps=True))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting offline training for {total_steps} gradient steps…")
    step = 0
    save_interval = min(chkpt_save_every, buf_save_every)

    try:
        while step < total_steps:
            chunk = min(save_interval, total_steps - step)
            model.train(batch_size=batch_size, gradient_steps=chunk)
            step += chunk
            model.num_timesteps += chunk

            abs_step = trained_steps + step
            print(f"  Step {abs_step} ({step}/{total_steps})")

            if ckpt_c["is_saved"] and step % chkpt_save_every == 0:
                save_checkpoint(model, run_dir, abs_step)
            if buf_c.get("save_every") and step % buf_save_every == 0:
                save_buffers(model, run_dir, abs_step)

    except KeyboardInterrupt:
        abs_step = trained_steps + step
        print(f"\n[Interrupted at step {abs_step}] Saving final checkpoint…")
        save_checkpoint(model, run_dir, abs_step, final=True)
        save_buffers(model, run_dir, abs_step, final=True)
        return

    # ── Final save ────────────────────────────────────────────────────────────
    abs_step = trained_steps + step
    save_checkpoint(model, run_dir, abs_step, final=True)
    save_buffers(model, run_dir, abs_step, final=True)
    print(f"Offline training complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
