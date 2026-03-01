#!/usr/bin/env python3
"""rosbag.py — ROS2 bag recorder with tkinter GUI.

Loads config.yaml from the same directory (record/config.yaml).
Supports three recording modes: offline / online / deploy.
Each mode writes to a separate auto-incremented subdirectory, e.g.:
  models/offline/0001/bag/   (mcap format)
  models/online/0002/bag/

GUI elements:
  Mode selector  — choose offline / online / deploy before starting
  [Record]       — start a new recording session
  [Stop]         — stop the current recording (keep bag file)
  [Quit]         — stop recording if active, then exit
  Ticks display  — dynamically updated recorded message count
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.serialization import serialize_message

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

import rosbag2_py


# ─────────────────────────────────────────────────────────────────────────────
# Topic type map
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_TYPE_MAP: Dict[str, Tuple[str, Any]] = {
    # camera info
    "/camera/camera/color/camera_info":                                  ("sensor_msgs/msg/CameraInfo",     CameraInfo),
    "/camera/camera/aligned_depth_to_color/camera_info":                 ("sensor_msgs/msg/CameraInfo",     CameraInfo),
    # raw images
    "/camera/camera/color/image_raw":                                    ("sensor_msgs/msg/Image",          Image),
    "/camera/camera/aligned_depth_to_color/image_raw":                   ("sensor_msgs/msg/Image",          Image),
    # compressed images
    "/camera/camera/color/image_raw/compressed":                         ("sensor_msgs/msg/CompressedImage", CompressedImage),
    # "/camera/camera/aligned_depth_to_color/image_raw/compressed":        ("sensor_msgs/msg/CompressedImage", CompressedImage),
    "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth":   ("sensor_msgs/msg/CompressedImage", CompressedImage),
    # control
    "/stretch/cmd_vel_teleop":                                           ("geometry_msgs/msg/Twist",        Twist),
    "/stretch/cmd_vel":                                                  ("geometry_msgs/msg/Twist",        Twist),
    "/stretch/is_teleop":                                                ("std_msgs/msg/Bool",              Bool),
    "/tf":                                                               ("tf2_msgs/msg/TFMessage",         TFMessage),
    "/tf_static":                                                        ("tf2_msgs/msg/TFMessage",         TFMessage),
}

PVP_ROOT = Path(__file__).parent.parent.parent  # pvp4real/pvp4real/
MODES = ["offline", "online", "deploy"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
# ROS2 subscriber node
# ─────────────────────────────────────────────────────────────────────────────

class RecorderNode(Node):
    """Caches the latest message for each subscribed topic."""

    def __init__(self, topics: list[str]):
        super().__init__("pvp_bag_recorder")
        self._latest: Dict[str, Optional[Tuple[int, int, Any]]] = {t: None for t in topics}
        self._last_bag_ts: Dict[str, int] = {}
        self._lock = threading.Lock()

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

        for topic in topics:
            if topic not in TOPIC_TYPE_MAP:
                self.get_logger().warn(f"Unknown topic type for {topic}, skipping.")
                continue
            _, msg_cls = TOPIC_TYPE_MAP[topic]
            qos = qos_sensor if ("image" in topic or "compressed" in topic or "compressedDepth" in topic or "camera_info" in topic) else qos_default
            self.create_subscription(
                msg_cls, topic,
                lambda msg, t=topic: self._cache(t, msg),
                qos,
            )

    def _cache(self, topic: str, msg: Any) -> None:
        now_ns = self.get_clock().now().nanoseconds

        # Prefer sensor stamp, else fallback to now
        bag_ts_ns = now_ns
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            s = msg.header.stamp
            cand = s.sec * 1_000_000_000 + s.nanosec
            # If stamp is 0 or clearly invalid, fallback to now
            if cand > 0:
                bag_ts_ns = cand

        # Enforce strictly increasing timestamps per topic (Foxglove-friendly)
        last = self._last_bag_ts.get(topic)
        if last is not None and bag_ts_ns <= last:
            bag_ts_ns = last + 1
        self._last_bag_ts[topic] = bag_ts_ns

        recv_ts_ns = now_ns
        with self._lock:
            self._latest[topic] = (bag_ts_ns, recv_ts_ns, msg)

    def get_latest(self) -> Dict[str, Optional[Tuple[int, int, Any]]]:
        # return a snapshot copy for writer thread
        with self._lock:
            return dict(self._latest)


# ─────────────────────────────────────────────────────────────────────────────
# Recording session
# ─────────────────────────────────────────────────────────────────────────────

class RecordingSession:
    """Manages a single .mcap recording session."""

    def __init__(self, run_dir: Path, topics: list[str], frequency: float, node: RecorderNode):
        self._run_dir       = run_dir
        self._topics        = topics
        self._period        = 1.0 / frequency
        self._node          = node
        self._tick_count    = 0
        self._running       = False
        self._writer: Optional[rosbag2_py.SequentialWriter] = None
        self._thread: Optional[threading.Thread] = None
        # Dedup by reception clock (not header.stamp) to handle raw Image with stamp=0.
        self._last_recv_ts: Dict[str, int] = {}

    def start(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        storage_opts = rosbag2_py.StorageOptions(uri=str(self._run_dir / "bag_c"), storage_id="mcap")
        conv_opts    = rosbag2_py.ConverterOptions("", "")

        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(storage_opts, conv_opts)

        for topic in self._topics:
            if topic not in TOPIC_TYPE_MAP:
                continue
            type_str, _ = TOPIC_TYPE_MAP[topic]
            self._writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic, type=type_str, serialization_format="cdr"
            ))

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._writer = None  # closes the writer

        # Rename the .mcap file to {steps:05d}.mcap and update metadata.yaml
        bag_dir = self._run_dir / "bag_c"
        if bag_dir.exists():
            mcap_files = sorted(bag_dir.glob("*.mcap"))
            if mcap_files:
                old_path = mcap_files[0]
                new_name = f"{self._tick_count:05d}.mcap"
                new_path = bag_dir / new_name
                old_path.rename(new_path)
                meta_path = bag_dir / "metadata.yaml"
                if meta_path.exists():
                    text = meta_path.read_text()
                    text = text.replace(old_path.name, new_name)
                    meta_path.write_text(text)

    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            self._write_tick()
            sleep_t = self._period - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _write_tick(self) -> None:
        if self._writer is None:
            return
        written = 0
        for topic, data in self._node.get_latest().items():
            if data is None:
                continue
            bag_ts_ns, recv_ts_ns, msg = data
            # Skip if this exact reception hasn't changed (same message as last tick).
            if self._last_recv_ts.get(topic) == recv_ts_ns:
                continue
            try:
                self._writer.write(topic, serialize_message(msg), bag_ts_ns)
                self._last_recv_ts[topic] = recv_ts_ns
                written += 1
            except Exception as e:
                self._node.get_logger().warn(f"Write error on {topic}: {e}")
        if written > 0:
            self._tick_count += 1

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def run_dir(self) -> Path:
        return self._run_dir


# ─────────────────────────────────────────────────────────────────────────────
# tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────

class RecordGUI:
    def __init__(self, cfg: dict, node: RecorderNode):
        self._cfg       = cfg
        self._node      = node
        self._session: Optional[RecordingSession] = None

        rec_cfg         = cfg["record"]
        self._frequency = float(rec_cfg["frequency"])
        self._topics    = rec_cfg["topics"]
        self._paths: Dict[str, Path] = {
            mode: PVP_ROOT / rec_cfg["paths"][mode] for mode in MODES
        }

        self._root = tk.Tk()
        self._root.title("PVP4Real — Bag Recorder")

        # ✅ Make it adjustable
        self._root.resizable(True, True)

        # ✅ Grid expansion policy: allow main area to grow
        self._root.columnconfigure(0, weight=1)
        self._root.columnconfigure(1, weight=1)
        self._root.rowconfigure(2, weight=1)  # row with topics/info frames

        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._mode_var  = tk.StringVar(value=MODES[0])
        self._mode_btns: list[ttk.Radiobutton] = []
        self._build_ui()

        # ✅ Let Tk compute natural size (fits all widgets), then set it as minimum
        self._root.update_idletasks()
        self._root.minsize(self._root.winfo_width(), self._root.winfo_height())

        self._poll()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        r = self._root

        ttk.Label(r, text="ROS2 Bag Recorder", font=("Arial", 15, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(14, 6), padx=14, sticky="ew"
        )

        # ── Mode selector ────────────────────────────────────────────────────
        mode_frame = ttk.LabelFrame(r, text="Recording Mode", padding=8)
        mode_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        # allow the radio buttons area to stretch
        for c in range(len(MODES)):
            mode_frame.columnconfigure(c, weight=1)

        for col, mode in enumerate(MODES):
            btn = ttk.Radiobutton(
                mode_frame, text=mode.capitalize(),
                variable=self._mode_var, value=mode,
                command=self._on_mode_change,
            )
            btn.grid(row=0, column=col, padx=20, pady=2, sticky="w")
            self._mode_btns.append(btn)

        # ── Left column: topic list ───────────────────────────────────────────
        topics_frame = ttk.LabelFrame(r, text="Record Topics", padding=8)
        topics_frame.grid(row=2, column=0, sticky="nsew", padx=(14, 4), pady=4)

        topics_frame.rowconfigure(0, weight=1)
        topics_frame.columnconfigure(0, weight=1)

        lb = tk.Listbox(
            topics_frame,
            height=1,          # ✅ don't force fixed size
            width=1,           # ✅ let layout decide
            font=("Courier", 9),
            selectmode=tk.NONE, activestyle="none",
        )
        for t in self._topics:
            lb.insert(tk.END, t)
        lb.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(topics_frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        # ── Right column: session info ────────────────────────────────────────
        info_frame = ttk.LabelFrame(r, text="Session", padding=10)
        info_frame.grid(row=2, column=1, sticky="nsew", padx=(4, 14), pady=4)

        info_frame.columnconfigure(1, weight=1)
        info_frame.rowconfigure(99, weight=1)  # keep some elasticity

        ttk.Label(info_frame, text="Folder:", anchor="w").grid(row=0, column=0, sticky="w")
        self._path_label = ttk.Label(
            info_frame, text=self._current_path_str(),
            foreground="gray", font=("Arial", 9), wraplength=260, justify="left"
        )
        self._path_label.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Separator(info_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(info_frame, text="ID:", anchor="w").grid(row=2, column=0, sticky="w")
        self._id_label = ttk.Label(info_frame, text="————", font=("Courier", 16, "bold"), foreground="gray")
        self._id_label.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(info_frame, text="Messages:", anchor="w").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self._count_label = ttk.Label(info_frame, text="0", font=("Arial", 16, "bold"))
        self._count_label.grid(row=3, column=1, sticky="w", padx=6, pady=(10, 0))

        ttk.Label(info_frame, text="Status:", anchor="w").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self._status_label = ttk.Label(info_frame, text="Idle", foreground="gray", font=("Arial", 11, "bold"))
        self._status_label.grid(row=4, column=1, sticky="w", padx=6, pady=(10, 0))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(r)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=14, sticky="ew")
        for c in range(3):
            btn_frame.columnconfigure(c, weight=1)

        self._record_btn = ttk.Button(btn_frame, text="Record", width=16, command=self._on_record)
        self._record_btn.grid(row=0, column=0, padx=10)

        self._stop_btn = ttk.Button(btn_frame, text="Stop", width=16, command=self._on_stop, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=10)

        ttk.Button(btn_frame, text="Quit", width=16, command=self._on_quit).grid(row=0, column=2, padx=10)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _current_path_str(self) -> str:
        return str(self._paths[self._mode_var.get()])

    def _on_mode_change(self) -> None:
        self._path_label.config(text=self._current_path_str())

    def _on_record(self) -> None:
        base    = self._paths[self._mode_var.get()]
        run_dir = get_next_run_dir(base)
        self._session = RecordingSession(
            run_dir=run_dir, topics=self._topics,
            frequency=self._frequency, node=self._node,
        )
        self._session.start()
        self._id_label.config(text=run_dir.name, foreground="black")
        self._path_label.config(text=str(run_dir), foreground="black")
        self._count_label.config(text="0")
        self._status_label.config(text="Recording…", foreground="red")
        self._record_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        for btn in self._mode_btns:
            btn.config(state="disabled")

    def _on_stop(self) -> None:
        if self._session:
            self._session.stop()
            self._count_label.config(text=str(self._session.tick_count))
            self._status_label.config(text="Stopped", foreground="orange")
            self._session = None
        self._record_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        for btn in self._mode_btns:
            btn.config(state="normal")

    def _on_quit(self) -> None:
        if self._session:
            self._session.stop()
        self._root.quit()
        self._root.destroy()

    def _poll(self) -> None:
        if self._session:
            self._count_label.config(text=str(self._session.tick_count))
        self._root.after(200, self._poll)

    def run(self) -> None:
        self._root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    topics = cfg["record"]["topics"]

    rclpy.init()
    node = RecorderNode(topics)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        gui = RecordGUI(cfg=cfg, node=node)
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()