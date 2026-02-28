#!/usr/bin/env python3
"""decompress.py — Decompress a bag_c recording into a raw bag.

Reads config.yaml from the same directory (data/config.yaml).
Uses `decompress.target_bag_path` (relative to pvp4real/pvp4real/) to locate
the source run directory, then:

  Input : {run_dir}/bag_c/   (CompressedImage for rgb & depth)
  Output: {run_dir}/bag/     (raw Image for rgb & depth, others pass-through)

Usage:
    python3 pvp4real/scripts/data/decompress.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PVP_ROOT = Path(__file__).parent.parent.parent  # pvp4real/pvp4real/

# Compressed → raw topic mapping: (raw_topic, raw_type_str, raw_msg_cls, encoding)
DECOMPRESS_MAP: Dict[str, Tuple[str, str, Any, str]] = {
    "/camera/camera/color/image_raw/compressed": (
        "/camera/camera/color/image_raw",
        "sensor_msgs/msg/Image",
        Image,
        "rgb8",
    ),
    "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth": (
        "/camera/camera/aligned_depth_to_color/image_raw",
        "sensor_msgs/msg/Image",
        Image,
        "16UC1",
    ),
}

# Pass-through topic type map (topic → (type_str, msg_cls))
PASSTHROUGH_TYPE_MAP: Dict[str, Tuple[str, Any]] = {
    "/camera/camera/color/camera_info":   ("sensor_msgs/msg/CameraInfo",  CameraInfo),
    "/stretch/cmd_vel_teleop":            ("geometry_msgs/msg/Twist",     Twist),
    "/stretch/cmd_vel":                   ("geometry_msgs/msg/Twist",     Twist),
    "/stretch/is_teleop":                 ("std_msgs/msg/Bool",           Bool),
    "/tf":                                ("tf2_msgs/msg/TFMessage",      TFMessage),
    "/tf_static":                         ("tf2_msgs/msg/TFMessage",      TFMessage),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def decompress_image(comp_msg: CompressedImage, encoding: str) -> Image:
    """Decompress a CompressedImage into a raw Image message."""
    raw_bytes = np.frombuffer(comp_msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        # JPEG → BGR → RGB
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode color compressed image")
        data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        step = data.shape[1] * 3
    elif encoding == "16UC1":
        # image_transport compressedDepth prepends a 12-byte ConfigHeader
        # (float depthParam[2] = 8 bytes, int format = 4 bytes) before the PNG.
        # Try stripping the header first; fall back to raw bytes if that fails.
        data = None
        for offset in (12, 0):
            if len(raw_bytes) > offset:
                candidate = cv2.imdecode(raw_bytes[offset:], cv2.IMREAD_UNCHANGED)
                if candidate is not None:
                    data = candidate
                    break
        if data is None:
            raise ValueError(
                f"Failed to decode depth compressed image "
                f"(data_len={len(raw_bytes)}, format={comp_msg.format!r})"
            )
        if data.dtype != np.uint16:
            data = data.astype(np.uint16)
        step = data.shape[1] * 2
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    img = Image()
    img.header    = comp_msg.header
    img.height    = data.shape[0]
    img.width     = data.shape[1]
    img.encoding  = encoding
    img.is_bigendian = 0
    img.step      = step
    img.data      = data.tobytes()
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    rel_path = cfg.get("decompress", {}).get("target_bag_path")
    if not rel_path:
        print("[ERROR] 'decompress.target_bag_path' not set in config.yaml", file=sys.stderr)
        sys.exit(1)

    run_dir  = PVP_ROOT / rel_path
    src_dir  = run_dir / "bag_c"
    dst_dir  = run_dir / "bag"

    if not src_dir.exists():
        print(f"[ERROR] Source bag_c not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if dst_dir.exists():
        print(f"[WARN] Destination already exists, will overwrite: {dst_dir}")

    print(f"Source : {src_dir}")
    print(f"Dest   : {dst_dir}")

    # Tick window: messages within this tolerance (ns) belong to the same step
    TICK_TOL_NS = 150_000_000  # 150 ms  (recording is 5 Hz → 200 ms/tick)

    # Image compressed topics we must decode
    IMG_TOPICS = set(DECOMPRESS_MAP.keys())

    # ── Pass 1: read all messages into memory, try to decode image topics ────
    print("Pass 1: scanning messages …")

    # Each entry: (topic, raw_data, ts_ns, decoded_msg_or_None, ok: bool)
    records: list[tuple[str, bytes, int, Any, bool]] = []

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(src_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_type_map: Dict[str, str] = {
        t.name: t.type for t in reader.get_all_topics_and_types()
    }

    while reader.has_next():
        topic, raw_data, ts_ns = reader.read_next()
        if topic in IMG_TOPICS:
            try:
                _, raw_type, _, encoding = DECOMPRESS_MAP[topic]
                comp_msg = deserialize_message(raw_data, CompressedImage)
                if len(comp_msg.data) == 0:
                    raise ValueError(f"empty compressed data (format={comp_msg.format!r})")
                img_msg = decompress_image(comp_msg, encoding)
                records.append((topic, raw_data, ts_ns, img_msg, True))
            except Exception as e:
                records.append((topic, raw_data, ts_ns, None, False))
                print(f"  [WARN] decode failed @ ts={ts_ns}: {e}")
        else:
            records.append((topic, raw_data, ts_ns, None, True))  # pass-through, always ok

    print(f"  total messages in bag_c: {len(records)}")

    # ── Identify bad tick windows based on image decode failures ─────────────
    # Collect timestamps of all failed image messages
    failed_ts: list[int] = [ts for (topic, _, ts, _, ok) in records
                             if topic in IMG_TOPICS and not ok]

    def in_bad_window(ts_ns: int) -> bool:
        return any(abs(ts_ns - bad) <= TICK_TOL_NS for bad in failed_ts)

    # Count how many image messages fall in bad windows
    skipped_img_ts: set[int] = set()
    for topic, _, ts, _, ok in records:
        if topic in IMG_TOPICS and in_bad_window(ts):
            skipped_img_ts.add(ts)

    # ── Pass 2: write output ─────────────────────────────────────────────────
    print("Pass 2: writing output …")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(dst_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )

    registered: set[str] = set()

    def ensure_topic(name: str, type_str: str) -> None:
        if name not in registered:
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=name, type=type_str, serialization_format="cdr"
            ))
            registered.add(name)

    for raw_topic, raw_type, _, _ in DECOMPRESS_MAP.values():
        ensure_topic(raw_topic, raw_type)
    for topic, (type_str, _) in PASSTHROUGH_TYPE_MAP.items():
        ensure_topic(topic, type_str)

    written = 0
    skipped = 0

    for topic, raw_data, ts_ns, decoded_msg, ok in records:
        if topic in IMG_TOPICS:
            if in_bad_window(ts_ns):
                skipped += 1
                continue
            raw_topic, raw_type, _, _ = DECOMPRESS_MAP[topic]
            writer.write(raw_topic, serialize_message(decoded_msg), ts_ns)
            written += 1
        else:
            src_type = topic_type_map.get(topic, "")
            if src_type:
                ensure_topic(topic, src_type)
            writer.write(topic, raw_data, ts_ns)
            written += 1

    del writer  # flush & close

    total = len(records)
    bad_ticks = len(set(round(ts / TICK_TOL_NS) for ts in failed_ts))
    print(f"\n{'─'*50}")
    print(f"  Total messages in source : {total}")
    print(f"  Written to output        : {written}")
    print(f"  Skipped (bad timesteps)  : {skipped}  ({bad_ticks} tick(s) dropped)")
    print(f"  Decode failures          : {len(failed_ts)}")
    print(f"{'─'*50}")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
