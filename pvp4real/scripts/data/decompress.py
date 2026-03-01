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
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

# Compressed → raw topic mapping: (raw_topic, raw_type_str, raw_msg_cls, encoding_hint)
# NOTE: depth encoding is determined from CompressedImage.format at runtime (16UC1 or 32FC1).
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
        "depth",  # hint only; real encoding parsed from comp_msg.format
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

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def parse_depth_encoding_from_format(fmt: str) -> str:
    """
    Typical compressedDepth format strings:
      "16UC1; compressedDepth png"
      "32FC1; compressedDepth png"
    We return the left side before ';'.
    """
    left = fmt.split(";", 1)[0].strip()
    return left if left else "16UC1"


def split_header_and_png(data: bytes) -> Tuple[Optional[Tuple[float, float]], bytes]:
    """
    Robustly split ConfigHeader + PNG payload for compressedDepth.

    compressed_depth_image_transport prepends a ConfigHeader (commonly 12 bytes):
      enum format (4 bytes) + float depthParam[2] (8 bytes)
    followed by PNG bytes.

    We detect PNG magic; if found at offset 12/8/16 we treat preceding bytes as header.
    Returns:
      (depthQuantA, depthQuantB) or None, png_payload_bytes
    """
    if data.startswith(PNG_MAGIC):
        return None, data

    for off in (12, 8, 16):
        if len(data) > off + 8 and data[off:off + 8] == PNG_MAGIC:
            header = data[:off]
            payload = data[off:]
            if len(header) >= 8:
                qA, qB = struct.unpack("<ff", header[-8:])
                return (qA, qB), payload
            return None, payload

    # Can't find PNG magic; return as-is (may fail later in imdecode)
    return None, data


def decompress_image(comp_msg: CompressedImage, encoding_hint: str) -> Image:
    """Decompress a CompressedImage into a raw Image message."""
    if encoding_hint == "rgb8":
        # JPEG → BGR → RGB
        raw = np.frombuffer(comp_msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode color compressed image")
        data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        step = data.shape[1] * 3

        img = Image()
        img.header = comp_msg.header
        img.height = data.shape[0]
        img.width = data.shape[1]
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = step
        img.data = data.tobytes()
        return img

    if encoding_hint == "depth":
        # Determine actual encoding from CompressedImage.format
        encoding = parse_depth_encoding_from_format(comp_msg.format)  # "16UC1" or "32FC1"

        # Split header and PNG payload (robust)
        quant_params, png_bytes = split_header_and_png(bytes(comp_msg.data))

        # Decode PNG
        buf = np.frombuffer(png_bytes, dtype=np.uint8)
        dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if dec is None:
            raise ValueError(
                f"Failed to decode depth compressed image "
                f"(data_len={len(comp_msg.data)}, format={comp_msg.format!r})"
            )

        img = Image()
        img.header = comp_msg.header
        img.height = int(dec.shape[0])
        img.width = int(dec.shape[1])
        img.is_bigendian = 0

        if encoding == "16UC1":
            if dec.dtype != np.uint16:
                dec = dec.astype(np.uint16, copy=False)
            img.encoding = "16UC1"
            img.step = img.width * 2
            img.data = dec.tobytes(order="C")
            return img

        if encoding == "32FC1":
            # compressedDepth for 32FC1 uses inverse-depth with quant params.
            if dec.dtype != np.uint16:
                dec = dec.astype(np.uint16, copy=False)

            if quant_params is None:
                raise ValueError("Missing depth quantization parameters (depthQuantA/B) for 32FC1 compressedDepth")

            qA, qB = quant_params
            inv = dec.astype(np.float32)
            denom = inv - np.float32(qB)

            depth = np.zeros_like(inv, dtype=np.float32)
            valid = (inv != 0) & (np.abs(denom) > 1e-12)
            depth[valid] = np.float32(qA) / denom[valid]
            depth[~valid] = 0.0

            img.encoding = "32FC1"
            img.step = img.width * 4
            img.data = depth.tobytes(order="C")
            return img

        raise ValueError(f"Unsupported depth encoding parsed from format: {encoding} (format={comp_msg.format!r})")

    raise ValueError(f"Unsupported encoding hint: {encoding_hint}")


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
                _, raw_type, _, encoding_hint = DECOMPRESS_MAP[topic]
                comp_msg = deserialize_message(raw_data, CompressedImage)
                if len(comp_msg.data) == 0:
                    raise ValueError(f"empty compressed data (format={comp_msg.format!r})")
                img_msg = decompress_image(comp_msg, encoding_hint)
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