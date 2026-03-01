#!/usr/bin/env python3
"""merge.py — Merge multiple ROS2 bags into a single bag.

Reads config.yaml from the same directory (data/config.yaml).
Uses the ``merge`` section to configure source/output dirs and options.

Usage:
    python3 pvp4real/scripts/data/merge.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import rosbag2_py

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

PVP_ROOT = Path(__file__).parent.parent.parent  # pvp4real/pvp4real/

_RANGE_RE = re.compile(r"^(.+?)(\d+)~(\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def expand_input_dirs(raw: List[str]) -> List[Path]:
    """
    Expand each token in *raw* using range notation or keep as-is.

    Range notation:   ``prefix/0001~0006``
      → resolves to prefix/0001, prefix/0002, …, prefix/0006
      (zero-padding length matches the start token)

    Plain path:       ``datasets/offline/0001``
      → resolved relative to PVP_ROOT if not absolute
    """
    paths: List[Path] = []
    for token in raw:
        m = _RANGE_RE.match(token)
        if m:
            prefix, start_s, end_s = m.group(1), m.group(2), m.group(3)
            width = len(start_s)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            # prefix already ends with the directory separator, so use it as
            # the parent directly (e.g. "datasets/offline/" → Path(…/offline))
            base = Path(prefix.rstrip("/\\"))
            if not base.is_absolute():
                base = PVP_ROOT / base
            for i in range(start, end + 1):
                paths.append(base / str(i).zfill(width))
        else:
            p = Path(token)
            if not p.is_absolute():
                p = PVP_ROOT / p
            paths.append(p)
    return paths


def resolve_bag_dir(run_dir: Path, only_compress: bool) -> Path:
    """Return the bag sub-directory to read from inside *run_dir*."""
    sub = "bag_c" if only_compress else "bag"
    return run_dir / sub


# ─────────────────────────────────────────────────────────────────────────────
# Core merge
# ─────────────────────────────────────────────────────────────────────────────

# Topic used as step counter: one compressed color frame is published per
# recording tick regardless of teleop state.
_STEP_TOPIC = "/camera/camera/color/image_raw/compressed"


def count_steps(messages: List[Tuple[str, bytes, int]]) -> int:
    """Count recording steps = number of _STEP_TOPIC messages."""
    return sum(1 for topic, _, _ in messages if topic == _STEP_TOPIC)


def collect_messages(
    bag_dir: Path,
) -> Tuple[Dict[str, str], List[Tuple[str, bytes, int]]]:
    """
    Read all messages from *bag_dir* (MCAP).

    Returns:
        topic_type_map  – {topic_name: type_string}
        messages        – [(topic, raw_bytes, timestamp_ns), …]
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_type_map: Dict[str, str] = {
        t.name: t.type for t in reader.get_all_topics_and_types()
    }
    messages: List[Tuple[str, bytes, int]] = []
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        messages.append((topic, data, ts_ns))
    del reader
    return topic_type_map, messages


def merge_bags(
    source_dirs: List[Path],
    output_bag_dir: Path,
    only_compress: bool,
    save_raw: bool,
) -> None:
    # ── Discover & validate source bags ─────────────────────────────────────
    bag_dirs: List[Path] = []
    for run_dir in source_dirs:
        if not run_dir.exists():
            print(f"[ERROR] Input directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        bag_dir = resolve_bag_dir(run_dir, only_compress)
        if not bag_dir.exists():
            print(f"[ERROR] Bag sub-directory not found: {bag_dir}", file=sys.stderr)
            sys.exit(1)
        bag_dirs.append(bag_dir)

    print(f"\nMerging {len(bag_dirs)} bag(s) → {output_bag_dir}")
    for bd in bag_dirs:
        print(f"  {bd}")

    if output_bag_dir.exists():
        print(f"[WARN] Output already exists, it will be overwritten: {output_bag_dir}")
        shutil.rmtree(output_bag_dir)

    # ── Read all messages ────────────────────────────────────────────────────
    all_topic_types: Dict[str, str] = {}
    all_messages:    List[Tuple[str, bytes, int]] = []
    per_bag_steps:   List[Tuple[Path, int]] = []

    for bag_dir in bag_dirs:
        print(f"\nReading {bag_dir} …")
        ttm, msgs = collect_messages(bag_dir)
        steps = count_steps(msgs)
        print(f"  {len(msgs):,} messages, {len(ttm)} topic(s), steps: {steps}")
        per_bag_steps.append((bag_dir, steps))
        all_topic_types.update(ttm)
        all_messages.extend(msgs)

    # Sort by timestamp (ascending)
    print(f"\nSorting {len(all_messages):,} messages by timestamp …")
    all_messages.sort(key=lambda x: x[2])

    # ── Write merged output ──────────────────────────────────────────────────
    print(f"Writing merged bag to {output_bag_dir} …")
    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )

    registered: set[str] = set()
    for topic, type_str in all_topic_types.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=type_str, serialization_format="cdr"
        ))
        registered.add(topic)

    written = 0
    for topic, data, ts_ns in all_messages:
        # Register any topic not yet known (safety fallback)
        if topic not in registered:
            print(f"  [WARN] Late-registered topic: {topic}")
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic, type="", serialization_format="cdr"
            ))
            registered.add(topic)
        writer.write(topic, data, ts_ns)
        written += 1

    del writer  # flush & close

    # ── Rename the generated .mcap to reflect step count ────────────────────
    total_in    = len(all_messages)
    total_steps = count_steps(all_messages)

    mcap_files = sorted(output_bag_dir.glob("*.mcap"))
    if mcap_files:
        new_name = f"{total_steps:05d}.mcap"
        mcap_files[0].rename(output_bag_dir / new_name)
        print(f"Renamed mcap → {new_name}")
    else:
        print("[WARN] No .mcap file found to rename.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Sources              : {len(bag_dirs)}")
    for bd, s in per_bag_steps:
        print(f"    {bd.parent.name:>8}  →  {s} steps")
    print(f"  {'─' * 44}")
    print(f"  Total steps          : {total_steps}")
    print(f"  Total messages in    : {total_in:,}")
    print(f"  Written to output    : {written:,}")
    print(f"  Output               : {output_bag_dir}")
    print(f"{'─' * 50}")

    # ── Optionally delete originals ──────────────────────────────────────────
    if not save_raw:
        print("\nDeleting source run directories …")
        for run_dir in source_dirs:
            shutil.rmtree(run_dir)
            print(f"  Deleted: {run_dir}")
    else:
        print("\nOriginal data preserved (--save_raw).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    mcfg = cfg.get("merge", {})

    # ── Required fields ──────────────────────────────────────────────────────
    raw_input_dirs = mcfg.get("input_dirs")
    raw_output_dir = mcfg.get("output_dir")

    if not raw_input_dirs:
        print("[ERROR] 'merge.input_dirs' not set in config.yaml", file=sys.stderr)
        sys.exit(1)
    if not raw_output_dir:
        print("[ERROR] 'merge.output_dir' not set in config.yaml", file=sys.stderr)
        sys.exit(1)

    # ── Optional fields (with defaults) ──────────────────────────────────────
    only_compress: bool = mcfg.get("only_compress", True)
    save_raw:      bool = mcfg.get("save_raw",      True)

    # ── Resolve paths ────────────────────────────────────────────────────────
    # input_dirs may be a list or a single string in YAML
    if isinstance(raw_input_dirs, str):
        raw_input_dirs = [raw_input_dirs]
    source_dirs = expand_input_dirs([str(d) for d in raw_input_dirs])

    out_root = Path(str(raw_output_dir))
    if not out_root.is_absolute():
        out_root = PVP_ROOT / out_root

    sub = "bag_c" if only_compress else "bag"
    output_bag_dir = out_root / sub

    print("=" * 50)
    print("  ROS2 Bag Merger")
    print("=" * 50)
    print(f"  only_compress : {only_compress}  (reading '{sub}/' sub-dirs)")
    print(f"  save_raw      : {save_raw}")

    merge_bags(
        source_dirs=source_dirs,
        output_bag_dir=output_bag_dir,
        only_compress=only_compress,
        save_raw=save_raw,
    )


if __name__ == "__main__":
    main()
