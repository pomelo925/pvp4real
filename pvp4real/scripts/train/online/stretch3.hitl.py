#!/usr/bin/env python3
"""stretch3.hitl.py — Stretch3-side HITL runtime (authority arbitration).

Runs the authority arbitration loop:
  is_teleop → forward teleop cmd
  else      → forward policy cmd (or zero on stale)

Note: stretch_driver and d435i camera are launched by the helloRobot_stretch3
container. This script only handles cmd_vel arbitration.

This script is intentionally algorithm-agnostic (no PVP/SB3/Torch deps).

Key events logged:
  - startup config (hz, stale threshold)
  - mode change: Teleop ↔ Policy
  - policy stale / recovered
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config.yaml from the same directory as this script."""
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> float:
    return time.monotonic()


def _zero_twist() -> Twist:
    t = Twist()
    t.linear.x  = 0.0
    t.angular.z = 0.0
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Authority node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeCfg:
    hz:              float = 5.0
    policy_stale_s:  float = 0.5
    zero_on_stale:   bool  = True


class Stretch3Authority(Node):
    """Arbitrates cmd_vel between teleop and policy."""

    def __init__(self, cfg: RuntimeCfg):
        super().__init__("stretch3_hitl_authority")
        self.cfg = cfg

        self._is_teleop:      bool  = False
        self._teleop_cmd:     Twist = _zero_twist()
        self._policy_cmd:     Twist = _zero_twist()
        self._last_policy_t:  float = -1.0

        # state-change tracking (avoid log spam)
        self._prev_teleop:  bool = False
        self._prev_stale:   bool = True

        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Bool,  "/stretch/is_teleop",       self._on_is_teleop,   qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop",  self._on_teleop_cmd,  qos_default)
        self.create_subscription(Twist, "/pvp/novice_cmd_vel",      self._on_policy_cmd,  qos_default)

        self._cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", qos_default)
        self.create_timer(1.0 / float(cfg.hz), self._tick)

        self.get_logger().info(
            f"[stretch3.hitl] Authority started @ {cfg.hz}Hz | "
            f"stale threshold: {cfg.policy_stale_s}s"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_is_teleop(self, msg: Bool)   -> None: self._is_teleop    = bool(msg.data)
    def _on_teleop_cmd(self, msg: Twist) -> None: self._teleop_cmd   = msg
    def _on_policy_cmd(self, msg: Twist) -> None:
        self._policy_cmd    = msg
        self._last_policy_t = _now()

    def _policy_is_stale(self) -> bool:
        if self._last_policy_t < 0:
            return True
        return (_now() - self._last_policy_t) > float(self.cfg.policy_stale_s)

    def _tick(self) -> None:
        stale = self._policy_is_stale()

        # ── log mode change ───────────────────────────────────────────────────
        if self._is_teleop != self._prev_teleop:
            mode = "Teleop" if self._is_teleop else "Policy"
            self.get_logger().info(f"[stretch3.hitl] mode → {mode}")
            self._prev_teleop = self._is_teleop

        # ── log policy stale / recovered ──────────────────────────────────────
        if stale != self._prev_stale:
            if stale:
                self.get_logger().warn("[stretch3.hitl] policy STALE (no cmd received)")
            else:
                self.get_logger().info("[stretch3.hitl] policy RECOVERED")
            self._prev_stale = stale

        if self._is_teleop:
            cmd = self._teleop_cmd
        elif self.cfg.zero_on_stale and stale:
            cmd = _zero_twist()
        else:
            cmd = self._policy_cmd
        self._cmd_pub.publish(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stretch3 HITL runtime (authority arbitration).")
    parser.add_argument("--hz",             type=float, default=None)
    parser.add_argument("--policy_stale_s", type=float, default=0.5)
    parser.add_argument("--no_zero_on_stale", action="store_true")
    args = parser.parse_args()

    # ── Load config.yaml ──────────────────────────────────────────────────────
    raw    = load_config()
    common = raw.get("common", {})

    # CLI args override config values where provided
    hz = args.hz if args.hz is not None else float(common.get("hz", 5.0))

    cfg = RuntimeCfg(
        hz=hz,
        policy_stale_s=args.policy_stale_s,
        zero_on_stale=(not args.no_zero_on_stale),
    )

    # ── ROS2 node ─────────────────────────────────────────────────────────────
    rclpy.init()
    node = None
    try:
        node = Stretch3Authority(cfg)
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("[stretch3.hitl] Interrupted, shutting down.")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
