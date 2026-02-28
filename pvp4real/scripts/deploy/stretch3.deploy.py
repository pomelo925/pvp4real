#!/usr/bin/env python3
"""stretch3.deploy.py — Stretch3-side deploy runtime (authority arbitration).

Runs the authority arbitration loop:
  is_teleop → forward teleop cmd
  else      → forward policy cmd (or zero on stale)

Note: stretch_driver and d435i camera are launched by the helloRobot_stretch3
container. This script only handles cmd_vel arbitration.

This script is algorithm-agnostic (no PVP/SB3/Torch deps).

GUI:
  Shows current mode, policy staleness, and a Quit button.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


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
    hz:             float = 10.0
    policy_stale_s: float = 0.5
    zero_on_stale:  bool  = True


class Stretch3DeployAuthority(Node):
    """Arbitrates cmd_vel between teleop and policy (deploy mode)."""

    def __init__(self, cfg: RuntimeCfg):
        super().__init__("stretch3_deploy_authority")
        self.cfg = cfg

        self._is_teleop:     bool  = False
        self._teleop_cmd:    Twist = _zero_twist()
        self._policy_cmd:    Twist = _zero_twist()
        self._last_policy_t: float = -1.0

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
            f"[stretch3.deploy] Authority started @ {cfg.hz}Hz | "
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
        if self._is_teleop:
            cmd = self._teleop_cmd
        elif self.cfg.zero_on_stale and self._policy_is_stale():
            cmd = _zero_twist()
        else:
            cmd = self._policy_cmd
        self._cmd_pub.publish(cmd)

    # ── Status accessors for GUI ──────────────────────────────────────────────

    def is_teleop(self)    -> bool: return self._is_teleop
    def policy_stale(self) -> bool: return self._policy_is_stale()


# ─────────────────────────────────────────────────────────────────────────────
# tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────

class DeployAuthorityGUI:

    def __init__(self, node: Stretch3DeployAuthority):
        self._node          = node
        self.quit_requested = False

        self.root = tk.Tk()
        self.root.title("Stretch3 Deploy Runtime")
        self.root.geometry("380x200")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        r = self.root
        ttk.Label(r, text="Stretch3 Deploy Authority", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(14, 8)
        )

        sf = ttk.LabelFrame(r, text="Status", padding=10)
        sf.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        ttk.Label(sf, text="Mode:").grid(row=0, column=0, sticky="w")
        self._mode_lbl = ttk.Label(sf, text="—", font=("Arial", 11, "bold"))
        self._mode_lbl.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(sf, text="Policy:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._policy_lbl = ttk.Label(sf, text="—", font=("Arial", 11))
        self._policy_lbl.grid(row=1, column=1, sticky="w", padx=8, pady=(4, 0))

        sf.columnconfigure(1, weight=1)

        ttk.Button(r, text="Quit", width=22, command=self._on_quit).grid(
            row=2, column=0, columnspan=2, pady=14
        )
        r.columnconfigure(0, weight=1)

    def _poll(self) -> None:
        if self._node.is_teleop():
            self._mode_lbl.config(text="Gamepad (Teleop)", foreground="orange")
        else:
            self._mode_lbl.config(text="Navigation (Policy)", foreground="green")

        if self._node.policy_stale():
            self._policy_lbl.config(text="Stale / no cmd", foreground="red")
        else:
            self._policy_lbl.config(text="Active", foreground="green")

        self.root.after(300, self._poll)

    def _on_quit(self) -> None:
        self.quit_requested = True
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stretch3 deploy runtime (authority arbitration).")
    parser.add_argument("--hz",               type=float, default=10.0)
    parser.add_argument("--policy_stale_s",   type=float, default=0.5)
    parser.add_argument("--no_zero_on_stale", action="store_true")
    args = parser.parse_args()

    cfg = RuntimeCfg(
        hz=args.hz,
        policy_stale_s=args.policy_stale_s,
        zero_on_stale=(not args.no_zero_on_stale),
    )

    rclpy.init()
    node: Optional[Stretch3DeployAuthority] = None
    try:
        node = Stretch3DeployAuthority(cfg)

        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        gui = DeployAuthorityGUI(node=node)
        gui.run()

    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
