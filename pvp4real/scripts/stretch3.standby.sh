#!/usr/bin/env bash
set -euo pipefail

# Stretch3 container: bring up base + camera only.
# This script is intentionally independent of the PVP4Real code.

ros2 launch stretch_launch stretch_driver.launch.py &
DRIVER_PID=$!

ros2 launch stretch_launch d435i_basic.launch.py &
CAM_PID=$!

echo "[stretch3.standby.sh] Driver PID=${DRIVER_PID}, Camera PID=${CAM_PID}"
echo "[stretch3.standby.sh] Press Ctrl-C to stop."

trap 'echo "Stopping..."; kill ${DRIVER_PID} ${CAM_PID} 2>/dev/null || true; wait || true' INT TERM
wait
