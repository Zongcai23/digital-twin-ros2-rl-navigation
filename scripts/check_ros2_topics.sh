#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/foxy/setup.bash

echo "--- Topic list ---"
ros2 topic list | sort | grep -E "object|xform|robot|rl_vel|reset|Dyna|OT" || true

echo
for topic in /object /xform /robot01 /robot02 /rl_vel /resetSignal; do
  echo "--- $topic ---"
  ros2 topic info "$topic" -v || true
  echo
done
