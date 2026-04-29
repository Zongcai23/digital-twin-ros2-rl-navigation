#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${ROS2_RL_VENV:-$HOME/venvs/ros2foxy_rl}/bin/activate"
source /opt/ros/foxy/setup.bash
python rl_navigation/deploy_best_model.py "$@"
