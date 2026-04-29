#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-$HOME/venvs/ros2foxy_rl}"

sudo apt update || true
sudo apt install -y python3.8-venv python3-pip

/usr/bin/python3 -m venv --system-site-packages "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade "pip==24.0" wheel "setuptools<70"
python -m pip install -r requirements.txt
python -m pip install --no-deps torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

cat <<EOF

Done.
Activate the environment with:
  source "$VENV_PATH/bin/activate"
  source /opt/ros/foxy/setup.bash
EOF
