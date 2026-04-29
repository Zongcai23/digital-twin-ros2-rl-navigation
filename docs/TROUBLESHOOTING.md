# Troubleshooting

## 1. `ModuleNotFoundError: No module named 'rclpy._rclpy'`

Do not run the ROS 2 deployment with Isaac Sim's `python.sh`.

Use:

```bash
source ~/venvs/ros2foxy_rl/bin/activate
source /opt/ros/foxy/setup.bash
python rl_navigation/deploy_best_model.py
```

The reason is that ROS 2 Foxy on Ubuntu 20.04 uses Python 3.8, while Isaac Sim may bundle a different Python version.

## 2. `ModuleNotFoundError: No module named 'torch'`

Activate the venv and install dependencies:

```bash
source ~/venvs/ros2foxy_rl/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

Then check:

```bash
python -c "import torch; print(torch.__version__)"
```

## 3. `missing=['/object', '/xform']`

This means the Python controller did not receive required Isaac Sim state topics. Check:

```bash
source /opt/ros/foxy/setup.bash
ros2 topic list | sort | grep -E "object|xform|robot|rl_vel|reset"
ros2 topic info /xform -v
ros2 topic echo /xform --once
```

In Isaac Sim, confirm:

1. Timeline is **Play**.
2. ROS 2 Bridge is enabled.
3. ROS 1 Bridge is disabled.
4. The Action Graph ROS 2 publish nodes for `/object` and `/xform` are connected to a tick/physics-step execution signal.
5. The message type is `tf2_msgs/msg/TFMessage`.
6. The topic names are exactly `/object` and `/xform`.
7. The published message is not empty; `transforms:` must contain at least one transform.

## 4. The robot moves in one fixed direction too fast

This usually means `/xform` is stale or zero, so the controller thinks the robot is at `[0, 0]` while the path starts around `(1.99, -15.00)`.

The current code refuses to move if `/xform` is missing or still `[0, 0]`. If this happens, fix the Isaac Sim `/xform` publisher instead of disabling the safety check.

## 5. The command is published but Isaac does not move

Check that the ROS 2 Subscribe Twist node in Isaac Sim maps the fields correctly:

```text
/rl_vel.linear.x   -> x velocity input
/rl_vel.linear.y   -> y velocity input
/rl_vel.angular.z  -> yaw angular velocity input
/resetSignal.angular.x -> reset flag input
```

Also check:

```bash
ros2 topic echo /rl_vel
ros2 topic echo /resetSignal
```

## 6. The USD opens with missing assets

If Isaac Sim reports missing referenced assets, update the USD asset paths in Isaac Sim's Layer/Reference tools or keep the repository folder path simple and ASCII-only.
