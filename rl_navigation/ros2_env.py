"""ROS 2 environment wrapper for Isaac Sim optical microrobot navigation.

The trajectory/action math is intentionally kept consistent with the original
ROS 1 implementation. The ROS 2-specific changes are limited to rclpy node
management, QoS, and safety checks that prevent motion when Isaac state topics
are missing or stale.
"""

import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gym
from gym import spaces
import numpy as np

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class TransformListener(Node):
    """Receive Isaac Sim TF-style state topics and publish RL velocity commands."""

    def __init__(self, node_name: str = "transform_listener"):
        super().__init__(node_name)

        self.object_position = np.zeros(3)
        self.xform_position = np.zeros(3)
        self.xform_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.contact_force = 0.0
        self.object_lost = False
        self.collision_occurred = False
        self.collision_force = 0.0
        self.object_frame = None
        self.xform_time = None
        self.object_frame_carry = 0
        self.xform_time_carry = 0

        self.robot01_position = np.zeros(3)
        self.robot01_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.robot02_position = np.zeros(3)
        self.robot02_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.dynaOb_positions = [np.zeros(3) for _ in range(4)]
        self.collision01 = [0, 0]
        self.collision02 = [0, 0]
        self.collision03 = [0, 0]
        self.collision04 = [0, 0]

        self.first_ot_position = np.zeros(3)
        self.second_ot_position = np.zeros(3)

        self.required_topics = ["/object", "/xform", "/robot01", "/robot02"]
        self.received_topics = {name: False for name in self.required_topics}
        self.topic_counts = {name: 0 for name in self.required_topics}
        self.topic_last_wall_time = {name: 0.0 for name in self.required_topics}

        # Isaac Sim ROS 2 sensor streams are often best-effort. This subscriber
        # QoS is intentionally permissive so it can match common Action Graph settings.
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = 10

        self.create_subscription(TFMessage, "/object", self.object_callback, sub_qos)
        self.create_subscription(TFMessage, "/xform", self.xform_callback, sub_qos)
        self.create_subscription(TFMessage, "/robot01", self.robot01_callback, sub_qos)
        self.create_subscription(TFMessage, "/robot02", self.robot02_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb01Collide", self.dynaOb01_collide_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb02Collide", self.dynaOb02_collide_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb03Collide", self.dynaOb03_collide_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb04Collide", self.dynaOb04_collide_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb01", self.dynaOb01_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb02", self.dynaOb02_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb03", self.dynaOb03_callback, sub_qos)
        self.create_subscription(TFMessage, "/DynaOb04", self.dynaOb04_callback, sub_qos)
        self.create_subscription(TFMessage, "/firstOTposition", self.first_ot_callback, sub_qos)
        self.create_subscription(TFMessage, "/secondOTposition", self.second_ot_callback, sub_qos)

        self.velocity_publisher = self.create_publisher(Twist, "/rl_vel", pub_qos)
        self.reset_publisher = self.create_publisher(Twist, "/resetSignal", pub_qos)

    @staticmethod
    def _first_transform(msg: TFMessage):
        return msg.transforms[0] if len(msg.transforms) > 0 else None

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
        nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
        return float(sec) + float(nanosec) / 1e9

    def _mark_received(self, topic_name: str) -> None:
        if topic_name in self.received_topics:
            self.received_topics[topic_name] = True
            self.topic_counts[topic_name] += 1
            self.topic_last_wall_time[topic_name] = time.time()

    def snapshot_counts(self) -> Dict[str, int]:
        return dict(self.topic_counts)

    def wait_for_initial_messages(self, timeout_sec: float = 10.0, raise_on_timeout: bool = True) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            missing = [name for name, ok in self.received_topics.items() if not ok]
            if not missing:
                self.get_logger().info(
                    "Received initial Isaac Sim topics: "
                    f"xform={self.xform_position[:2]}, object={self.object_position[:2]}, "
                    f"robot01={self.robot01_position[:2]}, robot02={self.robot02_position[:2]}"
                )
                return True
            time.sleep(0.05)

        missing = [name for name, ok in self.received_topics.items() if not ok]
        msg = (
            "Did not receive all required Isaac Sim topics before deployment; "
            f"missing={missing}. Refusing to move to avoid sending velocity from zero/stale state."
        )
        if raise_on_timeout:
            raise RuntimeError(msg)
        self.get_logger().warning(msg)
        return False

    def wait_for_fresh_messages(
        self,
        before_counts: Dict[str, int],
        timeout_sec: float = 3.0,
        min_advanced_topics: Iterable[str] = ("/xform", "/object"),
    ) -> bool:
        """Wait until selected topics have advanced after a reset command."""
        start = time.time()
        min_advanced_topics = tuple(min_advanced_topics)
        while time.time() - start < timeout_sec:
            advanced = [
                name for name in min_advanced_topics
                if self.topic_counts.get(name, 0) > before_counts.get(name, 0)
            ]
            if len(advanced) == len(min_advanced_topics):
                return True
            time.sleep(0.05)

        not_advanced = [
            name for name in min_advanced_topics
            if self.topic_counts.get(name, 0) <= before_counts.get(name, 0)
        ]
        self.get_logger().warning(
            "After reset, these state topics did not publish fresh messages: "
            f"{not_advanced}. Current xform={self.xform_position[:2]}, object={self.object_position[:2]}"
        )
        return False

    def _read_tf_position_orientation(self, msg: TFMessage):
        transform = self._first_transform(msg)
        if transform is None:
            return None, None, None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        stamp = transform.header.stamp
        position = np.array([translation.x, translation.y, translation.z], dtype=float)
        orientation = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
        return position, orientation, stamp

    def object_callback(self, msg: TFMessage) -> None:
        position, orientation, stamp = self._read_tf_position_orientation(msg)
        if position is None:
            return
        self.object_position = position
        self._mark_received("/object")
        self.contact_force = orientation[0]
        self.object_lost = orientation[1] == 0
        self.collision_occurred = orientation[2] == 1
        self.collision_force = orientation[3] if self.collision_occurred else 0.0

        frame = int(getattr(stamp, "sec", getattr(stamp, "secs", 0))) % 101
        if frame == 0 and self.object_frame is not None and self.object_frame != 0:
            self.object_frame_carry += 1
        self.object_frame = frame

    def xform_callback(self, msg: TFMessage) -> None:
        position, orientation, stamp = self._read_tf_position_orientation(msg)
        if position is None:
            return
        self.xform_position = position
        self.xform_orientation = orientation
        self._mark_received("/xform")

        sim_time = (self._stamp_to_sec(stamp) * 60) % 100
        if sim_time == 0 and self.xform_time is not None and self.xform_time != 0:
            self.xform_time_carry += 1
        self.xform_time = sim_time

    def robot01_callback(self, msg: TFMessage) -> None:
        position, orientation, _ = self._read_tf_position_orientation(msg)
        if position is not None:
            self.robot01_position = position
            self.robot01_orientation = orientation
            self._mark_received("/robot01")

    def robot02_callback(self, msg: TFMessage) -> None:
        position, orientation, _ = self._read_tf_position_orientation(msg)
        if position is not None:
            self.robot02_position = position
            self.robot02_orientation = orientation
            self._mark_received("/robot02")

    def _collision_callback(self, msg: TFMessage, index: int) -> None:
        transform = self._first_transform(msg)
        if transform is None:
            return
        t = transform.transform.translation
        value = [t.x, t.y]
        if index == 1:
            self.collision01 = value
        elif index == 2:
            self.collision02 = value
        elif index == 3:
            self.collision03 = value
        elif index == 4:
            self.collision04 = value

    def dynaOb01_collide_callback(self, msg: TFMessage) -> None:
        self._collision_callback(msg, 1)

    def dynaOb02_collide_callback(self, msg: TFMessage) -> None:
        self._collision_callback(msg, 2)

    def dynaOb03_collide_callback(self, msg: TFMessage) -> None:
        self._collision_callback(msg, 3)

    def dynaOb04_collide_callback(self, msg: TFMessage) -> None:
        self._collision_callback(msg, 4)

    def _dynamic_obstacle_callback(self, msg: TFMessage, index: int) -> None:
        transform = self._first_transform(msg)
        if transform is None:
            return
        t = transform.transform.translation
        self.dynaOb_positions[index] = np.array([t.x, t.y, 0.0], dtype=float)

    def dynaOb01_callback(self, msg: TFMessage) -> None:
        self._dynamic_obstacle_callback(msg, 0)

    def dynaOb02_callback(self, msg: TFMessage) -> None:
        self._dynamic_obstacle_callback(msg, 1)

    def dynaOb03_callback(self, msg: TFMessage) -> None:
        self._dynamic_obstacle_callback(msg, 2)

    def dynaOb04_callback(self, msg: TFMessage) -> None:
        self._dynamic_obstacle_callback(msg, 3)

    def first_ot_callback(self, msg: TFMessage) -> None:
        transform = self._first_transform(msg)
        if transform is not None:
            t = transform.transform.translation
            self.first_ot_position = np.array([t.x, t.y, t.z], dtype=float)

    def second_ot_callback(self, msg: TFMessage) -> None:
        transform = self._first_transform(msg)
        if transform is not None:
            t = transform.transform.translation
            self.second_ot_position = np.array([t.x, t.y, t.z], dtype=float)

    def publish_velocity(self, linear_velocity_x: float, linear_velocity_y: float, angular_velocity_z: float) -> None:
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_velocity_x)
        twist_msg.linear.y = float(linear_velocity_y)
        twist_msg.angular.z = float(angular_velocity_z)
        self.velocity_publisher.publish(twist_msg)

    def publish_reset_signal(self, reset_needed: float) -> None:
        twist_msg = Twist()
        twist_msg.angular.x = float(reset_needed)
        self.reset_publisher.publish(twist_msg)

    def get_state(self, step_index: int, csv_length: int, step_size: float, yaw_error: float) -> np.ndarray:
        progress = step_index / csv_length
        distance_to_object = np.linalg.norm(self.xform_position[:2] - self.object_position[:2])
        distance_to_robot02 = (
            np.linalg.norm(self.xform_position[:2] - self.robot02_position[:2])
            - np.linalg.norm(np.array([2, -15]) - np.array([1.898, -15]))
        )
        return np.array([progress, step_size, distance_to_object, yaw_error, distance_to_robot02])


class NewDDPGEnv(gym.Env):
    """Gym-style environment wrapping the original DQN trajectory controller."""

    def __init__(self, listener: TransformListener, csv_path: str):
        super().__init__()
        self.listener = listener
        self.cumulative_reward = 0.0
        self.action_space = spaces.Discrete(10)

        self.csv_path = str(csv_path)
        self.path: List[Tuple[float, float]] = []
        self.load_path(self.csv_path)

        self.current_index = 0
        self.object_lost_counter = 0
        self.ignore_rotation = False
        self.ignore_rotation_end_index = -1
        self.move_duration = 1 / 5

        # These limits only prevent stale-state runaway. They do not change the path index or target selection.
        self.MAX_ABS_LINEAR_VEL = 30.0
        self.MAX_ABS_ANGULAR_VEL = 500.0

        self.last_time = time.time()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)

    def load_path(self, csv_path: str) -> None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find trajectory CSV: {csv_path}")
        with path.open("r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            self.path = [(float(row[0]), float(row[1])) for row in reader]
        if not self.path:
            raise RuntimeError(f"CSV file has no usable path rows: {csv_path}")
        self.listener.get_logger().info(
            f"Path loaded from CSV file: {csv_path}; points={len(self.path)}; "
            f"first={self.path[0]}, last={self.path[-1]}"
        )

    def _clip_command(self, value: float, limit: float) -> float:
        return float(np.clip(value, -limit, limit)) if limit is not None else float(value)

    def _assert_state_ready_for_motion(self) -> None:
        missing = [name for name, ok in self.listener.received_topics.items() if not ok]
        if missing:
            raise RuntimeError(
                f"Refusing to move because required Isaac state topics are missing: {missing}. "
                "Check ros2 topic list/echo and the Isaac Sim Action Graph Play state."
            )
        if np.linalg.norm(self.listener.xform_position[:2]) < 1e-9:
            raise RuntimeError(
                "Refusing to move because xform_position is still [0, 0]. "
                "This would make the controller command a large fixed-direction velocity. "
                "Check the /xform ROS 2 publish node and reset timing."
            )

    def step(self, action: int):
        self._assert_state_ready_for_motion()

        # Original trajectory/action math.
        step_size = 10.344 + (action + 1) * 0.156
        next_index = min(int(self.current_index + step_size), len(self.path) - 1)

        current_time = time.time()
        _dt = current_time - self.last_time
        self.last_time = current_time

        current_x, current_y = self.listener.xform_position[:2]
        target_x, target_y = self.path[next_index]

        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.linalg.norm([dx, dy])

        v_factor = 0.28
        w_factor = 1
        vx = v_factor * dx / self.move_duration
        vy = v_factor * dy / self.move_duration

        yaw = self.get_current_yaw()
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.compute_yaw_error(target_yaw, yaw)

        if not self.ignore_rotation:
            self.check_future_yaw_difference(next_index)
        if abs(yaw_error) > math.pi / 10:
            yaw_error = 0
        if self.ignore_rotation and self.current_index < self.ignore_rotation_end_index:
            yaw_error = 0
        else:
            self.ignore_rotation = False

        angular_velocity_z = w_factor * math.degrees(yaw_error) / self.move_duration

        vx = self._clip_command(vx, self.MAX_ABS_LINEAR_VEL)
        vy = self._clip_command(vy, self.MAX_ABS_LINEAR_VEL)
        angular_velocity_z = self._clip_command(angular_velocity_z, self.MAX_ABS_ANGULAR_VEL)

        reward, done = self.calculate_reward(next_index, action)
        self.cumulative_reward += reward

        print(
            "[CTRL] "
            f"idx {self.current_index}->{next_index}, action={action}, step_size={step_size:.3f}, "
            f"cur=({current_x:.4f},{current_y:.4f}), target=({target_x:.4f},{target_y:.4f}), "
            f"d=({dx:.4f},{dy:.4f}), dist={distance:.4f}, "
            f"cmd=({vx:.4f},{vy:.4f},{angular_velocity_z:.4f}), yaw_err={yaw_error:.4f}"
        )

        if done:
            self.publish_velocity(0.0, 0.0, 0.0)
            self.publish_reset_signal(1.0)
            time.sleep(2)
            self.publish_reset_signal(0.0)
        else:
            self.publish_velocity(vx, vy, angular_velocity_z)
            self.publish_reset_signal(0.0)

        time.sleep(self.move_duration)
        observation = self.get_observation(next_index, len(self.path), step_size, yaw_error)
        self.current_index = next_index
        return observation, reward, done, {}

    def publish_velocity(self, linear_velocity_x: float, linear_velocity_y: float, angular_velocity_z: float) -> None:
        self.listener.publish_velocity(linear_velocity_x, linear_velocity_y, angular_velocity_z)

    def publish_reset_signal(self, signal: float) -> None:
        self.listener.publish_reset_signal(signal)

    def get_current_yaw(self) -> float:
        x, y, z, w = self.listener.xform_orientation
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def compute_yaw_error(self, target_yaw: float, current_yaw: float) -> float:
        yaw_error_1 = target_yaw - current_yaw
        yaw_error_2 = target_yaw - (current_yaw + math.pi)

        if yaw_error_1 > math.pi:
            yaw_error_1 -= 2 * math.pi
        elif yaw_error_1 < -math.pi:
            yaw_error_1 += 2 * math.pi

        if yaw_error_2 > math.pi:
            yaw_error_2 -= 2 * math.pi
        elif yaw_error_2 < -math.pi:
            yaw_error_2 += 2 * math.pi

        if abs(yaw_error_1) < abs(yaw_error_2):
            return 0 if abs(yaw_error_1) > math.pi / 10 else yaw_error_1
        return 0 if abs(yaw_error_2) > math.pi / 10 else yaw_error_2

    def check_future_yaw_difference(self, next_index: int) -> None:
        future_steps = 100
        if next_index + future_steps + 1 < len(self.path):
            current_target_yaw = math.atan2(
                self.path[next_index + 1][1] - self.path[next_index][1],
                self.path[next_index + 1][0] - self.path[next_index][0],
            )
            future_target_yaw = math.atan2(
                self.path[next_index + future_steps + 1][1] - self.path[next_index + future_steps][1],
                self.path[next_index + future_steps + 1][0] - self.path[next_index + future_steps][0],
            )
            yaw_diff = future_target_yaw - current_target_yaw
            if yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            elif yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi
            if abs(yaw_diff) > math.pi / 2:
                self.ignore_rotation = True
                self.ignore_rotation_end_index = next_index + future_steps

    def calculate_reward(self, step_index: int, action: int):
        reward = -1
        done = False
        reset_reason = ""

        distance_to_object = np.linalg.norm(self.listener.xform_position[:2] - self.listener.object_position[:2])

        if distance_to_object < 0.05:
            reward += 3
        elif 0.05 <= distance_to_object <= 0.1:
            reward -= 0
        elif distance_to_object > 0.1:
            reward -= (distance_to_object - 0.1) * 150
            if distance_to_object > 0.20:
                reward -= 15
                reset_reason = "Distance to object too large"

        if self.listener.object_lost:
            self.object_lost_counter += 1
            reward -= 15
            if self.object_lost_counter >= 10:
                reward -= 17
                reset_reason = "Object lost for too long"
        else:
            self.object_lost_counter = 0

        if 0.05 < self.listener.contact_force < 10:
            reward += 5
        elif self.listener.contact_force <= 0.05:
            reward -= -5 + (0.05 - self.listener.contact_force) * 350
        elif 10 <= self.listener.contact_force < 20:
            reward -= -5 + (self.listener.contact_force - 10) * 2.5
        elif self.listener.contact_force >= 20:
            reward -= 20

        if action > 5:
            reward += 0.85 * (30 + (action - 5) * (40 - 8) / 4)
        elif action < 2:
            reward += 0.85 * (-10 + (action - 2) * (50 - 40) / 2)
        else:
            reward += 17 + (action - 2) * (23 - 8) / 4

        progress = step_index / len(self.path)
        if progress >= 0.999:
            reward += 300
            done = True
            reset_reason = "Progress exceeded 99.9%"

        if done:
            self.listener.get_logger().info(f"Episode reset due to: {reset_reason}")
        return reward, done

    def get_observation(self, step_index: int, csv_length: int, step_size: float, yaw_error: float) -> np.ndarray:
        return self.listener.get_state(step_index, csv_length, step_size, yaw_error)

    def reset(self) -> np.ndarray:
        self.publish_velocity(0.0, 0.0, 0.0)
        time.sleep(0.1)

        self.current_index = 0
        self.object_lost_counter = 0
        self.cumulative_reward = 0.0

        # Do not clear listener positions in ROS 2. If we set xform/object to zeros
        # and Isaac does not publish a fresh state immediately, the controller can
        # command a large fixed-direction velocity from a stale zero pose.
        before_counts = self.listener.snapshot_counts()

        self.publish_reset_signal(1.0)
        time.sleep(1.0)
        self.publish_reset_signal(0.0)

        self.listener.wait_for_fresh_messages(before_counts, timeout_sec=3.0, min_advanced_topics=("/xform", "/object"))
        self._assert_state_ready_for_motion()

        observation = self.get_observation(self.current_index, len(self.path), 0, 0)
        self.listener.get_logger().info(
            f"Reset finished. xform={self.listener.xform_position[:2]}, "
            f"object={self.listener.object_position[:2]}, initial_state={observation}"
        )
        return observation

    def render(self, mode="human"):
        pass

    def close(self) -> None:
        self.publish_velocity(0.0, 0.0, 0.0)
        self.publish_reset_signal(0.0)
