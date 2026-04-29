"""Optional CSV logger for ROS 2 Isaac Sim state topics.

This logger is not required for deployment. It is useful when you want to record
object, robot, and optical tweezer positions during a rollout.
"""

import argparse
import csv
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from ros2_env import TransformListener


def as_xy_text(vec) -> str:
    arr = np.asarray(vec, dtype=float)
    return f"{arr[0]:.6f},{arr[1]:.6f}"


def start_data_logging(csv_path: str, rate_hz: float = 5.0) -> None:
    path = Path(csv_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    listener = TransformListener("data_logger")
    executor = MultiThreadedExecutor()
    executor.add_node(listener)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        listener.wait_for_initial_messages(timeout_sec=10.0, raise_on_timeout=False)
        start = time.time()
        period = 1.0 / rate_hz

        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "time_s",
                "object_xy",
                "xform_xy",
                "robot01_xy",
                "robot02_xy",
                "contact_force",
                "first_ot_xy",
                "second_ot_xy",
            ])

            print(f"[INFO] Logging to {path}. Press Ctrl+C to stop.")
            while rclpy.ok():
                writer.writerow([
                    f"{time.time() - start:.3f}",
                    as_xy_text(listener.object_position),
                    as_xy_text(listener.xform_position),
                    as_xy_text(listener.robot01_position),
                    as_xy_text(listener.robot02_position),
                    f"{listener.contact_force:.6f}",
                    as_xy_text(listener.first_ot_position),
                    as_xy_text(listener.second_ot_position),
                ])
                file.flush()
                time.sleep(period)
    finally:
        executor.shutdown()
        listener.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Isaac Sim ROS 2 state topics to CSV.")
    parser.add_argument("--output", default="logs/rollout.csv", help="Output CSV path.")
    parser.add_argument("--rate", type=float, default=5.0, help="Logging frequency in Hz.")
    args = parser.parse_args()
    start_data_logging(args.output, args.rate)


if __name__ == "__main__":
    main()
