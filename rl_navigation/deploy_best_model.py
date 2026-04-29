"""Deploy the trained DQN controller through ROS 2 topics.

Run from the repository root:

    python rl_navigation/deploy_best_model.py

Make sure Isaac Sim is already open, the USD scene is loaded, Timeline is Play,
and the ROS 2 bridge Action Graph is publishing /object, /xform, /robot01, and /robot02.
"""

import argparse
from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
import torch

from dqn_model import DQN
from ros2_env import NewDDPGEnv, TransformListener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "smoothed_path.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_750.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy ROS 2 DQN navigation controller for Isaac Sim.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Path to smoothed trajectory CSV.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to trained DQN .pth file.")
    parser.add_argument("--wait-timeout", type=float, default=15.0, help="Seconds to wait for initial Isaac state topics.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional maximum control steps. 0 means run until done.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find CSV path: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model path: {model_path}")

    rclpy.init()
    listener = TransformListener("dqn_deployment")
    executor = MultiThreadedExecutor()
    executor.add_node(listener)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    env = None
    try:
        device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        agent = DQN(
            state_dim=5,
            action_dim=10,
            learning_rate=0.0002,
            gamma=0.99,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            target_update=10,
            device=device,
        )

        print(f"[INFO] Loading model from: {model_path}")
        agent.q_net.load_state_dict(torch.load(str(model_path), map_location=device))
        agent.q_net.eval()

        env = NewDDPGEnv(listener, str(csv_path))
        listener.wait_for_initial_messages(timeout_sec=args.wait_timeout, raise_on_timeout=True)

        state = env.reset()
        done = False
        total_reward = 0.0
        count = 0

        while not done and rclpy.ok():
            if args.max_steps and count >= args.max_steps:
                print(f"[INFO] Reached --max-steps={args.max_steps}; stopping.")
                break

            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)

            print(f"Step: {count}, State: {state}, Action: {action}, Reward: {reward}, Done: {done}")
            state = next_state
            total_reward += reward
            count += 1

        print(f"Total reward obtained: {total_reward}")

    finally:
        if env is not None:
            env.close()
        executor.shutdown()
        listener.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
