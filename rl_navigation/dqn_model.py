"""Deep Q-Network model used by the ROS 2 deployment script.

This file intentionally contains only the model and action-selection logic needed
for deployment. The trained weights are stored in models/best_model_750.pth.
"""

import random
import collections
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Simple replay buffer retained for compatibility with the original DQN code."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, Iterable[int], Iterable[float], np.ndarray, Iterable[bool]]:
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self) -> int:
        return len(self.buffer)


class FullyConnectedQnet(torch.nn.Module):
    """Fully connected Q-network matching the original trained model."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.head = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQN:
    """DQN wrapper used for both deployment and optional retraining."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        target_update: int,
        device: torch.device,
    ):
        self.action_dim = action_dim
        self.q_net = FullyConnectedQnet(state_dim, self.action_dim).to(device)
        self.target_q_net = FullyConnectedQnet(state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        return int(self.q_net(state_tensor).argmax().item())

    def update(self, transition_dict: Dict[str, np.ndarray]) -> None:
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end
