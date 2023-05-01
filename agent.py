import random
from typing import Callable

import torch
from torch import nn

from catch import CatchEnv
from memory import ReplayBuffer, Trajectory


class QNetworkAgent:
    """Agent class for handeling the interactions with the environment using a Q-Network."""

    def __init__(self,
                 env: CatchEnv,
                 Q_network: nn.Module,
                 replay_buffer: ReplayBuffer,
                 epsilon_schedule: Callable[[int], float] = lambda t: 0.1,
                 ) -> None:
        self.env = env
        self.Q_network = Q_network
        self.replay_buffer = replay_buffer
        self.epsilon_schedule = epsilon_schedule

        self.device = next(Q_network.parameters()).device

        self.global_step = 0

        self.state = self.env.reset()

    def _sample_action(self) -> int:
        # Take a random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, self.env.get_num_actions() - 1)

        # Take the greedy action according to the Q-network
        state = torch.tensor(
            [self.state], dtype=torch.float32, device=self.device)
        Q_values = self.Q_network(state)
        return torch.argmax(Q_values).item()

    @torch.no_grad()
    def step(self):
        self._update_epsilon()

        action = self._sample_action()
        next_state, reward, terminal = self.env.step(action)

        trajectory = Trajectory(
            self.state, action, reward, next_state, terminal)
        self.replay_buffer.append(trajectory)

        self.state = next_state
        if terminal:
            self.state = self.env.reset()

        self.global_step += 1

        return reward, terminal

    def _update_epsilon(self):
        self.epsilon = self.epsilon_schedule(self.global_step)