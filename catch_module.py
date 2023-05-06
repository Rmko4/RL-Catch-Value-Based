from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

from agent import QNetworkAgent
from catch import CatchEnv
from dnn import DeepQNetwork
from memory import ReplayBuffer, ReplayBufferDataset, Trajectory
from scheduler import EpsilonDecay


class CatchRLModule(LightningModule):
    def __init__(self,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.1,
                 epsilon_end: float = 0.01,
                 epsilon_decay_rate: float = 1000,
                 buffer_capacity: int = 1000,
                 replay_warmup_steps: int = 10,
                 target_net_update_freq: int = 100,
                 hidden_size: int = 128,
                 n_filters: int = 32,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        self.env = CatchEnv()

        n_actions = self.env.get_num_actions()
        state_shape = self.env.state_shape()

        # Initialize networks
        self.Q_network = DeepQNetwork(n_actions, state_shape, hidden_size, n_filters)
        self.target_Q_network = DeepQNetwork(n_actions, state_shape, hidden_size, n_filters)

        # Initialize replay buffer and agent
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        epsilon_schedule = EpsilonDecay(
            epsilon_start, epsilon_end, epsilon_decay_rate)
        self.agent = QNetworkAgent(
            self.env, self.Q_network, self.replay_buffer, epsilon_schedule)

        self.loss = nn.MSELoss()
        self.episode = 0
        self.episode_reward = 0

    def forward(self, x):
        return self.Q_network(x)

    def replay_warmup(self):
        for _ in range(self.hparams.replay_warmup_steps):
            self.agent.step()

    def update_target_network(self):
        # TODO: Add soft update with parameter tau
        # Copy the weights from Q_network to target_Q_network
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())

    @torch.no_grad()
    def compute_td_target(self, batch: Trajectory) -> Tensor:
        Q_values = self.target_Q_network(batch.next_state).max(dim=-1).values
        Q_values[batch.terminal] = 0.  # Terminal state has 0 Q-value
        td_target = batch.reward + self.hparams.gamma * Q_values
        return td_target

    def training_step(self, batch: Trajectory, batch_idx: int) -> Tensor:
        reward, terminal = self.agent.step()

        td_target = self.compute_td_target(batch)
        # Unsqueeze to get (batch_size, 1) shape
        Q_values = self.Q_network(batch.state).gather(
            dim=-1, index=batch.action.unsqueeze(-1)).squeeze(-1)

        loss = self.loss(Q_values, td_target)

        if self.global_step % self.hparams.target_net_update_freq == 0:
            self.update_target_network()

        # Logging
        self.episode_reward += reward

        self.log('epsilon', self.agent.epsilon, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({'step': self.global_step,
                       'episode': self.episode,
                       }, on_step=True, on_epoch=False, prog_bar=True)

        if terminal:
            self.log("episode reward", self.episode_reward,
                     on_step=True, on_epoch=True, prog_bar=True)
            self.episode += 1
            self.episode_reward = 0

        return loss

    def train_dataloader(self) -> DataLoader:
        # First call
        if len(self.replay_buffer) < self.hparams.replay_warmup_steps:
            self.replay_warmup()
        dataset = ReplayBufferDataset(self.replay_buffer)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def configure_optimizers(self) -> Optimizer:
        # Only the Q-network's parameters are optimized
        return Adam(self.Q_network.parameters(), lr=self.hparams.learning_rate)
