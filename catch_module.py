from typing import Any, Callable

import torch
from numpy import indices
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

from agent import QNetworkAgent
from catch import CatchEnv
from dnn import DeepVNetwork, DeepQNetwork, DuelingDQN
from memory import (PRIORITIZED_TRAJECTORY, PrioritizedReplayBuffer,
                    ReplayBufferDataset, Trajectory, UniformReplayBuffer)
from scheduler import EpsilonDecay

MODES = {'DQN': DeepQNetwork,
         'Dueling_architecture': DuelingDQN,
         'DQV_max': DeepQNetwork}


class CatchRLModule(LightningModule):
    def __init__(self,
                 episodes_per_epoch: int = 10,
                 algorithm: str = 'DQN',
                 double_q_learning: bool = False,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.1,
                 epsilon_end: float = 0.01,
                 epsilon_decay_rate: float = 1000,
                 buffer_capacity: int = 1000,
                 replay_warmup_steps: int = 10,
                 prioritized_replay: bool = False,
                 prioritized_replay_alpha: float = 0.6,
                 prioritized_replay_beta: float = 0.4,
                 target_net_update_freq: int = None,
                 soft_update_tau: float = 1e-3,
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

        Q_net_cls = MODES[algorithm]

        # Initialize networks
        self.Q_network = Q_net_cls(
            n_actions, state_shape, hidden_size, n_filters)
        self.target_Q_network = Q_net_cls(
            n_actions, state_shape, hidden_size, n_filters)
        
        if algorithm == 'DQV_max':
            self.V_network = DeepVNetwork(state_shape, hidden_size, n_filters)

        # Initialize replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=prioritized_replay_alpha,
                beta=prioritized_replay_beta)
        else:
            self.replay_buffer = UniformReplayBuffer(capacity=buffer_capacity)

        epsilon_schedule = EpsilonDecay(
            epsilon_start, epsilon_end, epsilon_decay_rate)
        self.agent = QNetworkAgent(
            self.env, self.Q_network, self.replay_buffer, epsilon_schedule)

        self.update_target_network = self.target_update_fn()
        self.compute_next_Q = self.compute_next_Q_fn()

        self.hard_update_target_network()

        self.loss = nn.MSELoss()
        self.episode = 0
        self.episode_reward = 0

    def forward(self, x):
        return self.Q_network(x)

    def replay_warmup(self):
        for _ in range(self.hparams.replay_warmup_steps):
            self.agent.step(freeze_time=True, epsilon=1.)

    def target_update_fn(self) -> Callable:
        # if self.hparams.double_q_learning:
        #     return self.scheduled_update(self.swap_networks)
        # Presence of target_net_update_freq indicates hard update
        if self.hparams.target_net_update_freq:
            return self.scheduled_update(self.hard_update_target_network)
        return self.soft_update_target_network

    def swap_networks(self):
        self.Q_network, self.target_Q_network = \
            self.target_Q_network, self.Q_network
        self.agent.Q_network = self.Q_network

    def scheduled_update(self, fn: Callable):
        def scheduled_fn():
            if self.global_step % self.hparams.target_net_update_freq == 0:
                fn()
        return scheduled_fn

    def hard_update_target_network(self):
        # Copy the weights from Q_network to target_Q_network
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())

    def soft_update_target_network(self):
        Q_net_state = self.Q_network.state_dict()
        target_net_state = self.target_Q_network.state_dict()

        tau = self.hparams.soft_update_tau
        for key in Q_net_state:
            target_net_state[key] = Q_net_state[key]*tau + \
                target_net_state[key]*(1 - tau)

        self.target_Q_network.load_state_dict(target_net_state)

    def compute_next_Q_fn(self) -> Callable:
        if self.hparams.double_q_learning:
            return self.next_double_Q_values
        return self.next_Q_values

    @torch.no_grad()
    def next_Q_values(self, batch: Trajectory) -> Tensor:
        Q_values = self.target_Q_network(batch.next_state).max(dim=-1).values
        return Q_values

    @torch.no_grad()
    def next_double_Q_values(self, batch: Trajectory) -> Tensor:
        max_actions = self.Q_network(
            batch.next_state).argmax(dim=-1, keepdim=True)
        Q_values = self.target_Q_network(batch.next_state).gather(
            dim=-1, index=max_actions).squeeze(-1)
        return Q_values

    @torch.no_grad()
    def compute_td_target_Q(self, batch: Trajectory) -> Tensor:
        Q_values = self.compute_next_Q(batch)
        Q_values[batch.terminal] = 0.  # Terminal state has 0 Q-value
        td_target = batch.reward + self.hparams.gamma * Q_values
        return td_target
    
    @torch.no_grad()
    def compute_td_target_V(self, batch: Trajectory) -> Tensor:
        V_values = self.V_network(batch.next_state)
        V_values[batch.terminal] = 0.
        td_target = batch.reward + self.hparams.gamma * V_values
        return td_target

    def training_step(self, batch: Trajectory | PRIORITIZED_TRAJECTORY, batch_idx: int) -> Tensor:
        reward, terminal = self.agent.step()

        if self.hparams.prioritized_replay:
            batch, indices, weights = batch

        td_target_Q = self.compute_td_target_Q(batch)
        
        # Unsqueeze to get (batch_size, 1) shape
        Q_values = self.Q_network(batch.state).gather(
            dim=-1, index=batch.action.unsqueeze(-1)).squeeze(-1)

        if self.hparams.algorithm == 'DQV_max':
            V_values = self.V_network(batch.state)
            td_target_V = self.compute_td_target_V(batch)

            loss_V = self.loss(V_values, td_target_Q)
            loss_Q = self.loss(Q_values, td_target_V)

            self.log('train/loss_V', loss_V, on_step=True, on_epoch=False)
            self.log('train/loss_Q', loss_Q, on_step=True, on_epoch=False)

            loss = loss_V + loss_Q
        else:
            if not self.hparams.prioritized_replay:
                loss = self.loss(Q_values, td_target_Q)
            else:
                errors: Tensor = td_target_Q - Q_values
                loss = (errors ** 2 * weights).mean()

                priorities = errors.abs() + 1e-6
                priorities = priorities.detach().cpu().numpy()
                indices = indices.detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)

        self.update_target_network()

        # Logging
        self.episode_reward += reward

        self.log('epsilon', self.agent.epsilon, on_step=True,
                 on_epoch=False, prog_bar=False)
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log_dict({'step': self.global_step,
                       'episode': self.episode,
                       }, on_step=True, on_epoch=False, prog_bar=True)

        if terminal:
            self.log("episode reward", self.episode_reward,
                     on_step=True, on_epoch=True, prog_bar=False)
            self.episode += 1
            self.episode_reward = 0
            if self.episode % self.hparams.episodes_per_epoch == 0:
                self.dataset.end()

        return loss

    def on_train_epoch_end(self) -> None:
        total_reward = 0
        self.agent.reset()

        for _ in range(10):
            terminal = False
            while not terminal:
                reward, terminal = self.agent.step(
                    freeze_time=True, epsilon=0.)
                total_reward += reward

        self.log("test/total_reward", total_reward,
                 on_epoch=True, prog_bar=True)

    def train_dataloader(self) -> DataLoader:
        # First call
        if len(self.replay_buffer) < self.hparams.replay_warmup_steps:
            self.replay_warmup()
        self.dataset = ReplayBufferDataset(self.replay_buffer)
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size)

    def configure_optimizers(self) -> Optimizer:
        # Only the Q-network's parameters are optimized
        return Adam(self.Q_network.parameters(), lr=self.hparams.learning_rate)
