from typing import Any, Callable

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

from agent import QNetworkAgent
from catch import CatchEnv
from dnn import DeepQNetwork, DeepVNetwork, DuelingDQN
from memory import (PRIORITIZED_TRAJECTORY, PrioritizedReplayBuffer,
                    ReplayBufferDataset, Trajectory, UniformReplayBuffer)
from network_update import NetUpdater
from scheduler import EpsilonDecay

MODES = {'DQN': DeepQNetwork,
         'Dueling_architecture': DuelingDQN,
         'DQV': DeepQNetwork,
         'DQV_max': DeepQNetwork}

DQV_FAMILY = ['DQV', 'DQV_max']

OPTIMIZERS = {'Adam': Adam,
              'RMSprop': RMSprop,
              'SGD': SGD}


class CatchRLModule(LightningModule):
    def __init__(self,
                 episodes_per_epoch: int = 10,
                 algorithm: str = 'DQN',
                 double_q_learning: bool = False,
                 batch_size: int = 32,
                 batches_per_step: int = 1,
                 optimizer: str = 'Adam',
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
                 target_net_update_freq: int | None = None,
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

        self.algorithm = algorithm
        Q_net_cls = MODES[algorithm]

        # Initialize Q-networks
        self.Q_network: nn.Module = Q_net_cls(
            n_actions, state_shape, hidden_size, n_filters)
        if not algorithm == 'DQV':
            self.target_Q_network: nn.Module = Q_net_cls(
                n_actions, state_shape, hidden_size, n_filters)

        # Initialize V-networks
        if algorithm in DQV_FAMILY:
            self.V_network = DeepVNetwork(state_shape, hidden_size, n_filters)
            if algorithm == 'DQV':
                self.target_V_network = DeepVNetwork(
                    state_shape, hidden_size, n_filters)

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

        self.loss = nn.MSELoss()
        self.episode = 0
        self.episode_reward = 0
        self.batch_step = 0

    def compute_next_Q(self, batch: Trajectory) -> Tensor:
        raise NotImplementedError

    def update_target_network(self):
        raise NotImplementedError

    def forward(self, x):
        return self.Q_network(x)

    def replay_warmup(self):
        hp = self.hparams
        warmup_steps: int = hp.replay_warmup_steps
        for _ in range(warmup_steps):
            _, terminal = self.agent.step(freeze_time=True, epsilon=1.)
            if terminal:
                self.episode += 1

    def target_update_fn(self) -> Callable:
        if self.algorithm != 'DQV':
            policy_net = self.Q_network
            target_net = self.target_Q_network
        else:
            policy_net = self.V_network
            target_net = self.target_V_network
        # Presence of target_net_update_freq indicates hard update
        soft_update = self.hparams.target_net_update_freq is None
        return NetUpdater(policy_net,
                          target_net,
                          soft_update,
                          update_freq=self.hparams.target_net_update_freq,
                          soft_update_tau=self.hparams.soft_update_tau).update

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
        V_values = self.V_network(batch.next_state).squeeze()
        V_values[batch.terminal] = 0.
        td_target = batch.reward + self.hparams.gamma * V_values
        return td_target

    def training_step(self, batch: Trajectory | PRIORITIZED_TRAJECTORY, batch_idx: int) -> Tensor:
        do_step = self.batch_step % self.hparams.batches_per_step == 0
        self.batch_step += 1

        if do_step:
            reward, terminal = self.agent.step()

        if self.hparams.prioritized_replay:
            batch, indices, weights = batch

        alg = self.hparams.algorithm
        if alg != 'DQV':
            td_target_Q = self.compute_td_target_Q(batch)

        # Unsqueeze to get (batch_size, 1) shape
        Q_values = self.Q_network(batch.state).gather(
            dim=-1, index=batch.action.unsqueeze(-1)).squeeze(-1)

        if alg in DQV_FAMILY:
            if alg == 'DQV':
                V_values = self.V_network(batch.state).squeeze()
                td_target_V = self.compute_td_target_V(batch)

                loss_V = self.loss(V_values, td_target_V)
                loss_Q = self.loss(Q_values, td_target_V)
            else:  # DQV_max
                V_values = self.V_network(batch.state).squeeze()
                td_target_V = self.compute_td_target_V(batch)

                loss_V = self.loss(V_values, td_target_Q)
                loss_Q = self.loss(Q_values, td_target_V)

            self.log('train/loss_V', loss_V, on_step=False, on_epoch=True)
            self.log('train/loss_Q', loss_Q, on_step=False, on_epoch=True)

            loss = loss_Q + loss_V

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
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if do_step:
            self.episode_reward += reward
            self.log('epsilon', self.agent.epsilon, on_step=True,
                     on_epoch=False, prog_bar=False)
            self.log_dict({'step': float(self.global_step),
                           'episode': float(self.episode),
                           }, on_step=True, on_epoch=False, prog_bar=True)

            if terminal:
                self.log("episode reward", float(self.episode_reward),
                         on_step=True, on_epoch=True, prog_bar=False)
                self.episode += 1
                self.episode_reward = 0
                if self.episode % self.hparams.episodes_per_epoch == 0:
                    self.dataset.end()

        return loss

    def on_train_epoch_end(self) -> None:
        self.test_epoch()

    def test_epoch(self):
        total_reward = 0
        self.agent.reset()

        n_epochs = 10
        for _ in range(n_epochs):
            terminal = False
            while not terminal:
                reward, terminal = self.agent.step(
                    freeze_time=True, epsilon=0.)
                total_reward += reward

        average_reward = total_reward / n_epochs
        self.log("test/total_reward", float(total_reward),
                 on_epoch=True, prog_bar=False)
        self.log("test/average_reward", average_reward,
                    on_epoch=True, prog_bar=True)

    def train_dataloader(self) -> DataLoader:
        # First call
        if len(self.replay_buffer) < self.hparams.replay_warmup_steps:
            self.replay_warmup()
        self.dataset = ReplayBufferDataset(self.replay_buffer)
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size)

    def configure_optimizers(self) -> Optimizer:
        # Both Q-network and V-network parameters are optimized
        if self.algorithm in DQV_FAMILY:
            params = list(self.Q_network.parameters()) + \
                list(self.V_network.parameters())
            return OPTIMIZERS[self.hparams.optimizer](params, lr=self.hparams.learning_rate)

        # Only the Q-network's parameters are optimized
        return OPTIMIZERS[self.hparams.optimizer](self.Q_network.parameters(), lr=self.hparams.learning_rate)
