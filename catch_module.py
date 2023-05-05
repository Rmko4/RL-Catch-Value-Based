from typing import Any
from pytorch_lightning import LightningModule

from torch.utils.data import DataLoader
from torch.optim import Optimizer, RMSprop, Adam
from catch import CatchEnv
from dnn import DeepQNetwork

from memory import ReplayBuffer, ReplayBufferDataset
from agent import QNetworkAgent
from scheduler import EpsilonDecay


class CatchRLModule(LightningModule):
    def __init__(self,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.1,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: float = 1000,
                 buffer_capacity: int = 1000,
                 replay_warmup_steps: int = 10,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        self.env = CatchEnv

        n_actions = self.env.get_num_actions()
        state_shape = self.env.state_shape()[::-1]

        # Initialize networks
        self.Q_network = DeepQNetwork(n_actions, state_shape)
        self.target_Q_network = DeepQNetwork(n_actions, state_shape)
        self.update_target_network()

        # Initialize replay buffer and agent
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        epsilon_schedule = EpsilonDecay(
            epsilon_start, epsilon_end, epsilon_decay_steps)
        self.agent = QNetworkAgent(
            self.env, self.Q_network, self.replay_buffer, epsilon_schedule)

    def replay_warmup(self):
        for _ in range(self.hparams.replay_warmup_steps):
            self.agent.step()

    def update_target_network(self):
        # TODO: Add soft update with parameter tau
        # Copy the weights from Q_network to target_Q_network
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())

    def train_dataloader(self) -> DataLoader:
        if len(self.replay_buffer) < self.hparams.replay_warmup_steps:
            self.replay_warmup()
        dataset = ReplayBufferDataset(self.replay_buffer)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
