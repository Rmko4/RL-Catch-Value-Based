from typing import Any
from pytorch_lightning import LightningModule

from torch.utils.data import DataLoader
from torch.optim import Optimizer, RMSprop, Adam
from catch import CatchEnv
from dnn import DeepQNetwork

from memory import ReplayBuffer, ReplayBufferDataset


class CatchRLModule(LightningModule):
    def __init__(self,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,

                 buffer_capacity: int = 1000,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        self.env = CatchEnv

        n_actions = self.env.get_num_actions()
        state_shape = self.env.state_shape()[::-1]

        self.Q_network = DeepQNetwork(n_actions, state_shape)
        self.target_Q_network = DeepQNetwork(n_actions, state_shape)

        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def train_dataloader(self) -> DataLoader:
        dataset = ReplayBufferDataset(self.buffer)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
