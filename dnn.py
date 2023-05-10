from typing import Tuple

from torch import nn, Tensor

DEFAULT_STATE_SHAPE = (84, 84, 4)


class ConvBackboneOld(nn.Module):
    def __init__(self,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 n_filters: int = 32
                 ) -> None:
        super().__init__()

        def _out_size(size):
            return (size - 28) // 8

        f1 = n_filters
        f2 = 2*n_filters

        self.output_size = f2 * \
            _out_size(state_shape[1]) * _out_size(state_shape[2])

        self.net = nn.Sequential(
            nn.Conv2d(state_shape[0], f1, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(f2, f2, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvBackbone(nn.Module):
    def __init__(self,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 n_filters: int = 32
                 ) -> None:
        super().__init__()

        def _out_size(size):
            return (size - 28) // 8

        f1 = n_filters
        f2 = 2*n_filters

        self.output_size = f2 * \
            _out_size(state_shape[1]) * _out_size(state_shape[2])

        self.net = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(state_shape[0], f1, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeepQNetwork(nn.Module):
    def __init__(self,
                 n_actions: int = 3,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 hidden_size: int = 128,
                 n_filters: int = 32
                 ) -> None:
        super().__init__()

        self.conv = ConvBackbone(state_shape, n_filters)

        self.ff = nn.Sequential(
            nn.Linear(self.conv.output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return self.ff(x)


class DuelingDQN(nn.Module):
    pass

    def __init__(self,
                 n_actions: int = 3,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 hidden_size: int = 128,
                 n_filters: int = 32
                 ) -> None:
        super().__init__()

        self.conv = ConvBackbone(state_shape, n_filters)

        self.ff_value = nn.Sequential(
            nn.Linear(self.conv.output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.ff_advantage = nn.Sequential(
            nn.Linear(self.conv.output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        value: Tensor = self.ff_value(x)
        advantage: Tensor = self.ff_advantage(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
