from typing import Tuple

from torch import nn, Tensor

DEFAULT_STATE_SHAPE = (84, 84, 4)


class DeepQNetwork(nn.Module):
    pass

    def __init__(self,
                 n_actions: int = 3,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 hidden_size: int = 128,
                 n_filters: int = 32
                 ) -> None:
        super().__init__()

        def _out_size(size):
            return (size - 28) // 8

        f1 = n_filters
        f2 = 2*n_filters

        flatten_input_size = f2 * \
            _out_size(state_shape[1]) * _out_size(state_shape[2])

        self.net = nn.Sequential(
            nn.Conv2d(state_shape[0], f1, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(f1, f2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(f2, f2, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flatten_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
