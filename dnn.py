from typing import Tuple

from torch import nn, Tensor

DEFAULT_STATE_SHAPE = (84, 84, 4)


class DeepQNetwork(nn.Module):
    pass

    def __init__(self,
                 n_actions: int = 3,
                 state_shape: Tuple[int] = DEFAULT_STATE_SHAPE,
                 ) -> None:
        super().__init__()

        def _out_size(size):
            return (size - 12) // 8

        flatten_input_size = 64 * \
            _out_size(state_shape[0]) * _out_size(state_shape[1])

        self.net = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flatten_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
