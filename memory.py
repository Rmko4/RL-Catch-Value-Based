# %%
import random
from collections import deque
from collections.abc import Iterator
from typing import List, NamedTuple

import numpy as np
from torch.utils.data import IterableDataset


class Trajectory(NamedTuple):
    # Using typing.NamedTuple
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class ReplayBuffer():
    def __init__(self, capacity: int) -> None:
        # Class does not extend deque to expose only the required methods
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, trajectory: Trajectory) -> None:
        self.buffer.append(trajectory)

    def sample(self, batch_size: int) -> List[Trajectory]:
        return random.sample(self.buffer, batch_size)

    def choice(self) -> Trajectory:
        return random.choice(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    pass


class ReplayBufferDataset(IterableDataset):
    """ Iterable dataset for replay buffer
        Supports random sampling of dynamic replay buffer
    """

    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        self.replay_buffer = replay_buffer

    def __iter__(self) -> Iterator[Trajectory]:
        while True:
            # TODO: Might need to cast: tuple()
            yield self.replay_buffer.choice()
           