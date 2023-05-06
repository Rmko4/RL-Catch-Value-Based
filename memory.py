import random
from collections import deque
from collections.abc import Iterator
from typing import List, NamedTuple

import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset


class Trajectory(NamedTuple):
    # Using typing.NamedTuple
    state: np.ndarray | Tensor
    action: int | Tensor
    reward: float | Tensor
    next_state: np.ndarray | Tensor
    terminal: bool | Tensor


class ReplayBuffer:
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

class ReplayBufferFastAccess:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self, trajectory: Trajectory) -> None:
        self.buffer[self.index] = trajectory
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Trajectory]:
        if self.size < batch_size:
            return []
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[i] for i in indices]

    def choice(self) -> Trajectory:
        if self.size == 0:
            return None
        index = random.randint(0, self.size - 1)
        return self.buffer[index]

class PrioritizedReplayBuffer(ReplayBuffer):
    pass


class ReplayBufferDataset(IterableDataset):
    """ Iterable dataset for replay buffer
        Supports random sampling of dynamic replay buffer
    """

    def __init__(self, replay_buffer: ReplayBuffer, epoch_length: int = 2000) -> None:
        self.replay_buffer = replay_buffer
        self.epoch_length = epoch_length

    def __iter__(self) -> Iterator[Trajectory]:
        for _ in range(self.epoch_length):
            yield self.replay_buffer.choice()


def main():
    from torch.utils.data import DataLoader
    buffer = ReplayBuffer(10)
    for i in range(10):
        buffer.append(Trajectory(i, i, i, i, False))
    dataset = ReplayBufferDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
