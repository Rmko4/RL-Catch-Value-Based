import random
from collections import deque
from collections.abc import Iterator
from typing import List, NamedTuple, Tuple

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
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, trajectory: Trajectory) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
        else:
            self.buffer[self.index] = trajectory
        self.index = (self.index + 1) % self.capacity

    # TODO: Consider output type
    def sample(self, batch_size: int, *args, **kwargs) -> List[Trajectory]:
        raise NotImplementedError

    def choice(self) -> Trajectory:
        raise NotImplementedError


class UniformReplayBuffer(ReplayBuffer):
    def sample(self, batch_size: int) -> List[Trajectory]:
        return random.sample(self.buffer, batch_size)

    def choice(self) -> Trajectory:
        return random.choice(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        super().__init__(capacity)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.p = np.zeros(capacity, dtype=np.float32)
        self.weights = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta

    # TODO: Consider appending while trajectory is still to be updated.
    def append(self, trajectory: Trajectory) -> None:
        self.priorities[self.index] = self.priorities.max() \
            if self.buffer else 1.0
        super().append(trajectory)

    def sample(self, batch_size: int) -> Tuple[List[Trajectory], np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, p=self.p)
        weights = self.weights[indices]
        trajectories = [self.buffer[i] for i in indices]
        return trajectories, indices, weights

    def choice(self) -> Tuple[Trajectory, int, float]:
        index = np.random.choice(len(self.buffer), p=self.p)
        return self.buffer[index], index, self.weights[index]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        p = self.priorities[:len(self.buffer)] ** self.alpha
        self.p /= p.sum()
        self.weights[:len(self.buffer)] = (
            len(self.buffer) * p) ** (-self.beta)
        self.weights /= self.weights.max()


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
