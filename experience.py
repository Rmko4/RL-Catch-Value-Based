from collections import namedtuple, deque
import random
from typing import List

# Trajectory class
Trajectory = namedtuple(
    'Trajectory', ['state', 'action', 'reward', 'next_state', 'done'])
#TODO: Add type hints


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
        #TODO: Unpack here
