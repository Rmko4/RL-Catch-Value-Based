import math
from dataclasses import dataclass


@dataclass
class epsilon_decay():
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: float

    def __call__(self, t):
        epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * t / self.epsilon_decay_steps)
        return epsilon
