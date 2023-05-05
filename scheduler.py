import math
from dataclasses import dataclass


@dataclass
class EpsilonDecay():
    epsilon_start: float
    epsilon_end: float
    decay_steps: float

    def __call__(self, t):
        epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            math.exp(-t / self.decay_steps)
        return epsilon
