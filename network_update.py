from typing import Callable
from torch import nn


class NetUpdater:
    def __init__(self,
                 policy_net: nn.Module,
                 target_net: nn.Module,
                 soft_update: bool = False,
                 update_freq: int = 50,
                 soft_update_tau: float = 1e-2,
                 ) -> None:
        self.policy_net = policy_net
        self.target_net = target_net
        self.soft_update = soft_update
        self.update_freq = update_freq
        self.soft_update_tau = soft_update_tau

        self.global_step = 0
        self.update_fn = self.target_update_fn()
        self.hard_update_target_network()

    def update(self):
        self.global_step += 1
        self.update_fn()

    def target_update_fn(self) -> Callable:
        if self.soft_update:
            return self.soft_update_target_network
        else:
            return self.scheduled_update(self.hard_update_target_network)

    def scheduled_update(self, fn: Callable):
        def scheduled_fn():
            if self.global_step % self.update_freq == 0:
                fn()
        return scheduled_fn

    def hard_update_target_network(self):
        # Copy the weights from Q_network to target_Q_network
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self):
        Q_net_state = self.policy_net.state_dict()
        target_net_state = self.target_net.state_dict()

        tau = self.soft_update_tau
        for key in Q_net_state:
            target_net_state[key] = Q_net_state[key]*tau + \
                target_net_state[key]*(1 - tau)

        self.target_net.load_state_dict(target_net_state)
