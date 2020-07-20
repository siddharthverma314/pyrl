from typing import List
import torch
from torch import nn
from .mlp import MLP
from ..logger import Loggable, LogType


class DoubleQCritic(nn.Module, Loggable):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: List[int]) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        self.qf_1 = MLP(obs_dim + action_dim, hidden_dim, 1)
        self.qf_2 = MLP(obs_dim + action_dim, hidden_dim, 1)

    def double_q(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        self.q1 = self.qf_1(obs_action)
        self.q2 = self.qf_2(obs_action)
        return self.q1, self.q2

    def forward(self, obs, action):
        return torch.min(*self.double_q(obs, action))

    def log_local_hyperparams(self):
        return {}

    def log_local_epoch(self):
        return {
            "qval_1": LogType.histogram(self.q1),
            "qval_2": LogType.histogram(self.q2),
        }
