from typing import List
import torch
import gym
from torch import nn
from .mlp import MLP
from .logger import Loggable, LogType


class DoubleVCritic(nn.Module, Loggable):
    """Value network, employes double Q-learning."""

    def __init__(self, env: gym.Env, hidden_dim: List[int]) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_dim = len(env.observation_space.low)

        self.vf_1 = MLP(obs_dim, hidden_dim, 1)
        self.vf_2 = MLP(obs_dim, hidden_dim, 1)

    def double_v(self, obs):
        self.v1 = self.vf_1(obs)
        self.v2 = self.vf_2(obs)
        return self.v1, self.v2

    def forward(self, obs):
        return torch.min(*self.double_v(obs))

    def log_local_hyperparams(self):
        return {}

    def log_local_epoch(self):
        return {
            "val_1": LogType.histogram(self.v1),
            "val_2": LogType.histogram(self.v2),
        }
