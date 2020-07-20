from typing import List
import torch
from torch import nn
import gym
from ..logger import Loggable, LogType


def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module, Loggable):
    """Defines a standard Multi Layer Perceptron"""

    def __init__(
        self, input_dim: int, hidden_dim: List[int], output_dim: int, output_mod=None
    ) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        # create mlp
        sizes = [input_dim] + hidden_dim + [output_dim]
        mods = sum(
            [
                [nn.Linear(i, j), nn.ReLU(inplace=True)]
                for i, j in zip(sizes, sizes[1:])
            ],
            [],
        )
        mods = mods[:-1]
        if output_mod is not None:
            mods.append(output_mod)
        self.mlp = nn.Sequential(*mods)
        self.apply(weight_init)

        # hyperparameters
        self._mlp_hyperparams = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "output_mod": output_mod.__class__.__name__ if output_mod else None,
        }

        self._losses = []

    def forward(self, x):
        return self.mlp.forward(x)

    def log_local_hyperparams(self):
        return self._mlp_hyperparams

    def log_local_epoch(self):
        return {k: LogType.histogram(v) for k, v in self.mlp.state_dict().items()}

    # TODO: fix this
    def log_local_save(self, obj):
        return self.mlp
