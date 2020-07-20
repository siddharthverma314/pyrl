from typing import List, Tuple, Union
import torch
from torch import nn, distributions as pyd
from .mlp import MLP
from ..logger import Loggable, LogType


class SquashedNormal(pyd.TransformedDistribution):
    def __init__(self, loc, scale):
        super().__init__(
            pyd.Normal(loc, scale),
            pyd.TanhTransform(
                cache_size=1
            ),  # TODO: check performance of this vs cache_size=0
        )
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class GaussianActor(nn.Module, Loggable):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: List[int],
        log_std_bounds: Tuple[float, float],
        use_squashed_normal=False
    ) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        self.log_std_bounds = log_std_bounds
        self.policy = MLP(obs_dim, hidden_dim, act_dim * 2)
        self.outputs = dict()

    def forward(self, obs: Union[torch.Tensor, dict]):
        self._mu, log_std = self.policy(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = log_std.tanh()
        ls_min, ls_max = self.log_std_bounds
        self._log_std = ls_min + (ls_max - ls_min) * (log_std + 1) / 2

        return SquashedNormal(self._mu, self._log_std.exp())

    def log_local_hyperparams(self):
        return {"log_std_bounds": self.log_std_bounds}

    def log_local_epoch(self):
        return {
            "mu": LogType.histogram(self._mu),
            "log_std": LogType.histogram(self._log_std),
        }
