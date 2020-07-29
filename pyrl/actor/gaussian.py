from typing import List, Tuple, Union
import torch
from torch import nn, distributions as pyd
from gym import Space
from pyrl.utils import MLP, Flatten, Unflatten
from pyrl.logger import simpleloggable


class SquashedNormal(pyd.TransformedDistribution):
    def __init__(self, loc, scale):
        super().__init__(
            pyd.Normal(loc, scale),
            pyd.TanhTransform(
                cache_size=1,
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

    def log_prob(self, tensor):
        # hack to prevent nans
        mask = (tensor > 0.999).float() + (tensor < -0.999).float()
        mask = 1. - 0.001 * mask
        return super().log_prob(mask * tensor)


@simpleloggable
class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        hidden_dim: List[int],
        _log_std_bounds: Tuple[float, float] = (-5, 2),
        _use_squashed_normal: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        self.obs_flat = Flatten(obs_spec)
        self.act_flat = Flatten(act_spec)
        self.act_unflat = Unflatten(act_spec)

        self.log_std_bounds = _log_std_bounds
        self.policy = MLP(self.obs_flat.dim, hidden_dim, self.act_flat.dim * 2)
        self.dist = SquashedNormal if _use_squashed_normal else pyd.Normal

    def log_prob(self, obs, act):
        dist = self.forward(obs)
        act = self.act_flat(act)
        return dist.log_prob(act).sum(-1, keepdim=True)

    def action(self, obs, deterministic=False):
        # get dist
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        return self.act_unflat(action)

    def forward(self, obs) -> pyd.Distribution:
        # flatten
        obs = self.obs_flat(obs)

        # get mu and log_std
        mu, log_std = self.policy(obs).chunk(2, dim=-1)
        self.log("mu", mu)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = log_std.tanh()
        ls_min, ls_max = self.log_std_bounds
        log_std = ls_min + (ls_max - ls_min) * (log_std + 1) / 2
        self.log("log_std", log_std)

        return self.dist(mu, log_std.exp())
