from typing import List, Tuple
from torch import nn, distributions as pyd
from gym import Space
from pyrl.utils import MLP, Flatten, Unflatten
from pyrl.logger import simpleloggable


@simpleloggable
class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        hidden_dim: List[int],
        _log_std_bounds: Tuple[float, float] = (-5, 2),
    ) -> None:
        nn.Module.__init__(self)

        self.obs_flat = Flatten(obs_spec)
        self.act_flat = Flatten(act_spec, tanh=True)
        self.act_unflat = Unflatten(act_spec, tanh=True, is_logits=True) # TODO: Check this

        self.log_std_bounds = _log_std_bounds
        self.policy = MLP(self.obs_flat.dim, hidden_dim, self.act_flat.dim * 2)

    def _get_dist(self, obs_flat):
        # get mu and log_std
        mu, log_std = self.policy(obs_flat).chunk(2, dim=-1)
        self.log("mu", mu)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = log_std.tanh()
        ls_min, ls_max = self.log_std_bounds
        log_std = ls_min + (ls_max - ls_min) * (log_std + 1) / 2
        dist = pyd.Normal(mu, log_std.exp())
        self.log("log_std", log_std)

        return dist

    def log_prob(self, obs, act):
        dist = self._get_dist(self.obs_flat(obs))
        return dist.log_prob(self.act_flat(act)).sum(-1, keepdim=True)

    def action(self, obs, deterministic=False):
        return self.action_with_log_prob(obs, deterministic)[0]

    def action_with_log_prob(self, obs, deterministic=False):
        dist = self._get_dist(self.obs_flat(obs))

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return self.act_unflat(action), log_prob

    def forward(self, obs) -> pyd.Distribution:
        return self.action(obs, deterministic=False)
