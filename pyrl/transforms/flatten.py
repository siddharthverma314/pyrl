from collections import OrderedDict
import numpy as np
import torch
from torch.functional import F
import torch.distributions as pyd
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def flatdim(space: gym.Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space
    would have.
    """

    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.sum(space.nvec))
    else:
        raise NotImplementedError


def get_affine_transformation(space: gym.spaces.Box, device) -> tuple:
    if all(space.bounded_below) and all(space.bounded_above):
        m = (space.high - space.low) / 2
        b = (space.high + space.low) / 2
    else:
        m, b = 1, 0
    return tuple([torch.tensor(x).to(device) for x in (m, b)])


class Flatten(torch.nn.Module):
    def __init__(self, space, tanh=False):
        super().__init__()
        self.space = space
        self.tanh = tanh

    def flatten(self, space, x):
        if isinstance(space, Box):
            if not self.tanh:
                return x.float()
            m, b = get_affine_transformation(space, x.device)
            x = (x - b) / m

            # hack to prevent computing atanh(1.)
            mask = (x > 0.999) + (x < 0.999)
            mask = 0.999 * mask.float()
            x = (x * mask).atanh()

            return x.float()
        elif isinstance(space, Discrete):
            return F.one_hot(x.squeeze(-1).long(), space.n).float()
        elif isinstance(space, Tuple):
            return torch.cat(
                [self.flatten(s, xp) for xp, s in zip(x, space.spaces)], dim=1
            )
        elif isinstance(space, Dict):
            return torch.cat(
                [self.flatten(s, x[k]) for k, s in space.spaces.items()], dim=1
            )
        elif isinstance(space, MultiBinary):
            return x.float()
        elif isinstance(space, MultiDiscrete):
            return torch.cat(
                [
                    F.one_hot(x.long()[:, i], v).float()
                    for i, v in enumerate(space.nvec)
                ],
                dim=1,
            )
        else:
            raise NotImplementedError

    @property
    def dim(self) -> int:
        return flatdim(self.space)

    def forward(self, x):
        return self.flatten(self.space, x)


class Unflatten(torch.nn.Module):
    def __init__(self, space, is_logits=False, tanh=False):
        super().__init__()
        self.space = space
        self.is_logits = is_logits
        self.tanh = tanh

    def unflatten(self, space, x):
        if isinstance(space, Box):
            if not self.tanh:
                return x.float()
            m, b = get_affine_transformation(space, x.device)
            return m * x.tanh() + b
        elif isinstance(space, Discrete):
            if self.is_logits:
                return pyd.Categorical(logits=x).sample().unsqueeze(-1)
            else:
                return pyd.Categorical(probs=x).sample().unsqueeze(-1)
        elif isinstance(space, Tuple):
            list_flattened = torch.split(x, list(map(flatdim, space.spaces)), dim=-1)
            list_unflattened = [
                self.unflatten(s, flattened)
                for flattened, s in zip(list_flattened, space.spaces)
            ]
            return tuple(list_unflattened)
        elif isinstance(space, Dict):
            list_flattened = torch.split(
                x, list(map(flatdim, space.spaces.values())), dim=-1
            )
            list_unflattened = [
                (key, self.unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
            return OrderedDict(list_unflattened)
        elif isinstance(space, MultiBinary):
            return x
        elif isinstance(space, MultiDiscrete):
            outputs = []
            for t in torch.split(x, space.nvec.tolist(), dim=1):
                if self.is_logits:
                    outputs.append(pyd.Categorical(logits=t).sample().unsqueeze(-1))
                else:
                    outputs.append(pyd.Categorical(probs=t).sample().unsqueeze(-1))
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

    @property
    def dim(self) -> int:
        return flatdim(self.space)

    def forward(self, x):
        return self.unflatten(self.space, x)
