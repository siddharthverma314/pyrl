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


class Flatten(torch.nn.Module):
    def __init__(self, space):
        super().__init__()
        self.space = space

    @staticmethod
    def flatten(space, x):
        if isinstance(space, Box):
            return x.float()
        elif isinstance(space, Discrete):
            return F.one_hot(x.long()[0], space.n).float()
        elif isinstance(space, Tuple):
            return torch.cat(
                [Flatten.flatten(s, xp) for xp, s in zip(x, space.spaces)], dim=1
            )
        elif isinstance(space, Dict):
            return torch.cat(
                [Flatten.flatten(s, x[k]) for k, s in space.spaces.items()], dim=1
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

    def forward(self, x):
        return self.flatten(self.space, x)


class Unflatten(torch.nn.Module):
    def __init__(self, space):
        super().__init__()
        self.space = space

    @staticmethod
    def unflatten(space, x):
        "Makes a random choice if necessary"

        if isinstance(space, Box):
            return x
        elif isinstance(space, Discrete):
            return pyd.Categorical(logits=x).sample((1,))
        elif isinstance(space, Tuple):
            list_flattened = torch.split(x, list(map(flatdim, space.spaces)), dim=-1)
            list_unflattened = [
                Unflatten.unflatten(s, flattened)
                for flattened, s in zip(list_flattened, space.spaces)
            ]
            return tuple(list_unflattened)
        elif isinstance(space, Dict):
            list_flattened = torch.split(
                x, list(map(flatdim, space.spaces.values())), dim=-1
            )
            list_unflattened = [
                (key, Unflatten.unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
            return OrderedDict(list_unflattened)
        elif isinstance(space, MultiBinary):
            return x
        elif isinstance(space, MultiDiscrete):
            outputs = []
            for t in torch.split(x, space.nvec.tolist(), dim=1):
                outputs.append(pyd.Categorical(logits=t).sample((1,)))
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.unflatten(self.space, x)
