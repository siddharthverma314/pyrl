from torch.nn import Module
from torch.functional import F
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def flatdim(space: gym.Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space
    would have.
    """

    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


class OneHot(Module):
    def __init__(self, space):
        super().__init__()
        self.space = space
        self.dim = flatdim(space)

    def one_hot(self, space, x):
        if isinstance(space, Tuple):
            return tuple([self.one_hot(s, xp) for s, xp in zip(space.spaces, x)])
        elif isinstance(space, Dict):
            return {k: self.one_hot(s, x[k]) for k, s in space.spaces.items()}
        elif isinstance(space, MultiDiscrete):
            return torch.cat(
                self.one_hot(
                    Tuple([Discrete(d) for d in space.nvec]), x.split(1, dim=1)
                ),
                dim=1,
            )
        elif isinstance(space, Discrete):
            return F.one_hot(x.squeeze(1).long(), space.n)
        else:
            return x

    def forward(self, x):
        return self.one_hot(self.space, x)


class UnOneHot(Module):
    def __init__(self, space):
        super().__init__()
        self.space = space
        self.dim = flatdim(self.space)

    def un_one_hot(self, space, x):
        if isinstance(space, Tuple):
            return tuple([self.un_one_hot(s, xp) for s, xp in zip(space.spaces, x)])
        elif isinstance(space, Dict):
            return {k: self.un_one_hot(s, x[k]) for k, s in space.spaces.items()}
        elif isinstance(space, MultiDiscrete):
            return torch.cat(
                self.un_one_hot(
                    Tuple([Discrete(d) for d in space.nvec]),
                    x.split(tuple(space.nvec), dim=1),
                ),
                dim=1,
            )
        elif isinstance(space, Discrete):
            return torch.argmax(x, dim=1, keepdim=True)
        else:
            return x

    def forward(self, x):
        return self.un_one_hot(self.space, x)
