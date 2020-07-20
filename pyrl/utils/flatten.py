from collections import OrderedDict
import numpy as np
import torch
from torch.functional import F

from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def flatdim(space):
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
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, x):
    if isinstance(space, Box):
        return x.float()
    elif isinstance(space, Discrete):
        return F.one_hot(x, space.n)
    elif isinstance(space, Tuple):
        return torch.cat([flatten(s, xp) for xp, s in zip(x, space.spaces)], dim=1)
    elif isinstance(space, Dict):
        return torch.cat([flatten(s, x[k]) for k, s in space.spaces.items()], dim=1)
    elif isinstance(space, MultiBinary):
        return x.float()
    elif isinstance(space, MultiDiscrete):
        return torch.cat(
            [F.one_hot(x[:, i], v) for i, v in enumerate(space.nvec)], dim=1
        )
    else:
        raise NotImplementedError


def unflatten(space, x):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).reshape(space.shape)
    elif isinstance(space, Discrete):
        return int(np.nonzero(x)[0][0])
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)
    elif isinstance(space, MultiBinary):
        return np.asarray(x).reshape(space.shape)
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).reshape(space.shape)
    else:
        raise NotImplementedError


def flatten_space(space):
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n,))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n,))
    if isinstance(space, MultiDiscrete):
        return Box(low=np.zeros_like(space.nvec), high=space.nvec,)
    raise NotImplementedError


class Flatten(nn.Module):
    def __init__(self, input_space: gym.Space) -> None:
        self.input_space = input_space

    def flatten(self, obj) -> torch.Tensor:
        if isinstance(obj, dict):
            return torch.cat([self.flatten(obj[k]) for k in sorted(obj.keys())], dim=1)
        elif isinstance(obj, torch.FloatTensor):
            return obj
        else:
            raise NotImplementedError

    def forward(self, obj) -> torch.Tensor:
        return self.flatten(obj)
