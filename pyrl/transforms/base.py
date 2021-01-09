from pyrl.transforms.space_helper import make_recursive
from torch.nn import Module
from gym.spaces import Space, Box, Discrete, MultiBinary, MultiDiscrete
import numpy as np
from toolz import compose


@make_recursive(compose(int, sum))
def flatdim(space: Space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


class Transform(Module):
    def __init__(self, before_space: Space, after_space: Space):
        super().__init__()
        self.before_space = before_space
        self.after_space = after_space
        self.before_dim = flatdim(before_space)
        self.after_dim = flatdim(after_space)
