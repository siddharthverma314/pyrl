from collections import OrderedDict
import numpy as np
import torch
from torch.functional import F
import torch.distributions as pyd
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def create_random_space():
    "Create a random space that might be nested. Mostly for testing purposes."

    choice = np.random.randint(4)
    if choice == 0:
        dim = np.random.randint(5) + 1
        return Box(
            low=-5 * np.ones(dim, dtype=np.float32),
            high=5 * np.ones(dim, dtype=np.float32),
        )
    elif choice == 1:
        return Dict({"a": create_random_space(), "b": create_random_space()})
    elif choice == 2:
        return Discrete(np.random.randint(5) + 2)
    elif choice == 3:
        return MultiDiscrete(
            [np.random.randint(5) + 2 for _ in range(np.random.randint(10) + 1)]
        )
    else:
        raise NotImplementedError()


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

    @property
    def dim(self) -> int:
        return flatdim(self.space)

    def forward(self, x):
        return self.flatten(self.space, x)


class Unflatten(torch.nn.Module):
    def __init__(self, space, is_logits=True):
        super().__init__()
        self.space = space
        self.is_logits = is_logits

    def unflatten(self, space, x):
        if isinstance(space, Box):
            return x
        elif isinstance(space, Discrete):
            if self.is_logits:
                return pyd.Categorical(logits=x).sample((1,))
            else:
                return pyd.Categorical(probs=x).sample((1,))
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
                    outputs.append(pyd.Categorical(logits=t).sample((1,)))
                else:
                    outputs.append(pyd.Categorical(probs=t).sample((1,)))
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

    @property
    def dim(self) -> int:
        return flatdim(self.space)

    def forward(self, x):
        return self.unflatten(self.space, x)
