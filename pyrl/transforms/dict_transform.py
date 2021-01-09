from pyrl.transforms.space_helper import make_recursive
from toolz.itertoolz import mapcat
from torch.tensor import Tensor
from .base import Transform
from collections import OrderedDict
from gym.spaces import Tuple, Dict, Space
from .base import flatdim
from pyrl.types import NestedTensor
import torch


@make_recursive(lambda ts: Tuple(mapcat(lambda t: t.spaces, ts)))
def flatten_space(space: Space) -> Tuple:
    return Tuple([space])


class Flatten(Transform):
    def __init__(self, space: Space):
        super().__init__(space, flatten_space(space))

    @classmethod
    def flatten(cls, x: NestedTensor) -> Tensor:
        if isinstance(x, tuple):
            return torch.cat(list(map(cls.flatten, x)), dim=1)
        elif isinstance(x, dict):
            return torch.cat(list(map(cls.flatten, x.values())), dim=1)
        return x

    def forward(self, x: NestedTensor) -> Tensor:
        return self.flatten(x)


class Unflatten(Transform):
    def __init__(self, space: Space):
        super().__init__(flatten_space(space), space)

    @classmethod
    def unflatten(cls, space: Space, x: Tensor) -> NestedTensor:
        if isinstance(space, Tuple):
            list_flattened = torch.split(x, list(map(flatdim, space.spaces)), dim=-1)
            list_unflattened = [
                cls.unflatten(s, flattened)
                for flattened, s in zip(list_flattened, space.spaces)
            ]
            return tuple(list_unflattened)
        elif isinstance(space, Dict):
            list_flattened = torch.split(
                x, list(map(flatdim, space.spaces.values())), dim=-1
            )
            list_unflattened = [
                (key, cls.unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
            return OrderedDict(list_unflattened)
        else:
            return x

    def forward(self, x: Tensor) -> NestedTensor:
        return self.unflatten(self.after_space, x)
