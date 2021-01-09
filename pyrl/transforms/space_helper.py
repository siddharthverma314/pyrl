from __future__ import annotations
from typing import Callable, List, TypeVar
from functools import wraps
import gym.spaces as S

T = TypeVar("T")


def make_recursive(comb: Callable[[List[T]], T]):
    def decorator(fn: Callable[[S.Space], T]) -> Callable[[S.Space], T]:
        @wraps(fn)
        def new_fn(space: S.Space):
            if isinstance(space, S.Tuple):
                return comb([new_fn(s) for s in space.spaces])
            elif isinstance(space, S.Dict):
                return comb([new_fn(s) for s in space.spaces.values()])
            return fn(space)

        return new_fn

    return decorator
