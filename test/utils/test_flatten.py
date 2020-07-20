from pyrl.utils import Flatten, Unflatten, torchify
from torch.nn import Sequential
from gym.spaces import Dict, Box, MultiDiscrete, Discrete
import numpy as np
import torch


def create_random_space():
    choice = np.random.randint(4)
    if choice == 0:
        dim = np.random.randint(5)
        return Box(low=-5*np.ones(dim, dtype=np.float32), high=5*np.ones(dim, dtype=np.float32))
    elif choice == 1:
        return Dict({"a": create_random_space(), "b": create_random_space()})
    elif choice == 2:
        return Discrete(np.random.randint(5) + 2)
    elif choice == 3:
        return MultiDiscrete([np.random.randint(5) + 2 for _ in range(np.random.randint(10) + 1)])
    else:
        raise NotImplementedError()


def custom_equals(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and b.keys() == a.keys()
        for k, v in a.items():
            custom_equals(v, b[k])
    elif isinstance(a, torch.FloatTensor):
        assert isinstance(b, torch.Tensor) and torch.all(a == b)
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.size() == b.size()


def test_space():
    for _ in range(100):
        space = create_random_space()
        print(space)
        m = Sequential(Flatten(space), Unflatten(space))
        sample = torchify(space.sample())
        custom_equals(m.forward(sample), sample)
