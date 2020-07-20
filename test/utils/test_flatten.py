from pyrl.utils import Flatten, Unflatten, torchify, create_random_space
from torch.nn import Sequential
import torch


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
