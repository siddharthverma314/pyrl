from pyrl.utils import Flatten, Unflatten, torchify, create_random_space
from torch.nn import Sequential
import torch
from torch.functional import F
import gym
import numpy as np


def custom_equals(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and b.keys() == a.keys()
        for k, v in a.items():
            custom_equals(v, b[k])
    elif isinstance(a, torch.FloatTensor):
        assert isinstance(b, torch.Tensor) and F.mse_loss(a, b) < 1e-2
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.size() == b.size()


def test_space():
    for _ in range(100):
        space = create_random_space()
        print(space)
        m = Sequential(Flatten(space), Unflatten(space))
        sample = torchify(space.sample())
        forward = m.forward(sample)
        print("sample", sample, "forward", forward)
        custom_equals(forward, sample)


def test_nan():
    low = np.array([-1, -2, -3, -2, -1])
    high = np.array([1, 2, 3, 2, 1])
    space = gym.spaces.Box(low=low, high=high)

    flatten = Flatten(space, tanh=True)
    unflatten = Unflatten(space, tanh=True)

    for _ in range(1000):
        print("=" * 50)

        orig_sample = torchify(space.sample())
        sample = orig_sample
        print("orig", sample)

        sample = flatten(sample)
        print("flatten", sample)
        assert not torch.any(sample.isnan())

        sample = unflatten(sample)
        print("unflatten", sample)
        assert F.mse_loss(sample, orig_sample) < 1e-2
