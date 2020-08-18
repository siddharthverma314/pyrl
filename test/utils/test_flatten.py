from pyrl.utils import Flatten, Unflatten, torchify, create_random_space
from torch.nn import Sequential
from pyrl.utils import collate
import torch
from torch.functional import F
import gym
import numpy as np


def custom_equals(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and b.keys() == a.keys()
        for k, v in a.items():
            custom_equals(v, b[k])
    else:
        print("ISCLOSE:", torch.isclose(a.float(), b.float()))
        assert (
            isinstance(b, torch.Tensor)
            and b.dim() == 2
            and torch.all(torch.isclose(a.float(), b.float()))
        )


def test_single_space():
    for _ in range(100):
        space = create_random_space()
        m = Sequential(Flatten(space), Unflatten(space))

        sample = torchify(space.sample())

        print("=" * 50)
        print("SPACE:", space)
        print("SAMPLE:", sample)
        forward = m.forward(sample)
        print("FORWARD:", forward)
        custom_equals(forward, sample)


def test_multi_space():
    for _ in range(100):
        space = create_random_space()
        m = Sequential(Flatten(space), Unflatten(space))

        samples = []
        for _ in range(20):
            samples.append({"obs": space.sample()})
        sample = collate([torchify(s) for s in samples])["obs"]

        print("=" * 50)
        print("SPACE:", space)
        print("SAMPLE:", sample)
        forward = m.forward(sample)
        print("FORWARD:", forward)
        custom_equals(forward, sample)


def test_nan():
    low = np.array([-1, -2, -3, -2, -1], dtype=np.float32)
    high = np.array([1, 2, 3, 2, 1], dtype=np.float32)
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
