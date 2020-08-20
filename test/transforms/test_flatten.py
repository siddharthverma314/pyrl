from pyrl.transforms import Flatten, Unflatten
from torch.nn import Sequential
from pyrl.utils import collate, create_random_space, torchify
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
