from typing import Union
from pyrl.utils import create_random_space
from pyrl.utils import torchify, collate, uncollate
from flatten_dict import flatten
import gym
import torch


def assert_equal(d1: Union[dict, torch.Tensor], d2: Union[dict, torch.Tensor]) -> bool:
    d1, d2 = map(flatten, (d1, d2))
    for k, v in d1.items():
        assert k in d2
        assert torch.all(d2[k] == v)


def test_collate_uncollate():
    for _ in range(100):
        space = create_random_space()
        if not isinstance(space, gym.spaces.Dict):
            continue

        samples = []
        for _ in range(50):
            samples.append(torchify(space.sample()))

        new_samples = uncollate(collate(samples))
        for s1, s2 in zip(samples, new_samples):
            assert_equal(s1, s2)
