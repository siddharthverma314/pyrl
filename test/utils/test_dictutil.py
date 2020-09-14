from pyrl.utils import create_random_space
from pyrl.utils import dictutil, torchify, collate, uncollate


def test_collate_uncollate():
    for _ in range(100):
        space = create_random_space()
        samples = []
        for _ in range(50):
            samples.append(torchify(space.sample()))

        new_samples = uncollate(collate(samples))
        for s1, s2 in zip(samples, new_samples):
            assert s1 == s2
