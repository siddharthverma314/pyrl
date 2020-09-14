from pyrl.spaces import Box, Dict, Tuple, Discrete, MultiDiscrete
from pyrl.utils import collate
import torch


def make_test(space):
    def test():
        sample = space.sample()
        assert space.contains(sample)
    return test


b = Box(low=torch.zeros(2), high=torch.ones(2))
d = Dict({"a": b, "b": b})
t = Tuple([b, b])

test_box = make_test(b)
test_dict = make_test(d)
test_tuple = make_test(t)
test_discrete = make_test(Discrete(3))
test_multi_discrete = make_test(MultiDiscrete([3, 3, 3]))


def test_box_multi():
    s = []
    for _ in range(20):
        s.append(b.sample())
    s = torch.cat(s)
    print(s.shape)

    assert b.contains(s)
