from .base import make_test_single_space, make_test_multi_space
from pyrl.transforms import Flatten, Unflatten
from torch.nn import Sequential


m = lambda space: Sequential(Flatten(space), Unflatten(space))
test_single_space = make_test_single_space(m)
test_multi_space = make_test_single_space(m)
