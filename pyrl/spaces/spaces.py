from gym import spaces
from pyrl.utils import torchify, untorchify
from abc
import torch


class TorchSpace(spaces.Space, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, n=1):
        """Like gym.sample, but n is the number of samples"""

    @abc.abstractmethod
    def contains(self, sample):
        """gym.contains"""


class Box(TorchSpace):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        assert low.size() == high.size()
        self.low = low
        self.high = high

    def sample(self, n=1):
        return torch.rand((1, self.low.size()[0])) * (self.high - self.low) + self.low

    def contains(self, sample):
        return sample.size()[1] == self.low.size()[1]


class Discrete(TorchSpace):
    def __init__(self, n: int):
        self.n = n.item()

    def sample(self, n=1):
        return torch.randint(self.n, size=(n, 1))

    def contains(self, sample):
        sample = sample.long()
        return sample.size()[1] == 1 and torch.all(sample > 0 and sample < self.n)


class Dict(TorchSpace):
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self, n=1):
        return self.spaces


# Wrap all spaces to use pytorch
Box = wrap_space(spaces.Box)
Dict = wrap_space(spaces.Dict)
Tuple = wrap_space(spaces.Tuple)
Discrete = wrap_space(spaces.Discrete)
MultiDiscrete = wrap_space(spaces.MultiDiscrete)
MultiBinary = wrap_space(spaces.MultiBinary)
