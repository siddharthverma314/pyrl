from gym import spaces
from pyrl.utils import torchify, untorchify


def wrap_space(space):
    class NewSpace(space):
        def __init__(self, *args, **kwargs):
            args = [untorchify(a) for a in args]
            kwargs = {k: untorchify(v) for k, v in args}
            super().__init__(*args, **kwargs)

        def sample(self):
            return torchify(super().sample())

        def contains(self, x):
            return super().contains(untorchify(x))

    return NewSpace


# Wrap all spaces to use pytorch
Box = wrap_space(spaces.Box)
Dict = wrap_space(spaces.Dict)
Tuple = wrap_space(spaces.Tuple)
Discrete = wrap_space(spaces.Discrete)
MultiDiscrete = wrap_space(spaces.MultiDiscrete)
MultiBinary = wrap_space(spaces.MultiBinary)
