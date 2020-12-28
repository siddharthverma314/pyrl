from .base import Transform
from .dict_transform import Flatten, Unflatten
from .one_hot_transform import OneHot, UnOneHot


class OneHotFlatten(Transform):
    def __init__(self, space):
        self.one_hot = OneHot(space)
        self.flatten = Flatten(self.one_hot.after_space)
        super().__init__(self.one_hot.before_space, self.flatten.after_space)

    def forward(self, x):
        return self.flatten(self.one_hot(x))


class UnOneHotUnflatten(Transform):
    def __init__(self, space):
        self.unflatten = Unflatten(space)
        self.un_one_hot = UnOneHot(self.unflatten.after_space)
        super().__init__(self.unflatten.before_space, self.un_one_hot.after_space)

    def forward(self, x):
        return self.un_one_hot(self.unflatten(x))
