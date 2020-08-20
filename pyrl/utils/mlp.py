from typing import List, Union
from torch import nn
from gym import Space
from pyrl.logger import Loggable
from pyrl.transforms import Flatten, Unflatten


class MLP(nn.Module, Loggable):
    """Defines a standard Multi Layer Perceptron"""

    def __init__(
        self,
        input_spec: Union[Space, int],
        hidden_dim: List[int],
        output_spec: Union[Space, int],
    ) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        # create flatten if required
        input_dim, pre_mods = self._spec_to_modules(input_spec, Flatten)
        output_dim, post_mods = self._spec_to_modules(output_spec, Unflatten)

        # create mlp
        sizes = [input_dim] + hidden_dim + [output_dim]
        mods = sum([
            [nn.Linear(i, j), nn.ReLU(inplace=True)]
            for i, j in zip(sizes, sizes[1:])
        ], [])
        mods = mods[:-1]
        mods = pre_mods + mods + post_mods

        self.mlp = nn.Sequential(*mods)

        self._mlp_hyperparams = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
        }

    @staticmethod
    def _spec_to_modules(spec, mod):
        if isinstance(spec, int):
            dim = spec
            mods = []
        elif isinstance(spec, Space):
            mod = mod(spec)
            dim = mod.dim
            mods = [mod]
        else:
            raise NotImplementedError
        return dim, mods

    def forward(self, x):
        return self.mlp.forward(x)

    def log_local_hyperparams(self):
        return self._mlp_hyperparams

    def log_local_epoch(self):
        return self.mlp.state_dict()
