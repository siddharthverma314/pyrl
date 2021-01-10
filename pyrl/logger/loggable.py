from typing import Dict, Type, cast
from pyrl.types import NestedDictTensor
import torch
from torch.nn import Module
import inspect
import abc
from toolz.dicttoolz import merge, update_in
from torch.nn.modules.module import Module
from torch.tensor import Tensor


class BaseLoggable():
    """Most basic loggable class. No assumptions about structure."""

    @abc.abstractmethod
    def log_epoch(self) -> Dict[str, NestedDictTensor]:
        "Log the epoch parameters of the class"

    @abc.abstractmethod
    def log_hyperparams(self) -> Dict[str, NestedDictTensor]:
        "Log the hyperparameters of the class"

    @abc.abstractmethod
    def log_snapshot(self) -> Dict[str, NestedDictTensor]:
        "Log a snapshot of the class"

    @abc.abstractmethod
    def load_snapshot(self, snapshot: Dict[str, NestedDictTensor]) -> None:
        "Reset from a snapshot logged by :self.log_snapshot:"


class Loggable(BaseLoggable, Module):
    """The default loggable class. Assumes structure of parent and
    children.

    The class defines log_local_* methods which are to be
    overriden. Children are found via the log_collect method, and the
    log_* methods are filled in using both these values.

    """

    def log_collect(self):
        """Recursively collect parameters. Override to select which members
        get chosen and their respective names."""

        return {k: v for k, v in inspect.getmembers(self) if isinstance(v, Loggable)}

    @abc.abstractmethod
    def log_local_hyperparams(self) -> Dict[str, NestedDictTensor]:
        "Return the hyperparameters of the current object"

    @abc.abstractmethod
    def log_local_epoch(self) -> Dict[str, NestedDictTensor]:
        "Return logs at every epoch of the current object"

    def log_hyperparams(self) -> Dict[str, NestedDictTensor]:
        return {
            **{k: v.log_hyperparams() for k, v in self.log_collect().items()},
            **self.log_local_hyperparams(),
        }

    def log_epoch(self) -> Dict[str, NestedDictTensor]:
        return {
            **{k: v.log_epoch() for k, v in self.log_collect().items()},
            **self.log_local_epoch(),
        }

    def log_snapshot(self) -> Dict[str, NestedDictTensor]:
        if isinstance(self, Module):
            return {"state_dict": cast(Module, self).state_dict()}
        return {k: v.log_snapshot() for k, v in self.log_collect().items()}

    def load_snapshot(self, snapshot: dict) -> None:
        if isinstance(self, Module):
            return cast(Module, self).load_state_dict(snapshot)
        for k, v in self.log_collect().items():
            v.load_snapshot(snapshot[k])


class SimpleLoggable(type):
    """A MetaClass for loggable objects.

    All function parameters in __init__ starting with an underscore
    are logged as hyperparameters.

    In order to log an object, simply call self.log(obj). All the
    collected objects will be available through an interface to be
    logged at each epoch.

    All submodules which inherit loggable are automatically logged.

    """

    def __new__(mcls, name, classes, methods):
        def log(self, name, val: object) -> None:
            """Log an object per epoch.

            Expects value to either be :torch.Tensor: or a numeric value.

            The values are logged under :param name: as either a scalar or
            a histogram appropriately.

            """
            self.__log_epoch[name] = val

        def wrap_init(fn):
            def new_init(self, *args, **kwargs):
                Loggable.__init__(self)
                args = inspect.signature(fn).parameters.keys()

                fn(self, *args, **kwargs)

                # set hyperparms
                self.__hyperparams = {}
                # local hyperparams
                args = inspect.signature(fn).bind(self, *args, **kwargs).arguments
                self.__hyperparams.update(
                    {k[1:]: v for k, v in args.items() if k.startswith("_")}
                )

                self.__log_epoch = {}

            return new_init

        return super(SimpleLoggable, mcls).__new__(
            mcls,
            name,
            classes + (Loggable,),
            merge(
                update_in(methods, ["__init__"], wrap_init),
                {
                    "log_local_hyperparams": lambda self: self.__hyperparams,
                    "log_local_epoch": lambda self: self.__log_epoch,
                    "log": log,
                },
            ),
        )
