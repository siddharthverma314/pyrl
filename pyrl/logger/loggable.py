from typing import Dict, Type, cast
from pyrl.types import NestedDictTensor
import torch
from torch.nn import Module
import inspect
import abc


class BaseLoggable(abc.ABC):
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


class Loggable(BaseLoggable):
    """The default loggable class. Assumes structure of parent and children.

    The class defines log_local_* methods which are to be
    overrided. Children are found via the log_collect method, and the
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


def simpleloggable(cls) -> Type[Loggable]:
    """A class decorator for loggable objects.

    All function parameters in __init__ starting with an underscore
    are logged as hyperparameters.

    In order to log an object, simply call self.log(obj). All the
    collected objects will be available through an interface to be
    logged at each epoch.

    All submodules which inherit loggable are automatically logged.

    """

    class newcls(cls, Loggable):
        __name__ = cls.__name__

        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            Loggable.__init__(self)

            # set hyperparms
            if "_hyperparams" not in self.__dict__:
                self._hyperparams = {}
            # local hyperparams
            args = inspect.signature(cls.__init__).bind(self, *args, **kwargs).arguments
            self._hyperparams.update(
                {k[1:]: v for k, v in args.items() if k.startswith("_")}
            )
            # parent hyperparams
            if Loggable in cls.mro():
                self._hyperparams.update(cls.log_hyperparams(self))

            self._args = (args, kwargs)
            self.__log_epoch = {}

        def log(self, name, val: object) -> None:
            """Log an object per epoch.

            Expects value to either be :torch.Tensor: or a numeric value.

            The values are logged under :param name: as either a scalar or
            a histogram appropriately.

            """
            self.__log_epoch[name] = val

        def log_local_hyperparams(self):
            return self._hyperparams

        def log_local_epoch(self):
            return self.__log_epoch

    return newcls
