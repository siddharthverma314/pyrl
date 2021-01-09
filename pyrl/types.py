from __future__ import annotations
from typing import Callable, List, TypeVar, Tuple, Union, Dict
from torch import Tensor
from numpy import ndarray

NestedTensor = Union[Tensor, Dict[str, 'NestedTensor'], Tuple['NestedTensor']]
NestedNdArray = Union[ndarray, Dict[str, 'NestedNdArray'], Tuple['NestedNdArray']]
