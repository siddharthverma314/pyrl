from __future__ import annotations
from typing import Tuple, Union, Dict
from torch import Tensor
from numpy import ndarray

NestedDictTensor = Union[Tensor, Dict[str, 'NestedDictTensor']]
NestedTensor = Union[Tensor, Dict[str, 'NestedTensor'], Tuple['NestedTensor']]
NestedNdArray = Union[ndarray, Dict[str, 'NestedNdArray'], Tuple['NestedNdArray']]
