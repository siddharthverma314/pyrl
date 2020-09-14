from typing import List
from flatten_dict import flatten, unflatten
import torch


def collate(dicts: List[dict]) -> dict:
    keys = flatten(dicts[0]).keys()
    new_dict = {}
    for k in keys:
        new_dict[k] = torch.cat([flatten(d)[k] for d in dicts])
    return unflatten(new_dict)


def uncollate(d: dict) -> List[dict]:
    d = flatten(d)
    dicts = []
    for i in range(len(next(d.values()))):
        dicts = unflatten({k: v[i] for k, v in d.items()})
    return dicts
