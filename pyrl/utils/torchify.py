from typing import Union
import torch
import numbers
import numpy as np

untorchified = Union[np.ndarray, dict]
torchified = Union[torch.Tensor, dict]

def torchify(obs: untorchified, device: Union[str, torch.device] = "cpu") -> torchified:
    if isinstance(obs, dict):
        ret = {}
        for k, v in obs.items():
            tv = torchify(v, device)
            if tv is not None:
                ret[k] = tv
        return ret
    obs = torch.tensor(obs)
    if obs.dtype != torch.int64 or obs.dtype != torch.int32:
        obs = obs.float()
    while obs.dim() < 2:
        obs = obs.unsqueeze(0)
    return obs.to(device)


def untorchify(obs: torchified) -> untorchified:
    if isinstance(obs, dict):
        return {k: untorchify(v) for k, v in obs.items()}
    obs = obs.detach().cpu()
    if torch.is_floating_point(obs):
        obs = obs.squeeze(0)
    else:
        obs = obs.squeeze()
    return obs.numpy()
