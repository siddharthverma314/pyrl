import torch
import numbers
import numpy as np


def torchify(obs, device="cpu"):
    if isinstance(obs, dict):
        ret = {}
        for k, v in obs.items():
            if (tv := torchify(v, device)) is not None:
                ret[k] = tv
        return ret
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs).float()
    elif isinstance(obs, numbers.Number):
        obs = torch.tensor(obs).float()
    elif not isinstance(obs, torch.Tensor):
        return None
    while obs.dim() < 2:
        obs = obs.unsqueeze(0)
    return obs.to(device)


def untorchify(obs):
    if isinstance(obs, dict):
        return {k: untorchify(v) for k, v in obs.items()}
    obs = obs.detach().cpu()
    if torch.is_floating_point(obs):
        obs = obs.squeeze(0)
    else:
        obs = obs.squeeze()
    return obs.numpy()
