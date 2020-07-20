import torch


def torchify(obs, device="cpu"):
    if isinstance(obs, dict):
        return {k: torchify(v) for k, v in obs.items()}
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs).float()
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
