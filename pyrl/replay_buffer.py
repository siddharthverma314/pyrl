import numpy as np
import torch
import gym
import cpprb
from flatten_dict import flatten, unflatten
from .logger import Loggable


def space_to_spec(space: gym.Space) -> dict:
    if isinstance(space, gym.spaces.Dict):
        obs_spec = {}
        for k, v in space.spaces.items():
            obs_spec[k] = space_to_spec(v)
        return obs_spec
    elif isinstance(space, gym.spaces.Box):
        return (np.float32, space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return (np.int32, 1)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return (np.int32, space.shape)
    else:
        raise NotImplementedError


class ReplayBuffer(Loggable):
    def __init__(self, env: gym.Env, capacity: int, batch_size: int, device: str):
        obs_spec = space_to_spec(env.observation_space)
        act_spec = space_to_spec(env.action_space)
        spec = {
            "obs": obs_spec,
            "act": act_spec,
            "next_obs": obs_spec,
            "rew": (np.float32, 1),
            "done": (np.float32, 1),
        }
        spec = {
            k: {"dtype": d, "shape": s}
            for k, (d, s) in flatten(spec, reducer="dot").items()
        }

        self.buffer: cpprb.ReplayBuffer = cpprb.create_buffer(capacity, spec)
        self.capacity = capacity  # only for logging
        self.batch_size = batch_size
        self.device = torch.device(device)

        self._hyp = {
            "capacity": capacity,
            "batch_size": batch_size,
            "device": device,
        }

    def __len__(self):
        return len(self.buffer)

    def add(self, step):
        step = flatten(step, reducer="dot")
        step = {k: v.detach().cpu().numpy() for k, v in step.items()}

        self.buffer.add(**step)

    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size

        return unflatten(
            {
                k: torch.tensor(v).to(self.device)
                for k, v in self.buffer.sample(batch_size).items()
            },
            splitter="dot",
        )

    def log_local_hyperparams(self):
        return self._hyp

    def log_local_epoch(self):
        return {}
