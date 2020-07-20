import gym
import torch
from adversarial.env.dict_wrapper import DictWrapper
from adversarial.env.torch_wrapper import TorchWrapper
from adversarial.sac.replay_buffer import ReplayBuffer
from flatten_dict import flatten


def test_integration():
    for device in "cpu", "cuda":
        env = gym.make("InvertedPendulum-v2")
        env = DictWrapper(TorchWrapper(env, device))
        buf = ReplayBuffer(env, int(1e5), 1)
        print(buf.log_hyperparams())

        obs = env.reset()
        act = torch.tensor(env.action_space.sample()).to(device).unsqueeze(0)
        nobs, reward, done, _ = env.step(act)
        step = {"obs": obs, "act": act, "rew": reward, "next_obs": nobs, "done": done}
        buf.add(step)
        step2 = buf.sample()
        step = flatten(step)
        step2 = flatten(step2)
        assert step.keys() == step2.keys()
        for k in step:
            assert torch.all(step[k].cpu() == step2[k].cpu())

        print(buf.log_epoch())
