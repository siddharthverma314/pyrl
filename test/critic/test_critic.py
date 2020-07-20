from adversarial.sac.mlp import MLP
from adversarial.sac.critic import DoubleQCritic
import torch


def test_integration():
    critic = DoubleQCritic(10, 3, [60, 50])
    obs = torch.rand((100, 10))
    act = torch.rand((100, 3))
    qs = critic.forward(obs, act)
    assert qs.shape == (100, 1)


def test_logging():
    critic = DoubleQCritic(10, 3, [60, 50])
    obs = torch.rand((100, 10))
    act = torch.rand((100, 3))
    qs = critic.forward(obs, act)
    critic.log_hyperparams()
    critic.log_epoch()
