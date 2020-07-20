import gym
from adversarial.sac import actor, critic, sac
import torch


def test_integration():
    env = gym.make("InvertedPendulum-v2")
    obs_dim = len(env.observation_space.low)
    act_dim = len(env.action_space.low)

    for device in "cpu", "cuda":
        a = actor.DiagGaussianActor(obs_dim, act_dim, [256, 256], (-100, 100))
        c = critic.DoubleQCritic(obs_dim, act_dim, [256, 256])
        s = sac.SAC(actor=a, critic=c, device=device, act_dim=act_dim,)

        print(s.log_hyperparams())

        batch = {
            "obs": torch.tensor(
                [env.observation_space.sample() for _ in range(100)]
            ).float(),
            "act": torch.tensor(
                [env.action_space.sample() for _ in range(100)]
            ).float(),
            "rew": torch.rand((100, 1)).float(),
            "next_obs": torch.tensor(
                [env.observation_space.sample() for _ in range(100)]
            ).float(),
            "done": torch.randint(0, 2, (100, 1)).float(),
        }
        for t in range(10):
            s.update(batch, t)

        print(s.log_epoch())


def test_no_nan():
    """Test for no nans in all parameters.

    The main takeaway from this test is that you must set the learning
    rates low or else the parameters will tend to nan.

    """

    env = gym.make("InvertedPendulum-v2")
    obs_dim = len(env.observation_space.low)
    act_dim = len(env.action_space.low)

    a = actor.DiagGaussianActor(obs_dim, act_dim, [256, 256], (-100, 100))
    c = critic.DoubleQCritic(obs_dim, act_dim, [256, 256])
    s = sac.SAC(actor=a, critic=c, device="cuda", act_dim=act_dim,)

    batch = {
        "obs": torch.tensor(
            [env.observation_space.sample() for _ in range(100)]
        ).float(),
        "act": torch.tensor([env.action_space.sample() for _ in range(100)]).float(),
        "rew": torch.rand((100, 1)).float(),
        "next_obs": torch.tensor(
            [env.observation_space.sample() for _ in range(100)]
        ).float(),
        "done": torch.randint(0, 2, (100, 1)).float(),
    }

    for t in range(200):
        print("iteration", t)
        s.update(batch, t)
        for key, v in a.state_dict().items():
            print("actor", key)
            assert torch.any(torch.isnan(v)) == False
        for key, v in c.state_dict().items():
            print("actor", key)
            assert torch.any(torch.isnan(v)) == False


def test_critic_target_update():
    env = gym.make("InvertedPendulum-v2")
    obs_dim = len(env.observation_space.low)
    act_dim = len(env.action_space.low)

    a = actor.DiagGaussianActor(obs_dim, act_dim, [256, 256], (-100, 100))
    c = critic.DoubleQCritic(obs_dim, act_dim, [256, 256])

    s = sac.SAC(actor=a, critic=c, device="cuda", act_dim=act_dim,)

    batch = {
        "obs": torch.tensor(
            [env.observation_space.sample() for _ in range(100)]
        ).float(),
        "act": torch.tensor([env.action_space.sample() for _ in range(100)]).float(),
        "rew": torch.rand((100, 1)).float(),
        "next_obs": torch.tensor(
            [env.observation_space.sample() for _ in range(100)]
        ).float(),
        "done": torch.randint(0, 2, (100, 1)).float(),
    }

    cp_before = s.critic_target.state_dict()
    for t in range(100):
        s.update(batch, t + 1)
        cp_after = s.critic_target.state_dict()

    for k, v in cp_before.items():
        v2 = cp_after[k]
        assert torch.all(v == v2)

    cp_before = s.critic_target.state_dict()
    s.update(batch, 1000)
    cp_after = s.critic_target.state_dict()

    for k, v in cp_before.items():
        v2 = cp_after[k]
        assert not torch.all(v == v2)


def test_actor_loss_decrease():
    env = gym.make("InvertedPendulum-v2")
    obs_dim = len(env.observation_space.low)
    act_dim = len(env.action_space.low)

    a = actor.DiagGaussianActor(obs_dim, act_dim, [256, 256], (-100, 100))
    c = critic.DoubleQCritic(obs_dim, act_dim, [256, 256])

    s = sac.SAC(actor=a, critic=c, device="cuda", act_dim=act_dim,)

    batch = {
        "obs": (
            torch.tensor([env.observation_space.sample() for _ in range(100)])
            .float()
            .cuda()
        )
    }

    s.update_actor_and_alpha(**batch)
    loss_before = s.log_local_epoch()["actor"]["loss"][1]
    for _ in range(200):
        s.update_actor_and_alpha(**batch)
    loss_after = s.log_local_epoch()["actor"]["loss"][1]
    assert loss_after < loss_before + 0.2


def test_critic_value_increase():
    env = gym.make("InvertedPendulum-v2")
    obs_dim = len(env.observation_space.low)
    act_dim = len(env.action_space.low)

    a = actor.DiagGaussianActor(obs_dim, act_dim, [256, 256], (-100, 100))
    c = critic.DoubleQCritic(obs_dim, act_dim, [256, 256])

    s = sac.SAC(actor=a, critic=c, device="cuda", act_dim=act_dim,)

    batch = {
        "obs": torch.tensor([env.observation_space.sample() for _ in range(100)])
        .float()
        .cuda(),
        "act": torch.tensor([env.action_space.sample() for _ in range(100)])
        .float()
        .cuda(),
        "rew": torch.rand((100, 1)).float().cuda(),
        "next_obs": torch.tensor([env.observation_space.sample() for _ in range(100)])
        .float()
        .cuda(),
        "done": torch.randint(0, 2, (100, 1)).float().cuda(),
    }

    s.update_critic(**batch)
    q1_before = s.log_local_epoch()["critic"]["q1"]["mean"][1]
    q2_before = s.log_local_epoch()["critic"]["q2"]["mean"][1]
    for _ in range(200):
        s.update_critic(**batch)
    q1_after = s.log_local_epoch()["critic"]["q1"]["mean"][1]
    q2_after = s.log_local_epoch()["critic"]["q2"]["mean"][1]
    assert q1_after > q1_before - 0.2
    assert q2_after > q2_before - 0.2
