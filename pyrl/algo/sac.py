import numpy as np
import torch
import torch.nn.functional as F
import copy
from flatten_dict import flatten, unflatten
from adversarial.dict_util import join_obs

from .actor import DiagGaussianActor
from .critic import DoubleQCritic
from ..logger import LogType, Loggable


class SAC(Loggable):
    """SAC algorithm."""

    def __init__(
        self,
        actor: DiagGaussianActor,
        critic: DoubleQCritic,
        device: torch.device,
        act_dim: int,
        critic_tau: float = 0.1,
        discount: float = 0.99,
        init_temperature: float = 0.1,
        learnable_temperature: bool = True,
        actor_update_frequency: int = 2,
        critic_target_update_frequency: int = 2,
        alpha_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> None:

        super().__init__()

        # set other parameters
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature

        # instantiate actor and critic
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(critic)

        # instantiate log alpha
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -act_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr,)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr,
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr,)

        # logging
        self._epoch_log = {}

    @property
    def alpha(self):
        return self.log_alpha.exp().to(self.device)

    def update_critic(self, obs, act, rew, next_obs, done):
        dist = self.actor.forward(next_obs)
        next_act = dist.rsample()

        # compute target Q
        log_prob = dist.log_prob(next_act).sum(-1, keepdim=True)
        target_Q = self.critic_target.forward(next_obs, next_act)
        target_V = target_Q - self.alpha.detach() * log_prob
        target_Q = rew + (1 - done) * self.discount * target_V
        target_Q = target_Q.detach()

        # compute critic loss
        current_Q1, current_Q2 = self.critic.double_q(obs, act)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # logging
        self._epoch_log["critic"] = {
            "loss": LogType.scalar(critic_loss),
            "target_q": LogType.collate(target_Q),
            "q1": LogType.collate(current_Q1),
            "q2": LogType.collate(current_Q2),
        }

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q = self.critic.forward(obs, action)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self._epoch_log["actor"] = {
            "loss": LogType.scalar(actor_loss),
            "entropy": LogType.collate(log_prob),
        }

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            self._epoch_log["alpha"] = {
                "loss": LogType.scalar(alpha_loss),
                "value": LogType.scalar(self.alpha),
            }
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, batch, step):
        # done should not include the last step. Basically, infinite horizon task

        # preprocessing
        batch = unflatten({k: v.to(self.device) for k, v in flatten(batch).items()})
        batch["obs"] = join_obs(batch["obs"])
        batch["next_obs"] = join_obs(batch["next_obs"])

        self._epoch_log["batch_reward"] = LogType.collate(batch["rew"])

        batch["obs"] = batch["obs"]
        self.update_critic(**batch)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(batch["obs"])

        if step % self.critic_target_update_frequency == 0:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(
                    self.critic_tau * p.data + (1 - self.critic_tau) * tp.data
                )

    def log_local_hyperparams(self):
        return {
            "discount": self.discount,
            "critic_tau": self.critic_tau,
            "actor_update_frequency": self.actor_update_frequency,
            "critic_target_update_frequency": self.critic_target_update_frequency,
            "learnable_temperature": self.learnable_temperature,
            "target_entropy": self.target_entropy,
            "actor_optim": self.actor_optimizer.defaults,
            "critic_optim": self.critic_optimizer.defaults,
            "log_alpha_optim": self.log_alpha_optimizer.defaults,
        }

    def log_local_epoch(self):
        return self._epoch_log
