import torch
import torch.nn.functional as F
import copy
from flatten_dict import flatten, unflatten

from .actor import DiagGaussianActor
from .critic import DoubleVCritic
from .logger import LogType, Loggable
from .replay_buffer import ReplayBuffer


class AWR(Loggable):
    def __init__(
        self,
        actor: DiagGaussianActor,
        critic: DoubleVCritic,
        replay_buffer: ReplayBuffer,
        device: torch.device = "cpu",
        discount: float = 0.99,
        max_weight: float = 20,
        beta: float = 0.95,
        actor_updates: int = 1000,
        critic_updates: int = 500,
        actor_lr: float = 0.00005,
        critic_lr: float = 0.01,
    ) -> None:

        super().__init__()

        # set other parameters
        self.device = torch.device(device)
        self.discount = discount
        self.beta = beta
        self.max_weight = max_weight
        self.actor_updates = actor_updates
        self.critic_updates = critic_updates

        # instantiate actor and critic
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.replay_buffer = replay_buffer
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # logging
        self._epoch_log = {}
        self._hyp = {
            "device": device,
            "discount": discount,
            "beta": beta,
            "actor_updates": actor_updates,
            "critic_updates": critic_updates,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
        }

    def update_critic(self, obs, act, rew, next_obs, done):
        # calc loss
        target_V = (
            rew + self.discount * (1.0 - done) * self.critic_target.forward(next_obs)
        ).detach()
        cur_V1, cur_V2 = self.critic.double_v(obs)
        critic_loss = F.mse_loss(cur_V1, target_V) + F.mse_loss(cur_V2, target_V)

        # optimize
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # logging
        self._epoch_log["critic"] = {
            "loss": LogType.scalar(critic_loss),
            "target_v": LogType.collate(target_V),
            "v1": LogType.collate(cur_V1),
            "v2": LogType.collate(cur_V2),
        }

    def update_actor(self, obs, act, rew, next_obs, done):
        dist = self.actor(obs)
        log_prob = dist.log_prob(act).sum(-1, keepdim=True)

        weight = (
            torch.exp(
                1
                / self.beta
                * (
                    rew
                    + self.discount
                    * (1.0 - done)
                    * self.critic_target.forward(next_obs)
                    - self.critic_target.forward(obs)
                )
            )
            .clamp(max=self.max_weight)
            .detach()
        )
        actor_loss = -(log_prob * weight).mean()

        self._epoch_log["actor"] = {
            "loss": LogType.scalar(actor_loss),
            "entropy": LogType.collate(log_prob),
        }

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def update(self):
        # done should not include the last step. Basically, infinite horizon task

        # update critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        for _ in range(self.critic_updates):
            self.update_critic(**self.replay_buffer.sample())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # update actor
        for _ in range(self.actor_updates):
            self.update_actor(**self.replay_buffer.sample())

    def log_local_hyperparams(self):
        return self._hyp

    def log_local_epoch(self):
        return self._epoch_log
