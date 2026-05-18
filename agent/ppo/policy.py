"""Actor-critic for the unit-interval action of AMMEnv."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as dists


HIDDEN: int = 256
N_TRUNK_LAYERS: int = 3
LOG_STD_INIT: float = 0.0
EPS: float = 1e-6


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        *,
        with_cvar_head: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(f"obs_dim and act_dim must be >=1, got {obs_dim}, {act_dim}")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.with_cvar_head = bool(with_cvar_head)
        layers = [nn.Linear(obs_dim, HIDDEN), nn.Tanh()]
        for _ in range(N_TRUNK_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(HIDDEN, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))
        self.critic = nn.Linear(HIDDEN, 1)
        if self.with_cvar_head:
            self.critic_cvar = nn.Linear(HIDDEN, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_log_std)
        dist = dists.Normal(mean, std)
        value = self.critic(h).squeeze(-1)
        if self.with_cvar_head:
            value_cvar = self.critic_cvar(h).squeeze(-1)
            return dist, value, value_cvar
        return dist, value

    @staticmethod
    def squash(raw_action: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_action)

    @staticmethod
    def log_prob_squashed(
        dist: dists.Distribution, raw_action: torch.Tensor
    ) -> torch.Tensor:
        a = torch.sigmoid(raw_action)
        a = torch.clamp(a, EPS, 1.0 - EPS)
        log_prob_raw = dist.log_prob(raw_action).sum(-1)
        jac = torch.log(a * (1.0 - a)).sum(-1)
        return log_prob_raw - jac
