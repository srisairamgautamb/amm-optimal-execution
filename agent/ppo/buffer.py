"""Rollout buffer + GAE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    raw_actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    advantages: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    episode_returns: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    values_cvar: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    cvar_targets: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    advantages_cvar: torch.Tensor = field(default_factory=lambda: torch.empty(0))


def step_episode_ids(dones: torch.Tensor) -> torch.Tensor:
    T = dones.shape[0]
    out = torch.zeros(T, dtype=torch.long)
    cur = 0
    for i in range(T):
        out[i] = cur
        if dones[i].item() > 0.5:
            cur += 1
    return out


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype)
    last_adv = torch.tensor(0.0, dtype=rewards.dtype)
    next_value = torch.tensor(float(last_value), dtype=rewards.dtype)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_adv = delta + gamma * gae_lambda * not_done * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def episode_returns_from_rewards(
    rewards: torch.Tensor, dones: torch.Tensor
) -> torch.Tensor:
    out: List[float] = []
    acc = 0.0
    for r, d in zip(rewards.tolist(), dones.tolist()):
        acc += float(r)
        if d > 0.5:
            out.append(acc)
            acc = 0.0
    if acc != 0.0 or not out:
        out.append(acc)
    return torch.tensor(out, dtype=rewards.dtype)
