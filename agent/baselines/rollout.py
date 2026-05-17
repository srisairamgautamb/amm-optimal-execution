"""Rollout harness for execution policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from env.amm import AMMEnv
from agent.baselines.protocol import AMMEnvView, ExecutionPolicy


@dataclass(frozen=True)
class RolloutResult:
    rewards: List[float]
    cum_reward: float
    infos: List[dict]
    policy_name: str


def _view_from_env(env: AMMEnv, info: dict) -> AMMEnvView:
    return AMMEnvView(
        Q_remaining=float(info["Q_remaining"]),
        tau_remaining=int(info["tau_remaining"]),
        x=float(info["x_post"]),
        y=float(info["y_post"]),
        gamma=float(env._gamma),
        gas_c=float(env._c),
        T_total=int(env._T),
        Q0_initial=float(env._Q0),
    )


def run_episode(
    env: AMMEnv,
    policy: ExecutionPolicy,
    *,
    seed: int,
    policy_name: str = "",
) -> RolloutResult:
    obs, info = env.reset(seed=seed)
    policy.reset()
    rewards: List[float] = []
    infos: List[dict] = []
    terminated = False
    truncated = False
    cum = 0.0
    while not (terminated or truncated):
        view = _view_from_env(env, info)
        action = policy.act(obs, info, view)
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(float(r))
        infos.append(info)
        cum += float(r)
    return RolloutResult(
        rewards=rewards,
        cum_reward=cum,
        infos=infos,
        policy_name=policy_name or getattr(policy, "name", "unnamed"),
    )
