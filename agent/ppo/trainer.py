"""PPO + CVaR training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from env.amm import AMMEnv
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 200_000
    rollout_length: int = 2048
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 32
    seed: int = 42


@dataclass
class TrainResult:
    iterations: int
    final_metrics: Dict[str, float]
    eval_history: List[Dict[str, float]] = field(default_factory=list)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    env_factory: Callable[[], AMMEnv],
    ppo_config: PPOConfig,
    train_config: TrainConfig,
    *,
    log_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[CVaRPPO, TrainResult]:
    _seed_all(train_config.seed)
    agent = CVaRPPO(ppo_config)
    env = env_factory()

    n_iters = max(1, train_config.total_timesteps // train_config.rollout_length)
    eval_history: List[Dict[str, float]] = []
    rollout_seed = train_config.seed

    last_metrics: Dict[str, float] = {}
    for it in range(n_iters):
        buffer = agent.collect_rollout(
            env, n_steps=train_config.rollout_length, rng_seed=rollout_seed,
        )
        rollout_seed += 1000
        metrics = agent.update(buffer)
        last_metrics = dict(metrics)
        last_metrics["iteration"] = float(it)
        last_metrics["timesteps"] = float((it + 1) * train_config.rollout_length)
        if log_callback is not None and (it % train_config.log_interval == 0
                                         or it == n_iters - 1):
            log_callback(it, last_metrics)
        if (it + 1) % train_config.eval_interval == 0 or it == n_iters - 1:
            eval_metrics = _evaluate(
                agent, env_factory,
                episodes=train_config.eval_episodes,
                seed=train_config.seed + 10_000 + it,
            )
            eval_metrics["iteration"] = float(it)
            eval_history.append(eval_metrics)
    return agent, TrainResult(
        iterations=n_iters, final_metrics=last_metrics, eval_history=eval_history,
    )


def _evaluate(
    agent: CVaRPPO,
    env_factory: Callable[[], AMMEnv],
    *,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    env = env_factory()
    returns: List[float] = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        cum = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            cum += float(r)
        returns.append(cum)
    arr = np.asarray(returns, dtype=np.float64)
    losses = -arr
    losses_sorted = np.sort(losses)
    alpha = agent.config.cvar_alpha
    cutoff = int(np.ceil(alpha * len(losses_sorted)))
    tail = losses_sorted[cutoff:] if cutoff < len(losses_sorted) else losses_sorted[-1:]
    return {
        "eval_mean": float(arr.mean()),
        "eval_std": float(arr.std(ddof=0)),
        "eval_p5": float(np.percentile(arr, 5)),
        "eval_p95": float(np.percentile(arr, 95)),
        "eval_cvar_loss": float(np.mean(tail)),
    }
