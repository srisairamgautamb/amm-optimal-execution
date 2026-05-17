"""Phase 3: PPO+CVaR framework tests. Framework-level, not full training."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from env.amm import AMMEnv, AMMConfig, DEFAULT_GAMMA
from env.mev_bot import compute_sandwich
from agent.ppo.policy import ActorCritic
from agent.ppo.buffer import compute_gae, episode_returns_from_rewards
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig
from agent.ppo.trainer import train, TrainConfig
from agent.ppo.policy_adapter import PPOPolicyAdapter
from agent.baselines.protocol import AMMEnvView


def _env_factory() -> Callable[[], AMMEnv]:
    def factory() -> AMMEnv:
        return AMMEnv(AMMConfig(
            x0=1e6, y0=1e6, Q0=1e4, T=5,
            gamma=DEFAULT_GAMMA, gas_c=1.0,
            mev_adversary=compute_sandwich,
        ))
    return factory


def test_policy_forward_shapes():
    net = ActorCritic(obs_dim=4, act_dim=1)
    obs = torch.zeros(7, 4)
    dist, value = net(obs)
    assert value.shape == (7,)
    assert dist.mean.shape == (7, 1)


def test_policy_action_in_unit_interval():
    net = ActorCritic(obs_dim=4, act_dim=1)
    torch.manual_seed(0)
    obs = torch.randn(500, 4)
    dist, _ = net(obs)
    raw = dist.sample()
    a = ActorCritic.squash(raw)
    assert (a >= 0.0).all() and (a <= 1.0).all()


def test_log_prob_squashed_finite():
    net = ActorCritic(obs_dim=4, act_dim=1)
    torch.manual_seed(1)
    obs = torch.randn(10, 4)
    dist, _ = net(obs)
    raw = dist.sample()
    lp = ActorCritic.log_prob_squashed(dist, raw)
    assert torch.isfinite(lp).all()


def test_gae_correctness_hand_example():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = compute_gae(rewards, values, dones,
                           gamma=1.0, gae_lambda=1.0, last_value=0.0)
    assert torch.allclose(adv, torch.tensor([5.5, 4.5, 2.5]))
    assert torch.allclose(ret, torch.tensor([6.0, 5.0, 3.0]))


def test_episode_returns_split_on_done():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    out = episode_returns_from_rewards(rewards, dones)
    assert torch.allclose(out, torch.tensor([3.0, 12.0]))


def test_cvar_penalty_zero_when_lambda_zero():
    cfg = PPOConfig(cvar_lambda=0.0, minibatch_size=4)
    agent = CVaRPPO(cfg)
    env = _env_factory()()
    buf = agent.collect_rollout(env, n_steps=20, rng_seed=0)
    metrics = agent.update(buf)
    assert metrics["cvar_penalty"] == 0.0


def test_cvar_dual_recovers_correct_estimate():
    torch.manual_seed(0)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    alpha = 0.8
    cutoff = int(math.ceil(alpha * len(losses)))
    analytic_cvar = float(losses[cutoff:].mean()) if cutoff < len(losses) else float(losses[-1])
    best = float("inf")
    for t_candidate in torch.linspace(0.0, 10.0, 1001):
        relu = torch.clamp(losses - t_candidate, min=0.0)
        est = float(t_candidate + (1.0 / (1.0 - alpha)) * relu.mean())
        if est < best:
            best = est
    assert abs(best - analytic_cvar) < 0.05


def test_update_does_not_crash():
    cfg = PPOConfig(minibatch_size=8, cvar_lambda=0.5)
    agent = CVaRPPO(cfg)
    env = _env_factory()()
    buf = agent.collect_rollout(env, n_steps=32, rng_seed=42)
    metrics = agent.update(buf)
    for k in ("policy_loss", "value_loss", "entropy", "cvar_penalty",
              "mean_return", "cvar_t", "cvar_return"):
        assert k in metrics


def test_predict_deterministic():
    cfg = PPOConfig()
    agent = CVaRPPO(cfg)
    obs = np.array([0.5, 0.2, -0.1, 0.0], dtype=np.float64)
    a1 = agent.predict(obs, deterministic=True)
    a2 = agent.predict(obs, deterministic=True)
    assert np.allclose(a1, a2)
    assert a1.shape == (1,)
    assert 0.0 <= a1[0] <= 1.0


def test_save_load_roundtrip():
    cfg = PPOConfig(cvar_lambda=0.7)
    agent = CVaRPPO(cfg)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    agent.save(path)
    loaded = CVaRPPO.load(path)
    obs = np.array([0.5, 0.2, -0.1, 0.0], dtype=np.float64)
    a_orig = agent.predict(obs, deterministic=True)
    a_load = loaded.predict(obs, deterministic=True)
    assert np.allclose(a_orig, a_load)
    path.unlink()


def test_policy_adapter_act_shape():
    cfg = PPOConfig()
    agent = CVaRPPO(cfg)
    adapter = PPOPolicyAdapter(agent, deterministic=True)
    adapter.reset()
    obs = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
    view = AMMEnvView(
        Q_remaining=1.0, tau_remaining=5, x=1e6, y=1e6, gamma=0.997,
        gas_c=1.0, T_total=5, Q0_initial=1.0,
    )
    a = adapter.act(obs, {}, view)
    assert a.shape == (1,)
    assert 0.0 <= a[0] <= 1.0


def test_short_train_runs():
    ppo_cfg = PPOConfig(minibatch_size=16, n_epochs=2)
    train_cfg = TrainConfig(
        total_timesteps=128, rollout_length=64, log_interval=1,
        eval_interval=1, eval_episodes=4, seed=7,
    )
    agent, result = train(_env_factory(), ppo_cfg, train_cfg)
    assert result.iterations >= 1
    assert "mean_return" in result.final_metrics
    assert len(result.eval_history) >= 1
