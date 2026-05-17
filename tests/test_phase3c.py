"""Phase 3c: cyclic replayer + CVaR dual-head critic."""

from __future__ import annotations

import torch
import pytest

from env.amm import AMMEnv, AMMConfig, DEFAULT_GAMMA
from env.mev_bot import compute_sandwich
from data.swap_replay import SwapEvent
from env.flow_replay import SwapReplayer
from agent.ppo.policy import ActorCritic
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig


def _events():
    return [
        SwapEvent(block_number=100, log_index=0, amount0_in=1000,
                  amount1_in=0, amount0_out=0, amount1_out=500),
        SwapEvent(block_number=102, log_index=0, amount0_in=0,
                  amount1_in=2_000_000, amount0_out=300, amount1_out=0),
    ]


class TestCyclicReplayer:
    def test_cyclic_wraps(self):
        r = SwapReplayer(_events(), start_block=100, cyclic=True)
        first_pass = [r.next_block() for _ in range(4)]
        wrapped = [r.next_block() for _ in range(4)]
        assert first_pass == wrapped

    def test_non_cyclic_still_raises(self):
        r = SwapReplayer(_events(), start_block=100, cyclic=False)
        for _ in range(4):
            r.next_block()
        with pytest.raises(StopIteration):
            r.next_block()

    def test_default_is_non_cyclic_backcompat(self):
        r = SwapReplayer(_events(), start_block=100)
        for _ in range(4):
            r.next_block()
        with pytest.raises(StopIteration):
            r.next_block()


class TestActorCriticDualHead:
    def test_with_cvar_head_returns_three_tuple(self):
        net = ActorCritic(obs_dim=4, act_dim=1, with_cvar_head=True)
        obs = torch.zeros(5, 4)
        out = net(obs)
        assert len(out) == 3
        dist, value, value_cvar = out
        assert value.shape == (5,)
        assert value_cvar.shape == (5,)

    def test_backcompat_single_head_returns_two_tuple(self):
        net = ActorCritic(obs_dim=4, act_dim=1)
        obs = torch.zeros(5, 4)
        out = net(obs)
        assert len(out) == 2

    def test_dual_head_critic_distinct_parameters(self):
        net = ActorCritic(obs_dim=4, act_dim=1, with_cvar_head=True)
        std_params = {id(p) for p in net.critic.parameters()}
        cvar_params = {id(p) for p in net.critic_cvar.parameters()}
        assert std_params.isdisjoint(cvar_params)


def _env():
    return AMMEnv(AMMConfig(
        x0=1e6, y0=1e6, Q0=1e4, T=5, gamma=DEFAULT_GAMMA, gas_c=1.0,
        mev_adversary=compute_sandwich,
    ))


class TestCVaRCriticUpdate:
    def test_cvar_target_in_buffer(self):
        cfg = PPOConfig(use_cvar_critic=True, cvar_lambda=1.0, minibatch_size=8)
        agent = CVaRPPO(cfg)
        buf = agent.collect_rollout(_env(), n_steps=40, rng_seed=42)
        assert buf.cvar_targets.shape == buf.rewards.shape
        assert buf.values_cvar.shape == buf.rewards.shape
        assert buf.advantages_cvar.shape == buf.rewards.shape
        assert not torch.allclose(buf.cvar_targets,
                                  torch.zeros_like(buf.cvar_targets))

    def test_phase3a_backcompat_no_cvar_critic(self):
        cfg = PPOConfig(use_cvar_critic=False, cvar_lambda=0.0,
                        minibatch_size=8)
        agent = CVaRPPO(cfg)
        buf = agent.collect_rollout(_env(), n_steps=40, rng_seed=42)
        metrics = agent.update(buf)
        assert metrics["value_loss_cvar"] == 0.0

    def test_dual_critic_distinguishes_policies(self):
        cfg_van = PPOConfig(use_cvar_critic=False, cvar_lambda=0.0,
                            minibatch_size=8)
        cfg_cvar = PPOConfig(use_cvar_critic=True, cvar_lambda=2.0,
                             minibatch_size=8)
        ag_van = CVaRPPO(cfg_van)
        ag_cvar = CVaRPPO(cfg_cvar)
        ag_van.policy.load_state_dict(
            {k: v.clone() for k, v in ag_cvar.policy.state_dict().items()
             if "cvar" not in k},
            strict=False,
        )
        buf_van = ag_van.collect_rollout(_env(), n_steps=64, rng_seed=42)
        buf_cvar = ag_cvar.collect_rollout(_env(), n_steps=64, rng_seed=42)
        ag_van.update(buf_van)
        ag_cvar.update(buf_cvar)
        diffs = []
        for (k_v, p_v), (k_c, p_c) in zip(
            ag_van.policy.named_parameters(),
            ag_cvar.policy.named_parameters(),
        ):
            if "cvar" in k_c:
                continue
            if p_v.shape == p_c.shape:
                diffs.append(float((p_v - p_c).abs().max().item()))
        assert max(diffs) > 1e-6
