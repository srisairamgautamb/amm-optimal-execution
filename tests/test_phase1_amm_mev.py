"""CFMM execution and MEV sandwich adversary tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.amm import AMMEnv, AMMConfig, DEFAULT_GAMMA
from env.mev_bot import compute_sandwich, SandwichOutcome, pi_mev_x


REL_TOL = 1e-12

POOL_SMALL = (1000.0, 1000.0)
POOL_MILLION = (1_000_000.0, 1_000_000.0)
POOL_HUGE = (1e18, 1e18)


class TestCFMMMath:
    def test_golden_delta_y_small_pool(self):
        expected = 9.871580343970614
        result = AMMEnv.cfmm_output(q=10.0, x=POOL_SMALL[0], y=POOL_SMALL[1],
                                    gamma=0.997)
        assert result == pytest.approx(expected, rel=REL_TOL)

    def test_golden_delta_y_million_pool(self):
        expected = 996.0069810399032
        result = AMMEnv.cfmm_output(q=1000.0, x=POOL_MILLION[0],
                                    y=POOL_MILLION[1], gamma=0.997)
        assert result == pytest.approx(expected, rel=REL_TOL)

    def test_zero_fee_invariant(self):
        x, y = POOL_HUGE
        q = 1e10
        gamma = 1.0
        dy = AMMEnv.cfmm_output(q=q, x=x, y=y, gamma=gamma)
        assert dy == pytest.approx(9999999900.0, rel=REL_TOL)
        k_pre = x * y
        k_post = (x + q) * (y - dy)
        assert k_post == pytest.approx(k_pre, rel=REL_TOL)

    def test_positive_fee_invariant(self):
        x, y = POOL_MILLION
        q = 1000.0
        gamma = 0.997
        dy = AMMEnv.cfmm_output(q=q, x=x, y=y, gamma=gamma)
        k_pre = x * y
        k_post = (x + q) * (y - dy)
        assert k_post > k_pre

    def test_zero_trade(self):
        x, y = POOL_MILLION
        dy = AMMEnv.cfmm_output(q=0.0, x=x, y=y, gamma=0.997)
        assert dy == 0.0

    def test_negative_trade_raises(self):
        with pytest.raises(ValueError):
            AMMEnv.cfmm_output(q=-1.0, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997)


def _basic_config(mev=None) -> AMMConfig:
    return AMMConfig(
        x0=POOL_MILLION[0],
        y0=POOL_MILLION[1],
        Q0=1e5,
        T=5,
        gamma=DEFAULT_GAMMA,
        gas_c=1e-3,
        mev_adversary=mev,
    )


class TestAMMEnv:
    def test_reset_obs_shape(self):
        env = AMMEnv(_basic_config())
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert obs.dtype == np.float64

    def test_action_clipping(self):
        env = AMMEnv(_basic_config())
        env.reset(seed=42)
        _, _, _, _, info_high = env.step(np.array([5.0], dtype=np.float64))
        Q0 = _basic_config().Q0
        assert info_high["q_t"] == pytest.approx(Q0, rel=REL_TOL)

    def test_inventory_drains(self):
        env = AMMEnv(_basic_config())
        env.reset(seed=42)
        terminated = False
        truncated = False
        info = None
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.5], dtype=np.float64)
            )
        assert info is not None
        assert info["Q_remaining"] < 1e-9

    def test_invariant_monotone(self):
        env = AMMEnv(_basic_config())
        env.reset(seed=42)
        terminated = False
        truncated = False
        prev_k = POOL_MILLION[0] * POOL_MILLION[1]
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.3], dtype=np.float64)
            )
            assert info["k_post"] >= info["k_pre"]
            assert info["k_post"] >= prev_k
            prev_k = info["k_post"]

    def test_no_mev_path(self):
        env = AMMEnv(_basic_config(mev=None))
        env.reset(seed=42)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.4], dtype=np.float64)
            )
            assert info["mev_triggered"] is False

    def test_token_conservation_no_mev(self):
        env = AMMEnv(_basic_config(mev=None))
        env.reset(seed=42)
        gamma = DEFAULT_GAMMA
        prev_x = POOL_MILLION[0]
        prev_y = POOL_MILLION[1]
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.4], dtype=np.float64)
            )
            q_t = info["q_t"]
            dy = info["delta_y"]
            if not terminated:
                expected_dy = AMMEnv.cfmm_output(q=q_t, x=prev_x, y=prev_y,
                                                 gamma=gamma)
                assert dy == pytest.approx(expected_dy, rel=REL_TOL)
                assert info["x_post"] == pytest.approx(prev_x + q_t,
                                                       rel=REL_TOL)
                assert info["y_post"] == pytest.approx(prev_y - dy,
                                                       rel=REL_TOL)
            prev_x = info["x_post"]
            prev_y = info["y_post"]

    def test_action_nan_raises(self):
        env = AMMEnv(_basic_config())
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(np.array([np.nan], dtype=np.float64))

    def test_action_inf_raises(self):
        env = AMMEnv(_basic_config())
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(np.array([np.inf], dtype=np.float64))

    def test_terminal_dump_in_info(self):
        env = AMMEnv(_basic_config(mev=None))
        env.reset(seed=42)
        infos = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.2], dtype=np.float64)
            )
            infos.append(info)
        assert "delta_y_term" in infos[-1]
        for info in infos[:-1]:
            assert info["delta_y_term"] == 0.0
        assert infos[-1]["delta_y_term"] > 0.0

    def test_obs_dim_with_history(self):
        cfg = AMMConfig(
            x0=POOL_MILLION[0], y0=POOL_MILLION[1], Q0=1e5, T=5,
            gamma=DEFAULT_GAMMA, gas_c=1e-3, gas_history_len=3,
        )
        env = AMMEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (7,)
        assert env.observation_space.shape == (7,)

    def test_history_zero_pad_at_reset(self):
        cfg = AMMConfig(
            x0=POOL_MILLION[0], y0=POOL_MILLION[1], Q0=1e5, T=5,
            gamma=DEFAULT_GAMMA, gas_c=1e-3, gas_history_len=3,
        )
        env = AMMEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert np.allclose(obs[4:], 0.0)

    def test_history_fifo_advances(self):
        from env.gas_models import AR1LognormalGasSampler
        from data.feehistory_loader import GasJointAR1Params
        params = GasJointAR1Params(
            mu_b=20.0, phi_b=0.9, sigma_b=0.1,
            alpha=18.0, beta=0.5, sigma_p=0.2,
        )
        sampler = AR1LognormalGasSampler(
            params, gas_limit=200_000, eth_quote_price=2000.0,
        )
        cfg = AMMConfig(
            x0=POOL_MILLION[0], y0=POOL_MILLION[1], Q0=1e5, T=10,
            gamma=DEFAULT_GAMMA, gas_c=1e-3,
            gas_sampler=sampler, gas_history_len=3,
        )
        env = AMMEnv(cfg)
        env.reset(seed=42)
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.array([0.1], dtype=np.float64))
        assert not np.allclose(obs[4:], 0.0)
        assert obs.shape == (7,)

    def test_history_disabled_returns_4dim(self):
        cfg = AMMConfig(
            x0=POOL_MILLION[0], y0=POOL_MILLION[1], Q0=1e5, T=5,
            gamma=DEFAULT_GAMMA, gas_c=1e-3,
        )
        env = AMMEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4,)


class TestMEVAdversary:
    def test_low_gas_triggers(self):
        out = compute_sandwich(q=1e4, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997, gas_c=1e-3)
        assert out.triggered is True

    def test_high_gas_skips(self):
        out = compute_sandwich(q=1e4, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997, gas_c=1e10)
        assert out.triggered is False
        assert out.mev_profit_y == pytest.approx(0.0, abs=REL_TOL)

    def test_delta_in_positive_when_triggered(self):
        out = compute_sandwich(q=1e4, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997, gas_c=1e-3)
        assert out.triggered is True
        assert out.delta_in > 0.0

    def test_back_run_profit_nonnegative_pre_gas(self):
        out = compute_sandwich(q=1e4, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997, gas_c=0.0)
        assert out.mev_profit_y >= 0.0

    def test_gas_threshold_monotone_in_q(self):
        x, y, gamma = POOL_MILLION[0], POOL_MILLION[1], 0.997
        q_grid = [5_000.0, 10_000.0, 20_000.0, 50_000.0, 100_000.0]
        outcomes = [
            compute_sandwich(q=q, x=x, y=y, gamma=gamma, gas_c=0.0)
            for q in q_grid
        ]
        c_stars = [o.mev_profit_y for o in outcomes]
        for prev, curr in zip(c_stars, c_stars[1:]):
            assert curr > prev
        upper_bound = max(q_grid[0] * 1000.0, x * 10.0)
        non_wall_deltas = [o.delta_in for o in outcomes
                           if o.delta_in < 0.95 * upper_bound]
        assert len(non_wall_deltas) >= 1, (
            "every grid point hit the optimizer wall — "
            "test is validating the bracket, not c*(q)"
        )

    def test_pool_after_sandwich_consistent(self):
        out = compute_sandwich(q=1e4, x=POOL_MILLION[0], y=POOL_MILLION[1],
                               gamma=0.997, gas_c=1e-3)
        k_pre = POOL_MILLION[0] * POOL_MILLION[1]
        k_post = out.x_post_back * out.y_post_back
        assert k_post >= k_pre


class TestIntegration:
    def test_env_with_adversary(self):
        config_no_mev = AMMConfig(
            x0=1e6, y0=1e6, Q0=1e5, T=5, gamma=DEFAULT_GAMMA,
            gas_c=1e-3, mev_adversary=None,
        )
        config_mev = AMMConfig(
            x0=1e6, y0=1e6, Q0=1e5, T=5, gamma=DEFAULT_GAMMA,
            gas_c=1e-3, mev_adversary=compute_sandwich,
        )
        actions = [np.array([0.5], dtype=np.float64) for _ in range(5)]

        env_no = AMMEnv(config_no_mev)
        env_no.reset(seed=42)
        reward_no_mev = 0.0
        for a in actions:
            _, r, term, trunc, _ = env_no.step(a)
            reward_no_mev += r
            if term or trunc:
                break

        env_yes = AMMEnv(config_mev)
        env_yes.reset(seed=42)
        reward_with_mev = 0.0
        for a in actions:
            _, r, term, trunc, _ = env_yes.step(a)
            reward_with_mev += r
            if term or trunc:
                break

        assert reward_with_mev < reward_no_mev


class TestDeterminism:
    def test_seed_reproducibility(self):
        config = _basic_config()
        actions = [np.array([0.2 + 0.1 * i], dtype=np.float64) for i in range(5)]

        env_a = AMMEnv(config)
        env_a.reset(seed=42)
        rewards_a = []
        for a in actions:
            _, r, term, trunc, _ = env_a.step(a)
            rewards_a.append(r)
            if term or trunc:
                break

        env_b = AMMEnv(config)
        env_b.reset(seed=42)
        rewards_b = []
        for a in actions:
            _, r, term, trunc, _ = env_b.step(a)
            rewards_b.append(r)
            if term or trunc:
                break

        assert rewards_a == rewards_b
