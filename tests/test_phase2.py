"""Phase 2: feehistory, swap replay, gas sampler, flow replayer, baselines."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from env.amm import AMMEnv, AMMConfig, DEFAULT_GAMMA
from env.mev_bot import compute_sandwich
from env.gas_models import AR1LognormalGasSampler, GasSample
from env.flow_replay import SwapReplayer, BlockFlow
from data.feehistory_loader import (
    FeeHistorySnapshot,
    GasJointAR1Params,
    fetch_fee_history,
    fit_gas_ar1,
)
from data.swap_replay import (
    SwapEvent,
    fetch_swap_logs,
    save_swaps_parquet,
    load_swaps_parquet,
)
from data.eth_rpc import SNAPSHOT_DIR
from agent.baselines.protocol import AMMEnvView, ExecutionPolicy
from agent.baselines.single_dump import SingleDumpPolicy
from agent.baselines.twap import TWAPPolicy
from agent.baselines.gas_aware_greedy import GasAwareGreedyPolicy
from agent.baselines.convex_no_mev import ConvexNoMEVPolicy
from agent.baselines.rollout import run_episode, RolloutResult


PINNED_BLOCK = 25_105_952
FEEHIST_SNAP = f"feehistory_200_{PINNED_BLOCK}"
SWAPS_PARQUET = SNAPSHOT_DIR / f"swaps_weth-usdc_{PINNED_BLOCK - 500}_{PINNED_BLOCK}.parquet"


class TestFeeHistory:
    def test_decode_offline(self):
        snap = fetch_fee_history(offline_snapshot=FEEHIST_SNAP)
        assert isinstance(snap, FeeHistorySnapshot)
        assert len(snap.base_fee_per_gas_wei) == len(snap.gas_used_ratio) + 1
        assert len(snap.priority_fee_p50_wei) == len(snap.gas_used_ratio)

    def test_fit_ar1_stationary(self):
        snap = fetch_fee_history(offline_snapshot=FEEHIST_SNAP)
        params = fit_gas_ar1(snap)
        assert isinstance(params, GasJointAR1Params)
        assert -1.0 < params.phi_b < 1.0, params.phi_b
        assert params.sigma_b >= 0
        assert params.sigma_p >= 0

    def test_priority_percentile_ordering(self):
        snap = fetch_fee_history(offline_snapshot=FEEHIST_SNAP)
        for p10, p50, p90 in zip(
            snap.priority_fee_p10_wei,
            snap.priority_fee_p50_wei,
            snap.priority_fee_p90_wei,
        ):
            assert p10 <= p50 <= p90

    def test_n_blocks_validation(self):
        with pytest.raises(ValueError):
            fetch_fee_history(n_blocks=0)


class TestSwapReplay:
    def test_load_parquet(self):
        evs = load_swaps_parquet(SWAPS_PARQUET)
        assert len(evs) > 0
        for e in evs:
            assert isinstance(e, SwapEvent)

    def test_event_ordering(self):
        evs = load_swaps_parquet(SWAPS_PARQUET)
        for prev, curr in zip(evs, evs[1:]):
            assert (prev.block_number, prev.log_index) <= (curr.block_number, curr.log_index)

    def test_parquet_roundtrip(self):
        evs = [
            SwapEvent(block_number=100, log_index=0, amount0_in=1000,
                      amount1_in=0, amount0_out=0, amount1_out=500),
            SwapEvent(block_number=100, log_index=2, amount0_in=0,
                      amount1_in=2_000_000, amount0_out=300, amount1_out=0),
        ]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        save_swaps_parquet(evs, path)
        loaded = load_swaps_parquet(path)
        assert len(loaded) == 2
        assert loaded[0].amount1_out == 500
        assert loaded[1].amount0_out == 300
        path.unlink()

    def test_net_amount_signs(self):
        ev_buy0 = SwapEvent(block_number=1, log_index=0, amount0_in=1000,
                            amount1_in=0, amount0_out=0, amount1_out=500)
        assert ev_buy0.amount0_net == 1000
        assert ev_buy0.amount1_net == -500


class TestGasSampler:
    def _params(self) -> GasJointAR1Params:
        return GasJointAR1Params(
            mu_b=20.0, phi_b=0.9, sigma_b=0.1,
            alpha=18.0, beta=0.5, sigma_p=0.2,
        )

    def test_seed_reproducibility(self):
        s = AR1LognormalGasSampler(self._params(), gas_limit=200_000,
                                   eth_quote_price=2000.0)
        s.reset(seed=7)
        a = [s.sample().total_in_quote for _ in range(20)]
        s.reset(seed=7)
        b = [s.sample().total_in_quote for _ in range(20)]
        assert a == b

    def test_different_seed_diverges(self):
        s = AR1LognormalGasSampler(self._params(), gas_limit=200_000,
                                   eth_quote_price=2000.0)
        s.reset(seed=1)
        a = [s.sample().total_in_quote for _ in range(20)]
        s.reset(seed=2)
        b = [s.sample().total_in_quote for _ in range(20)]
        assert a != b

    def test_ar1_mean_recovers_mu(self):
        p = self._params()
        s = AR1LognormalGasSampler(p, gas_limit=200_000,
                                   eth_quote_price=2000.0)
        s.reset(seed=1234)
        log_base = np.array([
            math.log(s.sample().base_fee_wei) for _ in range(5_000)
        ])
        assert abs(log_base.mean() - p.mu_b) < 0.05

    def test_invalid_inputs_raise(self):
        p = self._params()
        with pytest.raises(ValueError):
            AR1LognormalGasSampler(p, gas_limit=0, eth_quote_price=2000.0)
        with pytest.raises(ValueError):
            AR1LognormalGasSampler(p, gas_limit=200_000, eth_quote_price=0.0)


class TestFlowReplayer:
    def _events(self):
        return [
            SwapEvent(block_number=100, log_index=0, amount0_in=1000,
                      amount1_in=0, amount0_out=0, amount1_out=500),
            SwapEvent(block_number=100, log_index=2, amount0_in=0,
                      amount1_in=2_000_000, amount0_out=300, amount1_out=0),
            SwapEvent(block_number=102, log_index=0, amount0_in=0,
                      amount1_in=0, amount0_out=100, amount1_out=50),
        ]

    def test_aggregation_and_empty_block(self):
        r = SwapReplayer(self._events(), start_block=100)
        f0 = r.next_block()
        assert f0.amount0_net == (1000 + 0) - (0 + 300)
        assert f0.amount1_net == (0 + 2_000_000) - (500 + 0)
        f1 = r.next_block()
        assert f1.amount0_net == 0 and f1.amount1_net == 0
        f2 = r.next_block()
        assert f2.amount0_net == -100 and f2.amount1_net == -50

    def test_reset_replays_identical(self):
        r = SwapReplayer(self._events(), start_block=100)
        first = [r.next_block() for _ in range(3)]
        r.reset()
        second = [r.next_block() for _ in range(3)]
        assert first == second

    def test_exhaust_raises(self):
        r = SwapReplayer(self._events(), start_block=100)
        for _ in range(4):
            r.next_block()
        with pytest.raises(StopIteration):
            r.next_block()

    def test_invalid_start_block(self):
        with pytest.raises(ValueError):
            SwapReplayer(self._events(), start_block=-1)


def _phase1_config(**kwargs) -> AMMConfig:
    base = dict(
        x0=1e6, y0=1e6, Q0=1e4, T=5, gamma=DEFAULT_GAMMA, gas_c=1.0,
        mev_adversary=compute_sandwich,
    )
    base.update(kwargs)
    return AMMConfig(**base)


class TestEnvStochasticBackCompat:
    def test_zero_features_reward_bit_identical(self):
        cfg = _phase1_config()
        env_a = AMMEnv(cfg)
        env_a.reset(seed=42)
        env_b = AMMEnv(cfg)
        env_b.reset(seed=42)
        rs_a = []
        rs_b = []
        actions = [np.array([0.3], dtype=np.float64) for _ in range(5)]
        for a in actions:
            _, ra, *_ = env_a.step(a)
            _, rb, *_ = env_b.step(a)
            rs_a.append(ra)
            rs_b.append(rb)
        assert rs_a == rs_b

    def test_zero_features_info_superset(self):
        cfg = _phase1_config()
        env = AMMEnv(cfg)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.5], dtype=np.float64))
        for key in ["q_t", "delta_y", "mev_triggered", "x_post", "y_post",
                    "k_pre", "k_post", "Q_remaining", "tau_remaining"]:
            assert key in info
        assert info["flow_amount0_net"] == 0
        assert info["flow_amount1_net"] == 0

    def test_zero_features_no_gas_sample_key(self):
        cfg = _phase1_config()
        env = AMMEnv(cfg)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.5], dtype=np.float64))
        assert "gas_c_realized" not in info


class TestEnvStochasticForward:
    def _gas_sampler(self) -> AR1LognormalGasSampler:
        params = GasJointAR1Params(
            mu_b=20.0, phi_b=0.9, sigma_b=0.1,
            alpha=18.0, beta=0.5, sigma_p=0.2,
        )
        return AR1LognormalGasSampler(params, gas_limit=200_000,
                                      eth_quote_price=2000.0)

    def test_seed_reproducibility_with_sampler(self):
        cfg_a = _phase1_config(gas_sampler=self._gas_sampler())
        cfg_b = _phase1_config(gas_sampler=self._gas_sampler())
        env_a = AMMEnv(cfg_a)
        env_b = AMMEnv(cfg_b)
        env_a.reset(seed=99)
        env_b.reset(seed=99)
        actions = [np.array([0.4], dtype=np.float64) for _ in range(5)]
        rs_a = [env_a.step(a)[1] for a in actions]
        rs_b = [env_b.step(a)[1] for a in actions]
        assert rs_a == rs_b

    def test_gas_realized_in_info(self):
        cfg = _phase1_config(gas_sampler=self._gas_sampler())
        env = AMMEnv(cfg)
        env.reset(seed=11)
        _, _, _, _, info = env.step(np.array([0.4], dtype=np.float64))
        assert "gas_c_realized" in info
        assert info["gas_c_realized"] > 0
        assert "base_fee_wei" in info
        assert info["base_fee_wei"] > 0

    def test_flow_shifts_pool_pre_action(self):
        ev = SwapEvent(block_number=0, log_index=0,
                       amount0_in=10**12, amount1_in=0,
                       amount0_out=0, amount1_out=0)
        replayer = SwapReplayer([ev], start_block=0)
        cfg = AMMConfig(
            x0=1e6, y0=1e6, Q0=1e4, T=2, gamma=DEFAULT_GAMMA, gas_c=1.0,
            mev_adversary=None, flow_replayer=replayer,
            token0_decimals=6, token1_decimals=18, flow_token0_is_x=True,
        )
        env = AMMEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float64))
        assert info["flow_amount0_net"] == 10**12
        assert info["x_post"] == pytest.approx(1e6 + (10**12) / 1e6, rel=1e-12)


class TestBaselines:
    def _cfg(self, mev=True):
        return AMMConfig(
            x0=1e6, y0=1e6, Q0=1e4, T=5, gamma=DEFAULT_GAMMA, gas_c=1.0,
            mev_adversary=compute_sandwich if mev else None,
        )

    def test_single_dump_exhausts_first_step(self):
        env = AMMEnv(self._cfg())
        res = run_episode(env, SingleDumpPolicy(), seed=42)
        assert res.infos[0]["q_t"] == pytest.approx(1e4, rel=1e-12)
        for info in res.infos[1:]:
            assert info["q_t"] == 0.0 or info["q_t"] == pytest.approx(0.0, abs=1e-9)

    def test_twap_action_one_over_tau(self):
        cfg = self._cfg(mev=False)
        env = AMMEnv(cfg)
        env.reset(seed=42)
        policy = TWAPPolicy()
        policy.reset()
        view = AMMEnvView(
            Q_remaining=1e4, tau_remaining=5, x=1e6, y=1e6, gamma=0.997,
            gas_c=1.0, T_total=5, Q0_initial=1e4,
        )
        a = policy.act(np.zeros(4), {}, view)
        assert a[0] == pytest.approx(0.2, rel=1e-12)

    def test_gas_aware_avoids_mev_with_high_gas(self):
        cfg = AMMConfig(
            x0=1e6, y0=1e6, Q0=1e4, T=5, gamma=DEFAULT_GAMMA,
            gas_c=1e6, mev_adversary=compute_sandwich,
        )
        env = AMMEnv(cfg)
        res = run_episode(env, GasAwareGreedyPolicy(), seed=42)
        for info in res.infos:
            assert info["mev_triggered"] is False

    def test_convex_matches_twap_symmetric_no_mev(self):
        cfg = AMMConfig(
            x0=1e6, y0=1e6, Q0=1e3, T=10, gamma=DEFAULT_GAMMA,
            gas_c=1.0, mev_adversary=None,
        )
        env = AMMEnv(cfg)
        r_twap = run_episode(env, TWAPPolicy(), seed=42).cum_reward
        r_conv = run_episode(env, ConvexNoMEVPolicy(), seed=42).cum_reward
        assert abs(r_twap - r_conv) < 1e-6

    def test_rollout_determinism(self):
        cfg = self._cfg()
        env = AMMEnv(cfg)
        r1 = run_episode(env, GasAwareGreedyPolicy(), seed=42).rewards
        r2 = run_episode(env, GasAwareGreedyPolicy(), seed=42).rewards
        assert r1 == r2
