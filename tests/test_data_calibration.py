"""Data-layer tests. Offline-only via committed snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from env.amm import AMMEnv
from env.mev_bot import compute_sandwich
from data.eth_rpc import load_snapshot, save_snapshot, SNAPSHOT_DIR
from data.uniswap_v2_loader import (
    PAIRS,
    PairInfo,
    ReservesSnapshot,
    fetch_reserves,
    snapshot_name,
)
from data.gas_history import (
    GasObservation,
    fetch_latest_gas,
    fetch_gas_window,
    gas_in_quote,
)


REL_TOL = 1e-12
PINNED_BLOCK = 25103456
RESERVES_SNAP = f"reserves_weth-usdc_{PINNED_BLOCK}"
GAS_LATEST_SNAP = f"gas_latest_{PINNED_BLOCK}"
GAS_WINDOW_SNAP = f"gas_window_3_{PINNED_BLOCK}"


class TestSnapshotsExist:
    def test_reserves_snapshot_present(self):
        assert (SNAPSHOT_DIR / f"{RESERVES_SNAP}.json").exists()

    def test_gas_latest_snapshot_present(self):
        assert (SNAPSHOT_DIR / f"{GAS_LATEST_SNAP}.json").exists()

    def test_gas_window_snapshot_present(self):
        assert (SNAPSHOT_DIR / f"{GAS_WINDOW_SNAP}.json").exists()


class TestPairRegistry:
    def test_weth_usdc_addresses_locked(self):
        p = PAIRS["weth-usdc"]
        assert p.address == "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"
        assert p.token0_address == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        assert p.token1_address == "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        assert p.token0_symbol == "USDC"
        assert p.token1_symbol == "WETH"
        assert p.token0_decimals == 6
        assert p.token1_decimals == 18

    def test_snapshot_name_format(self):
        assert snapshot_name("weth-usdc", "latest") == \
            "reserves_weth-usdc_latest.json"
        assert snapshot_name("weth-usdc", 12345) == \
            "reserves_weth-usdc_12345.json"


class TestReservesDecode:
    def test_decode_pinned_snapshot(self):
        snap = fetch_reserves("weth-usdc", offline_snapshot=RESERVES_SNAP)
        assert isinstance(snap, ReservesSnapshot)
        assert snap.block_number == PINNED_BLOCK
        assert snap.reserve0_raw > 0
        assert snap.reserve1_raw > 0
        assert snap.reserve0_human == pytest.approx(snap.reserve0_raw / 1e6,
                                                    rel=REL_TOL)
        assert snap.reserve1_human == pytest.approx(snap.reserve1_raw / 1e18,
                                                    rel=REL_TOL)

    def test_implied_eth_price_in_band(self):
        snap = fetch_reserves("weth-usdc", offline_snapshot=RESERVES_SNAP)
        price = snap.reserve0_human / snap.reserve1_human
        assert 500.0 < price < 10000.0

    def test_unknown_pair_raises(self):
        with pytest.raises(KeyError):
            fetch_reserves("doge-shib", offline_snapshot=RESERVES_SNAP)

    def test_offline_snapshot_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            fetch_reserves("weth-usdc", offline_snapshot="does_not_exist")


class TestGasLoader:
    def test_fetch_latest_offline(self):
        g = fetch_latest_gas(offline_snapshot=GAS_LATEST_SNAP)
        assert isinstance(g, GasObservation)
        assert g.block_number == PINNED_BLOCK
        assert g.base_fee_per_gas_wei > 0
        assert g.gas_price_wei > 0
        assert g.base_fee_gwei == pytest.approx(g.base_fee_per_gas_wei / 1e9,
                                                rel=REL_TOL)
        assert g.gas_price_gwei == pytest.approx(g.gas_price_wei / 1e9,
                                                 rel=REL_TOL)

    def test_fetch_window_offline(self):
        w = fetch_gas_window(n_blocks=3, offline_snapshot=GAS_WINDOW_SNAP)
        assert len(w) == 3
        for o in w:
            assert isinstance(o, GasObservation)
            assert o.base_fee_per_gas_wei > 0

    def test_gas_in_quote_unit(self):
        obs = GasObservation(
            block_number=1,
            base_fee_per_gas_wei=10**9,
            gas_price_wei=10**9,
            base_fee_gwei=1.0,
            gas_price_gwei=1.0,
        )
        cost = gas_in_quote(obs, eth_quote_price=3500.0, gas_limit=200_000)
        assert cost == pytest.approx(0.7, rel=REL_TOL)

    def test_gas_in_quote_invalid_price_raises(self):
        obs = GasObservation(
            block_number=1, base_fee_per_gas_wei=10**9, gas_price_wei=10**9,
            base_fee_gwei=1.0, gas_price_gwei=1.0,
        )
        with pytest.raises(ValueError):
            gas_in_quote(obs, eth_quote_price=-1.0)

    def test_window_invalid_n_blocks_raises(self):
        with pytest.raises(ValueError):
            fetch_gas_window(n_blocks=0)


class TestCalibrate:
    def test_build_config_offline(self):
        from scripts.calibrate import build_config_from_snapshots
        cfg = build_config_from_snapshots(
            pair_name="weth-usdc",
            Q0=1.0,
            T=5,
            eth_quote_price=2221.5,
            sell_token="token1",
            reserves_snapshot=RESERVES_SNAP,
            gas_snapshot=GAS_LATEST_SNAP,
        )
        assert cfg.x0 > 1000.0
        assert cfg.x0 < 100_000.0
        assert cfg.y0 > 1_000_000.0
        assert cfg.y0 < 1e9
        assert cfg.gas_c > 0.0
        assert cfg.Q0 == 1.0
        assert cfg.T == 5
        assert cfg.gamma == 0.997
        assert cfg.mev_adversary is compute_sandwich

    def test_env_step_runs_on_real_pool(self):
        from scripts.calibrate import build_config_from_snapshots
        cfg = build_config_from_snapshots(
            pair_name="weth-usdc",
            Q0=10.0,
            T=5,
            eth_quote_price=2221.5,
            sell_token="token1",
            reserves_snapshot=RESERVES_SNAP,
            gas_snapshot=GAS_LATEST_SNAP,
            with_adversary=False,
        )
        env = AMMEnv(cfg)
        env.reset(seed=42)
        _, reward, _, _, info = env.step(np.array([0.5], dtype=np.float64))
        q_t = info["q_t"]
        expected_dy = AMMEnv.cfmm_output(q=q_t, x=cfg.x0, y=cfg.y0,
                                         gamma=cfg.gamma)
        assert info["delta_y"] == pytest.approx(expected_dy, rel=REL_TOL)
        assert reward > 0.0

    def test_invalid_inputs_raise(self):
        from scripts.calibrate import build_config_from_snapshots
        with pytest.raises(KeyError):
            build_config_from_snapshots(
                pair_name="bogus", Q0=1.0, T=5, eth_quote_price=2000.0,
                reserves_snapshot=RESERVES_SNAP, gas_snapshot=GAS_LATEST_SNAP,
            )
        with pytest.raises(ValueError):
            build_config_from_snapshots(
                pair_name="weth-usdc", Q0=-1.0, T=5, eth_quote_price=2000.0,
                reserves_snapshot=RESERVES_SNAP, gas_snapshot=GAS_LATEST_SNAP,
            )
        with pytest.raises(ValueError):
            build_config_from_snapshots(
                pair_name="weth-usdc", Q0=1.0, T=0, eth_quote_price=2000.0,
                reserves_snapshot=RESERVES_SNAP, gas_snapshot=GAS_LATEST_SNAP,
            )


class TestSnapshotRoundtrip:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.eth_rpc.SNAPSHOT_DIR", tmp_path)
        payload = {"result_hex": "0xdeadbeef", "block_number": 42}
        path = save_snapshot("tmp_test", payload)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == payload
