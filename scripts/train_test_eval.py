"""Train PPO on TRAIN flow window, eval on disjoint TEST window."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from data.eth_rpc import SNAPSHOT_DIR
from data.feehistory_loader import fetch_fee_history, fit_gas_ar1, GasJointAR1Params
from data.swap_replay import load_swaps_parquet
from env.amm import AMMEnv, AMMConfig
from env.mev_bot import compute_sandwich
from env.gas_models import AR1LognormalGasSampler
from env.flow_replay import SwapReplayer
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig
from agent.ppo.trainer import train, TrainConfig
from agent.ppo.policy_adapter import PPOPolicyAdapter
from agent.baselines.single_dump import SingleDumpPolicy
from agent.baselines.twap import TWAPPolicy
from agent.baselines.gas_aware_greedy import GasAwareGreedyPolicy
from agent.baselines.convex_no_mev import ConvexNoMEVPolicy
from scripts.evaluate import evaluate_policy


TRAIN_PARQUET = SNAPSHOT_DIR / "swaps_weth-usdc_TRAIN_25056742_25096741.parquet"
TEST_PARQUET = SNAPSHOT_DIR / "swaps_weth-usdc_TEST_25096742_25106741.parquet"
FEEHIST_SNAP = "feehistory_200_25105952"


@dataclass(frozen=True)
class Phase3DConfig:
    Q0: float = 2e5
    T: int = 20
    pool_x0: float = 1e6
    pool_y0: float = 1e6
    gamma: float = 0.997
    base_gas_c: float = 1.0
    gas_limit: int = 200_000
    eth_quote_price: float = 2_221.5
    sigma_b_override: float = 1.5
    sigma_p_override: float = 1.0
    gas_history_len: int = 3
    total_timesteps: int = 100_000
    rollout_length: int = 1024
    seeds: Tuple[int, ...] = (42, 43)
    cvar_lambdas: Tuple[float, ...] = (0.0, 0.3, 1.0, 3.0)
    cvar_alpha: float = 0.95
    use_cvar_critic: bool = True
    eval_episodes: int = 64
    out_dir: str = "artifacts/phase3d"


def _params(cfg: Phase3DConfig) -> GasJointAR1Params:
    base = fit_gas_ar1(fetch_fee_history(offline_snapshot=FEEHIST_SNAP))
    return GasJointAR1Params(
        mu_b=base.mu_b, phi_b=base.phi_b, sigma_b=cfg.sigma_b_override,
        alpha=base.alpha, beta=base.beta, sigma_p=cfg.sigma_p_override,
    )


def make_factory(cfg: Phase3DConfig, parquet_path: Path) -> Callable[[], AMMEnv]:
    params = _params(cfg)
    events = load_swaps_parquet(parquet_path)
    start_block = events[0].block_number if events else 0

    def factory() -> AMMEnv:
        sampler = AR1LognormalGasSampler(
            params, gas_limit=cfg.gas_limit, eth_quote_price=cfg.eth_quote_price,
        )
        replayer = SwapReplayer(events, start_block=start_block, cyclic=True)
        return AMMEnv(AMMConfig(
            x0=cfg.pool_x0, y0=cfg.pool_y0, Q0=cfg.Q0, T=cfg.T,
            gamma=cfg.gamma, gas_c=cfg.base_gas_c,
            mev_adversary=compute_sandwich,
            gas_sampler=sampler, flow_replayer=replayer,
            token0_decimals=6, token1_decimals=18, flow_token0_is_x=False,
            gas_history_len=cfg.gas_history_len,
        ))
    return factory


@dataclass
class Run:
    run_id: str
    seed: int
    cvar_lambda: float
    checkpoint: str
    train_seconds: float
    train_final_mean_R: float
    test_mean: float
    test_std: float
    test_cvar95_loss: float


def run_one(cfg: Phase3DConfig, seed: int, cvar_lambda: float,
            out_dir: Path) -> Run:
    train_factory = make_factory(cfg, TRAIN_PARQUET)
    test_factory = make_factory(cfg, TEST_PARQUET)
    ppo_cfg = PPOConfig(
        obs_dim=4 + cfg.gas_history_len, act_dim=1,
        lr_policy=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, n_epochs=4, minibatch_size=64,
        entropy_coef=0.005, value_coef=0.5,
        cvar_alpha=cfg.cvar_alpha, cvar_lambda=cvar_lambda,
        use_cvar_critic=cfg.use_cvar_critic, device="cpu",
    )
    train_cfg = TrainConfig(
        total_timesteps=cfg.total_timesteps,
        rollout_length=cfg.rollout_length,
        log_interval=max(1, (cfg.total_timesteps // cfg.rollout_length) // 5),
        eval_interval=max(1, (cfg.total_timesteps // cfg.rollout_length) // 2),
        eval_episodes=16, seed=seed,
    )
    t0 = time.time()
    agent, result = train(train_factory, ppo_cfg, train_cfg)
    t1 = time.time()
    run_id = f"seed{seed}_lam{cvar_lambda}"
    ckpt = out_dir / f"{run_id}.pt"
    agent.save(ckpt)

    adapter = PPOPolicyAdapter(agent, deterministic=True)
    test_res = evaluate_policy(adapter, test_factory,
                               n_episodes=cfg.eval_episodes, seed=10_000)
    return Run(
        run_id=run_id, seed=seed, cvar_lambda=cvar_lambda,
        checkpoint=str(ckpt), train_seconds=float(t1 - t0),
        train_final_mean_R=float(result.final_metrics.get("mean_return", 0.0)),
        test_mean=float(test_res.mean), test_std=float(test_res.std),
        test_cvar95_loss=float(test_res.cvar_loss_alpha95),
    )


def run_baselines_on_test(cfg: Phase3DConfig) -> Dict[str, dict]:
    test_factory = make_factory(cfg, TEST_PARQUET)
    out: Dict[str, dict] = {}
    for pol in [SingleDumpPolicy(), TWAPPolicy(),
                GasAwareGreedyPolicy(), ConvexNoMEVPolicy()]:
        r = evaluate_policy(pol, test_factory,
                            n_episodes=cfg.eval_episodes, seed=10_000)
        out[pol.name] = {
            "mean": float(r.mean), "std": float(r.std),
            "cvar95_loss": float(r.cvar_loss_alpha95),
        }
    return out


def run_all(cfg: Phase3DConfig) -> Dict[str, object]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Run] = []
    configs = list(product(cfg.seeds, cfg.cvar_lambdas))
    print(f"phase 3d: {len(configs)} runs, {cfg.total_timesteps} steps each",
          flush=True)
    for i, (s, lam) in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] seed={s} lambda={lam}", flush=True)
        r = run_one(cfg, s, lam, out_dir)
        runs.append(r)
        print(f"  train={r.train_seconds:.1f}s "
              f"train_R={r.train_final_mean_R:.0f} "
              f"test_mean={r.test_mean:.0f} "
              f"test_cvar95_loss={r.test_cvar95_loss:.0f}",
              flush=True)
    baselines = run_baselines_on_test(cfg)
    manifest = {
        "config": asdict(cfg),
        "runs": [asdict(r) for r in runs],
        "baselines_on_test": baselines,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.3, 1.0, 3.0])
    p.add_argument("--Q0", type=float, default=2e5)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--out-dir", default="artifacts/phase3d")
    p.add_argument("--sigma-b", type=float, default=1.5,
                   help="AR(1) base-fee log-shock std override.")
    p.add_argument("--obs-dim", type=int, default=7,
                   help="Total observation dim. gas_history_len = obs_dim - 4.")
    return p.parse_args()


def main() -> None:
    a = _args()
    if a.obs_dim < 4:
        raise ValueError(f"--obs-dim must be >= 4, got {a.obs_dim}")
    cfg = Phase3DConfig(
        Q0=a.Q0, T=a.T, total_timesteps=a.total_timesteps,
        seeds=tuple(a.seeds), cvar_lambdas=tuple(a.lambdas),
        eval_episodes=a.eval_episodes, out_dir=a.out_dir,
        sigma_b_override=a.sigma_b,
        gas_history_len=a.obs_dim - 4,
    )
    run_all(cfg)


if __name__ == "__main__":
    main()
