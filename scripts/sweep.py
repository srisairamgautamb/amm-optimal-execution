"""Sweep orchestrator: train PPO across seeds x cvar lambdas."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from data.feehistory_loader import GasJointAR1Params, fetch_fee_history, fit_gas_ar1
from env.amm import AMMEnv, AMMConfig
from env.mev_bot import compute_sandwich
from env.gas_models import AR1LognormalGasSampler
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig
from agent.ppo.trainer import train, TrainConfig


@dataclass(frozen=True)
class SweepConfig:
    Q0: float = 2e5
    T: int = 20
    pool_x0: float = 1e6
    pool_y0: float = 1e6
    gamma: float = 0.997
    base_gas_c: float = 1.0
    gas_limit: int = 200_000
    eth_quote_price: float = 2000.0
    feehistory_snapshot: str = "feehistory_200_25105952"
    sigma_b_override: float = 0.5
    sigma_p_override: float = 1.0
    total_timesteps: int = 30_000
    rollout_length: int = 1024
    seeds: Tuple[int, ...] = (42, 43, 44)
    cvar_lambdas: Tuple[float, ...] = (0.0, 0.5, 2.0, 5.0)
    cvar_alpha: float = 0.95
    use_cvar_critic: bool = False
    out_dir: str = "artifacts/sweep"


def _calibrated_params(snapshot: str) -> GasJointAR1Params:
    snap = fetch_fee_history(offline_snapshot=snapshot)
    return fit_gas_ar1(snap)


def _params_with_variance(base: GasJointAR1Params, sigma_b: float,
                          sigma_p: float) -> GasJointAR1Params:
    return GasJointAR1Params(
        mu_b=base.mu_b, phi_b=base.phi_b, sigma_b=sigma_b,
        alpha=base.alpha, beta=base.beta, sigma_p=sigma_p,
    )


def make_env_factory(sw: SweepConfig) -> Callable[[], AMMEnv]:
    base_params = _calibrated_params(sw.feehistory_snapshot)
    params = _params_with_variance(base_params, sw.sigma_b_override,
                                   sw.sigma_p_override)

    def factory() -> AMMEnv:
        sampler = AR1LognormalGasSampler(
            params, gas_limit=sw.gas_limit, eth_quote_price=sw.eth_quote_price,
        )
        return AMMEnv(AMMConfig(
            x0=sw.pool_x0, y0=sw.pool_y0, Q0=sw.Q0, T=sw.T,
            gamma=sw.gamma, gas_c=sw.base_gas_c,
            mev_adversary=compute_sandwich,
            gas_sampler=sampler,
        ))
    return factory


@dataclass
class RunRecord:
    run_id: str
    checkpoint_path: str
    seed: int
    cvar_lambda: float
    cvar_alpha: float
    total_timesteps: int
    train_seconds: float
    final_metrics: Dict[str, float]
    eval_history: List[Dict[str, float]] = field(default_factory=list)


def run_single(sw: SweepConfig, *, seed: int, cvar_lambda: float,
               out_dir: Path) -> RunRecord:
    tag = "critic" if sw.use_cvar_critic else "weight"
    run_id = f"{tag}_seed{seed}_lam{cvar_lambda}"
    ckpt = out_dir / f"{run_id}.pt"
    factory = make_env_factory(sw)
    ppo_cfg = PPOConfig(
        obs_dim=4, act_dim=1,
        lr_policy=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, n_epochs=4, minibatch_size=64,
        entropy_coef=0.005, value_coef=0.5,
        cvar_alpha=sw.cvar_alpha, cvar_lambda=cvar_lambda,
        use_cvar_critic=sw.use_cvar_critic,
        device="cpu",
    )
    train_cfg = TrainConfig(
        total_timesteps=sw.total_timesteps,
        rollout_length=sw.rollout_length,
        log_interval=max(1, (sw.total_timesteps // sw.rollout_length) // 5),
        eval_interval=max(1, (sw.total_timesteps // sw.rollout_length) // 3),
        eval_episodes=16,
        seed=seed,
    )
    t0 = time.time()
    agent, result = train(factory, ppo_cfg, train_cfg)
    t1 = time.time()
    agent.save(ckpt)
    return RunRecord(
        run_id=run_id,
        checkpoint_path=str(ckpt),
        seed=seed,
        cvar_lambda=cvar_lambda,
        cvar_alpha=sw.cvar_alpha,
        total_timesteps=sw.total_timesteps,
        train_seconds=float(t1 - t0),
        final_metrics={k: float(v) for k, v in result.final_metrics.items()},
        eval_history=[{k: float(v) for k, v in e.items()} for e in result.eval_history],
    )


def run_sweep(sw: SweepConfig) -> List[RunRecord]:
    out_dir = Path(sw.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[RunRecord] = []
    configs = list(product(sw.seeds, sw.cvar_lambdas))
    print(f"Sweep: {len(configs)} runs "
          f"({len(sw.seeds)} seeds x {len(sw.cvar_lambdas)} lambdas), "
          f"{sw.total_timesteps} steps each")
    for i, (seed, lam) in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] seed={seed} lambda={lam}")
        rec = run_single(sw, seed=seed, cvar_lambda=lam, out_dir=out_dir)
        records.append(rec)
        print(f"   train_seconds={rec.train_seconds:.1f}s "
              f"final_mean_R={rec.final_metrics.get('mean_return', float('nan')):.2f}")
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps([asdict(r) for r in records], indent=2))
    print(f"\nSaved manifest: {manifest}")
    return records


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=30_000)
    p.add_argument("--rollout-length", type=int, default=1024)
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    p.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.5, 2.0, 5.0])
    p.add_argument("--out-dir", default="artifacts/sweep")
    p.add_argument("--Q0", type=float, default=5e4)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--use-cvar-critic", action="store_true",
                   help="Enable dual-head CVaR critic.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sw = SweepConfig(
        Q0=args.Q0, T=args.T,
        total_timesteps=args.total_timesteps,
        rollout_length=args.rollout_length,
        seeds=tuple(args.seeds),
        cvar_lambdas=tuple(args.lambdas),
        use_cvar_critic=args.use_cvar_critic,
        out_dir=args.out_dir,
    )
    run_sweep(sw)


if __name__ == "__main__":
    main()
