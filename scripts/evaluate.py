"""Pareto evaluation harness: mean reward vs CVaR_0.95 of loss."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.amm import AMMEnv, AMMConfig
from env.mev_bot import compute_sandwich
from agent.baselines.protocol import ExecutionPolicy
from agent.baselines.single_dump import SingleDumpPolicy
from agent.baselines.twap import TWAPPolicy
from agent.baselines.gas_aware_greedy import GasAwareGreedyPolicy
from agent.baselines.convex_no_mev import ConvexNoMEVPolicy
from agent.baselines.rollout import run_episode
from agent.ppo.cvar_ppo import CVaRPPO
from agent.ppo.policy_adapter import PPOPolicyAdapter


@dataclass(frozen=True)
class PolicyResult:
    name: str
    rewards: List[float]
    mean: float
    std: float
    cvar_loss_alpha95: float
    p5: float
    p95: float


def _cvar_loss(rewards: np.ndarray, alpha: float = 0.95) -> float:
    losses = -np.asarray(rewards, dtype=np.float64)
    losses.sort()
    cutoff = int(np.ceil(alpha * len(losses)))
    tail = losses[cutoff:] if cutoff < len(losses) else losses[-1:]
    return float(np.mean(tail))


def evaluate_policy(
    policy: ExecutionPolicy,
    env_factory: Callable[[], AMMEnv],
    *,
    n_episodes: int,
    seed: int,
) -> PolicyResult:
    rewards: List[float] = []
    env = env_factory()
    for i in range(n_episodes):
        res = run_episode(env, policy, seed=seed + i)
        rewards.append(res.cum_reward)
    arr = np.asarray(rewards, dtype=np.float64)
    return PolicyResult(
        name=getattr(policy, "name", policy.__class__.__name__),
        rewards=rewards,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        cvar_loss_alpha95=_cvar_loss(arr, alpha=0.95),
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
    )


def render_pareto(
    results: List[PolicyResult],
    output_path: Path,
    *,
    title: str = "Pareto: mean reward vs CVaR_{0.95} of loss",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        ax.scatter(r.cvar_loss_alpha95, r.mean, s=80, label=r.name)
        ax.annotate(r.name, (r.cvar_loss_alpha95, r.mean), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("CVaR_{0.95}(loss = -reward) [lower is better]")
    ax.set_ylabel("mean reward [higher is better]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def default_baselines() -> List[ExecutionPolicy]:
    return [
        SingleDumpPolicy(),
        TWAPPolicy(),
        GasAwareGreedyPolicy(),
        ConvexNoMEVPolicy(),
    ]


def default_env_factory(
    *, Q0: float = 1e4, T: int = 20,
    x0: float = 1e6, y0: float = 1e6,
    gas_c: float = 1.0, gamma: float = 0.997,
) -> Callable[[], AMMEnv]:
    def factory() -> AMMEnv:
        return AMMEnv(AMMConfig(
            x0=x0, y0=y0, Q0=Q0, T=T, gamma=gamma, gas_c=gas_c,
            mev_adversary=compute_sandwich,
        ))
    return factory


def evaluate_all(
    *,
    ppo_checkpoints: Optional[List[Path]] = None,
    env_factory: Optional[Callable[[], AMMEnv]] = None,
    n_episodes: int = 32,
    seed: int = 1000,
) -> List[PolicyResult]:
    env_factory = env_factory or default_env_factory()
    results: List[PolicyResult] = []
    for pol in default_baselines():
        results.append(evaluate_policy(
            pol, env_factory, n_episodes=n_episodes, seed=seed,
        ))
    if ppo_checkpoints:
        for ck in ppo_checkpoints:
            agent = CVaRPPO.load(Path(ck))
            adapter = PPOPolicyAdapter(agent, deterministic=True)
            adapter.name = f"ppo_{Path(ck).stem}"
            results.append(evaluate_policy(
                adapter, env_factory, n_episodes=n_episodes, seed=seed,
            ))
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Pareto plot for policies.")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--output", default="artifacts/pareto.png")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--Q0", type=float, default=1e4)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--gas-c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    factory = default_env_factory(Q0=args.Q0, T=args.T, gas_c=args.gas_c)
    ckpts = [Path(c) for c in args.checkpoints]
    results = evaluate_all(
        ppo_checkpoints=ckpts, env_factory=factory,
        n_episodes=args.episodes, seed=args.seed,
    )
    out_png = render_pareto(results, Path(args.output))
    out_json = out_png.with_suffix(".json")
    out_json.write_text(json.dumps(
        [{k: v for k, v in asdict(r).items() if k != "rewards"} for r in results],
        indent=2,
    ))
    print(f"saved {out_png}")
    print(f"saved {out_json}")
    for r in results:
        print(f"  {r.name:22s} mean={r.mean:,.2f} cvar95_loss={r.cvar_loss_alpha95:,.2f}")


if __name__ == "__main__":
    main()
