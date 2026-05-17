"""Single-shot live end-to-end run. No snapshots, live mainnet RPC."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.eth_rpc import call
from data.uniswap_v2_loader import fetch_reserves
from data.feehistory_loader import fetch_fee_history, fit_gas_ar1, GasJointAR1Params
from data.swap_replay import fetch_swap_logs
from env.amm import AMMEnv, AMMConfig
from env.mev_bot import compute_sandwich
from env.gas_models import AR1LognormalGasSampler
from env.flow_replay import SwapReplayer
from agent.baselines.single_dump import SingleDumpPolicy
from agent.baselines.twap import TWAPPolicy
from agent.baselines.gas_aware_greedy import GasAwareGreedyPolicy
from agent.baselines.convex_no_mev import ConvexNoMEVPolicy
from agent.ppo.cvar_ppo import CVaRPPO, PPOConfig
from agent.ppo.trainer import train, TrainConfig
from agent.ppo.policy_adapter import PPOPolicyAdapter
from scripts.evaluate import evaluate_policy


STAGES = []


def stage(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            print(f"\n=== STAGE {len(STAGES) + 1}: {name} ===", flush=True)
            try:
                result = fn(*args, **kwargs)
                dt = time.time() - t0
                STAGES.append((name, "PASS", dt, None))
                print(f"  -> PASS in {dt:.1f}s", flush=True)
                return result
            except Exception as exc:
                dt = time.time() - t0
                STAGES.append((name, "FAIL", dt, str(exc)))
                print(f"  -> FAIL in {dt:.1f}s: {exc}", flush=True)
                traceback.print_exc()
                raise
        return wrapper
    return decorator


@stage("pull live reserves")
def stage_reserves():
    latest = int(call("eth_blockNumber", []), 16)
    snap = fetch_reserves("weth-usdc", block=latest)
    assert snap.reserve0_human > 0 and snap.reserve1_human > 0
    eth_price = snap.reserve0_human / snap.reserve1_human
    assert 500 < eth_price < 10000, f"eth_price out of band: {eth_price}"
    print(f"  block={latest} USDC={snap.reserve0_human:,.0f} "
          f"WETH={snap.reserve1_human:,.2f} mid=${eth_price:,.2f}")
    return latest, snap, eth_price


@stage("pull live fee history + fit AR(1)")
def stage_fee_history():
    fh = fetch_fee_history(n_blocks=200)
    params = fit_gas_ar1(fh)
    assert -1.0 < params.phi_b < 1.0
    assert params.sigma_b >= 0
    print(f"  mu_b={params.mu_b:.3f} phi_b={params.phi_b:.3f} "
          f"sigma_b={params.sigma_b:.4f} sigma_p={params.sigma_p:.4f}")
    return params


@stage("pull live Swap window + TRAIN/TEST split")
def stage_flow(latest: int):
    window = 5000
    fb = latest - window + 1
    events = fetch_swap_logs("weth-usdc", from_block=fb, to_block=latest,
                             chunk_size=2000)
    assert len(events) > 10, f"too few events: {len(events)}"
    split = int(0.8 * len(events))
    train_e = events[:split]
    test_e = events[split:]
    train_start = train_e[0].block_number
    test_start = test_e[0].block_number
    print(f"  total={len(events)} TRAIN={len(train_e)} (block {train_start}) "
          f"TEST={len(test_e)} (block {test_start})")
    return train_e, train_start, test_e, test_start


@stage("build stochastic env factories")
def stage_env(snap, params, train_e, train_start, test_e, test_start, eth_price):
    params_demo = GasJointAR1Params(
        mu_b=params.mu_b, phi_b=params.phi_b, sigma_b=0.5,
        alpha=params.alpha, beta=params.beta, sigma_p=1.0,
    )

    def make_factory(events, start_block):
        def factory():
            sampler = AR1LognormalGasSampler(
                params_demo, gas_limit=200_000, eth_quote_price=eth_price,
            )
            replayer = SwapReplayer(events, start_block=start_block, cyclic=True)
            return AMMEnv(AMMConfig(
                x0=snap.reserve1_human, y0=snap.reserve0_human,
                Q0=snap.reserve1_human * 0.20, T=20,
                gamma=0.997, gas_c=1.0,
                mev_adversary=compute_sandwich,
                gas_sampler=sampler, flow_replayer=replayer,
                token0_decimals=6, token1_decimals=18, flow_token0_is_x=False,
            ))
        return factory

    train_factory = make_factory(train_e, train_start)
    test_factory = make_factory(test_e, test_start)
    e = train_factory()
    assert e._Q0 > 0 and e._x0 > 0 and e._y0 > 0
    print(f"  Q0={e._Q0:,.2f} WETH ({100*e._Q0/e._x0:.1f}% of pool) T={e._T}")
    return train_factory, test_factory


@stage("baseline tournament on TEST")
def stage_baselines(test_factory):
    pols = [SingleDumpPolicy(), TWAPPolicy(),
            GasAwareGreedyPolicy(), ConvexNoMEVPolicy()]
    results = []
    for pol in pols:
        r = evaluate_policy(pol, test_factory, n_episodes=32, seed=10_000)
        results.append(r)
        assert r.mean > 0 or pol.name == "single_dump", (
            f"baseline {pol.name} gave non-positive mean reward"
        )
        print(f"  {r.name:22s} mean={r.mean:>12,.0f} std={r.std:>8,.0f} "
              f"cvar95_loss={r.cvar_loss_alpha95:>12,.0f}")
    return results


@stage("train PPO (vanilla + CVaR)")
def stage_train(train_factory):
    configs = [("ppo_lam0.0", 0.0), ("ppo_lam1.0", 1.0)]
    trained = {}
    for name, lam in configs:
        ppo_cfg = PPOConfig(
            obs_dim=4, act_dim=1, lr_policy=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_ratio=0.2, n_epochs=4, minibatch_size=64,
            entropy_coef=0.005, value_coef=0.5,
            cvar_alpha=0.95, cvar_lambda=lam,
            use_cvar_critic=(lam > 0.0), device="cpu",
        )
        train_cfg = TrainConfig(
            total_timesteps=30_000, rollout_length=1024,
            log_interval=20, eval_interval=40, eval_episodes=4, seed=42,
        )
        t0 = time.time()
        agent, result = train(train_factory, ppo_cfg, train_cfg)
        dt = time.time() - t0
        final = result.final_metrics["mean_return"]
        print(f"  {name}: {dt:.1f}s, final_mean_R={final:,.0f}")
        assert final > 0, f"{name} trained to non-positive return"
        trained[name] = agent
    return trained


@stage("evaluate PPO on TEST")
def stage_eval_ppo(trained, test_factory):
    results = []
    for name, agent in trained.items():
        adapter = PPOPolicyAdapter(agent, deterministic=True)
        adapter.name = name
        r = evaluate_policy(adapter, test_factory, n_episodes=32, seed=10_000)
        results.append(r)
        assert r.mean > 0, f"{name} eval mean non-positive: {r.mean}"
        print(f"  {r.name:22s} mean={r.mean:>12,.0f} std={r.std:>8,.0f} "
              f"cvar95_loss={r.cvar_loss_alpha95:>12,.0f}")
    return results


@stage("render Pareto + save artifact")
def stage_render(baseline_results, ppo_results, out_path: Path):
    palette = {
        "single_dump": "#888888", "twap": "#1f77b4",
        "gas_aware_greedy": "#2ca02c", "convex_no_mev": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in baseline_results:
        c = palette.get(r.name, "k")
        ax.scatter(r.cvar_loss_alpha95, r.mean, s=140, marker="s", color=c,
                   edgecolors="black", linewidths=0.6, label=r.name, alpha=0.9)
        ax.annotate(r.name, (r.cvar_loss_alpha95, r.mean), fontsize=8,
                    textcoords="offset points", xytext=(6, 6))
    ppo_colors = ["#d62728", "#ff7f0e"]
    for r, c in zip(ppo_results, ppo_colors):
        ax.scatter(r.cvar_loss_alpha95, r.mean, s=110, marker="o", color=c,
                   edgecolors="black", linewidths=0.6, label=r.name, alpha=0.85)
        ax.annotate(r.name, (r.cvar_loss_alpha95, r.mean), fontsize=8,
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("CVaR_{0.95}(loss = -reward)")
    ax.set_ylabel("mean cumulative reward on TEST [USDC]")
    ax.set_title("Live end-to-end Pareto (no caches, fresh RPC)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    assert out_path.exists()
    print(f"  saved {out_path}")
    return out_path


def main() -> int:
    t_total = time.time()
    print("=" * 60)
    print("FULL LIVE END-TO-END RUN")
    print("=" * 60)

    latest, snap, eth_price = stage_reserves()
    params = stage_fee_history()
    train_e, train_start, test_e, test_start = stage_flow(latest)
    train_factory, test_factory = stage_env(
        snap, params, train_e, train_start, test_e, test_start, eth_price,
    )
    baseline_results = stage_baselines(test_factory)
    trained = stage_train(train_factory)
    ppo_results = stage_eval_ppo(trained, test_factory)
    out_path = Path("artifacts/pareto_live_endtoend.png")
    stage_render(baseline_results, ppo_results, out_path)

    dt_total = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(STAGES)} stages, total {dt_total:.1f}s")
    print("=" * 60)
    for n, st, dt, err in STAGES:
        marker = "OK" if st == "PASS" else "XX"
        print(f"  [{marker}] {n:42s} {dt:>6.1f}s")
    fails = [s for s in STAGES if s[1] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} FAILURE(S):")
        for n, _, _, err in fails:
            print(f"  {n}: {err}")
        return 1
    print("\nALL STAGES PASS. live pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
