"""Publication figures from the three 1M-step sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0]

BUGGY = {
    "name": "buggy env (terminal = no MEV)",
    "Q0_pct": 20,
    "means": [167_757, 159_229, 167_900, 168_359, 168_406],
    "stds":  [  1_108,   6_212,     892,      45,      22],
}

FIXED_OLD_CKPTS_NEW_ENV = {
    "name": "buggy checkpoints, eval on fixed env",
    "Q0_pct": 20,
    "means_seed42": [46_845, 51_095, 52_795, 37_387, 19_792],
}

GAS_AWARE_FIXED_Q20 = 58_751

SWEEP_05 = {
    "name": r"$\sigma_b = 0.5$, obs_dim = 4",
    "Q0_pct": 10,
    "means": [45_863, 31_952, 32_742, 34_646, 27_342],
    "stds":  [ 1_357,    764,  1_075,  1_443,  6_708],
}

SWEEP_15 = {
    "name": r"$\sigma_b = 1.5$, obs_dim = 7 (3-block gas history)",
    "Q0_pct": 10,
    "means": [46_542, 31_356, 30_921, 30_954, 24_732],
    "stds":  [   673,    619,  1_706,  4_166,  5_303],
}

BASELINES_Q10 = {
    "single_dump":      (    82,     0),
    "twap":             (37_847, 2_466),
    "gas_aware_greedy": (58_786,   616),
    "convex_no_mev":    (37_847, 2_466),
}

BASELINE_COLORS = {
    "single_dump": "#888888",
    "twap": "#1f77b4",
    "gas_aware_greedy": "#2ca02c",
    "convex_no_mev": "#9467bd",
}


def fig1_specification_gaming() -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LAMBDAS))
    width = 0.38

    ax.bar(x - width / 2, BUGGY["means"], width=width, yerr=BUGGY["stds"],
           color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5,
           label="trained on buggy env, evaluated on buggy env",
           capsize=4)
    ax.bar(x + width / 2, FIXED_OLD_CKPTS_NEW_ENV["means_seed42"],
           width=width, color="#1f77b4", alpha=0.85, edgecolor="black",
           linewidth=0.5,
           label="same checkpoints, evaluated on fixed env (seed 42)")
    ax.axhline(GAS_AWARE_FIXED_Q20, color="#2ca02c", linestyle="--",
               linewidth=1.5, label=f"gas_aware_greedy on fixed env ({GAS_AWARE_FIXED_Q20:,})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"lambda = {l}" for l in LAMBDAS])
    ax.set_ylabel("TEST mean reward (USDC)")
    ax.set_title("Specification gaming: terminal-dump MEV escape hatch\n"
                 "Q0 = 20% pool, 1M-step PPO checkpoints")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    path = OUT_DIR / "publication_fig1_specgaming.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig2_pareto_corrected() -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, (mean, std) in BASELINES_Q10.items():
        col = BASELINE_COLORS[name]
        ax.errorbar(std, mean, fmt="s", color=col, markersize=11,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=name)
        ax.annotate(name, (std, mean), fontsize=8,
                    textcoords="offset points", xytext=(8, 6))

    cmap_low = plt.cm.YlOrRd
    cmap_high = plt.cm.GnBu

    for i, lam in enumerate(LAMBDAS):
        c_low = cmap_low(0.35 + 0.55 * i / max(len(LAMBDAS) - 1, 1))
        c_high = cmap_high(0.45 + 0.5 * i / max(len(LAMBDAS) - 1, 1))
        ax.errorbar(SWEEP_05["stds"][i], SWEEP_05["means"][i],
                    fmt="o", color=c_low, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.4, alpha=0.85)
        ax.errorbar(SWEEP_15["stds"][i], SWEEP_15["means"][i],
                    fmt="^", color=c_high, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.4, alpha=0.85)

    ax.scatter([], [], marker="o", color="#cc6600", s=80, label=SWEEP_05["name"])
    ax.scatter([], [], marker="^", color="#1f77b4", s=80, label=SWEEP_15["name"])

    ax.set_xlabel("cross-seed standard deviation of TEST mean (USDC)")
    ax.set_ylabel("TEST mean reward, 5-seed average (USDC)")
    ax.set_title("Pareto on the corrected env: Q0 = 10% pool, T = 20, 1M steps\n"
                 "PPO does not beat the myopic gas_aware_greedy baseline")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / "publication_fig2_pareto.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig3_lambda_response() -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(LAMBDAS))
    width = 0.38

    ax.bar(x - width / 2, SWEEP_05["means"], width=width,
           yerr=SWEEP_05["stds"], color="#ff7f0e", alpha=0.85,
           edgecolor="black", linewidth=0.5,
           label=SWEEP_05["name"], capsize=4)
    ax.bar(x + width / 2, SWEEP_15["means"], width=width,
           yerr=SWEEP_15["stds"], color="#1f77b4", alpha=0.85,
           edgecolor="black", linewidth=0.5,
           label=SWEEP_15["name"], capsize=4)
    ax.axhline(BASELINES_Q10["gas_aware_greedy"][0], color="#2ca02c",
               linestyle="--", linewidth=1.5,
               label=f"gas_aware_greedy baseline ({BASELINES_Q10['gas_aware_greedy'][0]:,})")
    ax.axhline(BASELINES_Q10["twap"][0], color="#1f77b4",
               linestyle=":", linewidth=1.2,
               label=f"twap baseline ({BASELINES_Q10['twap'][0]:,})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"lambda = {l}" for l in LAMBDAS])
    ax.set_ylabel("TEST mean reward, 5-seed avg (USDC)")
    ax.set_title("Gas-history extension does not improve PPO on the corrected env\n"
                 "Q0 = 10% pool, T = 20, 1M steps, error bars = cross-seed std")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    path = OUT_DIR / "publication_fig3_lambda_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    paths = [fig1_specification_gaming(),
             fig2_pareto_corrected(),
             fig3_lambda_response()]
    for p in paths:
        print(f"saved {p}")


if __name__ == "__main__":
    main()
