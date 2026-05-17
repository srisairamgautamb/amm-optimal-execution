"""Pareto with multi-seed aggregation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE_COLORS = {
    "single_dump": "#888888",
    "twap": "#1f77b4",
    "gas_aware_greedy": "#2ca02c",
    "convex_no_mev": "#9467bd",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="artifacts/phase3e/manifest.json")
    p.add_argument("--output", default="artifacts/pareto_phase3e_test.png")
    args = p.parse_args()
    m = json.loads(Path(args.manifest).read_text())

    by_lambda: dict = defaultdict(list)
    for r in m["runs"]:
        by_lambda[r["cvar_lambda"]].append(r)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for name, b in m["baselines_on_test"].items():
        color = BASELINE_COLORS.get(name, "k")
        ax.scatter(b["cvar95_loss"], b["mean"], s=160, marker="s",
                   color=color, alpha=0.95, edgecolors="black", linewidths=0.6,
                   label=name)
        ax.annotate(name, (b["cvar95_loss"], b["mean"]), fontsize=8,
                    textcoords="offset points", xytext=(6, 6))

    lambdas = sorted(by_lambda.keys())
    cmap = plt.cm.YlOrRd
    for i, lam in enumerate(lambdas):
        col = cmap(0.3 + 0.7 * i / max(len(lambdas) - 1, 1))
        runs = by_lambda[lam]
        means = [r["test_mean"] for r in runs]
        cvars = [r["test_cvar95_loss"] for r in runs]
        for r in runs:
            ax.scatter(r["test_cvar95_loss"], r["test_mean"], s=45,
                       marker="o", color=col, alpha=0.5,
                       edgecolors="black", linewidths=0.3)
        mu_m = mean(means); mu_c = mean(cvars)
        sd_m = stdev(means) if len(means) > 1 else 0.0
        sd_c = stdev(cvars) if len(cvars) > 1 else 0.0
        ax.errorbar(mu_c, mu_m, xerr=sd_c, yerr=sd_m, fmt="o",
                    color=col, ecolor=col, capsize=4, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=f"PPO lambda={lam} (n={len(runs)})")

    ax.set_xlabel("CVaR_{0.95}(loss = -reward)")
    ax.set_ylabel("mean cumulative reward on TEST window")
    ax.set_title("Phase 3e: 5 seeds x 5 lambdas, train on TRAIN, eval on TEST")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    print("\nper-lambda aggregate:")
    print(f"  {'lambda':>8} {'n':>3} {'mean':>12} {'mean_std':>10} {'cvar95':>12}")
    for lam in lambdas:
        runs = by_lambda[lam]
        means = [r["test_mean"] for r in runs]
        cvars = [r["test_cvar95_loss"] for r in runs]
        n = len(means)
        mu_m = mean(means); sd_m = stdev(means) if n > 1 else 0.0
        mu_c = mean(cvars)
        print(f"  {lam:>8} {n:>3} {mu_m:>12,.0f} {sd_m:>10,.0f} {mu_c:>12,.0f}")


if __name__ == "__main__":
    main()
