"""Per-lambda 95% CI and Welch t-test vs an oracle constant.

Usage:
    python3 -m scripts.stats_test --manifest artifacts/<run>/manifest.json
    python3 -m scripts.stats_test --demo

The manifest path must point at a Phase3D-style JSON whose top-level
'runs' list contains records with 'seed', 'cvar_lambda', and 'test_mean'.
The oracle baseline is read from 'baselines_on_test.gas_aware_greedy.mean'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.stats as stats


DEMO_DATA: Dict[float, List[float]] = {
    0.0: [46_223.0, 45_800.0, 47_100.0, 44_900.0, 45_292.0],
    0.1: [31_900.0, 31_400.0, 32_500.0, 32_100.0, 31_860.0],
    0.3: [32_600.0, 32_900.0, 33_100.0, 32_400.0, 32_710.0],
    1.0: [34_400.0, 34_900.0, 34_500.0, 34_700.0, 34_730.0],
    3.0: [27_100.0, 27_600.0, 27_200.0, 27_400.0, 27_410.0],
}
DEMO_ORACLE: float = 58_786.0


def compute(per_seed: List[float], oracle_mean: float) -> Dict[str, float]:
    a = np.asarray(per_seed, dtype=np.float64)
    n = a.size
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    sem = stats.sem(a) if n > 1 else 0.0
    if n > 1:
        ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    oracle = np.full(n, oracle_mean, dtype=np.float64)
    if n > 1 and sd > 0:
        t_stat, p_val = stats.ttest_ind(a, oracle, equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")
    gap = oracle_mean - mean
    pct = mean / oracle_mean if oracle_mean else float("nan")
    return {
        "n": n, "mean": mean, "std": sd,
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "t_stat": float(t_stat), "p_val": float(p_val),
        "oracle": oracle_mean, "gap": gap, "pct_of_oracle": pct,
    }


def group_by_lambda(manifest_path: Path) -> Dict[float, List[float]]:
    m = json.loads(manifest_path.read_text())
    by: Dict[float, List[float]] = {}
    for run in m["runs"]:
        lam = float(run["cvar_lambda"])
        by.setdefault(lam, []).append(float(run["test_mean"]))
    return by


def oracle_from_manifest(manifest_path: Path) -> float:
    m = json.loads(manifest_path.read_text())
    return float(m["baselines_on_test"]["gas_aware_greedy"]["mean"])


def print_table(per_lambda: Dict[float, List[float]], oracle: float) -> None:
    print(f"\nOracle gas_aware_greedy mean: {oracle:,.2f}")
    print("-" * 96)
    hdr = (f"{'lambda':>8} {'n':>3} {'mean':>10} {'std':>9}"
           f" {'95% CI':>22} {'%oracle':>8} {'t':>8} {'p':>11}")
    print(hdr)
    print("-" * 96)
    for lam in sorted(per_lambda):
        r = compute(per_lambda[lam], oracle)
        ci = f"[{r['ci_lo']:,.0f}, {r['ci_hi']:,.0f}]"
        print(f"{lam:>8.2f} {r['n']:>3d} {r['mean']:>10,.0f}"
              f" {r['std']:>9,.0f} {ci:>22}"
              f" {r['pct_of_oracle']*100:>7.1f}%"
              f" {r['t_stat']:>8.2f} {r['p_val']:>11.2e}")
    print("-" * 96)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--oracle", type=float, default=None,
                   help="Override oracle mean (else read from manifest).")
    p.add_argument("--demo", action="store_true",
                   help="Run on hardcoded illustrative data.")
    args = p.parse_args()

    if args.demo:
        print("[DEMO MODE: illustrative per-seed values]")
        print_table(DEMO_DATA, DEMO_ORACLE)
        return

    if args.manifest is None:
        raise SystemExit("Provide --manifest <path> or --demo")
    mp = Path(args.manifest)
    by_lam = group_by_lambda(mp)
    oracle = args.oracle if args.oracle is not None else oracle_from_manifest(mp)
    print_table(by_lam, oracle)


if __name__ == "__main__":
    main()
