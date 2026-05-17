"""EIP-1559 fee-history loader and AR(1)-lognormal gas-model calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from data import eth_rpc


@dataclass(frozen=True)
class FeeHistorySnapshot:
    oldest_block: int
    base_fee_per_gas_wei: List[int]
    gas_used_ratio: List[float]
    priority_fee_p10_wei: List[int]
    priority_fee_p50_wei: List[int]
    priority_fee_p90_wei: List[int]


@dataclass(frozen=True)
class GasJointAR1Params:
    mu_b: float
    phi_b: float
    sigma_b: float
    alpha: float
    beta: float
    sigma_p: float


def _extract_percentile_columns(
    reward: List[List[str]],
    reward_percentiles: Tuple[float, ...],
) -> Tuple[List[int], List[int], List[int]]:
    try:
        idx_p10 = reward_percentiles.index(10.0)
        idx_p50 = reward_percentiles.index(50.0)
        idx_p90 = reward_percentiles.index(90.0)
    except ValueError as exc:
        raise ValueError(
            f"reward_percentiles must contain 10.0, 50.0, 90.0; got {reward_percentiles}"
        ) from exc

    p10: List[int] = []
    p50: List[int] = []
    p90: List[int] = []
    for row in reward:
        p10.append(int(row[idx_p10], 16))
        p50.append(int(row[idx_p50], 16))
        p90.append(int(row[idx_p90], 16))
    return p10, p50, p90


def fetch_fee_history(
    n_blocks: int = 200,
    *,
    reward_percentiles: Tuple[float, ...] = (10.0, 50.0, 90.0),
    offline_snapshot: Optional[str] = None,
) -> FeeHistorySnapshot:
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

    if offline_snapshot is not None:
        payload = eth_rpc.load_snapshot(offline_snapshot)
        return FeeHistorySnapshot(
            oldest_block=int(payload["oldest_block"]),
            base_fee_per_gas_wei=[int(v) for v in payload["base_fee_per_gas_wei"]],
            gas_used_ratio=[float(v) for v in payload["gas_used_ratio"]],
            priority_fee_p10_wei=[int(v) for v in payload["priority_fee_p10_wei"]],
            priority_fee_p50_wei=[int(v) for v in payload["priority_fee_p50_wei"]],
            priority_fee_p90_wei=[int(v) for v in payload["priority_fee_p90_wei"]],
        )

    result = eth_rpc.call(
        "eth_feeHistory",
        [hex(n_blocks), "latest", list(reward_percentiles)],
    )

    oldest_block = int(result["oldestBlock"], 16)
    base_fee_per_gas_wei = [int(v, 16) for v in result["baseFeePerGas"]]
    gas_used_ratio = [float(v) for v in result["gasUsedRatio"]]
    reward = result.get("reward") or []
    p10, p50, p90 = _extract_percentile_columns(reward, reward_percentiles)

    return FeeHistorySnapshot(
        oldest_block=oldest_block,
        base_fee_per_gas_wei=base_fee_per_gas_wei,
        gas_used_ratio=gas_used_ratio,
        priority_fee_p10_wei=p10,
        priority_fee_p50_wei=p50,
        priority_fee_p90_wei=p90,
    )


def fit_gas_ar1(snap: FeeHistorySnapshot) -> GasJointAR1Params:
    base_fee = np.array(snap.base_fee_per_gas_wei[:-1], dtype=np.float64)
    log_base = np.log(base_fee)

    p50 = np.array(snap.priority_fee_p50_wei, dtype=np.float64)
    p50_clamped = np.maximum(p50, 1.0)
    log_p50 = np.log(p50_clamped)

    y = log_base[1:]
    x = log_base[:-1]
    phi_b, intercept_b = np.polyfit(x, y, 1)
    phi_b = float(phi_b)
    intercept_b = float(intercept_b)
    if abs(1.0 - phi_b) > 1e-12:
        mu_b = intercept_b / (1.0 - phi_b)
    else:
        mu_b = float(np.mean(log_base))
    resid_b = y - (intercept_b + phi_b * x)
    sigma_b = float(np.std(resid_b, ddof=1)) if resid_b.size > 1 else 0.0

    beta, alpha = np.polyfit(log_base, log_p50, 1)
    beta = float(beta)
    alpha = float(alpha)
    resid_p = log_p50 - (alpha + beta * log_base)
    sigma_p = float(np.std(resid_p, ddof=1)) if resid_p.size > 1 else 0.0

    return GasJointAR1Params(
        mu_b=float(mu_b),
        phi_b=phi_b,
        sigma_b=sigma_b,
        alpha=alpha,
        beta=beta,
        sigma_p=sigma_p,
    )
