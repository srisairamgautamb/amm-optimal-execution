"""AR(1)-lognormal sampler over (baseFeePerGas, priorityFee)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.feehistory_loader import GasJointAR1Params


@dataclass(frozen=True)
class GasSample:
    base_fee_wei: int
    priority_fee_wei: int
    total_wei: int
    total_in_quote: float


class AR1LognormalGasSampler:
    # base_log_{t+1} = mu_b + phi_b * (base_log_t - mu_b) + sigma_b * eps_b
    # prio_log       = alpha + beta * base_log_{t+1}      + sigma_p * eps_p

    def __init__(
        self,
        params: GasJointAR1Params,
        *,
        gas_limit: int = 200_000,
        eth_quote_price: float,
        initial_base_log: Optional[float] = None,
    ) -> None:
        if gas_limit <= 0:
            raise ValueError(f"gas_limit must be > 0, got {gas_limit}")
        if eth_quote_price <= 0:
            raise ValueError(f"eth_quote_price must be > 0, got {eth_quote_price}")
        if params.sigma_b < 0 or params.sigma_p < 0:
            raise ValueError("sigma_b and sigma_p must be >= 0")

        self._params = params
        self._gas_limit = int(gas_limit)
        self._eth_quote_price = float(eth_quote_price)
        self._initial_base_log = (
            float(initial_base_log) if initial_base_log is not None else float(params.mu_b)
        )
        self._base_log = self._initial_base_log
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._base_log = self._initial_base_log

    def sample(self) -> GasSample:
        p = self._params
        eps_b = float(self._rng.standard_normal())
        eps_p = float(self._rng.standard_normal())
        next_base_log = p.mu_b + p.phi_b * (self._base_log - p.mu_b) + p.sigma_b * eps_b
        prio_log = p.alpha + p.beta * next_base_log + p.sigma_p * eps_p

        base_fee_wei = max(1, int(round(float(np.exp(next_base_log)))))
        priority_fee_wei = max(0, int(round(float(np.exp(prio_log)))))
        total_wei = base_fee_wei + priority_fee_wei
        total_in_quote = float(
            np.float64(total_wei) * np.float64(self._gas_limit)
            / np.float64(1e18) * np.float64(self._eth_quote_price)
        )

        self._base_log = next_base_log

        return GasSample(
            base_fee_wei=base_fee_wei,
            priority_fee_wei=priority_fee_wei,
            total_wei=total_wei,
            total_in_quote=total_in_quote,
        )
