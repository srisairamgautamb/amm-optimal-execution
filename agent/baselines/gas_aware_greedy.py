"""Gas-aware greedy baseline.

At each block, bisect to find the largest q such that
compute_sandwich(q, x, y, gamma, gas_c).triggered is False. Trade
min(q_max, TWAP share).
"""

from __future__ import annotations

import numpy as np

from agent.baselines.protocol import AMMEnvView
from env.mev_bot import compute_sandwich


class GasAwareGreedyPolicy:
    name = "gas_aware_greedy"

    def __init__(
        self,
        q_max_search_upper_factor: float = 1.0,
        bisection_tol: float = 1e-6,
        bisection_max_iter: int = 60,
    ) -> None:
        if q_max_search_upper_factor <= 0:
            raise ValueError("q_max_search_upper_factor must be > 0")
        self._upper_factor = float(q_max_search_upper_factor)
        self._tol = float(bisection_tol)
        self._max_iter = int(bisection_max_iter)

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict, env_view: AMMEnvView) -> np.ndarray:
        Q = env_view.Q_remaining
        tau = max(env_view.tau_remaining, 1)
        if Q <= 0.0:
            return np.array([0.0], dtype=np.float64)

        x, y, g, c = env_view.x, env_view.y, env_view.gamma, env_view.gas_c
        upper = max(Q * self._upper_factor, 1e-12)
        upper_trig = compute_sandwich(q=upper, x=x, y=y, gamma=g, gas_c=c).triggered

        if not upper_trig:
            q_max = upper
        else:
            lo, hi = 0.0, upper
            for _ in range(self._max_iter):
                mid = 0.5 * (lo + hi)
                if compute_sandwich(q=mid, x=x, y=y, gamma=g, gas_c=c).triggered:
                    hi = mid
                else:
                    lo = mid
                if (hi - lo) < self._tol:
                    break
            q_max = lo

        twap_share = Q / tau
        q = min(q_max, twap_share)
        u = max(0.0, min(1.0, q / Q))
        return np.array([u], dtype=np.float64)
