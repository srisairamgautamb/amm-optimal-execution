"""Analytical optimum under pure CFMM (no MEV adversary).

SLSQP over (q_1,..., q_T) with equality sum(q_t) = Q0 and bounds [0, Q0].
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from agent.baselines.protocol import AMMEnvView
from env.amm import AMMEnv


class ConvexNoMEVPolicy:
    name = "convex_no_mev"

    def __init__(self) -> None:
        self._schedule: Optional[List[float]] = None
        self._cursor = 0

    def reset(self) -> None:
        self._schedule = None
        self._cursor = 0

    def _solve_schedule(self, env_view: AMMEnvView) -> List[float]:
        T = int(env_view.T_total)
        Q0 = float(env_view.Q0_initial)
        x0 = float(env_view.x)
        y0 = float(env_view.y)
        gamma = float(env_view.gamma)

        def neg_cum_dy(q_vec: np.ndarray) -> float:
            x = x0
            y = y0
            s = 0.0
            for qt in q_vec:
                qt = float(max(qt, 0.0))
                if qt > 0.0 and y > 0.0:
                    dy = AMMEnv.cfmm_output(q=qt, x=x, y=y, gamma=gamma)
                    x += qt
                    y -= dy
                    s += dy
            return -s

        q0_guess = np.full(T, Q0 / T, dtype=np.float64)
        cons = [{"type": "eq", "fun": lambda q: float(np.sum(q) - Q0)}]
        bounds = [(0.0, Q0)] * T
        res = minimize(
            neg_cum_dy,
            q0_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not res.success or abs(float(np.sum(res.x) - Q0)) > 1e-5:
            return [Q0 / T] * T
        return [float(v) for v in res.x]

    def act(self, obs: np.ndarray, info: dict, env_view: AMMEnvView) -> np.ndarray:
        if self._schedule is None:
            self._schedule = self._solve_schedule(env_view)
            self._cursor = 0

        Q = env_view.Q_remaining
        if Q <= 0.0 or self._cursor >= len(self._schedule):
            return np.array([0.0], dtype=np.float64)
        q = self._schedule[self._cursor]
        self._cursor += 1
        u = max(0.0, min(1.0, q / max(Q, 1e-12)))
        return np.array([u], dtype=np.float64)
