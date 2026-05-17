"""TWAP baseline: equal split q_t = Q_t / tau_t, action u_t = 1/tau_t."""

from __future__ import annotations

import numpy as np

from agent.baselines.protocol import AMMEnvView


class TWAPPolicy:
    name = "twap"

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict, env_view: AMMEnvView) -> np.ndarray:
        tau = max(env_view.tau_remaining, 1)
        return np.array([1.0 / tau], dtype=np.float64)
