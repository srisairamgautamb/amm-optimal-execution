"""Adapt CVaRPPO to the ExecutionPolicy protocol."""

from __future__ import annotations

import numpy as np

from agent.baselines.protocol import AMMEnvView
from agent.ppo.cvar_ppo import CVaRPPO


class PPOPolicyAdapter:
    name = "ppo_cvar"

    def __init__(self, ppo: CVaRPPO, deterministic: bool = True) -> None:
        self._ppo = ppo
        self._deterministic = bool(deterministic)

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict, env_view: AMMEnvView) -> np.ndarray:
        return self._ppo.predict(obs, deterministic=self._deterministic)
