"""Single-block dump baseline: liquidate all inventory on the first step."""

from __future__ import annotations

import numpy as np

from agent.baselines.protocol import AMMEnvView


class SingleDumpPolicy:
    name = "single_dump"

    def __init__(self) -> None:
        self._done_first = False

    def reset(self) -> None:
        self._done_first = False

    def act(self, obs: np.ndarray, info: dict, env_view: AMMEnvView) -> np.ndarray:
        if not self._done_first:
            self._done_first = True
            return np.array([1.0], dtype=np.float64)
        return np.array([0.0], dtype=np.float64)
