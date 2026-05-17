"""ExecutionPolicy protocol and AMMEnvView."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AMMEnvView:
    Q_remaining: float
    tau_remaining: int
    x: float
    y: float
    gamma: float
    gas_c: float
    T_total: int
    Q0_initial: float


class ExecutionPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self, obs: np.ndarray, info: dict, env_view: AMMEnvView
    ) -> np.ndarray: ...
