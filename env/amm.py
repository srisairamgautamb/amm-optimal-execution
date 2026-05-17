"""Constant-product AMM execution environment."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces


DEFAULT_GAMMA: float = 0.997
DEFAULT_GAS: float = 1.0
DTYPE = np.float64


@dataclass(frozen=True)
class AMMConfig:
    x0: float
    y0: float
    Q0: float
    T: int
    gamma: float = DEFAULT_GAMMA
    gas_c: float = DEFAULT_GAS
    mev_adversary: Optional[Callable[..., Any]] = None
    gas_sampler: Optional[Any] = None
    flow_replayer: Optional[Any] = None
    token0_decimals: int = 6
    token1_decimals: int = 18
    flow_token0_is_x: bool = False
    gas_history_len: int = 0


class AMMEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: AMMConfig) -> None:
        super().__init__()

        if config.x0 <= 0.0 or config.y0 <= 0.0:
            raise ValueError("x0 and y0 must be strictly positive")
        if config.Q0 <= 0.0:
            raise ValueError("Q0 must be strictly positive")
        if config.T < 1:
            raise ValueError("T must be >= 1")
        if not (0.0 < config.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if config.gas_c <= 0.0:
            raise ValueError("gas_c must be strictly positive")

        self.config = config

        self._x0 = DTYPE(config.x0)
        self._y0 = DTYPE(config.y0)
        self._Q0 = DTYPE(config.Q0)
        self._T = int(config.T)
        self._gamma = DTYPE(config.gamma)
        self._c0 = DTYPE(config.gas_c)

        self._gas_history_len = int(config.gas_history_len)
        if self._gas_history_len < 0:
            raise ValueError("gas_history_len must be >= 0")
        self._gas_history: deque = deque(maxlen=self._gas_history_len)

        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + self._gas_history_len,),
            dtype=DTYPE,
        )
        self._action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=DTYPE,
        )

        self._x = self._x0
        self._y = self._y0
        self._Q = self._Q0
        self._tau = self._T
        self._c = self._c0

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @staticmethod
    def cfmm_output(q: float, x: float, y: float, gamma: float) -> float:
        q_f = DTYPE(q)
        x_f = DTYPE(x)
        y_f = DTYPE(y)
        g_f = DTYPE(gamma)

        if x_f <= DTYPE(0.0) or y_f <= DTYPE(0.0):
            raise ValueError("pool reserves x and y must be strictly positive")
        if q_f < DTYPE(0.0):
            raise ValueError("trade size q must be non-negative")
        if q_f == DTYPE(0.0):
            return float(DTYPE(0.0))

        numerator = y_f * g_f * q_f
        denominator = x_f + g_f * q_f
        return float(numerator / denominator)

    def _obs(self) -> np.ndarray:
        q_ratio = self._Q / self._Q0
        tau_ratio = DTYPE(self._tau) / DTYPE(self._T)
        log_x_ratio = np.log(self._x / self._x0)
        log_c_ratio = np.log(self._c / self._c0)
        base = [q_ratio, tau_ratio, log_x_ratio, log_c_ratio]
        if self._gas_history_len > 0:
            hist = list(self._gas_history)
            pad = self._gas_history_len - len(hist)
            base.extend([0.0] * pad)
            c0 = float(self._c0)
            for h in hist:
                base.append(float(np.log(max(h, 1e-12) / c0)))
        return np.array(base, dtype=DTYPE)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._x = self._x0
        self._y = self._y0
        self._Q = self._Q0
        self._tau = self._T
        self._c = self._c0
        self._gas_history.clear()

        if self.config.gas_sampler is not None:
            self.config.gas_sampler.reset(seed=seed)
        if self.config.flow_replayer is not None:
            self.config.flow_replayer.reset()

        info: dict = {
            "Q_remaining": float(self._Q),
            "tau_remaining": int(self._tau),
            "x_post": float(self._x),
            "y_post": float(self._y),
        }
        return self._obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=DTYPE).reshape(-1)
        if action_arr.size == 0 or not np.isfinite(action_arr[0]):
            raise ValueError(f"action must be a finite scalar, got {action!r}")

        flow_amount0_net = 0
        flow_amount1_net = 0
        if self.config.flow_replayer is not None:
            flow = self.config.flow_replayer.next_block()
            flow_amount0_net = int(flow.amount0_net)
            flow_amount1_net = int(flow.amount1_net)
            net0_human = DTYPE(flow_amount0_net) / DTYPE(10 ** self.config.token0_decimals)
            net1_human = DTYPE(flow_amount1_net) / DTYPE(10 ** self.config.token1_decimals)
            if self.config.flow_token0_is_x:
                self._x = self._x + net0_human
                self._y = self._y - net1_human
            else:
                self._x = self._x + net1_human
                self._y = self._y - net0_human
            if self._x <= DTYPE(0.0) or self._y <= DTYPE(0.0):
                raise RuntimeError(
                    f"Background flow drained pool: x={float(self._x)}, "
                    f"y={float(self._y)}. Check replay window or token-decimals."
                )

        gas_sample_info: dict = {}
        if self.config.gas_sampler is not None:
            sample = self.config.gas_sampler.sample()
            self._c = DTYPE(sample.total_in_quote)
            if self._gas_history_len > 0:
                self._gas_history.append(float(sample.total_in_quote))
            gas_sample_info = {
                "base_fee_wei": int(sample.base_fee_wei),
                "priority_fee_wei": int(sample.priority_fee_wei),
                "gas_c_realized": float(sample.total_in_quote),
            }

        u = DTYPE(np.clip(action_arr[0], 0.0, 1.0))
        q = u * self._Q

        x_pre = self._x
        y_pre = self._y
        k_pre = x_pre * y_pre

        adversary = self.config.mev_adversary
        mev_triggered = False

        if q <= DTYPE(0.0):
            delta_y = DTYPE(0.0)
            x_post = x_pre
            y_post = y_pre
        elif adversary is None:
            delta_y = DTYPE(
                AMMEnv.cfmm_output(float(q), float(x_pre), float(y_pre), float(self._gamma))
            )
            x_post = x_pre + q
            y_post = y_pre - delta_y
        else:
            outcome = adversary(
                float(q), float(x_pre), float(y_pre), float(self._gamma), float(self._c)
            )
            mev_triggered = bool(outcome.triggered)
            delta_y = DTYPE(outcome.delta_y_victim)
            x_post = DTYPE(outcome.x_post_back)
            y_post = DTYPE(outcome.y_post_back)

        self._x = x_post
        self._y = y_post
        self._Q = self._Q - q
        self._tau = self._tau - 1

        delta_y_term = DTYPE(0.0)
        mev_triggered_term = False
        if self._tau <= 0 and self._Q > DTYPE(0.0):
            q_term = self._Q
            if adversary is None:
                delta_y_term = DTYPE(
                    AMMEnv.cfmm_output(float(q_term), float(self._x), float(self._y), float(self._gamma))
                )
                self._x = self._x + q_term
                self._y = self._y - delta_y_term
            else:
                term_out = adversary(
                    float(q_term), float(self._x), float(self._y),
                    float(self._gamma), float(self._c),
                )
                mev_triggered_term = bool(term_out.triggered)
                delta_y_term = DTYPE(term_out.delta_y_victim)
                self._x = DTYPE(term_out.x_post_back)
                self._y = DTYPE(term_out.y_post_back)
            self._Q = DTYPE(0.0)

        reward = delta_y + delta_y_term
        mev_triggered = mev_triggered or mev_triggered_term

        if self._tau < 0:
            self._tau = 0

        k_post = self._x * self._y
        terminated = bool(self._tau <= 0)
        truncated = False

        info = {
            "q_t": float(q),
            "delta_y": float(delta_y),
            "delta_y_term": float(delta_y_term),
            "mev_triggered": mev_triggered,
            "x_post": float(self._x),
            "y_post": float(self._y),
            "k_pre": float(k_pre),
            "k_post": float(k_post),
            "Q_remaining": float(self._Q),
            "tau_remaining": int(self._tau),
            "flow_amount0_net": flow_amount0_net,
            "flow_amount1_net": flow_amount1_net,
        }
        info.update(gas_sample_info)
        return self._obs(), float(reward), terminated, truncated, info
