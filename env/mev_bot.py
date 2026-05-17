"""MEV sandwich adversary. Deterministic, gas-aware, float64 throughout."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


_OPT_LOWER_BOUND = 1e-9
_OPT_XATOL = 1e-10
_OPT_MAXITER = 500
_BRACKET_BIND_TOL = 0.999


@dataclass(frozen=True)
class SandwichOutcome:
    triggered: bool
    delta_in: float
    x_pre_victim: float
    y_pre_victim: float
    x_post_victim: float
    y_post_victim: float
    x_post_back: float
    y_post_back: float
    mev_profit_y: float
    delta_y_victim: float


def _cfmm_out(q: float, x: float, y: float, gamma: float) -> float:
    return (y * gamma * q) / (x + gamma * q)


def pi_mev_x(delta_in: float, q: float, x: float, y: float, gamma: float) -> float:
    d = np.float64(delta_in)
    qv = np.float64(q)
    xv = np.float64(x)
    yv = np.float64(y)
    g = np.float64(gamma)

    delta_y_front = _cfmm_out(d, xv, yv, g)
    x_prime = xv + d
    y_prime = yv - delta_y_front

    delta_y_victim = _cfmm_out(qv, x_prime, y_prime, g)
    x_double = x_prime + qv
    y_double = y_prime - delta_y_victim

    delta_x_back = (x_double * g * delta_y_front) / (y_double + g * delta_y_front)

    return float(delta_x_back - d)


def _validate(q: float, x: float, y: float, gamma: float, gas_c: float) -> None:
    if q < 0.0:
        raise ValueError(f"q must be >= 0, got {q}")
    if x <= 0.0:
        raise ValueError(f"x must be > 0, got {x}")
    if y <= 0.0:
        raise ValueError(f"y must be > 0, got {y}")
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if gas_c < 0.0:
        raise ValueError(f"gas_c must be >= 0, got {gas_c}")


def compute_sandwich(
    q: float, x: float, y: float, gamma: float, gas_c: float
) -> SandwichOutcome:
    _validate(q, x, y, gamma, gas_c)

    xv = np.float64(x)
    yv = np.float64(y)
    gv = np.float64(gamma)
    gc = np.float64(gas_c)
    qv = np.float64(q)

    if qv == 0.0:
        return SandwichOutcome(
            triggered=False,
            delta_in=0.0,
            x_pre_victim=float(xv),
            y_pre_victim=float(yv),
            x_post_victim=float(xv),
            y_post_victim=float(yv),
            x_post_back=float(xv),
            y_post_back=float(yv),
            mev_profit_y=0.0,
            delta_y_victim=0.0,
        )

    p_t = yv / xv

    def neg_net_profit_y(delta_in: float) -> float:
        profit_x = pi_mev_x(delta_in, float(qv), float(xv), float(yv), float(gv))
        return float(-(p_t * profit_x - gc))

    upper = float(max(qv * 1000.0, xv * 10.0))
    res = minimize_scalar(
        neg_net_profit_y,
        bounds=(_OPT_LOWER_BOUND, upper),
        method="bounded",
        options={"xatol": _OPT_XATOL, "maxiter": _OPT_MAXITER},
    )

    delta_star = float(res.x)
    net_profit_y = float(-res.fun)

    if net_profit_y > 0.0 and delta_star > _BRACKET_BIND_TOL * upper:
        raise RuntimeError(
            f"MEV optimizer hit upper bracket: delta*={delta_star:.6g} "
            f"upper={upper:.6g}. Widen bracket."
        )

    if net_profit_y <= 0.0:
        delta_y_victim_noatk = _cfmm_out(qv, xv, yv, gv)
        x_post_victim = float(xv + qv)
        y_post_victim = float(yv - delta_y_victim_noatk)
        return SandwichOutcome(
            triggered=False,
            delta_in=0.0,
            x_pre_victim=float(xv),
            y_pre_victim=float(yv),
            x_post_victim=x_post_victim,
            y_post_victim=y_post_victim,
            x_post_back=x_post_victim,
            y_post_back=y_post_victim,
            mev_profit_y=0.0,
            delta_y_victim=float(delta_y_victim_noatk),
        )

    delta_y_front = _cfmm_out(np.float64(delta_star), xv, yv, gv)
    x_prime = xv + np.float64(delta_star)
    y_prime = yv - delta_y_front

    delta_y_victim_atk = _cfmm_out(qv, x_prime, y_prime, gv)
    x_double = x_prime + qv
    y_double = y_prime - delta_y_victim_atk

    delta_x_back = (x_double * gv * delta_y_front) / (y_double + gv * delta_y_front)
    x_triple = x_double - delta_x_back
    y_triple = y_double + delta_y_front

    return SandwichOutcome(
        triggered=True,
        delta_in=float(delta_star),
        x_pre_victim=float(x_prime),
        y_pre_victim=float(y_prime),
        x_post_victim=float(x_double),
        y_post_victim=float(y_double),
        x_post_back=float(x_triple),
        y_post_back=float(y_triple),
        mev_profit_y=net_profit_y,
        delta_y_victim=float(delta_y_victim_atk),
    )
