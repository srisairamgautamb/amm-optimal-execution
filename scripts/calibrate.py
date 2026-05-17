"""Build an AMMConfig from real-data snapshots."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional

from env.amm import AMMConfig, DEFAULT_GAMMA
from env.mev_bot import compute_sandwich
from data.uniswap_v2_loader import PAIRS, fetch_reserves
from data.gas_history import fetch_latest_gas, gas_in_quote


def build_config_from_snapshots(
    *,
    pair_name: str,
    Q0: float,
    T: int,
    eth_quote_price: float,
    sell_token: str = "token1",
    reserves_snapshot: Optional[str] = None,
    gas_snapshot: Optional[str] = None,
    block: "int | str" = "latest",
    gamma: float = DEFAULT_GAMMA,
    gas_limit: int = 200_000,
    with_adversary: bool = True,
) -> AMMConfig:
    if pair_name not in PAIRS:
        raise KeyError(f"unknown pair {pair_name!r}; known: {list(PAIRS)}")
    if Q0 <= 0:
        raise ValueError(f"Q0 must be > 0, got {Q0}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if eth_quote_price <= 0:
        raise ValueError(f"eth_quote_price must be > 0, got {eth_quote_price}")
    if sell_token not in ("token0", "token1"):
        raise ValueError(f"sell_token must be token0 or token1, got {sell_token!r}")

    reserves = fetch_reserves(pair_name, block=block, offline_snapshot=reserves_snapshot)
    gas = fetch_latest_gas(offline_snapshot=gas_snapshot)

    if sell_token == "token1":
        x0 = reserves.reserve1_human
        y0 = reserves.reserve0_human
    else:
        x0 = reserves.reserve0_human
        y0 = reserves.reserve1_human

    gas_c = gas_in_quote(gas, eth_quote_price=eth_quote_price, gas_limit=gas_limit)
    if gas_c <= 0:
        gas_c = 1e-9

    adversary = compute_sandwich if with_adversary else None

    return AMMConfig(
        x0=x0, y0=y0, Q0=Q0, T=T, gamma=gamma, gas_c=gas_c, mev_adversary=adversary,
    )


def _config_to_json(cfg: AMMConfig) -> dict:
    raw = asdict(cfg)
    raw["mev_adversary"] = (
        f"{cfg.mev_adversary.__module__}.{cfg.mev_adversary.__name__}"
        if cfg.mev_adversary is not None else None
    )
    return raw


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate an AMMConfig from real pool data.")
    p.add_argument("--pool", required=True, choices=sorted(PAIRS.keys()))
    p.add_argument("--block", default="latest")
    p.add_argument("--Q0", type=float, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--eth-quote-price", type=float, required=True,
                   help="Price of 1 ETH in the quote token (e.g. USDC).")
    p.add_argument("--sell-token", default="token1", choices=("token0", "token1"))
    p.add_argument("--reserves-snapshot", default=None)
    p.add_argument("--gas-snapshot", default=None)
    p.add_argument("--no-adversary", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = build_config_from_snapshots(
        pair_name=args.pool,
        Q0=args.Q0,
        T=args.T,
        eth_quote_price=args.eth_quote_price,
        sell_token=args.sell_token,
        reserves_snapshot=args.reserves_snapshot,
        gas_snapshot=args.gas_snapshot,
        block=int(args.block) if args.block != "latest" else "latest",
        with_adversary=not args.no_adversary,
    )
    print(json.dumps(_config_to_json(cfg), indent=2))


if __name__ == "__main__":
    main()
