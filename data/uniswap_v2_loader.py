"""Uniswap V2 reserves loader. Calls getReserves() via eth_call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from data import eth_rpc


GET_RESERVES_SELECTOR: str = "0x0902f1ac"
_EXPECTED_HEX_PAYLOAD_LEN: int = 192


@dataclass(frozen=True)
class PairInfo:
    name: str
    address: str
    token0_address: str
    token1_address: str
    token0_symbol: str
    token0_decimals: int
    token1_symbol: str
    token1_decimals: int


PAIRS: Dict[str, PairInfo] = {
    "weth-usdc": PairInfo(
        name="weth-usdc",
        address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        token0_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        token1_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        token0_symbol="USDC",
        token0_decimals=6,
        token1_symbol="WETH",
        token1_decimals=18,
    ),
}


@dataclass(frozen=True)
class ReservesSnapshot:
    pair: PairInfo
    block_number: int
    reserve0_raw: int
    reserve1_raw: int
    block_timestamp: int
    reserve0_human: float
    reserve1_human: float


def snapshot_name(pair_name: str, block: Union[int, str]) -> str:
    return f"reserves_{pair_name}_{block}.json"


def _resolve_block_tag(block: Union[int, str]) -> tuple[str, Optional[int]]:
    if isinstance(block, int):
        return hex(block), block
    if block == "latest":
        return "latest", None
    if isinstance(block, str) and block.startswith("0x"):
        return block, int(block, 16)
    raise ValueError(f"unsupported block specifier: {block!r}")


def _decode_reserves(result_hex: str) -> tuple[int, int, int]:
    if not isinstance(result_hex, str) or not result_hex.startswith("0x"):
        raise RuntimeError(f"unexpected getReserves payload (no 0x prefix): {result_hex!r}")
    raw = result_hex[2:]
    if len(raw) < _EXPECTED_HEX_PAYLOAD_LEN:
        raise RuntimeError(
            f"getReserves payload too short: got {len(raw)} hex chars, "
            f"expected at least {_EXPECTED_HEX_PAYLOAD_LEN}"
        )
    reserve0_raw = int(raw[0:64], 16)
    reserve1_raw = int(raw[64:128], 16)
    block_timestamp = int(raw[128:192], 16)
    return reserve0_raw, reserve1_raw, block_timestamp


def fetch_reserves(
    pair_name: str,
    *,
    block: Union[int, str] = "latest",
    offline_snapshot: Optional[str] = None,
) -> ReservesSnapshot:
    if pair_name not in PAIRS:
        raise KeyError(f"unknown pair_name {pair_name!r}; known: {sorted(PAIRS)}")
    pair = PAIRS[pair_name]

    if offline_snapshot is not None:
        payload = eth_rpc.load_snapshot(offline_snapshot)
        result_hex = payload["result_hex"]
        block_number = int(payload["block_number"])
    else:
        if block == "latest":
            latest_hex = eth_rpc.call("eth_blockNumber", [])
            block_number = int(latest_hex, 16)
        else:
            _, resolved = _resolve_block_tag(block)
            if resolved is None:
                raise ValueError(f"could not resolve block tag {block!r}")
            block_number = resolved

        block_tag = hex(block_number)
        result_hex = eth_rpc.call(
            "eth_call",
            [{"to": pair.address, "data": GET_RESERVES_SELECTOR}, block_tag],
        )

    reserve0_raw, reserve1_raw, block_timestamp = _decode_reserves(result_hex)

    reserve0_human = float(
        np.float64(reserve0_raw) / np.float64(10 ** pair.token0_decimals)
    )
    reserve1_human = float(
        np.float64(reserve1_raw) / np.float64(10 ** pair.token1_decimals)
    )

    return ReservesSnapshot(
        pair=pair,
        block_number=block_number,
        reserve0_raw=reserve0_raw,
        reserve1_raw=reserve1_raw,
        block_timestamp=block_timestamp,
        reserve0_human=reserve0_human,
        reserve1_human=reserve1_human,
    )
