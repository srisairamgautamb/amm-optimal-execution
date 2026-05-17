"""Uniswap V2 Swap event log replay loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data import eth_rpc
from data.uniswap_v2_loader import PAIRS


SWAP_TOPIC: str = (
    "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
)
_DATA_HEX_LEN: int = 256


@dataclass(frozen=True)
class SwapEvent:
    block_number: int
    log_index: int
    amount0_in: int
    amount1_in: int
    amount0_out: int
    amount1_out: int

    @property
    def amount0_net(self) -> int:
        return self.amount0_in - self.amount0_out

    @property
    def amount1_net(self) -> int:
        return self.amount1_in - self.amount1_out


def _decode_swap_log(log: dict) -> SwapEvent:
    data = log["data"]
    if not isinstance(data, str) or not data.startswith("0x"):
        raise RuntimeError(f"Swap log data missing 0x prefix: {data!r}")
    payload = data[2:]
    if len(payload) < _DATA_HEX_LEN:
        raise RuntimeError(
            f"Swap log data too short: got {len(payload)} hex chars, "
            f"expected at least {_DATA_HEX_LEN}"
        )
    amount0_in = int(payload[0:64], 16)
    amount1_in = int(payload[64:128], 16)
    amount0_out = int(payload[128:192], 16)
    amount1_out = int(payload[192:256], 16)
    return SwapEvent(
        block_number=int(log["blockNumber"], 16),
        log_index=int(log["logIndex"], 16),
        amount0_in=amount0_in,
        amount1_in=amount1_in,
        amount0_out=amount0_out,
        amount1_out=amount1_out,
    )


def fetch_swap_logs(
    pair_name: str,
    *,
    from_block: int,
    to_block: int,
    offline_snapshot: Optional[str] = None,
    chunk_size: int = 2000,
) -> List[SwapEvent]:
    if pair_name not in PAIRS:
        raise KeyError(f"unknown pair_name {pair_name!r}; known: {sorted(PAIRS)}")
    if from_block > to_block:
        raise ValueError(
            f"from_block must be <= to_block, got {from_block} > {to_block}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    if offline_snapshot is not None:
        payload = eth_rpc.load_snapshot(offline_snapshot)
        events = [
            SwapEvent(
                block_number=int(row["block_number"]),
                log_index=int(row["log_index"]),
                amount0_in=int(row["amount0_in"]),
                amount1_in=int(row["amount1_in"]),
                amount0_out=int(row["amount0_out"]),
                amount1_out=int(row["amount1_out"]),
            )
            for row in payload
        ]
        events.sort(key=lambda e: (e.block_number, e.log_index))
        return events

    pair = PAIRS[pair_name]
    events: List[SwapEvent] = []
    for window_start in range(from_block, to_block + 1, chunk_size):
        window_end = min(window_start + chunk_size - 1, to_block)
        logs = eth_rpc.call(
            "eth_getLogs",
            [
                {
                    "address": pair.address,
                    "fromBlock": hex(window_start),
                    "toBlock": hex(window_end),
                    "topics": [SWAP_TOPIC],
                }
            ],
        )
        for log in logs:
            events.append(_decode_swap_log(log))

    events.sort(key=lambda e: (e.block_number, e.log_index))
    return events


def save_swaps_parquet(events: List[SwapEvent], path: Path) -> None:
    columns = ["block_number", "log_index", "amount0_in", "amount1_in",
               "amount0_out", "amount1_out"]
    if not events:
        df = pd.DataFrame(columns=columns)
    else:
        rows = []
        for e in events:
            rows.append({
                "block_number": int(e.block_number),
                "log_index": int(e.log_index),
                "amount0_in": str(e.amount0_in),
                "amount1_in": str(e.amount1_in),
                "amount0_out": str(e.amount0_out),
                "amount1_out": str(e.amount1_out),
            })
        df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(path, index=False)


def load_swaps_parquet(path: Path) -> List[SwapEvent]:
    df = pd.read_parquet(path)
    events: List[SwapEvent] = []
    for row in df.to_dict("records"):
        events.append(
            SwapEvent(
                block_number=int(row["block_number"]),
                log_index=int(row["log_index"]),
                amount0_in=int(row["amount0_in"]),
                amount1_in=int(row["amount1_in"]),
                amount0_out=int(row["amount0_out"]),
                amount1_out=int(row["amount1_out"]),
            )
        )
    return events
