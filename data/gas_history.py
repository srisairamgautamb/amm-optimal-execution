"""Ethereum gas-history loader. Pulls baseFeePerGas and gas price."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from data import eth_rpc


@dataclass(frozen=True)
class GasObservation:
    block_number: int
    base_fee_per_gas_wei: int
    gas_price_wei: int
    base_fee_gwei: float
    gas_price_gwei: float


def _parse_hex(value: Any, field: str) -> int:
    if not isinstance(value, str):
        raise RuntimeError(f"expected hex string for {field}, got {type(value).__name__}: {value!r}")
    return int(value, 16)


def _observation_from_block(block: dict, gas_price_wei: int) -> GasObservation:
    if "baseFeePerGas" not in block or block.get("baseFeePerGas") is None:
        raise RuntimeError(
            f"baseFeePerGas missing on block {block.get('number')!r}; "
            "pre-EIP-1559 block not supported by this loader"
        )
    block_number = _parse_hex(block["number"], "block.number")
    base_fee_per_gas_wei = _parse_hex(block["baseFeePerGas"], "block.baseFeePerGas")
    base_fee_gwei = float(base_fee_per_gas_wei) / 1e9
    gas_price_gwei = float(gas_price_wei) / 1e9
    return GasObservation(
        block_number=block_number,
        base_fee_per_gas_wei=base_fee_per_gas_wei,
        gas_price_wei=gas_price_wei,
        base_fee_gwei=base_fee_gwei,
        gas_price_gwei=gas_price_gwei,
    )


def _observation_from_payload(payload: dict) -> GasObservation:
    required = {
        "block_number",
        "base_fee_per_gas_wei",
        "gas_price_wei",
        "base_fee_gwei",
        "gas_price_gwei",
    }
    missing = required - payload.keys()
    if missing:
        raise RuntimeError(f"offline snapshot missing fields: {sorted(missing)}")
    return GasObservation(
        block_number=int(payload["block_number"]),
        base_fee_per_gas_wei=int(payload["base_fee_per_gas_wei"]),
        gas_price_wei=int(payload["gas_price_wei"]),
        base_fee_gwei=float(payload["base_fee_gwei"]),
        gas_price_gwei=float(payload["gas_price_gwei"]),
    )


def fetch_latest_gas(*, offline_snapshot: Optional[str] = None) -> GasObservation:
    if offline_snapshot is not None:
        payload = eth_rpc.load_snapshot(offline_snapshot)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"offline snapshot {offline_snapshot!r} must be a dict for fetch_latest_gas"
            )
        return _observation_from_payload(payload)

    latest_hex = eth_rpc.call("eth_blockNumber", [])
    latest_block_number = _parse_hex(latest_hex, "eth_blockNumber")
    block_tag = hex(latest_block_number)
    block = eth_rpc.call("eth_getBlockByNumber", [block_tag, False])
    if not isinstance(block, dict):
        raise RuntimeError(f"eth_getBlockByNumber returned non-dict: {type(block).__name__}")
    gas_price_hex = eth_rpc.call("eth_gasPrice", [])
    gas_price_wei = _parse_hex(gas_price_hex, "eth_gasPrice")
    return _observation_from_block(block, gas_price_wei)


def fetch_gas_window(
    *,
    n_blocks: int = 20,
    offline_snapshot: Optional[str] = None,
) -> List[GasObservation]:
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

    if offline_snapshot is not None:
        payload = eth_rpc.load_snapshot(offline_snapshot)
        if not isinstance(payload, list):
            raise RuntimeError(
                f"offline snapshot {offline_snapshot!r} must be a list for fetch_gas_window"
            )
        return [_observation_from_payload(item) for item in payload]

    latest_hex = eth_rpc.call("eth_blockNumber", [])
    latest_block_number = _parse_hex(latest_hex, "eth_blockNumber")
    gas_price_hex = eth_rpc.call("eth_gasPrice", [])
    gas_price_wei = _parse_hex(gas_price_hex, "eth_gasPrice")

    observations: List[GasObservation] = []
    for offset in range(n_blocks):
        target = latest_block_number - offset
        if target < 0:
            raise RuntimeError(f"requested block {target} is below genesis")
        block = eth_rpc.call("eth_getBlockByNumber", [hex(target), False])
        if not isinstance(block, dict):
            raise RuntimeError(
                f"eth_getBlockByNumber({hex(target)}) returned non-dict: {type(block).__name__}"
            )
        observations.append(_observation_from_block(block, gas_price_wei))
    return observations


def gas_in_quote(
    observation: GasObservation,
    *,
    eth_quote_price: float,
    gas_limit: int = 200_000,
) -> float:
    if gas_limit <= 0:
        raise ValueError(f"gas_limit must be > 0, got {gas_limit}")
    if eth_quote_price <= 0:
        raise ValueError(f"eth_quote_price must be > 0, got {eth_quote_price}")
    cost_eth = float(gas_limit) * float(observation.base_fee_per_gas_wei) / 1e18
    return cost_eth * float(eth_quote_price)
