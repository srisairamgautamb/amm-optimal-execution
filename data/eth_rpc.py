"""Minimal Ethereum JSON-RPC client. Stdlib + certifi only.

Failover across multiple public endpoints, configurable timeout and retry
budget, and an offline-snapshot mode for deterministic CI replay.
"""

from __future__ import annotations

import json
import ssl
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib import request as urlrequest
from urllib.error import URLError

import certifi


_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

DEFAULT_ENDPOINTS: List[str] = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.drpc.org",
]
DEFAULT_TIMEOUT_S: float = 10.0
DEFAULT_RETRIES: int = 3
SNAPSHOT_DIR: Path = Path(__file__).parent / "snapshots"


class RpcError(RuntimeError):
    pass


def _post_json(endpoint: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "defi-execution/0.1 (research; python-urllib)",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def call(
    method: str,
    params: list,
    *,
    endpoints: Sequence[str] = DEFAULT_ENDPOINTS,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    offline_snapshot: Optional[str] = None,
) -> Any:
    if offline_snapshot is not None:
        return load_snapshot(offline_snapshot)

    body = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    last_exc: Optional[BaseException] = None

    for attempt in range(retries):
        for endpoint in endpoints:
            try:
                resp = _post_json(endpoint, body, timeout=timeout)
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                continue
            if "error" in resp:
                last_exc = RpcError(
                    f"{endpoint} returned {resp['error'].get('code')}: "
                    f"{resp['error'].get('message')}"
                )
                continue
            if "result" not in resp:
                last_exc = RpcError(f"{endpoint} malformed response: {resp}")
                continue
            return resp["result"]
        if attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))

    raise RpcError(
        f"all {len(endpoints)} endpoints failed after {retries} attempts: {last_exc}"
    )


def save_snapshot(name: str, payload: Any) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load_snapshot(name: str) -> Any:
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")
    with path.open("r") as f:
        return json.load(f)
