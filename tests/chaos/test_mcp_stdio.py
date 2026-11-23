"""Chaos test: 20 threads x 50 calls through one MCP client — zero argument
loss (plan P21.T3, issue #156143).

Drives a single shared :class:`~hermes.mcp.client.McpStdioClient` (one
subprocess, one reader thread, one write lock) from 20 concurrent caller
threads. Every one of the 1000 responses must echo the exact arguments sent,
proving the id-correlation and locked framing hold under concurrency.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from hermes.mcp.client import McpStdioClient

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ECHO_BOOTSTRAP = """\
import sys

from hermes.mcp.protocol import serve_stdio


def handler(request):
    params = request.get("params") or {}
    if request.get("method") == "tools/call":
        return {
            "ok": True,
            "name": params.get("name"),
            "arguments_received": params.get("arguments"),
        }
    return {"ok": True, "method": request.get("method")}


serve_stdio(handler)
"""

_THREADS = 20
_CALLS_PER_THREAD = 50
_JOIN_TIMEOUT = 120.0
_CALL_TIMEOUT = 30.0


@pytest.mark.chaos
def test_high_concurrency_no_argument_loss() -> None:
    """1000 concurrent calls through one client: every echo matches exactly."""
    failures: list[str] = []
    failures_lock = threading.Lock()
    client = McpStdioClient([sys.executable, "-c", _ECHO_BOOTSTRAP], cwd=_REPO_ROOT)
    try:

        def worker(worker_id: int) -> None:
            local_failures: list[str] = []
            for call in range(_CALLS_PER_THREAD):
                arguments: dict[str, Any] = {
                    "worker": worker_id,
                    "call": call,
                    "payload": {
                        "ids": [worker_id, call],
                        "note": f"w{worker_id}-c{call}",
                        "flags": {"ok": call % 2 == 0, "nested": {"depth": call}},
                    },
                }
                try:
                    result = client.request(
                        "tools/call", {"name": "echo", "arguments": arguments},
                        timeout=_CALL_TIMEOUT,
                    )
                    if result.get("arguments_received") != arguments:
                        local_failures.append(
                            f"worker {worker_id} call {call}: echo mismatch "
                            f"(sent={arguments!r} got={result!r})"
                        )
                except Exception as exc:  # noqa: BLE001 - chaos audit collects all
                    local_failures.append(
                        f"worker {worker_id} call {call}: {type(exc).__name__}: {exc}"
                    )
            with failures_lock:
                failures.extend(local_failures)

        threads = [
            threading.Thread(target=worker, args=(wid,), name=f"mcp-caller-{wid}")
            for wid in range(_THREADS)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_JOIN_TIMEOUT)
        assert not any(thread.is_alive() for thread in threads), (
            "a caller thread hung on a lost response"
        )

        assert failures == [], (
            f"{len(failures)} argument-loss/mismatch events in "
            f"{_THREADS * _CALLS_PER_THREAD} calls: {failures[:5]}"
        )
    finally:
        client.close()
