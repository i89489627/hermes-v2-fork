"""Integration tests: MCP stdio client against real subprocess servers
(plan P21.T3, issue #156143).

Spawns actual ``python -c`` bootstrap servers built on
:mod:`hermes.mcp.protocol` and drives :class:`~hermes.mcp.client.McpStdioClient`
over real pipes: strict framing, caller-side validation, and the
``arguments_received`` echo that makes dropped arguments impossible to miss.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from hermes.mcp.argument_validator import validate_arguments
from hermes.mcp.client import (
    McpArgumentError,
    McpEchoMismatchError,
    McpStdioClient,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Echo server: answers tools/call with the EXACT arguments it received,
#: under "arguments_received" (the hermes server contract, issue #156143).
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

#: Buggy server: simulates the upstream drop — arguments replaced by defaults
#: — which the client's echo verification must turn into a loud error.
_DROPPING_BOOTSTRAP = """\
import sys

from hermes.mcp.protocol import serve_stdio


def handler(request):
    params = request.get("params") or {}
    if request.get("method") == "tools/call":
        return {
            "ok": True,
            "name": params.get("name"),
            "arguments_received": {"dropped": True},
        }
    return {"ok": True, "method": request.get("method")}


serve_stdio(handler)
"""

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "recursive": {"type": "boolean"},
        "max_depth": {"type": "integer"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


@pytest.mark.integration
def test_arguments_not_dropped() -> None:
    """200 consecutive nested-argument calls: every echo matches exactly.

    Each call's arguments exercise the full JSON surface (unicode strings,
    ints, bools, nested dicts, lists) that the upstream framing used to
    corrupt intermittently.
    """
    client = McpStdioClient(
        [sys.executable, "-c", _ECHO_BOOTSTRAP], cwd=_REPO_ROOT
    )
    try:
        for index in range(200):
            arguments: dict[str, Any] = {
                "query": f"call-{index} — ünïcode ✓",
                "filters": {
                    "tags": [f"t{index}", "shared"],
                    "limit": index,
                    "exact": index % 2 == 0,
                },
                "nested": {"deep": {"value": index * 2, "empty": {}}},
            }
            result = client.request(
                "tools/call", {"name": "search", "arguments": arguments}
            )
            assert result["ok"] is True
            assert result["name"] == "search"
            assert result["arguments_received"] == arguments, (
                f"arguments dropped or altered on call {index}"
            )
    finally:
        client.close()


@pytest.mark.integration
def test_server_echoes_arguments() -> None:
    """The echo contract: present and exact on a good call; mismatches and
    schema violations raise before a dropped-arguments call can complete."""
    client = McpStdioClient(
        [sys.executable, "-c", _ECHO_BOOTSTRAP], cwd=_REPO_ROOT
    )
    try:
        arguments = {"path": "docs/", "recursive": True, "max_depth": 3}

        # 1. Single call: the server's result carries the exact echo.
        result = client.request(
            "tools/call", {"name": "list_files", "arguments": arguments}
        )
        assert result["ok"] is True
        assert result["arguments_received"] == arguments

        # 2. A schema-checked call passes validation first, then echoes.
        assert validate_arguments(_TOOL_SCHEMA, arguments) == []
        result = client.call_tool("list_files", arguments, schema=_TOOL_SCHEMA)
        assert result["arguments_received"] == arguments

        # 3. Missing required argument: rejected client-side, BEFORE sending.
        with pytest.raises(McpArgumentError) as excinfo:
            client.call_tool(
                "list_files", {"recursive": True}, schema=_TOOL_SCHEMA
            )
        assert "'path'" in excinfo.value.errors[0]

        # The rejected request never reached the server: the very next call
        # still gets a clean, exact echo (no stray request was consumed).
        assert client.call_tool(
            "list_files", arguments, schema=_TOOL_SCHEMA
        )["arguments_received"] == arguments

        # 4. A server that drops arguments gets caught by echo verification.
        dropping = McpStdioClient(
            [sys.executable, "-c", _DROPPING_BOOTSTRAP], cwd=_REPO_ROOT
        )
        try:
            with pytest.raises(McpEchoMismatchError):
                dropping.request(
                    "tools/call", {"name": "list_files", "arguments": arguments}
                )
        finally:
            dropping.close()
    finally:
        client.close()
