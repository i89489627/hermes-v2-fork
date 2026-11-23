"""P2.T3 integration test: the verify tool over the MCP protocol (plan Phase 2).

Three layers are driven against the DEFAULT persistence layout
(``memory-repo`` under the per-test ``HERMES_HOME``):

1. **Real stdio MCP**: :class:`~hermes.mcp.client.McpStdioClient` spawns
   ``python -c "from hermes.mcp.verify_tool import serve; serve()"`` (the
   subprocess inherits the per-test ``HERMES_HOME``, so its store is the same
   git-backed ``memory-repo``) and proves an end-to-end ``tools/call`` lands
   a verification event in the parent's store.
2. **The MCP dispatcher**: :func:`~hermes.mcp.verify_tool.register_verify_tool`
   answering ``tools/list`` / ``tools/call`` in-process.
3. **The Python API**: :func:`~hermes.mcp.verify_tool.verify` itself.

The echo contract (``arguments_received``, issue #156143) is exercised for
free: ``McpStdioClient.call_tool`` fails the call when the server does not
echo the exact arguments back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes.mcp.client import McpStdioClient
from hermes.mcp.verify_tool import (
    VERIFY_TOOL_NAME,
    build_request_handler,
    register_verify_tool,
    verify,
)
from hermes.memory.store import MemoryStore
from hermes.memory.types import SOURCE_WEIGHTS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Zero-argument subprocess bootstrap: default store (hermes_home/memory-repo)
#: + stdio JSON-RPC loop, per the P2.T3 MCP-accessibility design.
_BOOTSTRAP = "from hermes.mcp.verify_tool import serve\nserve()\n"


@pytest.mark.integration
def test_verify_via_mcp_adds_event() -> None:
    """tools/call through a real McpStdioClient subprocess appends the event.

    The subprocess serves on stdio with its own store rooted at
    ``hermes_home() / "memory-repo"`` — the same git repository the parent's
    default store points at, so the persisted event is visible here.
    """
    store = MemoryStore()  # default root: <HERMES_HOME>/memory-repo
    entry = store.add("The deploy flag flips staging to canary", type="Project")
    before = store.get(entry.id)
    assert before is not None
    events_before = len(before.verification_events)

    with McpStdioClient([sys.executable, "-c", _BOOTSTRAP], cwd=_REPO_ROOT) as client:
        listing = client.request("tools/list")
        tools = [tool.get("name") for tool in listing.get("tools", [])]
        assert VERIFY_TOOL_NAME in tools

        result = client.call_tool(
            VERIFY_TOOL_NAME,
            {
                "entry_id": entry.id,
                "source": "user-explicit",
                "evidence": "user confirmed the flag in chat",
            },
        )
        assert result["entry"]["entry_id"] == entry.id
        # Echo contract (#156143): exact arguments received, nothing dropped.
        assert result["arguments_received"] == {
            "entry_id": entry.id,
            "source": "user-explicit",
            "evidence": "user confirmed the flag in chat",
        }

    after = store.get(entry.id)
    assert after is not None
    assert len(after.verification_events) == events_before + 1
    added = after.verification_events[-1]
    assert added.source == "user-explicit"
    assert added.verifier == "main-model"
    assert added.evidence == "user confirmed the flag in chat"
    # Trust recomputed over the full history: two confirming events -> 1.0.
    assert after.trust_score == pytest.approx(1.0)


@pytest.mark.integration
def test_verify_with_evidence_text() -> None:
    """Evidence text lands verbatim in the persisted event, via both handler layers."""
    store = MemoryStore()
    entry = store.add("Quota resets on the first of each month", type="Reference")

    dispatch = register_verify_tool(store=store)
    descriptor = dispatch("tools/list", {})
    assert descriptor["tools"][0]["name"] == VERIFY_TOOL_NAME
    schema = descriptor["tools"][0]["inputSchema"]
    assert schema["properties"]["source"]["enum"] == list(SOURCE_WEIGHTS)

    evidence = "README section 3 states the quota reset explicitly"
    result = dispatch(
        "tools/call",
        {
            "name": VERIFY_TOOL_NAME,
            "arguments": {"entry_id": entry.id, "source": "file-read", "evidence": evidence},
        },
    )
    assert result["entry"]["evidence"] == evidence

    # Same dispatcher behind the serve_stdio-shaped request handler.
    request_handler = build_request_handler(dispatch)
    request_result = request_handler(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": VERIFY_TOOL_NAME,
                "arguments": {"entry_id": entry.id, "source": "file-read", "evidence": evidence},
            },
        }
    )
    assert request_result is not None
    assert request_result["arguments_received"] == {
        "entry_id": entry.id,
        "source": "file-read",
        "evidence": evidence,
    }

    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.verification_events[-1].evidence == evidence
    assert fetched.verification_events[-1].source == "file-read"


@pytest.mark.integration
def test_verify_test_green_event() -> None:
    """A test-green verification carries weight 1.0 and lifts the trust score."""
    store = MemoryStore()
    entry = store.add("Decay formula uses rate 0.01 per day", type="Project")
    before = store.get(entry.id)
    assert before is not None
    assert before.last_verified_at is not None

    verified = verify(
        store,
        entry.id,
        "test-green",
        "pytest -q tests/unit/memory/test_decay.py -> 5 passed",
    )
    assert verified.id == entry.id
    event = verified.verification_events[-1]
    assert event.source == "test-green"
    assert event.weight == pytest.approx(SOURCE_WEIGHTS["test-green"])  # 1.0
    assert event.value == 1.0
    assert event.verifier == "main-model"
    assert verified.trust_score == pytest.approx(1.0)
    assert verified.last_verified_at >= before.last_verified_at

    persisted = store.get(entry.id)
    assert persisted is not None
    assert persisted.verification_events[-1].source == "test-green"
