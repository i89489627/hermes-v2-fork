"""``verify`` MCP tool: explicit trust verification over MCP (plan P2.T3).

The plan Phase 2 component ``hermes/mcp/verify_tool.py`` — an MCP tool the
main agent calls to append a verification event to a memory entry and
recompute its trust score (plan Phase 2 acceptance: *``verify`` MCP tool
callable, updates trust score*). Persistence goes through
:meth:`hermes.memory.store.MemoryStore.record_event` (plan P2.T1): atomic
trust-card rewrite, one Git commit, weighted-average score recompute.

Layers
------
- :func:`verify` — the Python entry point: ``verify(store, entry_id, source,
  evidence, verifier="main-model") -> MemoryEntry``. Validates ``source``
  against the plan Section 2.4 literal set (:data:`SOURCE_WEIGHTS` keys) and
  records the event with ``value=1.0`` (a verification *confirms*; negative
  anchors are the P2.T5 test-hook's job, not the verify tool's).
- :func:`register_verify_tool` — builds the MCP surface: a ``(method,
  params) -> result`` dispatcher answering ``tools/list`` (the ``verify``
  descriptor with its JSON schema) and ``tools/call`` (runs :func:`verify`),
  delegating every other method to an optional wrapped ``server_handler``.
  The default store is rooted at ``hermes_home() / "memory-repo"``.
- :func:`build_request_handler` — adapts a dispatcher to the
  :func:`hermes.mcp.protocol.serve_stdio` handler contract (a full JSON-RPC
  request mapping in, result mapping out).
- :func:`serve` — runs the tool over real stdio (newline-delimited JSON-RPC
  2.0 via :mod:`hermes.mcp.protocol`); the zero-argument form is the
  subprocess bootstrap (``python -c "from hermes.mcp.verify_tool import
  serve; serve()"``) the integration suite drives with
  :class:`hermes.mcp.client.McpStdioClient`.

Echo contract (issue #156143): every ``tools/call`` result carries
``arguments_received`` — the EXACT arguments dict the server received — so a
:class:`~hermes.mcp.client.McpStdioClient` caller can never lose arguments
silently on the stdio hop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

from hermes.mcp.protocol import RequestHandler, serve_stdio
from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore
from hermes.memory.types import SOURCE_WEIGHTS
from hermes.memory.verification import build_event

__all__ = [
    "VERIFY_TOOL_NAME",
    "VERIFY_TOOL_SCHEMA",
    "build_request_handler",
    "register_verify_tool",
    "serve",
    "verify",
    "verify_tool_descriptor",
]

_LOGGER = logging.getLogger(__name__)

#: Name the tool registers under (plan P2.T3: ``verify(entry_id, source,
#: evidence)``).
VERIFY_TOOL_NAME: Final[str] = "verify"

#: JSON schema of the tool's arguments (plan P2.T3): ``entry_id`` string,
#: ``source`` enum over the plan Section 2.4 provenance set, ``evidence``
#: string. ``source`` is required alongside ``entry_id`` — an enum without a
#: value is meaningless and :func:`verify` would reject it anyway.
VERIFY_TOOL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "entry_id": {
            "type": "string",
            "description": "Id of the memory entry to verify.",
        },
        "source": {
            "type": "string",
            "enum": list(SOURCE_WEIGHTS),
            "description": (
                "Provenance of the verification (plan Section 2.4 weight table)."
            ),
        },
        "evidence": {
            "type": "string",
            "description": "Free-text evidence recorded with the event.",
        },
    },
    "required": ["entry_id", "source"],
    "additionalProperties": False,
}

#: ``(method, params) -> result`` dispatcher shape returned by
#: :func:`register_verify_tool` (plan P2.T3 handler signature).
DispatchHandler = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def verify(
    store: MemoryStore,
    entry_id: str,
    source: str,
    evidence: str = "",
    verifier: str = "main-model",
) -> MemoryEntry:
    """Append one verification event to ``entry_id`` and recompute trust.

    Validates ``source`` against the plan Section 2.4 literal set
    (:class:`ValueError` otherwise), builds the event (weight derived from
    the source, ``value=1.0`` — a verification confirms) with ``verifier``
    stamped (default ``main-model``: the calling agent) and persists it via
    :meth:`~hermes.memory.store.MemoryStore.record_event`. Unknown entry ids
    raise :class:`KeyError` (surfaced by the MCP layer as a JSON-RPC error).
    """
    if source not in SOURCE_WEIGHTS:
        raise ValueError(
            f"unknown verification source: {source!r}; must be one of "
            f"{', '.join(SOURCE_WEIGHTS)}"
        )
    event = build_event(
        source=source,
        verifier=verifier,
        evidence=evidence,
        value=1.0,
    )
    return store.record_event(entry_id, event)


def verify_tool_descriptor() -> dict[str, Any]:
    """MCP ``tools/list`` descriptor for the verify tool."""
    return {
        "name": VERIFY_TOOL_NAME,
        "description": (
            "Append a verification event (source + evidence) to a memory "
            "entry and recompute its trust score."
        ),
        "inputSchema": VERIFY_TOOL_SCHEMA,
    }


def _require_str(arguments: Mapping[str, Any], key: str) -> str:
    """Fetch a required string argument (``ValueError`` when missing/other)."""
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"verify tool requires a non-empty string argument {key!r}")
    return value


def _call_verify(store: MemoryStore, params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one ``tools/call`` for the verify tool (echo contract kept)."""
    name = params.get("name")
    if name != VERIFY_TOOL_NAME:
        raise ValueError(
            f"unknown tool {name!r}; this dispatcher serves {VERIFY_TOOL_NAME!r} only"
        )
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ValueError(f"tools/call arguments must be a mapping, got {type(arguments).__name__}")
    entry_id = _require_str(arguments, "entry_id")
    source = _require_str(arguments, "source")
    evidence_raw = arguments.get("evidence", "")
    evidence = evidence_raw if isinstance(evidence_raw, str) else str(evidence_raw)
    _LOGGER.info("verify tool: entry=%s source=%s", entry_id, source)
    entry = verify(store, entry_id, source, evidence)
    payload = {
        "entry_id": entry.id,
        "trust_score": entry.trust_score,
        "verification_events": len(entry.verification_events),
        "source": source,
        "verifier": entry.verification_events[-1].verifier,
        "evidence": evidence,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "entry": payload,
        # hermes MCP contract (issue #156143): echo the EXACT arguments back.
        "arguments_received": dict(arguments),
    }


def register_verify_tool(
    server_handler: DispatchHandler | None = None,
    store: MemoryStore | None = None,
) -> DispatchHandler:
    """Build the ``tools/list`` + ``tools/call`` dispatcher pair (plan P2.T3).

    The returned dispatcher answers ``tools/list`` with the verify descriptor
    and ``tools/call`` by running :func:`verify` against ``store`` (default:
    a :class:`~hermes.memory.store.MemoryStore` rooted at
    ``hermes_home() / "memory-repo"``). Any other method delegates to
    ``server_handler`` when provided, else raises :class:`ValueError` (which
    the stdio layer turns into a ``-32603`` error response). ``store`` is an
    embedding/testing seam beyond the plan signature; production callers use
    the default.
    """
    resolved_store = store if store is not None else MemoryStore()

    def dispatch(method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if method == "tools/list":
            return {"tools": [verify_tool_descriptor()]}
        if method == "tools/call":
            return _call_verify(resolved_store, params)
        if server_handler is not None:
            return server_handler(method, params)
        raise ValueError(f"unknown MCP method for the verify tool server: {method!r}")

    return dispatch


def build_request_handler(dispatch: DispatchHandler) -> RequestHandler:
    """Adapt a ``(method, params)`` dispatcher to the stdio handler contract.

    The returned handler takes the FULL JSON-RPC request mapping (as
    :func:`hermes.mcp.protocol.serve_stdio` hands them over), extracts
    ``method``/``params`` and returns the dispatcher's result for wrapping.
    Handler exceptions propagate — ``serve_stdio`` converts them into
    ``-32603`` error responses without wedging the framing loop.
    """

    def request_handler(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        method = request.get("method")
        if not isinstance(method, str):
            raise ValueError(f"request carries no string 'method': {dict(request)!r}")
        raw_params = request.get("params")
        params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}
        return dict(dispatch(method, params))

    return request_handler


def serve(handler: RequestHandler | None = None) -> None:
    """Serve the verify tool over stdio until EOF (plan P2.T3 MCP access).

    ``handler`` follows the :func:`hermes.mcp.protocol.serve_stdio` handler
    signature (full request mapping in, result mapping out); ``None`` builds
    the default verify server (:func:`register_verify_tool` adapted through
    :func:`build_request_handler`). The zero-argument form is the subprocess
    bootstrap: ``python -c "from hermes.mcp.verify_tool import serve; serve()"``.
    """
    request_handler = handler if handler is not None else build_request_handler(
        register_verify_tool()
    )
    serve_stdio(request_handler)
