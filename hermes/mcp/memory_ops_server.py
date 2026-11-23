"""``memory_ops`` MCP server: memory introspection over MCP (plan P11.T2).

The plan Phase 11 component ``hermes/mcp/memory_ops_server.py`` — a small MCP
server exposing the memory operations the v2 observability surface needs
(plan P11.T2 ``files_to_create``; tools: ``memory_stats``, ``memory_search
<query>``, ``memory_get <id>``) over the
:class:`~hermes.memory.store.MemoryStore`. Read-only by design: the write
paths stay with the pipeline (P3.T4) and the verify tool (P2.T3), so an MCP
client can inspect but never mutate the git-backed memory graph.

Layers (the P2.T3 verify-tool pattern)
--------------------------------------
- :func:`memory_stats` / :func:`memory_search` / :func:`memory_get` — the
  Python entry points over ``MemoryStore`` (``store.list`` / ``store.search``
  / ``store.get``; the BM25 search honors the optional ``context`` filter).
- :func:`register_memory_ops_server` — builds the MCP surface: a
  ``(method, params) -> result`` dispatcher answering ``tools/list`` (the
  three descriptors with their JSON schemas) and ``tools/call`` (runs the
  matching entry point), delegating every other method to an optional
  wrapped ``server_handler``. The default store is rooted at
  ``hermes_home() / "memory-repo"``.
- :func:`build_request_handler` — re-exported from
  :mod:`hermes.mcp.verify_tool`: adapts a dispatcher to the
  :func:`hermes.mcp.protocol.serve_stdio` handler contract (a full JSON-RPC
  request mapping in, result mapping out).
- :func:`serve` — runs the server over real stdio (newline-delimited
  JSON-RPC 2.0 via :mod:`hermes.mcp.protocol`); the zero-argument form is
  the subprocess bootstrap (``python -c "from hermes.mcp.memory_ops_server
  import serve; serve()"``), bootstrap-compatible like
  :mod:`hermes.mcp.verify_tool`.

Echo contract (issue #156143): every ``tools/call`` result carries
``arguments_received`` — the EXACT arguments dict the server received — so a
:class:`~hermes.mcp.client.McpStdioClient` caller can never lose arguments
silently on the stdio hop.

Errors: unknown entry ids for ``memory_get`` return
``{"found": False, "entry": None}`` (an introspection MISS is a result, not
a server fault — unlike the verify tool, which surfaces its unknown id as a
JSON-RPC error because a missed VERIFY is a caller bug). Invalid arguments
raise :class:`ValueError`, surfaced by the stdio layer as a ``-32603`` error
response.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Final

from hermes.mcp.protocol import RequestHandler, serve_stdio
from hermes.mcp.verify_tool import DispatchHandler, build_request_handler
from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore

__all__ = [
    "MEMORY_GET_TOOL_NAME",
    "MEMORY_SEARCH_TOOL_NAME",
    "MEMORY_STATS_TOOL_NAME",
    "build_request_handler",
    "memory_get",
    "memory_search",
    "memory_stats",
    "memory_tool_descriptors",
    "register_memory_ops_server",
    "serve",
]

_LOGGER = logging.getLogger(__name__)

#: Tool names (plan P11.T2: ``memory_stats``, ``memory_search <query>``,
#: ``memory_get <id>``).
MEMORY_STATS_TOOL_NAME: Final[str] = "memory_stats"
MEMORY_SEARCH_TOOL_NAME: Final[str] = "memory_search"
MEMORY_GET_TOOL_NAME: Final[str] = "memory_get"

#: Default ``top_k`` for ``memory_search`` (the store's own default).
_DEFAULT_TOP_K: Final[int] = 5

_MEMORY_STATS_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_MEMORY_SEARCH_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "BM25 query over entry text + tags.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "description": f"Maximum hits (default {_DEFAULT_TOP_K}).",
        },
        "context": {
            "type": "string",
            "description": (
                "Restrict to one memory context ('global' or a named "
                "context); omitted searches all contexts."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_MEMORY_GET_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "entry_id": {
            "type": "string",
            "description": "Id of the memory entry to fetch.",
        },
    },
    "required": ["entry_id"],
    "additionalProperties": False,
}


def memory_stats(store: MemoryStore) -> dict[str, Any]:
    """Entry counts by type and context plus the average trust score."""
    entries = store.list()
    by_type: dict[str, int] = {}
    by_context: dict[str, int] = {}
    trust_sum = 0.0
    for entry in entries:
        by_type[entry.type] = by_type.get(entry.type, 0) + 1
        by_context[entry.context] = by_context.get(entry.context, 0) + 1
        trust_sum += entry.trust_score
    average = trust_sum / len(entries) if entries else 0.0
    return {
        "total_entries": len(entries),
        "by_type": by_type,
        "by_context": by_context,
        "average_trust": round(average, 4),
    }


def memory_search(
    store: MemoryStore,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """BM25 search (plan P1.T4) projected to JSON-safe summary dicts."""
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")
    if context is not None and not isinstance(context, str):
        raise TypeError(f"context must be str or None, got {type(context).__name__}")
    return [_entry_summary(entry) for entry in store.search(query, top_k=top_k, context=context)]


def memory_get(store: MemoryStore, entry_id: str) -> dict[str, Any]:
    """Fetch one entry by id; a miss is ``{"found": False, "entry": None}``."""
    if not isinstance(entry_id, str):
        raise TypeError(f"entry_id must be str, got {type(entry_id).__name__}")
    entry = store.get(entry_id)
    if entry is None:
        return {"found": False, "entry": None}
    return {"found": True, "entry": _entry_detail(entry)}


def _entry_summary(entry: MemoryEntry) -> dict[str, Any]:
    """JSON-safe search-hit projection (id/type/context/trust/text)."""
    return {
        "id": entry.id,
        "type": entry.type,
        "context": entry.context,
        "trust_score": entry.trust_score,
        "text": entry.text,
    }


def _entry_detail(entry: MemoryEntry) -> dict[str, Any]:
    """JSON-safe full projection of one entry (trust card + body)."""
    last_verified = (
        entry.last_verified_at.isoformat() if entry.last_verified_at is not None else None
    )
    return {
        **_entry_summary(entry),
        "created_at": entry.created_at.isoformat(),
        "last_verified_at": last_verified,
        "tags": list(entry.tags),
        "verification_events": len(entry.verification_events),
        "attachments": list(entry.attachments),
    }


def memory_tool_descriptors() -> list[dict[str, Any]]:
    """MCP ``tools/list`` descriptors for the three memory-ops tools."""
    return [
        {
            "name": MEMORY_STATS_TOOL_NAME,
            "description": (
                "Memory store statistics: entry counts by type and context "
                "plus the average trust score."
            ),
            "inputSchema": _MEMORY_STATS_SCHEMA,
        },
        {
            "name": MEMORY_SEARCH_TOOL_NAME,
            "description": "BM25 search over the memory store (best matches first).",
            "inputSchema": _MEMORY_SEARCH_SCHEMA,
        },
        {
            "name": MEMORY_GET_TOOL_NAME,
            "description": "Fetch one memory entry by id (trust card + text).",
            "inputSchema": _MEMORY_GET_SCHEMA,
        },
    ]


def _optional_str(arguments: Mapping[str, Any], key: str) -> str | None:
    """Fetch an optional string argument (``None`` when absent)."""
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"argument {key!r} must be a string, got {type(value).__name__}")
    return value


def _optional_top_k(arguments: Mapping[str, Any]) -> int:
    """Fetch the optional ``top_k`` argument (default when absent)."""
    value = arguments.get("top_k")
    if value is None:
        return _DEFAULT_TOP_K
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"argument 'top_k' must be an integer, got {value!r}")
    return value


def _call_tool(store: MemoryStore, params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one ``tools/call`` (echo contract kept, issue #156143)."""
    name = params.get("name")
    arguments_raw = params.get("arguments")
    arguments: dict[str, Any] = dict(arguments_raw) if isinstance(arguments_raw, Mapping) else {}
    if name == MEMORY_STATS_TOOL_NAME:
        payload: dict[str, Any] = {"stats": memory_stats(store)}
    elif name == MEMORY_SEARCH_TOOL_NAME:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("memory_search requires a non-empty string argument 'query'")
        payload = {
            "query": query,
            "results": memory_search(
                store,
                query,
                top_k=_optional_top_k(arguments),
                context=_optional_str(arguments, "context"),
            ),
        }
    elif name == MEMORY_GET_TOOL_NAME:
        entry_id = arguments.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("memory_get requires a non-empty string argument 'entry_id'")
        payload = {"entry_id": entry_id, **memory_get(store, entry_id)}
    else:
        raise ValueError(
            f"unknown tool {name!r}; this server serves "
            f"{MEMORY_STATS_TOOL_NAME!r}, {MEMORY_SEARCH_TOOL_NAME!r}, "
            f"{MEMORY_GET_TOOL_NAME!r} only"
        )
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        **payload,
        # hermes MCP contract (issue #156143): echo the EXACT arguments back.
        "arguments_received": arguments,
    }


def register_memory_ops_server(
    server_handler: DispatchHandler | None = None,
    store: MemoryStore | None = None,
) -> DispatchHandler:
    """Build the ``tools/list`` + ``tools/call`` dispatcher pair (plan P11.T2).

    The returned dispatcher answers ``tools/list`` with the three memory-ops
    descriptors and ``tools/call`` by running the matching tool against
    ``store`` (default: a :class:`~hermes.memory.store.MemoryStore` rooted at
    ``hermes_home() / "memory-repo"``). Any other method delegates to
    ``server_handler`` when provided, else raises :class:`ValueError` (which
    the stdio layer turns into a ``-32603`` error response). ``store`` is an
    embedding/testing seam beyond the plan signature; production callers use
    the default.
    """
    resolved_store = store if store is not None else MemoryStore()

    def dispatch(method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if method == "tools/list":
            return {"tools": memory_tool_descriptors()}
        if method == "tools/call":
            return _call_tool(resolved_store, params)
        if server_handler is not None:
            return server_handler(method, params)
        raise ValueError(f"unknown MCP method for the memory-ops server: {method!r}")

    return dispatch


def serve(handler: RequestHandler | None = None) -> None:
    """Serve the memory-ops tools over stdio until EOF (plan P11.T2 MCP access).

    ``handler`` follows the :func:`hermes.mcp.protocol.serve_stdio` handler
    signature (full request mapping in, result mapping out); ``None`` builds
    the default memory-ops server (:func:`register_memory_ops_server`
    adapted through :func:`build_request_handler`). The zero-argument form
    is the subprocess bootstrap:
    ``python -c "from hermes.mcp.memory_ops_server import serve; serve()"``.
    """
    request_handler = (
        handler
        if handler is not None
        else build_request_handler(register_memory_ops_server())
    )
    serve_stdio(request_handler)
