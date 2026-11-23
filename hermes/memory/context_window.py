"""ContextWindow — recent-turn context for the extractor (plan P3.T6).

The extractor (P3.T2) sees only the current turn; P3.T6 adds a multi-turn
context window ("last 3-5 turns") so it can resolve references ("yes, use
that approach"), detect intent shifts and avoid re-extracting confirmed
facts. :class:`ContextWindow.get_recent_turns` assembles that window as a
list of :class:`Turn` records, ordered oldest-first, filtered to one session.

Turn sources — IMPORTANT TraceStore wiring note (P11.T2 integration point)
-----------------------------------------------------------------------
``store`` (a :class:`~hermes.traces.store.TraceStore`) is the PREFERRED
source: one :class:`~hermes.traces.schema.Trace` is one turn, so a trace
maps onto a :class:`Turn` (``turn_id`` + ``user_prompt``). However the
current TraceStore API is full-text search over prompt/payload — it has NO
session-indexed query, so the trace path is a best-effort recall (text
search for the session id, client-side session filter, client-side
chronological sort). In production this rarely recalls anything until
P11.T2 adds a session query to the TraceStore; therefore a
``fallback_provider`` callable (e.g. the in-memory session history — the
plan's r7 note "If Trace Store not yet available, can use in-memory session
history as fallback initially") is consulted whenever the trace path yields
nothing. Tests inject fakes for both.

Graceful degradation is part of the contract: no sources, an unknown
session, ``n < 1`` or a raising TraceStore all yield ``[]`` (with a logged
warning for the raising store) — context assembly must never break the
Stop-hook pipeline that consumes it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from hermes.memory.types import to_utc
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

logger = logging.getLogger(__name__)

__all__ = ["ContextWindow", "Turn"]

#: How many traces the best-effort trace-path recall asks the store for.
#: The store ranks by bm25 (FTS) or timestamp (LIKE); the client-side sort
#: below re-orders whatever comes back, so a generous recall limit only
#: costs a little memory.
_TRACE_RECALL_LIMIT: int = 100


@dataclass
class Turn:
    """One conversation turn as consumed by the multi-turn context (P3.T6).

    ``index`` is the turn's position within its session (0-based, provider-
    supplied). Trace-derived turns carry ``index=0`` — the current Trace
    schema has no turn ordinal; the P11.T2 session query is expected to
    supply one (see the module docstring).
    """

    turn_id: str
    session_id: str
    index: int
    text: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be a non-empty string")
        if not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        self.timestamp = to_utc(self.timestamp, "timestamp")


class ContextWindow:
    """Assembles the last-N-turn context block for one session (plan P3.T6).

    ``store`` is the preferred TraceStore source (P11.T2 integration point —
    see the module docstring); ``fallback_provider(session_id) -> list[Turn]``
    covers the gap while the TraceStore lacks a session query (and any trace-
    path miss). Both are optional; with neither, :meth:`get_recent_turns`
    returns ``[]``.
    """

    def __init__(
        self,
        store: TraceStore | None = None,
        fallback_provider: Callable[[str], list[Turn]] | None = None,
    ) -> None:
        self._store = store
        self._fallback_provider = fallback_provider

    # ── public API ────────────────────────────────────────────────────────
    def get_recent_turns(self, session_id: str, n: int = 3) -> list[Turn]:
        """Return the ``n`` most recent turns of ``session_id``, oldest first.

        Prefers the TraceStore trace path (best-effort recall — see the
        module docstring); when it yields nothing, consults the
        ``fallback_provider``. Returns ``[]`` gracefully for an empty/blank
        ``session_id``, ``n < 1``, an unknown session, or no usable source.
        Cross-session records are filtered out on both paths.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            return []
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            logger.warning("context_window: n must be a positive int, got %r; returning []", n)
            return []
        turns = self._turns_from_store(session_id)
        if not turns:
            turns = self._turns_from_fallback(session_id)
        return self._last_n(turns, n)

    # ── turn sources ──────────────────────────────────────────────────────
    def _turns_from_store(self, session_id: str) -> list[Turn]:
        """Best-effort trace-path recall (see module docstring for the caveat)."""
        if self._store is None:
            return []
        try:
            traces = self._store.search(session_id, limit=_TRACE_RECALL_LIMIT)
        except Exception:
            logger.warning(
                "context_window: TraceStore search failed for session %r; "
                "falling back",
                session_id,
                exc_info=True,
            )
            return []
        turns: list[Turn] = []
        for trace in traces:
            if not isinstance(trace, Trace) or trace.session_id != session_id:
                continue  # client-side session filter (recall is text-based)
            text = trace.user_prompt.strip()
            turn_id = trace.turn_id or trace.trace_id
            if not text or not turn_id:
                continue  # an unusable trace is skipped, never crashes the window
            turns.append(
                Turn(
                    turn_id=turn_id,
                    session_id=trace.session_id,
                    index=0,  # trace schema has no turn ordinal (P11.T2 supplies it)
                    text=text,
                    timestamp=datetime.fromtimestamp(float(trace.timestamp), tz=UTC),
                )
            )
        return turns

    def _turns_from_fallback(self, session_id: str) -> list[Turn]:
        """Consult the injected fallback provider (session-history design)."""
        if self._fallback_provider is None:
            return []
        turns = self._fallback_provider(session_id)
        return [turn for turn in turns if turn.session_id == session_id]

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _last_n(turns: list[Turn], n: int) -> list[Turn]:
        """Chronologically sort (oldest first) and keep the ``n`` most recent."""
        ordered = sorted(turns, key=lambda turn: (turn.timestamp, turn.index, turn.turn_id))
        return ordered[-n:]
