"""P3.T6 acceptance: ContextWindow assembles the recent-turn context window.

Covers the two turn sources — the injected ``fallback_provider`` (the plan's
in-memory session-history fallback) and the preferred TraceStore trace path
(fake store plus a REAL :class:`~hermes.traces.store.TraceStore` to pin the
graceful-degradation wiring that P11.T2 will replace with a session query) —
plus the last-N ordering, session filtering and the graceful empty cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.memory.context_window import ContextWindow, Turn
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _turn(turn_id: str, index: int, session: str = "s1", minutes: float = 0) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session,
        index=index,
        text=f"body of {turn_id}",
        timestamp=_T0 + timedelta(minutes=minutes),
    )


class _FakeTraceStore:
    """Fake TraceStore: records queries, replays a fixed trace list."""

    def __init__(self, traces: list[Trace]) -> None:
        self._traces = list(traces)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[Trace]:
        self.queries.append((query, limit))
        return list(self._traces)[:limit]


def _trace(turn_id: str, session: str, minutes: float, prompt: str = "") -> Trace:
    return Trace(
        turn_id=turn_id,
        session_id=session,
        timestamp=(_T0 + timedelta(minutes=minutes)).timestamp(),
        user_prompt=prompt or f"prompt of {turn_id}",
    )


@pytest.mark.unit
def test_returns_last_n_turns() -> None:
    """The last n turns come back oldest-first; the trace path is preferred."""
    provider_turns = [_turn(f"t{i}", index=i, minutes=float(i)) for i in range(5)]
    window = ContextWindow(
        fallback_provider=lambda session_id: list(provider_turns)
    )

    # last 3 of 5, ordered oldest-first
    turns = window.get_recent_turns("s1", n=3)
    assert [turn.turn_id for turn in turns] == ["t2", "t3", "t4"]
    assert [turn.index for turn in turns] == [2, 3, 4]

    # n larger than the session -> all turns, still oldest-first
    assert [turn.turn_id for turn in window.get_recent_turns("s1", n=10)] == [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
    ]

    # unsorted provider input is re-ordered chronologically
    shuffled = ContextWindow(
        fallback_provider=lambda _sid: [provider_turns[4], provider_turns[0], provider_turns[2]]
    )
    assert [turn.turn_id for turn in shuffled.get_recent_turns("s1", n=2)] == ["t2", "t4"]

    # graceful n handling
    assert window.get_recent_turns("s1", n=0) == []
    assert window.get_recent_turns("s1", n=-2) == []

    # the TraceStore path is preferred when it yields turns: the fallback is
    # not consulted and the traces map onto Turns (timestamp -> chronology).
    store = _FakeTraceStore([_trace("tr1", "s1", 5.0), _trace("tr2", "s1", 15.0)])
    fallback_marks = ContextWindow(
        store=store,  # type: ignore[arg-type]
        fallback_provider=lambda _sid: [_turn("FALLBACK", index=0)],
    )
    turns = fallback_marks.get_recent_turns("s1", n=5)
    assert [turn.turn_id for turn in turns] == ["tr1", "tr2"]
    assert [turn.text for turn in turns] == ["prompt of tr1", "prompt of tr2"]
    assert all(turn.session_id == "s1" for turn in turns)
    assert store.queries == [("s1", 100)]  # best-effort recall against the store

    # an unusable trace (no prompt) is skipped; a trace without turn_id falls
    # back to its trace_id for the referenceable turn id.
    messy = _FakeTraceStore([
        _trace("", "s1", 1.0, prompt=""),  # no prompt, no turn_id -> skipped
        _trace("", "s1", 2.0, prompt="has prompt"),  # trace_id missing too -> skipped
        Trace(
            trace_id="tid-9",
            turn_id="",
            session_id="s1",
            timestamp=(_T0 + timedelta(minutes=3.0)).timestamp(),
            user_prompt="usable prompt",
        ),
    ])
    window = ContextWindow(store=messy)  # type: ignore[arg-type]
    turns = window.get_recent_turns("s1", n=3)
    assert [turn.turn_id for turn in turns] == ["tid-9"]


@pytest.mark.unit
def test_filters_by_session(tmp_path: Path) -> None:
    """Only the requested session's turns survive; unknown sessions yield []."""
    provider_turns = [
        _turn("a0", index=0, session="s1", minutes=0.0),
        _turn("b0", index=0, session="s2", minutes=1.0),
        _turn("a1", index=1, session="s1", minutes=2.0),
        _turn("b1", index=1, session="s2", minutes=3.0),
        _turn("a2", index=2, session="s1", minutes=4.0),
    ]
    window = ContextWindow(fallback_provider=lambda _sid: list(provider_turns))

    turns = window.get_recent_turns("s1", n=5)
    assert [turn.turn_id for turn in turns] == ["a0", "a1", "a2"]
    assert all(turn.session_id == "s1" for turn in turns)

    assert [turn.turn_id for turn in window.get_recent_turns("s2", n=1)] == ["b1"]
    assert window.get_recent_turns("unknown-session") == []
    assert window.get_recent_turns("") == []
    assert window.get_recent_turns("   ") == []

    # the trace path filters cross-session traces client-side too (the
    # text-based recall can return other sessions' traces).
    store = _FakeTraceStore([
        _trace("tr-a", "s1", 1.0),
        _trace("tr-b", "s2", 2.0),
        _trace("tr-a2", "s1", 3.0),
    ])
    window = ContextWindow(store=store)  # type: ignore[arg-type]
    assert [turn.turn_id for turn in window.get_recent_turns("s1", n=5)] == ["tr-a", "tr-a2"]

    # graceful degradation against a REAL TraceStore (per-test temp root):
    # its text search finds no traces for the session id (the prompt text
    # never mentions it), so the fallback provider serves the window. This
    # pins the P11.T2 integration point: once TraceStore grows a session
    # query, this path switches to the store without touching the caller.
    real_store = TraceStore(root=tmp_path / "traces")
    real_store.append(_trace("kept-out", "s-other", 1.0, prompt="unrelated prompt"))
    provider_turns.append(_turn("late", index=9, session="s1", minutes=5.0))
    window = ContextWindow(
        store=real_store, fallback_provider=lambda _sid: list(provider_turns)
    )
    assert [turn.turn_id for turn in window.get_recent_turns("s1", n=5)] == [
        "a0",
        "a1",
        "a2",
        "late",
    ]
    real_store.close()
