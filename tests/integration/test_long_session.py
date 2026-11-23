"""P7.T4 acceptance: virtual session boundaries (plan Phase 7).

Covered: :class:`hermes.gateway.session_manager.VirtualSessionManager`
turns a long gateway chat into topic-scoped virtual sessions — on a topic
shift the active virtual session is closed (durable boundary record via
:class:`hermes.sessions.store.SessionStore` with topic/turn_ids/closed_at/
close_reason, plus ONE pipeline extraction run through
:meth:`hermes.memory.pipeline.MemoryPipeline.extract_session_scope` that
sees EXACTLY the closed session's turns — no leakage from the new session)
and a new virtual session starts under the detected topic; an explicit
``/session new`` (:meth:`VirtualSessionManager.force_new_session`) closes
the current session even without a topic shift and records the boundary
reason.

P7.T2 acceptance (appended): the memory tool wiring — ``store.on_update =
hot_reloader.invalidate`` (plan P7.T2) — makes a mid-session memory update
visible within 1 turn (no restart) while preserving the prefix cache
(hit rate >= 80%: only the updated entry's segment is invalidated).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from hermes.gateway.session_manager import (
    SESSION_NEW_REASON,
    TOPIC_SHIFT_REASON,
    ShiftDecision,
    VirtualSessionManager,
)
from hermes.gateway.topic_detector import ShiftResult
from hermes.memory.draft import DraftFact
from hermes.memory.hot_reload import MemoryHotReloader
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore
from hermes.sessions.store import SessionStore

_TEXT_A1 = "We deploy the acme service via GitHub Actions only"
_TEXT_A2 = "Acme deployments are pinned to the main branch"
_TEXT_B1 = "Now let's plan the kubernetes rollout instead"
_TEXT_B2 = "The kubernetes rollout needs three staged waves"
_TEXT_B3 = "Rollout wave three is the canary analysis"

_TOPIC_A = "acme deploy pipeline"
_TOPIC_B = "kubernetes rollout"

_PARENT = "chat-1"


class _ScriptedDetector:
    """Fake TopicDetector: scripted ShiftResults in call order."""

    def __init__(self, results: list[ShiftResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[list[str], str | None]] = []

    def detect_shift(
        self, recent_messages: Sequence[str], prev_topic: str | None = None
    ) -> ShiftResult:
        self.calls.append((list(recent_messages), prev_topic))
        if self._results:
            return self._results.pop(0)
        return ShiftResult(
            shifted=False, topic="quiet continuation", confidence=1.0, latency_ms=0.0
        )


class _RecordingExtractor:
    """Fake ExtractorAgent: records what each scoped extraction SAW."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        turn_text: str,
        context_hint: str = "global",
        context_turns: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[DraftFact]:
        turn_ids: list[str] = []
        texts: list[str] = []
        for item in context_turns or []:
            if isinstance(item, Mapping):
                turn_ids.append(str(item.get("turn_id", "")))
                texts.append(str(item.get("text", "")))
        self.calls.append(
            {
                "anchor": turn_text,
                "context_hint": context_hint,
                "turn_ids": turn_ids,
                "texts": texts,
            }
        )
        return [
            DraftFact(
                draft_id=f"scope-draft-{len(self.calls)}",
                text=f"fact from: {turn_text}",
                type="Project",
                proposed_source="user-explicit",
                confidence=0.9,
                context_hint=context_hint,
                source_turn_ids=list(turn_ids),
            )
        ]


def _make_pipeline(
    store: MemoryStore, extractor: _RecordingExtractor
) -> MemoryPipeline:
    """Real pipeline over the recording extractor and an always-add reviewer."""
    reviewer = ReviewerAgent(
        model=lambda _prompt: json.dumps({
            "action": "add",
            "target_id": None,
            "rationale": "new fact from the virtual session scope",
            "merged_text": None,
        }),
        search_fn=lambda _query: [],
    )
    return MemoryPipeline(extractor, reviewer, store)  # type: ignore[arg-type]


def _make_manager(
    tmp_path: Path, detector: _ScriptedDetector, extractor: _RecordingExtractor
) -> tuple[VirtualSessionManager, MemoryStore, SessionStore]:
    """Manager over a real tmp store/pipeline and a real tmp session store."""
    memory_store = MemoryStore(root=tmp_path / "memory-repo")
    pipeline = _make_pipeline(memory_store, extractor)
    session_store = SessionStore(root=tmp_path / "sessions")
    return (
        VirtualSessionManager(detector, pipeline, session_store),  # type: ignore[arg-type]
        memory_store,
        session_store,
    )


@pytest.mark.integration
def test_virtual_session_extraction_isolated(tmp_path: Path) -> None:
    """Topic shift -> boundary; scoped extraction sees ONLY the closed turns.

    2 topic-A turns + 3 topic-B turns, every_n_messages=3: the first check
    (after 3 further messages) shifts, closing virtual session v001 over
    turns t1-t4; extraction derives entries only from those turns, and the
    new session (v002, topic B) holds t5.
    """
    detector = _ScriptedDetector([
        ShiftResult(shifted=False, topic=_TOPIC_A, confidence=0.9, latency_ms=0.0),
        ShiftResult(shifted=True, topic=_TOPIC_B, confidence=0.85, latency_ms=0.0),
    ])
    extractor = _RecordingExtractor()
    manager, store, session_store = _make_manager(tmp_path, detector, extractor)

    decisions = [
        manager.feed(_PARENT, "t1", _TEXT_A1, every_n_messages=3),
        manager.feed(_PARENT, "t2", _TEXT_A2, every_n_messages=3),
        manager.feed(_PARENT, "t3", _TEXT_B1, every_n_messages=3),
        manager.feed(_PARENT, "t4", _TEXT_B2, every_n_messages=3),
        manager.feed(_PARENT, "t5", _TEXT_B3, every_n_messages=3),
    ]

    # session start established topic A; no shift before the first check
    assert decisions[0] == ShiftDecision(shifted=False, topic=_TOPIC_A, confidence=0.9)
    assert all(not decision.shifted for decision in decisions[:3])
    assert [decision.topic for decision in decisions[:3]] == [_TOPIC_A, _TOPIC_A, _TOPIC_A]

    # the topic shift at the first check created the boundary
    boundary = decisions[3]
    assert boundary.shifted is True
    assert boundary.topic == _TOPIC_B
    assert boundary.closed_session is not None
    closed = boundary.closed_session
    assert closed.session_id == "chat-1-v001"
    assert closed.topic == _TOPIC_A
    assert closed.turn_ids == ["t1", "t2", "t3", "t4"]
    assert closed.turns() == [
        ("t1", _TEXT_A1),
        ("t2", _TEXT_A2),
        ("t3", _TEXT_B1),
        ("t4", _TEXT_B2),
    ]

    # the new virtual session carries the detected topic and only t5
    active = manager.active_session(_PARENT)
    assert active.session_id == "chat-1-v002"
    assert active.topic == _TOPIC_B
    assert active.turn_ids == ["t5"]

    # the detector compared against the established topics
    assert detector.calls[0][1] is None  # session start: nothing to compare
    assert detector.calls[1][1] == _TOPIC_A

    # clean memory boundary: exactly ONE scoped extraction over v001's turns
    assert len(extractor.calls) == 1
    scope = extractor.calls[0]
    assert scope["context_hint"] == "global"
    assert scope["turn_ids"] == ["t1", "t2", "t3", "t4"]
    assert scope["texts"] == [_TEXT_A1, _TEXT_A2, _TEXT_B1, _TEXT_B2]
    assert _TEXT_B3 not in scope["texts"]  # the new session's turn never leaked
    assert scope["anchor"] == _TEXT_B2

    # the store holds only facts derived from the closed session's turns
    entries = store.list(context="global")
    assert [entry.text for entry in entries] == [f"fact from: {_TEXT_B2}"]
    assert entries[0].verification_events[0].evidence.startswith("extractor draft ")

    # durable boundary record: topic, turn_ids, closed_at, close_reason
    state = session_store.load_session("chat-1-v001")
    assert state is not None
    assert state["virtual"] is True
    assert state["topic"] == _TOPIC_A
    assert state["turn_ids"] == ["t1", "t2", "t3", "t4"]
    assert state["close_reason"] == TOPIC_SHIFT_REASON
    assert state["started_at"] and state["closed_at"]
    assert state["closed_at"] >= state["started_at"]
    # the still-active v002 has no boundary record yet
    assert session_store.list_sessions() == ["chat-1-v001"]


@pytest.mark.integration
def test_explicit_session_new_command(tmp_path: Path) -> None:
    """/session new closes the current session even without a topic shift."""
    detector = _ScriptedDetector([
        ShiftResult(shifted=False, topic=_TOPIC_A, confidence=0.9, latency_ms=0.0),
        ShiftResult(shifted=False, topic="release planning", confidence=0.9, latency_ms=0.0),
    ])
    extractor = _RecordingExtractor()
    manager, store, session_store = _make_manager(tmp_path, detector, extractor)

    # two turns, every_n_messages=5: the cadence never fires a shift check
    first = manager.feed(_PARENT, "t1", _TEXT_A1, every_n_messages=5)
    manager.feed(_PARENT, "t2", _TEXT_A2, every_n_messages=5)
    assert not first.shifted
    assert len(detector.calls) == 1  # establish only

    closed = manager.force_new_session(_PARENT)  # the /session new contract

    assert closed is not None
    assert closed.session_id == "chat-1-v001"
    assert closed.turn_ids == ["t1", "t2"]
    # a fresh, empty virtual session started (topic established on next feed)
    active = manager.active_session(_PARENT)
    assert active.session_id == "chat-1-v002"
    assert active.turn_ids == []
    assert active.topic == ""

    # the boundary ran the scoped extraction over exactly the closed turns
    assert len(extractor.calls) == 1
    assert extractor.calls[0]["turn_ids"] == ["t1", "t2"]
    assert extractor.calls[0]["anchor"] == _TEXT_A2
    assert [entry.text for entry in store.list(context="global")] == [
        f"fact from: {_TEXT_A2}"
    ]

    # boundary reason recorded durably
    state = session_store.load_session("chat-1-v001")
    assert state is not None
    assert state["close_reason"] == SESSION_NEW_REASON
    assert state["turn_ids"] == ["t1", "t2"]
    assert state["topic"] == _TOPIC_A
    assert state["closed_at"]

    # nothing to close: a second /session new is a no-op (no boundary, no extraction)
    assert manager.force_new_session(_PARENT, reason="user command /session new") is None
    assert session_store.list_sessions() == ["chat-1-v001"]
    assert len(extractor.calls) == 1

    # the fresh session establishes its topic on the next feed
    third = manager.feed(_PARENT, "t3", "let's plan the next release", every_n_messages=5)
    assert third.shifted is False
    assert third.topic == "release planning"
    assert manager.active_session(_PARENT).turn_ids == ["t3"]


# ── P7.T2: hot-reload wired to memory tool updates (plan Phase 7) ──────────

#: 10-entry session: with 1 update, 9/10 segments preserved = 90% >= 80%.
_P7_ENTRY_COUNT = 10

_P7_ORIGINAL = "the release train ships every tuesday"
_P7_UPDATED = "the release train now ships every wednesday"


def _p7_wired_session(tmp_path: Path) -> tuple[MemoryStore, MemoryHotReloader, str]:
    """Store + reloader with the P7.T2 wiring; returns ``(store, reloader, target_id)``.

    The reloader renders ONE segment per entry (``segments={}`` drops the
    default fixed segments so the hit-rate math is exactly entries vs
    invalidated). The memory-tool wiring under test is
    ``store.on_update = reloader.invalidate`` (plan P7.T2): every committed
    update invalidates exactly that entry's segment, automatically.
    """
    store = MemoryStore(root=tmp_path / "memory-repo")
    for index in range(_P7_ENTRY_COUNT - 1):
        store.add(f"long session fact {index:02d} about topic-{index % 3}", type="Reference")
    target = store.add(_P7_ORIGINAL, type="Project")
    reloader = MemoryHotReloader(store, segments={})
    store.on_update = reloader.invalidate  # the P7.T2 memory-tool wiring
    return store, reloader, target.id


@pytest.mark.integration
def test_memory_update_visible_mid_session(tmp_path: Path) -> None:
    """A mid-session store.update is visible in the NEXT render (within 1 turn).

    Turn 1 renders the prompt with the original text; the memory tool then
    commits ``store.update(...)`` — the on_update hook fires as part of the
    update itself, invalidating exactly that segment (no restart, no manual
    invalidate call) — and the turn-2 render shows the fresh text while every
    other segment stays byte-identical.
    """
    store, reloader, target_id = _p7_wired_session(tmp_path)

    # turn 1: reference render — the original text is visible
    turn_1 = reloader.render_prompt()
    assert turn_1[target_id] == f'<mem-segment id="{target_id}">{_P7_ORIGINAL}</mem-segment>'
    assert not reloader.pending_invalidations()

    # mid-session memory tool update: the store drives invalidation itself
    store.update(target_id, text=_P7_UPDATED)

    # the hook fired during update(): exactly the touched segment is pending
    assert reloader.pending_invalidations() == [target_id]

    # the SAME turn boundary: a fresh reload reads the new content from the store
    assert reloader.reload_segment(target_id) == (
        f'<mem-segment id="{target_id}">{_P7_UPDATED}</mem-segment>'
    )

    # turn 2 renders the prompt: the fresh text is already in (no restart)
    turn_2 = reloader.render_prompt(invalidated=set(reloader.pending_invalidations()))
    assert turn_2[target_id] == f'<mem-segment id="{target_id}">{_P7_UPDATED}</mem-segment>'
    assert _P7_ORIGINAL not in turn_2[target_id]

    # prefix cache preserved: every untouched segment is byte-identical
    for segment_id, reference_bytes in turn_1.items():
        if segment_id == target_id:
            continue
        assert turn_2[segment_id] == reference_bytes

    # the harness consumed the invalidations after re-injecting the segment
    reloader.clear_invalidations()
    assert not reloader.pending_invalidations()


@pytest.mark.integration
def test_prefix_cache_hit_rate_above_80_percent(tmp_path: Path) -> None:
    """One update among 10 entries preserves 9/10 segments = 90% >= 80%.

    The hit rate is computed from byte-identical segments between the
    pre-update and post-update renders — with the segment-level
    invalidation contract (plan Phase 7 trade-off note) that equals the
    segments-total vs invalidated ratio: 10 total, 1 invalidated, 9 served
    from the unchanged prefix bytes.
    """
    _store, reloader, target_id = _p7_wired_session(tmp_path)

    reference = reloader.render_prompt()
    total_segments = len(reference)
    assert total_segments == _P7_ENTRY_COUNT

    # exactly one memory tool update mid-session (the hook invalidates it)
    _store.update(target_id, text=_P7_UPDATED)
    invalidated = set(reloader.pending_invalidations())
    assert invalidated == {target_id}

    rendered = reloader.render_prompt(invalidated=invalidated)

    preserved = sum(
        1
        for segment_id, reference_bytes in reference.items()
        if segment_id not in invalidated and rendered[segment_id] == reference_bytes
    )
    hit_rate = preserved / total_segments
    assert hit_rate == pytest.approx(9 / 10)
    assert hit_rate >= 0.80

    # the invalidated segment is the ONLY one that changed bytes
    assert rendered[target_id] != reference[target_id]
    assert preserved == total_segments - len(invalidated)

