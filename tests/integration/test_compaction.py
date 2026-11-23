"""P7.T5 acceptance: compaction in the full session-close flow (plan Phase 7).

Covered: closing a virtual session with a :class:`SessionCompactor` wired
(topic shift close via :class:`hermes.gateway.session_manager.
VirtualSessionManager`) compacts the closed session's turns into a
Reference-type summary entry carrying the trace-range provenance built from
the session's recorded turn ids, and the boundary extraction — which runs for
the SAME session id AFTER the compaction — receives the summary FIRST in its
extractor context with the raw turns marked ``[secondary]`` (the DSH pattern:
extraction uses compaction summaries as primary input, raw turns secondary).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from hermes.gateway.session_manager import VirtualSessionManager
from hermes.gateway.topic_detector import ShiftResult
from hermes.memory.compaction import (
    COMPACTION_TAG,
    TRACE_RANGE_TAG_PREFIX,
    SessionCompactor,
    find_compaction_summary,
)
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore
from hermes.sessions.store import SessionStore

_PARENT = "chat-1"

_TOPIC_A = "acme deploy pipeline"
_TOPIC_B = "kubernetes rollout"

_TEXT_A1 = "We deploy the acme service via GitHub Actions only"
_TEXT_A2 = "Acme deployments are pinned to the main branch"
_TEXT_B1 = "Now let's plan the kubernetes rollout instead"
_TEXT_B2 = "The kubernetes rollout needs three staged waves"
_TEXT_B3 = "Rollout wave three is the canary analysis"

_CLOSED_SESSION = "chat-1-v001"
_TRACE_RANGE = "t1..t4"


class _ScriptedDetector:
    """Fake TopicDetector: establish topic A, then shift to topic B."""

    def __init__(self) -> None:
        self._results = [
            ShiftResult(shifted=False, topic=_TOPIC_A, confidence=0.9, latency_ms=0.0),
            ShiftResult(shifted=True, topic=_TOPIC_B, confidence=0.85, latency_ms=0.0),
        ]

    def detect_shift(
        self, recent_messages: Sequence[str], prev_topic: str | None = None
    ) -> ShiftResult:
        if self._results:
            return self._results.pop(0)
        return ShiftResult(
            shifted=False, topic="quiet continuation", confidence=1.0, latency_ms=0.0
        )


class _PromptRecordingExtractorModel:
    """Fake extractor model: records every prompt, proposes nothing."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "[]"  # no drafts: the boundary extraction stays a no-op write


@pytest.mark.integration
def test_extraction_uses_summary_first(tmp_path: Path) -> None:
    """Session close compacts, then the boundary extraction leads with the summary.

    4 topic-A turns + 1 topic-B turn (shift at the first check, every_n=3)
    closes virtual session chat-1-v001 over turns t1..t4: the store gains the
    compaction summary entry (trace range from the recorded turn ids) and the
    boundary extraction's prompt shows the summary as the FIRST context turn,
    followed by the raw turns each marked ``[secondary]``.
    """
    memory_store = MemoryStore(root=tmp_path / "memory-repo")
    extractor_model = _PromptRecordingExtractorModel()
    pipeline = MemoryPipeline(
        ExtractorAgent(model=extractor_model),
        ReviewerAgent(
            model=lambda _prompt: json.dumps({
                "action": "add",
                "target_id": None,
                "rationale": "add",
                "merged_text": None,
            }),
            search_fn=lambda _query: [],
        ),
        memory_store,
    )
    session_store = SessionStore(root=tmp_path / "sessions")
    manager = VirtualSessionManager(
        _ScriptedDetector(),
        pipeline,
        session_store,
        SessionCompactor(memory_store),  # plan P7.T5 additive wiring
    )

    feeds = [
        manager.feed(_PARENT, "t1", _TEXT_A1, every_n_messages=3),
        manager.feed(_PARENT, "t2", _TEXT_A2, every_n_messages=3),
        manager.feed(_PARENT, "t3", _TEXT_B1, every_n_messages=3),
        manager.feed(_PARENT, "t4", _TEXT_B2, every_n_messages=3),
        manager.feed(_PARENT, "t5", _TEXT_B3, every_n_messages=3),
    ]

    shift = feeds[3]
    assert shift.shifted is True
    assert shift.closed_session is not None
    assert shift.closed_session.turn_ids == ["t1", "t2", "t3", "t4"]

    # the compaction summary was created on the virtual session close
    summary = find_compaction_summary(memory_store, _CLOSED_SESSION)
    assert summary is not None
    assert summary.type == "Reference"
    assert summary.text.startswith(f"Compaction summary for session {_CLOSED_SESSION}")
    assert COMPACTION_TAG in summary.tags
    assert f"session:{_CLOSED_SESSION}" in summary.tags
    assert f"{TRACE_RANGE_TAG_PREFIX}{_TRACE_RANGE}" in summary.tags
    assert (
        summary.verification_events[0].evidence
        == f"source points to trace range {_TRACE_RANGE} — memory points, "
        "doesn't mirror"
    )

    # the boundary extraction ran exactly once, AFTER the compaction, and its
    # extractor prompt leads with the summary; raw turns are marked secondary
    assert len(extractor_model.prompts) == 1
    prompt = extractor_model.prompts[0]

    summary_pos = prompt.find(f"Compaction summary for session {_CLOSED_SESSION}")
    assert summary_pos != -1
    first_raw_pos = prompt.find(f"[secondary] {_TEXT_A1}")
    assert first_raw_pos != -1
    assert summary_pos < first_raw_pos  # SUMMARY FIRST (plan P7.T5)
    assert prompt.find(f"[compaction-{_CLOSED_SESSION}]") != -1

    # every raw turn of the closed session appears once, marked secondary,
    # in the extractor's context (raw turns SECONDARY, DSH pattern)
    for turn_text in (_TEXT_A1, _TEXT_A2, _TEXT_B1, _TEXT_B2):
        assert f"[secondary] {turn_text}" in prompt
    assert prompt.count("[secondary]") == 4
    assert f"[secondary] {_TEXT_B3}" not in prompt  # the new session never leaked

    # the closed session's boundary record keeps the turn ids the range points to
    state = session_store.load_session(_CLOSED_SESSION)
    assert state is not None
    assert state["turn_ids"] == ["t1", "t2", "t3", "t4"]
