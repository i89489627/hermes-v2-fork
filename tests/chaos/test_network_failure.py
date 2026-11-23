"""P20.T1 chaos: network failure to the main-model provider (plan P20.T1/5.5).

While the provider is unreachable, memory work must neither crash nor lose
data: the CBDCGate falls back to trust-score-only gating (no model), the
pipeline degrades extraction to "skip, retry next turn" (plan 5.5), and the
agent's memory ops go to the deferred-op queue (the conftest ``deferred_ops``
fixture — the plan Section 5.5 deferred queue). When the provider recovers,
the queue flushes through the REAL store in order and the gate returns to
model-driven decisions.

- ``test_provider_unreachable_defers_ops`` — outage phase: gate falls back,
  extraction degrades, the ops are deferred (never lost), the store stays
  empty (nothing applied, nothing partial).
- ``test_provider_recovery_flushes_queue`` — recovery phase: the gate is
  model-driven again (``fallback_used=False``) and the flush applies every
  deferred op through the store IN ORDER, each durably committed.
"""

from __future__ import annotations

import json

import pytest

from hermes.memory.extractor import ExtractorAgent
from hermes.memory.gating import CBDCGate
from hermes.memory.entry import MemoryEntry
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.retrieval import RetrievedEntry
from hermes.memory.reviewer import ReviewerAgent

from datetime import UTC, datetime

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)

#: Memory ops the (offline) agent wants to store during the outage.
OFFLINE_OPS = [
    {"text": "offline memory op one", "type": "Feedback"},
    {"text": "offline memory op two", "type": "Feedback"},
    {"text": "offline memory op three", "type": "Feedback"},
]

#: The recovered provider's decision list (covers every candidate id).
_GATE_DECISIONS = json.dumps(
    [
        {"candidate_id": "c1", "decision": "use", "reason": "relevant to the turn"},
        {"candidate_id": "c2", "decision": "ignore", "reason": "not relevant"},
        {"candidate_id": "c3", "decision": "verify", "reason": "plausible but stale"},
    ]
)


def _candidate(entry_id: str, text: str, trust: float, score: float) -> RetrievedEntry:
    """One gated candidate with an explicit trust score and BM25 ranking."""
    entry = MemoryEntry(
        id=entry_id, type="User", text=text, created_at=_NOW, trust_score=trust
    )
    return RetrievedEntry(entry=entry, score=score)


def _candidates() -> list[RetrievedEntry]:
    """Top-3 gate candidates spanning the trust-only fallback threshold."""
    return [
        _candidate("c1", "network chaos high-trust fact", trust=0.95, score=5.0),
        _candidate("c2", "network chaos low-trust fact", trust=0.40, score=4.0),
        _candidate("c3", "network chaos mid-trust fact", trust=0.80, score=1.0),
    ]


@pytest.mark.chaos
def test_provider_unreachable_defers_ops(chaos_store, failing_model, deferred_ops) -> None:
    """Outage: gate falls back, extraction degrades, ops defer — nothing lost."""
    store = chaos_store
    down = failing_model()

    # Gating during the outage: trust-only fallback, no exception escapes.
    gate = CBDCGate(gate_model=down)
    result = gate.gate(_candidates(), context="global", turn_text="offline turn")
    assert result.fallback_used is True
    assert down.calls == 1  # the one failed attempt; the gate never retries
    # trust-only fallback: both candidates above the 0.7 bar injected as "use"
    assert [gated.candidate_id for gated in result.injected] == ["c1", "c3"]
    assert all(gated.decision == "use" for gated in result.injected)

    # Extraction during the outage: skipped (retry next turn), no crash.
    pipeline = MemoryPipeline(
        extractor=ExtractorAgent(model=down),
        reviewer=ReviewerAgent(model=down, search_fn=lambda query: []),
        store=store,
    )
    outcome = pipeline.run_after_turn(
        "offline-turn-1", "user shares a fact while the provider is unreachable"
    )
    assert outcome.drafts == []
    assert outcome.applied == []

    # The agent defers its memory ops to the offline queue instead of
    # dropping them; order is preserved.
    for op in OFFLINE_OPS:
        deferred_ops.defer(op)
    assert deferred_ops.pending == 3
    assert [op["text"] for op in deferred_ops.ops] == [op["text"] for op in OFFLINE_OPS]

    # Nothing was applied while offline — and nothing was lost either.
    assert store.list() == []
    assert store.git.is_clean()


@pytest.mark.chaos
def test_provider_recovery_flushes_queue(
    chaos_store, failing_model, scripted_model, deferred_ops
) -> None:
    """Recovery: gate is model-driven again; the queue flushes in order."""
    store = chaos_store
    down = failing_model()
    gate_down = CBDCGate(gate_model=down)
    assert gate_down.gate(_candidates()).fallback_used is True
    for op in OFFLINE_OPS:
        deferred_ops.defer(op)
    assert deferred_ops.pending == 3

    # The provider recovers: the gate's next call is model-driven.
    recovered = scripted_model(_GATE_DECISIONS)
    gate_up = CBDCGate(gate_model=recovered)
    result = gate_up.gate(_candidates(), context="global", turn_text="back online")
    assert result.fallback_used is False
    assert [gated.decision for gated in result.decisions] == ["use", "ignore", "verify"]
    assert recovered.calls == 1

    # Flush: every deferred op applied through the real store, IN ORDER.
    applied = deferred_ops.flush(store)
    assert [entry.text for entry in applied] == [op["text"] for op in OFFLINE_OPS]
    assert deferred_ops.pending == 0

    # No data loss, no duplicates: exactly the deferred ops, nothing else.
    entries = store.list()
    assert sorted(entry.text for entry in entries) == sorted(
        op["text"] for op in OFFLINE_OPS
    )
    assert len(entries) == 3
    assert len({entry.id for entry in entries}) == 3
    assert store.git.is_clean()
    # Each flushed op is durably committed (one commit per add).
    for entry in applied:
        assert store.git.log(path=f"global/facts/{entry.id}.md", limit=5)
