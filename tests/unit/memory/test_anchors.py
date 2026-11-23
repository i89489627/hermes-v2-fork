"""P2.T5 acceptance: automatic ground-truth anchors (plan Phase 2).

Covers :func:`hermes.memory.anchors.record_test_anchor` over
:class:`hermes.memory.store.MemoryStore`: green runs append the weight-1.0
``test-green`` event, red runs append the documented negative-anchor mapping
(source ``tool-output``, weight 0.85, value 0.0, evidence prefixed
``test-red:`` — the plan's literal source set has no ``test-red``), and
entries unrelated to the run (no citation, no changed-file overlap, no
test-module tag) are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.anchors import record_test_anchor
from hermes.memory.store import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


@pytest.mark.unit
def test_test_green_appends_weight_1_event(tmp_path: Path) -> None:
    """A green run appends source=test-green weight=1.0 value=1.0 to cited entries."""
    store = _make_store(tmp_path)
    entry = store.add("BM25 prefilter keeps injection under budget", type="Project")
    assert entry.trust_score == 1.0  # single seeded agent-inferred event

    result = record_test_anchor(
        store,
        session_cited_entry_ids=[entry.id],
        changed_files=[],
        outcome="green",
        evidence="pytest tests/unit/memory/test_store.py (exit=0)",
    )

    assert result.events_added == 1
    assert result.related_ids == [entry.id]

    updated = store.get(entry.id)
    assert updated is not None
    assert len(updated.verification_events) == 2
    event = updated.verification_events[-1]
    assert event.source == "test-green"
    assert event.weight == 1.0
    assert event.value == 1.0
    assert event.verifier == "test-hook"
    assert event.evidence == "pytest tests/unit/memory/test_store.py (exit=0)"
    # test-green (w=1.0, v=1.0) joins agent-inferred (w=0.5, v=1.0): still 1.0.
    assert updated.trust_score == 1.0
    assert updated.last_verified_at == event.timestamp


@pytest.mark.unit
def test_test_red_downweights_related_entries(tmp_path: Path) -> None:
    """A red run downweights related entries via the negative-anchor mapping.

    DEVIATION NOTE: the plan's VerificationSource set has no ``test-red``;
    red is recorded as source ``tool-output`` (weight 0.85, value 0.0) with
    ``test-red:``-prefixed evidence (documented in
    :mod:`hermes.memory.verification` / :mod:`hermes.memory.anchors`).
    """
    store = _make_store(tmp_path)
    entry = store.add("hermes/memory/store.py holds the CRUD facade", type="Project")
    assert entry.trust_score == 1.0

    result = record_test_anchor(
        store,
        session_cited_entry_ids=None,
        changed_files=["hermes/memory/store.py"],
        outcome="red",
        evidence="pytest tests/unit/memory/test_store.py::test_add (exit=1)",
    )

    assert result.events_added == 1
    assert result.related_ids == [entry.id]

    updated = store.get(entry.id)
    assert updated is not None
    event = updated.verification_events[-1]
    assert event.source == "tool-output"
    assert event.weight == 0.85
    assert event.value == 0.0
    assert event.evidence.startswith("test-red:")
    assert "exit=1" in event.evidence
    # Weighted avg: (0.5*1.0 + 0.85*0.0) / 1.35 < 0.5 — the red run downweights.
    assert updated.trust_score < 0.5
    assert updated.trust_score == pytest.approx(0.5 / 1.35)


@pytest.mark.unit
def test_unrelated_entries_untouched(tmp_path: Path) -> None:
    """Entries with no citation / changed-file / test-module tie get NO event."""
    store = _make_store(tmp_path)
    cited = store.add("BM25 prefilter keeps injection under budget", type="Project")
    unrelated_a = store.add("user prefers dark theme in the TUI", type="User")
    unrelated_b = store.add("soul style stays within 2000 chars", type="Feedback")
    snapshots = {
        entry.id: (list(entry.verification_events), entry.trust_score)
        for entry in (unrelated_a, unrelated_b)
    }

    result = record_test_anchor(
        store,
        session_cited_entry_ids=[cited.id],
        changed_files=["hermes/memory/store.py"],
        outcome="green",
        evidence="pytest tests/unit/memory/test_store.py (exit=0)",
    )

    assert result.events_added == 1
    assert result.related_ids == [cited.id]
    for unrelated in (unrelated_a, unrelated_b):
        updated = store.get(unrelated.id)
        assert updated is not None
        assert updated.verification_events == snapshots[unrelated.id][0]
        assert updated.trust_score == snapshots[unrelated.id][1]
