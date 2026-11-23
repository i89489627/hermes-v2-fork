"""P7.T5 acceptance: SessionCompactor unit tests (plan Phase 7).

Covered: :meth:`SessionCompactor.compact` stores the session summary as a
**Reference-type** memory entry whose text begins
``"Compaction summary for session <id>"`` — one git commit via the store,
``compaction`` / ``session:<id>`` / ``trace_range:<first>..<last>`` tags and
the "memory points, doesn't mirror" evidence line as trust-card provenance
(test_compact_session_creates_reference_entry); the trace-range provenance is
pointed at exactly the requested range (or derived from the recorded turn ids
when omitted) and :func:`find_compaction_summary` resolves the LATEST summary
per session (test_summary_points_to_trace_range).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.compaction import (
    COMPACTION_TAG,
    SESSION_TAG_PREFIX,
    SUMMARY_TEXT_PREFIX,
    TRACE_RANGE_TAG_PREFIX,
    SessionCompactor,
    find_compaction_summary,
    summary_text,
)
from hermes.memory.store import MemoryStore

_SESSION = "chat-1-v001"

_TURNS = [
    ("t1", "We set up the acme deploy pipeline. It uses GitHub Actions."),
    ("t2", "Production deploys run only from main. That was pinned yesterday."),
]

_RANGE = "t1..t2"

#: Evidence line contract ("memory points, doesn't mirror", plan P7.T5).
_EXPECTED_EVIDENCE = (
    f"source points to trace range {_RANGE} — memory points, doesn't mirror"
)


@pytest.mark.unit
def test_compact_session_creates_reference_entry(tmp_path: Path) -> None:
    """Compaction stores a git-committed Reference entry with full provenance."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    compactor = SessionCompactor(store)

    entry = compactor.compact(_SESSION, _TURNS, trace_range=_RANGE)

    # Reference-type entry (plan P7.T5: "stored as a Reference-type memory entry")
    assert entry.type == "Reference"
    # the text BEGINS with the documented prefix + session id
    assert entry.text.startswith(f"{SUMMARY_TEXT_PREFIX} {_SESSION}")
    # deterministic extractive fallback: FIRST sentence of each turn, joined
    assert "We set up the acme deploy pipeline." in entry.text
    assert "Production deploys run only from main." in entry.text
    # second sentences stay OUT (the summary is compact, not a mirror)
    assert "It uses GitHub Actions" not in entry.text
    assert "That was pinned yesterday" not in entry.text
    # trust-card provenance: compaction + session + trace-range tags
    assert COMPACTION_TAG in entry.tags
    assert f"{SESSION_TAG_PREFIX}{_SESSION}" in entry.tags
    assert f"{TRACE_RANGE_TAG_PREFIX}{_RANGE}" in entry.tags
    # trust card carries the trace-range evidence line
    assert len(entry.verification_events) == 1
    event = entry.verification_events[0]
    assert event.source == "agent-inferred"
    assert event.verifier == "session-compactor"
    assert event.evidence == _EXPECTED_EVIDENCE
    # git-committed via the store: one add commit, nothing dangling
    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    assert path.is_file()
    commits = store.git.log(path=path)
    assert [commit.message for commit in commits] == [f"[memory] add {entry.id} global"]
    assert store.git.is_clean()

    # an empty session is refused (nothing to compact) — explicit contract
    with pytest.raises(ValueError):
        compactor.compact("empty-session", [])


@pytest.mark.unit
def test_summary_points_to_trace_range(tmp_path: Path) -> None:
    """The summary points at the trace range: explicit, derived, and lookup."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    compactor = SessionCompactor(store)

    # explicit range: tag + evidence carry EXACTLY the requested range
    explicit = compactor.compact(_SESSION, _TURNS, trace_range="t3..t7")
    assert f"{TRACE_RANGE_TAG_PREFIX}t3..t7" in explicit.tags
    assert (
        explicit.verification_events[0].evidence
        == "source points to trace range t3..t7 — memory points, doesn't mirror"
    )

    # derived range (trace_range omitted): first..last of the recorded turn ids
    derived = compactor.compact("chat-1-v002", _TURNS)
    assert f"{TRACE_RANGE_TAG_PREFIX}t1..t2" in derived.tags
    assert (
        derived.verification_events[0].evidence
        == "source points to trace range t1..t2 — memory points, doesn't mirror"
    )

    # lookup resolves the session's summary; an unknown session finds none
    found = find_compaction_summary(store, _SESSION)
    assert found is not None and found.id == explicit.id
    assert find_compaction_summary(store, "chat-1-v002").id == derived.id
    assert find_compaction_summary(store, "no-such-session") is None

    # a re-compaction supersedes: the LATEST summary is the one found
    second = compactor.compact(_SESSION, _TURNS, trace_range=_RANGE)
    assert second.id != explicit.id
    assert find_compaction_summary(store, _SESSION).id == second.id

    # the pure summarizer (store-free) honors the same trace-range contract
    assert summary_text(_TURNS) == (
        "We set up the acme deploy pipeline. Production deploys run only from main."
    )
