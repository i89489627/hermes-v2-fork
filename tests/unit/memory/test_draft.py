"""P3.T1 acceptance: DraftFact + ReviewAction dataclasses.

``to_dict``/``from_dict`` are JSON-safe inverse roundtrips (datetimes as
ISO 8601 UTC with ``Z`` suffix); ``DraftFact`` validates its enum fields and
clamps ``confidence``; ``ReviewAction.action`` is restricted to the
``add``/``update``/``skip``/``conflict`` literal set.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from hermes.memory.draft import DraftFact, ReviewAction

_TZ_PLUS_TWO = timezone(timedelta(hours=2))


def _fact(**overrides: Any) -> DraftFact:
    """A fully-populated DraftFact; created_at deliberately non-UTC."""
    kwargs: dict[str, Any] = {
        "draft_id": "draft-001",
        "text": "User prefers concise answers",
        "type": "User",
        "proposed_source": "agent-inferred",
        "confidence": 0.8,
        "context_hint": "work",
        "rationale": "stated twice in the session",
        "source_turn_ids": ["turn-7", "turn-11"],
        "created_at": datetime(2026, 11, 20, 12, 0, 0, tzinfo=_TZ_PLUS_TWO),
    }
    kwargs.update(overrides)
    return DraftFact(**kwargs)


@pytest.mark.unit
def test_draft_fact_serialization() -> None:
    """to_dict is JSON-safe; from_dict(json(to_dict)) restores every field."""
    fact = _fact()

    payload = fact.to_dict()
    json.dumps(payload)  # must be JSON-serializable by construction
    assert payload["created_at"] == "2026-11-20T10:00:00Z"  # normalized to UTC
    assert payload["draft_id"] == "draft-001"
    assert payload["text"] == "User prefers concise answers"
    assert payload["type"] == "User"
    assert payload["proposed_source"] == "agent-inferred"
    assert payload["confidence"] == 0.8
    assert payload["context_hint"] == "work"
    assert payload["rationale"] == "stated twice in the session"
    assert payload["source_turn_ids"] == ["turn-7", "turn-11"]

    restored = DraftFact.from_dict(json.loads(json.dumps(payload)))
    assert restored == fact  # aware datetimes compare equal after UTC normalization
    assert restored.created_at.tzinfo is not None

    # defaults per the plan's field table
    minimal = DraftFact(
        draft_id="draft-002",
        text="bare fact",
        type="Project",
        proposed_source="file-read",
        confidence=0.5,
    )
    assert minimal.context_hint == "global"
    assert minimal.rationale == ""
    assert minimal.source_turn_ids == []
    assert minimal.created_at.tzinfo is not None  # aware UTC default

    # construction invariants: enum fields validated, naive stamps rejected,
    # confidence clamped into [0, 1]
    with pytest.raises(ValueError, match="unknown entry type"):
        _fact(type="Snapshot")
    with pytest.raises(ValueError, match="unknown verification source"):
        _fact(proposed_source="gossip")
    with pytest.raises(ValueError, match="timezone-aware"):
        _fact(created_at=datetime(2026, 11, 20, 12, 0, 0))
    assert _fact(confidence=1.7).confidence == 1.0
    assert _fact(confidence=-0.2).confidence == 0.0
    with pytest.raises(ValueError):
        DraftFact.from_dict({"text": "missing everything else"})


@pytest.mark.unit
def test_review_action_validates_action_enum() -> None:
    """Only add/update/skip/conflict are accepted; everything else ValueError."""
    for action in ("add", "update", "skip", "conflict"):
        decision = ReviewAction(action, target_id=None, rationale=f"why: {action}")
        assert decision.action == action

    with pytest.raises(ValueError, match="invalid review action"):
        ReviewAction("delete", target_id=None, rationale="not a plan action")
    with pytest.raises(ValueError, match="invalid review action"):
        ReviewAction("ADD", target_id=None, rationale="case sensitive")

    decision = ReviewAction(
        "update",
        target_id="2026-11-20-old-fact-abcd1234",
        rationale="tightened wording",
        merged_text="the corrected body",
    )
    assert decision.merged_text == "the corrected body"

    payload = decision.to_dict()
    json.dumps(payload)  # JSON-safe by construction
    assert ReviewAction.from_dict(json.loads(json.dumps(payload))) == decision

    skip = ReviewAction("skip", target_id=None, rationale="duplicate of an existing fact")
    assert skip.merged_text is None
    assert skip.to_dict()["merged_text"] is None
    assert ReviewAction.from_dict(skip.to_dict()) == skip

    with pytest.raises(ValueError, match="missing field"):
        ReviewAction.from_dict({"rationale": "no action key"})
