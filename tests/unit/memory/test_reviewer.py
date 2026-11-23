"""P3.T3 acceptance: ReviewerAgent decides add/update/skip/conflict on drafts.

The LLM is injected as a fake callable (single-model architecture — never
called live in tests) and the ConflictDetector is the real default one, so
the RULES > MODEL precedence is exercised for real: exact duplicates are
skipped even when the model wrongly says "add", contradictions become
``conflict`` actions without trusting the model output, and validated model
decisions (add / update with merged text) pass through the schema
validation contract.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from hermes.memory.draft import DraftFact
from hermes.memory.entry import MemoryEntry
from hermes.memory.reviewer import (
    REVIEWER_SYSTEM_PROMPT,
    ReviewerAgent,
)

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class _ScriptedModel:
    """Fake model: records prompts, replays scripted completions in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected extra model call")
        return self._responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _entry(entry_id: str, text: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, type="User", text=text, created_at=_NOW)


def _draft(text: str) -> DraftFact:
    return DraftFact(
        draft_id="d-1",
        text=text,
        type="User",
        proposed_source="user-explicit",
        confidence=0.9,
    )


def _decision(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "action": "add",
        "target_id": None,
        "rationale": "because",
        "merged_text": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.unit
def test_add_action_for_new_fact() -> None:
    """A fact unrelated to the candidates is added; schema fields normalized."""
    existing = [_entry("e1", "User commutes by bicycle")]  # unrelated to the draft
    model = _ScriptedModel([
        _decision(
            action="add",
            target_id="e1",  # model hallucinating a target on add
            rationale="no entry covers this preference",
            merged_text="leftover merge text",  # dropped: merged_text is update-only
        )
    ])
    seen_queries: list[str] = []

    def _search_fn(query: str) -> list[MemoryEntry]:
        seen_queries.append(query)
        return list(existing)

    reviewer = ReviewerAgent(model=model, search_fn=_search_fn)

    action = reviewer.review(_draft("User prefers mechanical keyboards"))

    assert action.action == "add"
    assert action.target_id is None  # add never carries a target
    assert action.merged_text is None  # merged_text only for update
    assert action.rationale == "no entry covers this preference"
    assert seen_queries == ["User prefers mechanical keyboards"]  # draft text is the query
    assert model.calls == 1
    assert REVIEWER_SYSTEM_PROMPT in model.prompts[0]
    assert "User prefers mechanical keyboards" in model.prompts[0]
    assert "User commutes by bicycle" in model.prompts[0]  # candidates injected
    assert '"id": "e1"' in model.prompts[0]


@pytest.mark.unit
def test_update_action_for_related_existing() -> None:
    """A related candidate absorbs the draft: update with the merged text."""
    existing = [_entry("e2", "User prefers pytest for running Python tests")]
    model = _ScriptedModel([
        _decision(
            action="update",
            target_id="e2",
            rationale="extends the existing pytest preference",
            merged_text="User prefers pytest with the pytest-xdist plugin for parallel runs",
        )
    ])
    reviewer = ReviewerAgent(model=model)

    action = reviewer.review(
        _draft("User prefers pytest with the pytest-xdist plugin for parallel runs"),
        existing=existing,
    )

    assert action.action == "update"
    assert action.target_id == "e2"
    assert action.merged_text == "User prefers pytest with the pytest-xdist plugin for parallel runs"
    assert action.rationale == "extends the existing pytest preference"
    assert model.calls == 1


@pytest.mark.unit
def test_skip_action_for_exact_duplicate() -> None:
    """Exact duplicates are skipped — both when the model cooperates and when it doesn't."""
    existing = [_entry("e3", "User prefers concise answers")]
    draft = _draft("User prefers concise answers")

    # Case A: the model correctly names the duplicate -> skip with its target.
    honest = _ScriptedModel([
        _decision(action="skip", target_id="e3", rationale="verbatim duplicate")
    ])
    action = ReviewerAgent(model=honest).review(draft, existing=existing)
    assert action.action == "skip"
    assert action.target_id == "e3"
    assert action.merged_text is None
    assert action.rationale == "verbatim duplicate"
    assert honest.calls == 1

    # Case B: the model wrongly says "add" -> the duplicate rule overrides it.
    wrong = _ScriptedModel([
        _decision(action="add", target_id=None, rationale="seems new to me")
    ])
    action = ReviewerAgent(model=wrong).review(draft, existing=existing)
    assert action.action == "skip"  # rule > model: never stored as new
    assert action.target_id == "e3"
    assert "duplicate" in action.rationale
    assert wrong.calls == 1  # model consulted, then overruled


@pytest.mark.unit
def test_conflict_detection_returns_special_action() -> None:
    """A contradiction against an existing entry yields action=conflict, model untrusted."""
    existing = [_entry("e4", "The repo uses npm for installs")]
    draft = _draft("The repo uses pnpm for installs")  # antonym pair, similarity > 0.85
    model = _ScriptedModel([
        _decision(action="add", target_id=None, rationale="totally new fact")
    ])
    reviewer = ReviewerAgent(model=model)

    action = reviewer.review(draft, existing=existing)

    assert action.action == "conflict"
    assert action.target_id == "e4"  # the conflicting entry, from the detector
    assert action.merged_text is None
    assert "contradiction" in action.rationale or "contradicts" in action.rationale
    assert model.calls == 0  # the rule decides first; model output not trusted
