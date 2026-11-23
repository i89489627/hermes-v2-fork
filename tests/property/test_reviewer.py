"""P3.T3 property: ReviewerAgent always returns a valid, in-set ReviewAction.

For arbitrary draft texts, arbitrary candidate-entry sets and a fake model
returning anything (valid decisions for every action, invalid actions,
missing keys, unknown targets, misplaced merged_text, malformed JSON, empty
strings), ``review`` never raises and the emitted
:class:`~hermes.memory.draft.ReviewAction` always satisfies the schema
invariants: ``action`` is one of the four allowed literals, ``add`` never
carries a target or merged text, ``update`` always names a known candidate,
and ``merged_text`` survives only on ``update``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.draft import DraftFact, ReviewAction
from hermes.memory.entry import MemoryEntry
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.types import ENTRY_TYPES

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_ALLOWED_ACTIONS = frozenset({"add", "update", "skip", "conflict"})

#: Draft-text strategy: arbitrary unicode plus pinned rule-triggering texts
#: (negation cues, antonym pairs) so both ConflictDetector branches fire.
_DRAFT_STRATEGY = st.one_of(
    st.text(min_size=1, max_size=120),
    st.just("The repo uses pnpm for installs"),
    st.just("The repo uses npm for installs"),
    st.just("The user does not prefer vim"),
    st.just("Feature flags are enabled"),
    st.just("   "),
    st.just("turn with 🌍 unicode and 控 character"),
)

_ENTRY_TEXT_STRATEGY = st.one_of(
    st.text(max_size=120),
    st.just(""),
    st.just("The repo uses npm for installs"),
)

_BEHAVIORS = (
    "valid-add",
    "valid-update",
    "valid-skip",
    "valid-conflict",
    "invalid-action",
    "unhashable-action",
    "unhashable-target",
    "no-action-key",
    "malformed",
    "json-array",
    "empty",
    "merged-on-add",
    "target-on-add",
    "missing-rationale",
)


def _model_for(behavior: str, entry_count: int):
    """Deterministic fake reviewer model for the drawn behavior."""

    def _model(prompt: str) -> str:
        if behavior == "malformed":
            return "I am sorry, I cannot answer in JSON."
        if behavior == "json-array":
            return "[]"
        if behavior == "empty":
            return ""
        target = "e0" if entry_count >= 1 else None
        if behavior == "valid-add":
            return json.dumps({"action": "add", "target_id": None, "rationale": "new"})
        if behavior == "valid-update":
            return json.dumps({
                "action": "update",
                "target_id": target,
                "rationale": "merge",
                "merged_text": "merged body text",
            })
        if behavior == "valid-skip":
            return json.dumps({
                "action": "skip",
                "target_id": target,
                "rationale": "duplicate",
            })
        if behavior == "valid-conflict":
            return json.dumps({
                "action": "conflict",
                "target_id": target,
                "rationale": "contradicts entry",
            })
        if behavior == "invalid-action":
            return json.dumps({"action": "delete-everything", "target_id": target})
        if behavior == "unhashable-action":
            # a list action must downgrade to skip, never raise on the
            # allowed-set membership test
            return json.dumps({"action": ["add"], "target_id": target})
        if behavior == "unhashable-target":
            return json.dumps({"action": "update", "target_id": {"e0": 1}})
        if behavior == "no-action-key":
            return json.dumps({"target_id": target, "rationale": "no action"})
        if behavior == "merged-on-add":
            return json.dumps({
                "action": "add",
                "target_id": target,
                "rationale": "new",
                "merged_text": "should be dropped",
            })
        if behavior == "target-on-add":
            return json.dumps({"action": "add", "target_id": "e0", "rationale": "new"})
        if behavior == "missing-rationale":
            return json.dumps({"action": "add"})
        raise AssertionError(f"unknown behavior {behavior!r}")

    return _model


def _assert_invariants(action: ReviewAction, known_ids: set[str]) -> None:
    """The P3.T3 output invariants, regardless of what the model returned."""
    assert isinstance(action, ReviewAction)
    assert action.action in _ALLOWED_ACTIONS
    if action.action == "add":
        assert action.target_id is None
        assert action.merged_text is None
    if action.action == "update":
        assert action.target_id is not None
        assert action.target_id in known_ids
    if action.action in {"skip", "conflict"}:
        assert action.merged_text is None
    if action.action == "conflict":
        assert action.target_id is not None


@pytest.mark.property
@given(
    draft_text=_DRAFT_STRATEGY,
    entry_texts=st.lists(_ENTRY_TEXT_STRATEGY, max_size=4),
    entry_type=st.sampled_from(sorted(ENTRY_TYPES)),
    behavior=st.sampled_from(_BEHAVIORS),
)
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="valid-add")  # noqa: E501
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="malformed")  # noqa: E501
@example(draft_text="   ", entry_texts=[], entry_type="Project", behavior="valid-update")
@settings(max_examples=50, deadline=None)
def test_action_always_in_allowed_set(
    draft_text: str,
    entry_texts: list[str],
    entry_type: str,
    behavior: str,
) -> None:
    """Review never raises and the action is always valid with sane field values."""
    entries = [
        MemoryEntry(id=f"e{index}", type=entry_type, text=text, created_at=_NOW)
        for index, text in enumerate(entry_texts)
    ]
    draft = DraftFact(
        draft_id="prop-draft",
        text=draft_text,
        type="User",
        proposed_source="agent-inferred",
        confidence=0.5,
    )
    reviewer = ReviewerAgent(model=_model_for(behavior, len(entries)))

    action = reviewer.review(draft, existing=entries)

    _assert_invariants(action, {entry.id for entry in entries})


# ── P20.T2: the validity contract at 1000 examples + conflict target pin ─────


@pytest.mark.property
@given(
    draft_text=_DRAFT_STRATEGY,
    entry_texts=st.lists(_ENTRY_TEXT_STRATEGY, max_size=4),
    entry_type=st.sampled_from(sorted(ENTRY_TYPES)),
    behavior=st.sampled_from(_BEHAVIORS),
)
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="valid-add")  # noqa: E501
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="malformed")  # noqa: E501
@example(draft_text="   ", entry_texts=[], entry_type="Project", behavior="valid-conflict")
@settings(max_examples=1000, deadline=None)
def test_action_always_valid(
    draft_text: str,
    entry_texts: list[str],
    entry_type: str,
    behavior: str,
) -> None:
    """P20.T2: review always emits a valid action for any input (1000 examples).

    Same invariant wording as :func:`test_action_always_in_allowed_set` (the
    P3.T3 test), extended to the plan P20.T2-required 1000-example depth:
    the action is always one of {add, update, skip, conflict} with the P3.T3
    schema invariants, under arbitrary drafts, candidates and model output.
    """
    entries = [
        MemoryEntry(id=f"e{index}", type=entry_type, text=text, created_at=_NOW)
        for index, text in enumerate(entry_texts)
    ]
    draft = DraftFact(
        draft_id="prop-draft",
        text=draft_text,
        type="User",
        proposed_source="agent-inferred",
        confidence=0.5,
    )
    reviewer = ReviewerAgent(model=_model_for(behavior, len(entries)))

    action = reviewer.review(draft, existing=entries)

    _assert_invariants(action, {entry.id for entry in entries})


@pytest.mark.property
@given(
    draft_text=_DRAFT_STRATEGY,
    entry_texts=st.lists(_ENTRY_TEXT_STRATEGY, max_size=4),
    entry_type=st.sampled_from(sorted(ENTRY_TYPES)),
    behavior=st.sampled_from(_BEHAVIORS),
)
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="valid-add")  # noqa: E501
@example(draft_text="Feature flags are enabled", entry_texts=["Feature flags are enabled", "The repo uses npm for installs"], entry_type="Project", behavior="valid-add")  # noqa: E501
@example(draft_text="The user does not prefer vim", entry_texts=["The user prefers vim"], entry_type="User", behavior="valid-add")  # noqa: E501
@example(draft_text="The repo uses pnpm for installs", entry_texts=["The repo uses npm for installs"], entry_type="User", behavior="valid-conflict")  # noqa: E501
@settings(max_examples=1000, deadline=None)
def test_conflict_action_has_target_id(
    draft_text: str,
    entry_texts: list[str],
    entry_type: str,
    behavior: str,
) -> None:
    """P20.T2: a conflict action always names an existing candidate entry.

    Whichever path produced the conflict — the detector rule (rules > model,
    e.g. the pinned pnpm/npm and negation-cue pairs) or the validated model
    decision — ``target_id`` is never None and always names one of the
    candidate entries the reviewer was shown.
    """
    entries = [
        MemoryEntry(id=f"e{index}", type=entry_type, text=text, created_at=_NOW)
        for index, text in enumerate(entry_texts)
    ]
    draft = DraftFact(
        draft_id="prop-draft",
        text=draft_text,
        type="User",
        proposed_source="agent-inferred",
        confidence=0.5,
    )
    reviewer = ReviewerAgent(model=_model_for(behavior, len(entries)))

    action = reviewer.review(draft, existing=entries)

    if action.action == "conflict":
        known_ids = {entry.id for entry in entries}
        assert action.target_id is not None
        assert action.target_id in known_ids
