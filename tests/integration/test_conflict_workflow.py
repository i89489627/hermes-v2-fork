"""P8.T2 acceptance: conflict detection wired into the reviewer pipeline.

End-to-end over the REAL components with deterministic injected fakes
(single-model architecture — never called live): the :class:`ReviewerAgent`
runs the real default :class:`~hermes.memory.conflict_detector.ConflictDetector`,
whose rules-over-model rule 1 flags a contradiction BEFORE the model is
consulted ("conflicts caught before write"), and the
:class:`~hermes.memory.pipeline.MemoryPipeline` routes that detector-flagged
conflict into the explicit resolution protocol —
:meth:`ReviewerAgent.resolve_conflict` decides ``merge`` | ``replace`` |
``skip`` (with rationale), the pipeline applies it to the target entry and
appends the honest low-weight audit event (source ``llm`` / weight 0.3,
evidence ``conflict resolution <verb>: <rationale>``) to the target's trust
card ("audit trail in trust card").

Scenario texts are pinned to pairs the TF-IDF trigram detector measurably
classifies as contradictions (similarity > 0.85 plus a negation cue /
antonym pair): "frozen" vs "not frozen" and "npm" vs "pnpm".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.reviewer import (
    RESOLUTION_SYSTEM_PROMPT,
    ReviewerAgent,
)
from hermes.memory.store import MemoryStore
from hermes.memory.types import VerificationEvent

_PIPELINE_LOGGER = "hermes.memory.pipeline"

_TURN_TEXT = "Heads up - the deploy policy changed as of today"

_ENTRY_FROZEN = "Deployments to production are frozen"
_DRAFT_UNFROZEN = "Deployments to production are not frozen"

_ENTRY_NPM = "The acme deploy pipeline uses npm"
_DRAFT_PNPM = "The acme deploy pipeline uses pnpm"

_MERGE_BODY = (
    "Deployments to production were frozen until the January thaw; they are "
    "not frozen as of today"
)


class _ScriptedReviewerModel:
    """Fake reviewer model: records prompts, replays scripted completions."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected extra reviewer model call")
        return self._responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _extractor_for(draft_text: str) -> ExtractorAgent:
    """Stub extractor model: one user-explicit draft fact for the turn."""

    def _model(_prompt: str) -> str:
        return json.dumps([
            {
                "type": "Project",
                "text": draft_text,
                "proposed_source": "user-explicit",
                "confidence": 0.9,
            }
        ])

    return ExtractorAgent(model=_model)


def _resolution(
    verb: str, rationale: str, merged_text: str | None = None
) -> str:
    """Valid resolve_conflict model output for one resolution verb."""
    return json.dumps({
        "resolution": verb,
        "merged_text": merged_text,
        "rationale": rationale,
    })


def _pipeline(
    store: MemoryStore, draft_text: str, model: _ScriptedReviewerModel
) -> MemoryPipeline:
    reviewer = ReviewerAgent(model=model, search_fn=lambda q: store.search(q, top_k=5))
    return MemoryPipeline(_extractor_for(draft_text), reviewer, store)


def _audit_event(entry: MemoryEntry) -> VerificationEvent:
    """The entry's conflict-resolution audit event (asserts it is present)."""
    resolution_events = [
        event
        for event in entry.verification_events
        if event.evidence.startswith("conflict resolution")
    ]
    assert resolution_events, (
        f"no conflict-resolution audit event in {entry.verification_events!r}"
    )
    return resolution_events[-1]


@pytest.mark.integration
def test_conflict_triggers_reviewer_resolution(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Detector-flagged contradiction -> explicit resolution: target merged.

    The detector catches the contradiction BEFORE the model is consulted
    (first pass, zero model calls); the pipeline then asks the model ONCE how
    to resolve. A ``merge`` resolution updates the TARGET entry with the
    merged text and the draft is never added as a new entry.
    """
    store = MemoryStore(root=tmp_path / "memory-repo")
    existing = store.add(_ENTRY_FROZEN, type="Project")
    model = _ScriptedReviewerModel([
        _resolution(
            "merge",
            "the January thaw supersedes the freeze note",
            merged_text=_MERGE_BODY,
        )
    ])
    pipeline = _pipeline(store, _DRAFT_UNFROZEN, model)

    with caplog.at_level(logging.INFO, logger=_PIPELINE_LOGGER):
        result = pipeline.run_after_turn("t1", _TURN_TEXT, session_id="s1")

    # the conflict was caught before write: exactly one model call, and it is
    # the RESOLUTION pass (the first pass was decided by the detector rule)
    assert model.calls == 1
    resolution_prompt = model.prompts[0]
    assert "conflict resolver" in resolution_prompt
    assert RESOLUTION_SYSTEM_PROMPT.splitlines()[0] in resolution_prompt
    assert _DRAFT_UNFROZEN in resolution_prompt  # the draft under resolution
    assert _ENTRY_FROZEN in resolution_prompt  # the conflicting entry text
    assert existing.id in resolution_prompt  # ...with its id

    # resolution applied: merge -> the TARGET entry was updated
    assert [action.action for action in result.actions] == ["update"]
    final = result.actions[0]
    assert final.target_id == existing.id
    assert final.merged_text == _MERGE_BODY
    assert final.rationale.startswith("conflict resolution merge: ")
    assert "the January thaw supersedes the freeze note" in final.rationale

    assert len(result.applied) == 1
    applied = result.applied[0]
    assert applied.action == "update"
    assert applied.entry_id == existing.id

    # the target entry carries the merged text; the draft was NOT added
    # as a new entry
    assert len(store.list()) == 1
    target = store.get(existing.id)
    assert target is not None
    assert target.text == _MERGE_BODY

    # every resolution is logged with its rationale
    assert "resolution protocol" in caplog.text
    assert "resolved as merge" in caplog.text
    assert "conflict resolution merge" in caplog.text
    assert "the January thaw supersedes the freeze note" in caplog.text


@pytest.mark.integration
def test_resolution_creates_audit_trail(tmp_path: Path) -> None:
    """After merge/replace the target's trust card carries the audit event.

    The event uses the honest low-weight provenance (source ``llm``,
    weight 0.3) with evidence ``conflict resolution <verb>: <rationale>``.
    A ``skip`` resolution writes nothing and leaves no audit event.
    """
    # ── merge resolution: audit event on the merged target ───────────────
    merge_store = MemoryStore(root=tmp_path / "merge")
    merge_target = merge_store.add(_ENTRY_FROZEN, type="Project")
    merge_model = _ScriptedReviewerModel([
        _resolution("merge", "thaw supersedes the freeze", merged_text=_MERGE_BODY)
    ])
    merge_pipeline = _pipeline(merge_store, _DRAFT_UNFROZEN, merge_model)
    merge_result = merge_pipeline.run_after_turn("t1", _TURN_TEXT, session_id="s1")
    assert merge_result.applied  # the merge wrote the target

    merged_entry = merge_store.get(merge_target.id)
    assert merged_entry is not None
    event = _audit_event(merged_entry)
    assert event.source == "llm"
    assert event.weight == 0.3
    assert event.verifier == "conflict-resolver"
    evidence = event.evidence
    assert evidence.startswith("conflict resolution merge: ")
    assert "thaw supersedes the freeze" in evidence  # the verb's rationale rides along
    assert 0.0 <= merged_entry.trust_score <= 1.0

    # ── replace resolution: the draft text supersedes the entry ──────────
    replace_store = MemoryStore(root=tmp_path / "replace")
    replace_target = replace_store.add(_ENTRY_NPM, type="Project")
    replace_model = _ScriptedReviewerModel([
        _resolution("replace", "the team standardized on pnpm")
    ])
    replace_pipeline = _pipeline(replace_store, _DRAFT_PNPM, replace_model)
    replace_result = replace_pipeline.run_after_turn("t2", _TURN_TEXT, session_id="s1")
    assert [action.action for action in replace_result.actions] == ["update"]
    assert len(replace_store.list()) == 1  # still only the target entry
    replaced_entry = replace_store.get(replace_target.id)
    assert replaced_entry is not None
    assert replaced_entry.text == _DRAFT_PNPM  # replace writes the draft text
    replace_event = _audit_event(replaced_entry)
    assert replace_event.source == "llm"
    assert replace_event.evidence.startswith("conflict resolution replace: ")
    assert "the team standardized on pnpm" in replace_event.evidence

    # ── skip resolution: nothing written, no audit event ─────────────────
    skip_store = MemoryStore(root=tmp_path / "skip")
    skip_target = skip_store.add(_ENTRY_NPM, type="Project")
    skip_before = skip_store.get(skip_target.id)
    assert skip_before is not None
    skip_model = _ScriptedReviewerModel([
        _resolution("skip", "transient CI flakiness, not worth storing")
    ])
    skip_pipeline = _pipeline(skip_store, _DRAFT_PNPM, skip_model)
    skip_result = skip_pipeline.run_after_turn("t3", _TURN_TEXT, session_id="s1")

    assert [action.action for action in skip_result.actions] == ["conflict"]
    assert skip_result.actions[0].rationale.startswith("conflict resolution skip: ")
    assert skip_result.applied == []
    assert len(skip_store.list()) == 1  # the draft was never written
    skip_after = skip_store.get(skip_target.id)
    assert skip_after is not None
    assert skip_after.text == _ENTRY_NPM  # the target stayed untouched
    assert skip_after.verification_events == skip_before.verification_events


# ── model robustness: a resolution can never store anything ─────────────────


@pytest.mark.integration
def test_unresolvable_resolution_output_drops_draft(tmp_path: Path) -> None:
    """Malformed resolution output / an overruling 'add' degrade to a drop.

    The detector's contradiction verdict is final: unusable resolution output
    and a model attempt to overrule it must never write anything.
    """
    for label, raw_output in (
        ("malformed", "sorry, not JSON"),
        ("overruling-add", json.dumps({"action": "add", "rationale": "it is new"})),
        ("re-declared-conflict", json.dumps({"action": "conflict", "rationale": "still"})),
        ("merge-without-body", json.dumps({"resolution": "merge", "rationale": "merge it"})),
    ):
        store = MemoryStore(root=tmp_path / label)
        target = store.add(_ENTRY_NPM, type="Project")
        model = _ScriptedReviewerModel([raw_output])
        pipeline = _pipeline(store, _DRAFT_PNPM, model)

        result = pipeline.run_after_turn("t1", _TURN_TEXT, session_id="s1")

        assert [action.action for action in result.actions] == ["conflict"], label
        assert result.applied == [], label
        assert len(store.list()) == 1, label
        entry = store.get(target.id)
        assert entry is not None
        assert entry.text == _ENTRY_NPM, label
