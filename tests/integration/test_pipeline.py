"""P3.T4 + P3.T6 acceptance: memory pipeline (plan Phase 3).

P3.T6 workflow: end-to-end over the REAL v2 components with deterministic
injected fakes (single-model architecture — never called live):
:class:`ContextWindow` assembles the session window, the extractor consumes
it (context block, reference resolution across turns, confirmed-fact
suppression, ``source_turn_ids`` provenance), and the :class:`ReviewerAgent`
merges the refined fact into the existing entry. The latency acceptance
(context overhead < 200ms) is asserted structurally: with a stub model,
extraction with context must not be more than 0.2s slower than without —
context assembly is string ops only.

P3.T4 (appended): the :class:`~hermes.memory.pipeline.MemoryPipeline`
orchestrator over a REAL git-backed :class:`~hermes.memory.store.MemoryStore`
— turn to committed entry, async non-blocking run, per-run metrics, and the
conflict -> second-reviewer-pass -> resolve/drop flow with its documented
one-resolution-pass termination rule.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.memory.context_window import ContextWindow, Turn
from hermes.memory.entry import MemoryEntry
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.pipeline import AppliedAction, MemoryPipeline, PipelineResult
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore

_PIPELINE_LOGGER = "hermes.memory.pipeline"

_TURN_1_TEXT = "We deploy the acme service via GitHub Actions only."
_TURN_2_TEXT = "Yes - and production deploys happen only from the main branch."

_FACT_1 = "The acme service deploys via GitHub Actions only"
_FACT_2 = (
    "The acme service deploys to production only from the main branch via GitHub Actions"
)

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _extractor_model(prompt: str) -> str:
    """Stub extractor model: context-shaped output when context is present.

    With the RECENT TURNS block the model resolves the cross-turn reference
    into a self-contained fact and proposes turn provenance — plus a naive
    re-extraction of the confirmed turn-1 fact, which the P3.T6 suppression
    rule must drop. Without context it is the plain turn-1 extractor.
    """
    if "RECENT TURNS (oldest first):" in prompt:
        return json.dumps([
            {
                "type": "Project",
                "text": _FACT_2,
                "proposed_source": "user-explicit",
                "confidence": 0.9,
                "source_turn_ids": ["t1", "t2"],
            },
            {
                "type": "Project",
                "text": _FACT_1,
                "proposed_source": "user-explicit",
                "confidence": 0.8,
                "source_turn_ids": ["t2"],
            },
        ])
    return json.dumps([
        {
            "type": "Project",
            "text": _FACT_1,
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ])


@pytest.mark.integration
def test_multi_turn_extraction_workflow() -> None:
    """Turn 1 -> confirmed fact; turn 2 resolves, suppresses, and reviews."""
    turns: dict[str, list[Turn]] = {"s1": [Turn("t1", "s1", 0, _TURN_1_TEXT, _T0)]}
    window = ContextWindow(fallback_provider=lambda session_id: list(turns.get(session_id, [])))
    extractor = ExtractorAgent(model=_extractor_model)

    # ── turn 1: no context yet, plain single-turn extraction ─────────────
    drafts_1 = extractor.extract(_TURN_1_TEXT)
    assert [draft.text for draft in drafts_1] == [_FACT_1]
    assert drafts_1[0].source_turn_ids == []
    confirmed = [drafts_1[0].text]

    # ── turn 2: multi-turn context via ContextWindow + confirmed facts ───
    turns["s1"].append(Turn("t2", "s1", 1, _TURN_2_TEXT, _T0 + timedelta(minutes=1)))
    context = window.get_recent_turns("s1", n=3)
    assert [turn.turn_id for turn in context] == ["t1", "t2"]  # oldest-first window

    context_items = [*context, {"confirmed_facts": confirmed}]
    start = time.perf_counter()
    drafts_2 = extractor.extract(_TURN_2_TEXT, context_turns=context_items)
    with_context = time.perf_counter() - start

    assert len(drafts_2) == 1  # the confirmed re-extraction was suppressed
    fact_2 = drafts_2[0]
    assert fact_2.text == _FACT_2  # reference resolved into a self-contained fact
    assert fact_2.source_turn_ids == ["t1", "t2"]  # provenance populated (P3.T6)
    assert fact_2.type == "Project"

    # latency acceptance (plan P3.T6: <200ms overhead) — structural: context
    # assembly is string ops only, so the delta must stay far below budget.
    start = time.perf_counter()
    extractor.extract(_TURN_2_TEXT)
    without_context = time.perf_counter() - start
    assert with_context - without_context < 0.2

    # ── review stage: the refined fact merges into the turn-1 entry ──────
    entry_1 = MemoryEntry(
        id="2026-01-01-acme-deploy-1234abcd",
        type="Project",
        text=_FACT_1,
        created_at=_T0,
    )
    reviewer_prompts: list[str] = []

    def _reviewer_model(prompt: str) -> str:
        reviewer_prompts.append(prompt)
        return json.dumps({
            "action": "update",
            "target_id": entry_1.id,
            "rationale": "turn 2 narrows the deploy policy to main-branch production deploys",
            "merged_text": _FACT_2,
        })

    reviewer = ReviewerAgent(model=_reviewer_model)
    action = reviewer.review(fact_2, existing=[entry_1])

    assert action.action == "update"
    assert action.target_id == entry_1.id
    assert action.merged_text == _FACT_2
    assert action.rationale
    assert len(reviewer_prompts) == 1
    assert _FACT_2 in reviewer_prompts[0]  # the draft under review is in the prompt


# ── P3.T4: MemoryPipeline orchestrator ────────────────────────────────────────


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


def _json_decision(
    action: str,
    target_id: str | None = None,
    rationale: str = "because",
    merged_text: str | None = None,
) -> str:
    """Valid reviewer model output for one decision."""
    return json.dumps({
        "action": action,
        "target_id": target_id,
        "rationale": rationale,
        "merged_text": merged_text,
    })


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


@pytest.mark.integration
def test_full_pipeline_turn_to_committed_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Turn -> extraction -> review -> store.add: one committed memory entry."""
    store = _make_store(tmp_path)
    turn_text = "From now on I prefer dark mode in all my IDEs"
    draft_text = "User prefers dark mode IDE themes"
    extractor = ExtractorAgent(model=lambda _prompt: json.dumps([
        {"type": "User", "text": draft_text, "proposed_source": "user-explicit", "confidence": 0.9}
    ]))
    reviewer_model = _ScriptedReviewerModel([
        _json_decision("add", rationale="new durable user preference")
    ])
    reviewer = ReviewerAgent(model=reviewer_model, search_fn=lambda _q: store.search(_q, top_k=5))
    pipeline = MemoryPipeline(extractor, reviewer, store)

    with caplog.at_level(logging.INFO, logger=_PIPELINE_LOGGER):
        result = pipeline.run_after_turn("t1", turn_text, session_id="s1", context="global")

    assert isinstance(result, PipelineResult)
    assert result.turn_id == "t1"
    assert len(result.drafts) == 1
    draft = result.drafts[0]
    assert draft.text == draft_text
    assert [action.action for action in result.actions] == ["add"]
    assert len(result.applied) == 1
    applied = result.applied[0]
    assert isinstance(applied, AppliedAction)
    assert applied.action == "add"
    assert applied.draft_id == draft.draft_id
    assert applied.rationale == "new durable user preference"
    assert result.elapsed_ms >= 0

    # the entry is committed to the store with the pipeline's provenance
    entry = store.get(applied.entry_id)
    assert entry is not None
    assert entry.text == draft_text
    assert entry.type == "User"
    assert entry.context == "global"
    event = entry.verification_events[0]
    assert event.source == "user-explicit"  # draft provenance carried through
    assert event.evidence == f"extractor draft {draft.draft_id}"

    # acceptance: all actions logged with rationale
    assert "applied add" in caplog.text
    assert "new durable user preference" in caplog.text
    assert draft.draft_id in caplog.text


@pytest.mark.integration
def test_pipeline_async_does_not_block_turn(tmp_path: Path) -> None:
    """sync=False returns immediately; the flow completes on a worker thread."""
    store = _make_store(tmp_path)
    release = threading.Event()
    metrics_seen: list[dict] = []

    def _blocking_extractor_model(_prompt: str) -> str:
        release.wait(timeout=10.0)  # blocks until the test releases it
        return json.dumps([
            {"type": "User", "text": "User prefers async pipelines", "confidence": 0.9}
        ])

    extractor = ExtractorAgent(model=_blocking_extractor_model)
    reviewer_model = _ScriptedReviewerModel([_json_decision("add", rationale="async add")])
    reviewer = ReviewerAgent(model=reviewer_model, search_fn=lambda _q: store.search(_q, top_k=5))
    pipeline = MemoryPipeline(extractor, reviewer, store, metrics=metrics_seen.append)

    start = time.perf_counter()
    handle = pipeline.run_after_turn("t-async", "turn text", session_id="s1", sync=False)
    returned_after = time.perf_counter() - start

    assert returned_after < 1.0  # the calling (turn) thread was NOT blocked
    assert isinstance(handle, Future)
    assert not handle.done()  # the flow is still waiting on the blocked model

    release.set()
    result = handle.result(timeout=10.0)  # acceptance: completes within 10s (async)
    assert result.turn_id == "t-async"
    assert [action.action for action in result.actions] == ["add"]
    assert len(result.applied) == 1
    assert store.get(result.applied[0].entry_id) is not None
    # metrics are recorded by the worker thread once the flow resolved
    assert len(metrics_seen) == 1
    assert metrics_seen[0]["async"] is True


@pytest.mark.integration
def test_metrics_recorded(tmp_path: Path) -> None:
    """One metrics summary per run: drafts, actions by type, applied, elapsed."""
    store = _make_store(tmp_path)
    extractor = ExtractorAgent(model=lambda _prompt: json.dumps([
        {"type": "User", "text": "User drinks coffee every morning", "confidence": 0.8},
        {"type": "Feedback", "text": "Feedback: keep answers short", "confidence": 0.7},
    ]))
    reviewer_model = _ScriptedReviewerModel([
        _json_decision("add", rationale="new user habit"),
        _json_decision("skip", target_id=None, rationale="transient"),
    ])
    reviewer = ReviewerAgent(model=reviewer_model, search_fn=lambda _q: store.search(_q, top_k=5))
    recorder: list[dict] = []
    pipeline = MemoryPipeline(extractor, reviewer, store, metrics=recorder.append)

    result = pipeline.run_after_turn("t1", "two facts turn", session_id="s2", context="work")

    assert len(recorder) == 1
    metrics = recorder[0]
    assert metrics["turn_id"] == "t1"
    assert metrics["session_id"] == "s2"
    assert metrics["context"] == "work"
    assert metrics["drafts"] == 2
    assert metrics["actions_by_type"] == {"add": 1, "update": 0, "skip": 1, "conflict": 0}
    assert metrics["applied"] == 1
    assert metrics["async"] is False
    assert metrics["elapsed_ms"] >= 0
    assert metrics["applied"] == len(result.applied)
    assert len(store.list(context="work")) == 1  # only the add was written


@pytest.mark.integration
def test_conflict_triggers_resolution_flow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """First-pass conflict -> second reviewer pass with conflict context.

    Scenario 1: the resolution pass resolves the conflict -> the draft is
    applied. Scenario 2: the resolution pass still conflicts -> the draft is
    dropped (documented termination rule: at most ONE resolution pass).
    """
    caplog.set_level(logging.INFO, logger=_PIPELINE_LOGGER)
    draft_text = "Production deployments are frozen between 18:00 and 06:00"
    stored_text = "Production deploys run through GitHub Actions"
    turn_text = "Deployments to production are frozen between 18:00 and 06:00 from now on"

    def _extractor() -> ExtractorAgent:
        return ExtractorAgent(model=lambda _prompt: json.dumps([
            {
                "type": "Project",
                "text": draft_text,
                "proposed_source": "user-explicit",
                "confidence": 0.9,
            }
        ]))

    # ── scenario 1: the conflict RESOLVES to add in the second pass ──────
    store1 = _make_store(tmp_path / "one")
    existing = store1.add(stored_text, type="Project")
    model1 = _ScriptedReviewerModel([
        _json_decision("conflict", target_id=existing.id,
                       rationale="contradicts the stored deploy policy"),
        _json_decision("add", rationale="user overrode the freeze policy"),
    ])
    reviewer1 = ReviewerAgent(model=model1, search_fn=lambda _q: store1.search(_q, top_k=5))
    pipeline1 = MemoryPipeline(_extractor(), reviewer1, store1)

    result1 = pipeline1.run_after_turn("t1", turn_text, session_id="s1")

    assert model1.calls == 2  # first pass + conflict re-review
    assert "-conflict-pass" in model1.prompts[1]  # wrapper draft id marks the re-review
    assert existing.id in model1.prompts[1]  # conflict context: the conflicting entry...
    assert stored_text in model1.prompts[1]  # ...including its text, is in the prompt
    assert [action.action for action in result1.actions] == ["add"]
    assert len(result1.applied) == 1
    new_entry = store1.get(result1.applied[0].entry_id)
    assert new_entry is not None
    assert new_entry.text == draft_text
    assert new_entry.id != existing.id
    assert "second reviewer pass" in caplog.text
    assert "user overrode the freeze policy" in caplog.text

    # ── scenario 2: the conflict PERSISTS -> draft dropped, nothing written ──
    store2 = _make_store(tmp_path / "two")
    existing2 = store2.add(stored_text, type="Project")
    model2 = _ScriptedReviewerModel([
        _json_decision("conflict", target_id=existing2.id,
                       rationale="contradicts the stored deploy policy"),
        _json_decision("conflict", target_id=existing2.id, rationale="still contradicts"),
    ])
    reviewer2 = ReviewerAgent(model=model2, search_fn=lambda _q: store2.search(_q, top_k=5))
    pipeline2 = MemoryPipeline(_extractor(), reviewer2, store2)

    result2 = pipeline2.run_after_turn("t2", turn_text, session_id="s1")

    assert model2.calls == 2  # still exactly one resolution pass
    assert [action.action for action in result2.actions] == ["conflict"]
    assert result2.applied == []
    assert len(store2.list()) == 1  # the store was never written
    assert "still conflicts after the resolution pass" in caplog.text
    assert "dropping" in caplog.text
    assert "still contradicts" in caplog.text  # both rationales are in the drop log
