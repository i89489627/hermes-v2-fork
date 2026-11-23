"""P20.T1 chaos: main-model API failure mid-request and at startup (plan 5.5).

The main-model provider dies MID-REQUEST (the pipeline sits between the
extraction and review calls when the API goes away) or is unreachable AT
STARTUP (the first gate call of a turn). Both must degrade gracefully — the
turn completes, nothing crashes, nothing half-written (plan Section 5.5:
extraction skipped and retried next turn, reviewer failures degrade to a
conservative skip, gating falls back to trust-score-only with no LLM).

- ``test_api_failure_mid_request_degrades_gracefully`` — an always-dead
  provider degrades the whole run to an empty :class:`PipelineResult`; a
  provider that dies between the extractor's call and the reviewer's call
  degrades the extracted draft to a ``skip``. The store ends untouched in
  both runs (no partial write, no data loss, no exception escapes).
- ``test_api_unreachable_at_startup`` — the CBDCGate's very first model call
  fails: ``fallback_used=True`` and the injection payload is still composed
  from trust scores alone (high-trust candidates injected, low-trust ones
  ignored, no exception escapes the gate).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.gating import CBDCGate
from hermes.memory.pipeline import MemoryPipeline, PipelineResult
from hermes.memory.retrieval import RetrievedEntry
from hermes.memory.reviewer import ReviewerAgent

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)

#: Valid extractor completion for the scripted "first call succeeds" model.
_EXTRACT_JSON = json.dumps(
    [
        {
            "text": "chaos mid-request fact from the turn",
            "type": "Feedback",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ]
)

_MID_REQUEST_FACT = "chaos mid-request fact from the turn"


def _candidate(entry_id: str, text: str, trust: float, score: float) -> RetrievedEntry:
    """One gated candidate with an explicit trust score and BM25 ranking."""
    entry = MemoryEntry(
        id=entry_id, type="User", text=text, created_at=_NOW, trust_score=trust
    )
    return RetrievedEntry(entry=entry, score=score)


def _candidates() -> list[RetrievedEntry]:
    """Top-3 gate candidates: two above, one below the 0.7 fallback bar."""
    return [
        _candidate("c1", "chaos high-trust fact one", trust=0.95, score=5.0),
        _candidate("c2", "chaos low-trust fact two", trust=0.40, score=4.0),
        _candidate("c3", "chaos high-trust fact three", trust=0.80, score=1.0),
    ]


@pytest.mark.chaos
def test_api_failure_mid_request_degrades_gracefully(
    chaos_store, failing_model
) -> None:
    """A dying provider degrades the run; the store ends untouched."""
    store = chaos_store

    # Provider unreachable from the very first call: extraction is skipped
    # (plan 5.5, "retry next turn") and the run returns an empty result.
    dead = failing_model()
    pipeline = MemoryPipeline(
        extractor=ExtractorAgent(model=dead),
        reviewer=ReviewerAgent(model=dead, search_fn=lambda query: []),
        store=store,
    )
    result = pipeline.run_after_turn(
        "turn-dead-provider", "user tells a fact while the provider is down"
    )
    assert isinstance(result, PipelineResult)
    assert result.turn_id == "turn-dead-provider"
    assert result.drafts == []
    assert result.actions == []
    assert result.applied == []
    assert dead.calls == 1  # exactly one (batched) provider attempt, no retry storm
    assert dead.failures == 1

    # API failure MID-REQUEST: the extractor's call succeeds, the provider
    # dies before the reviewer's call. The draft degrades to a conservative
    # skip — never auto-added unaudited (plan 5.5).
    mid_request = failing_model(fail_after=1, success_response=_EXTRACT_JSON)
    pipeline = MemoryPipeline(
        extractor=ExtractorAgent(model=mid_request),
        reviewer=ReviewerAgent(model=mid_request, search_fn=lambda query: []),
        store=store,
    )
    result = pipeline.run_after_turn("turn-mid-request", "user states a reviewable fact")
    assert [draft.text for draft in result.drafts] == [_MID_REQUEST_FACT]
    assert mid_request.calls == 2  # extraction + the doomed review call
    assert mid_request.failures == 1
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action == "skip"
    assert "reviewer failed" in action.rationale
    assert "main model provider unreachable" in action.rationale
    assert result.applied == []  # the degraded run wrote NOTHING

    # No data loss, no partial write: the store holds exactly what it held.
    assert store.list() == []
    assert store.git.is_clean()


@pytest.mark.chaos
def test_api_unreachable_at_startup(chaos_store, failing_model) -> None:
    """First gate call fails -> trust-only fallback, injection still composed."""
    model = failing_model()
    gate = CBDCGate(gate_model=model)

    result = gate.gate(_candidates(), context="global", turn_text="which fact applies here?")

    # No exception escaped; the failure is reported via the fallback flag.
    assert result.fallback_used is True
    assert model.calls == 1  # one batched attempt, no retries (check 3 contract)

    # Trust-only gating: candidates above the 0.7 bar are injected as "use",
    # the rest ignored — the injection payload is still composed.
    injected = result.injected
    assert [gated.candidate_id for gated in injected] == ["c1", "c3"]
    assert all(gated.decision == "use" for gated in injected)
    assert [gated.text for gated in injected] == [
        "chaos high-trust fact one",
        "chaos high-trust fact three",
    ]
    assert result.decisions[1].candidate_id == "c2"
    assert result.decisions[1].decision == "ignore"  # trust 0.40 <= 0.7
    assert all("trust-only fallback" in gated.reason for gated in result.decisions)
    assert all(gated.needs_verification is False for gated in result.decisions)
