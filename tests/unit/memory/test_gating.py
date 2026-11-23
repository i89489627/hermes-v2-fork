"""P4.T2 acceptance: CBDCGate — one batched call, budgets, timeout fallback.

Covers the plan Phase 4 gate contract over injected fake models (single-model
architecture — never called live): use/verify/ignore decisions, the
deterministic budget enforcement (<= budget_claims claims for ``use``, <=
budget_chars characters for everything injected — including verify-flagged
entries), the ONE batched main-model call per turn (all top-3 candidates in
a single prompt, top-3 BM25 pre-filter), and the 3.5s-class hard timeout
falling back to trust-score-only gating (plan Section 5.5).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from hermes.memory import gating as hermes_memory_gating
from hermes.memory.entry import MemoryEntry
from hermes.memory.gating import GATE_TOP_K, CBDCGate, GateResult
from hermes.memory.retrieval import RetrievedEntry

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _entry(entry_id: str, text: str, trust: float) -> MemoryEntry:
    return MemoryEntry(id=entry_id, type="User", text=text, created_at=_NOW, trust_score=trust)


def _cand(entry_id: str, text: str, trust: float, score: float) -> RetrievedEntry:
    return RetrievedEntry(entry=_entry(entry_id, text, trust), score=score)


def _decision_list(**by_id: str) -> str:
    """Valid gate-model output: one decision object per candidate id."""
    return json.dumps([
        {"candidate_id": candidate_id, "decision": decision, "reason": f"because {decision}"}
        for candidate_id, decision in by_id.items()
    ])


class _CountingModel:
    """Fake gate model: records prompts, always returns the same completion."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.mark.unit
def test_use_decision_for_high_trust_relevant() -> None:
    """A high-trust, relevant candidate gets 'use' and lands in the payload."""
    model = _CountingModel(_decision_list(c1="use", c2="ignore"))
    gate = CBDCGate(gate_model=model)
    candidates = [
        _cand("c1", "User prefers dark mode IDEs", trust=0.95, score=5.0),
        _cand("c2", "Grandma's apple pie uses cinnamon", trust=0.95, score=1.0),
    ]

    result = gate.gate(candidates, context="global", turn_text="which theme should I use?")

    assert isinstance(result, GateResult)
    assert result.fallback_used is False
    assert len(result.decisions) == 2
    top = result.decisions[0]
    assert top.candidate_id == "c1"
    assert top.decision == "use"
    assert top.reason == "because use"
    assert top.needs_verification is False
    assert top.text == "User prefers dark mode IDEs"
    assert result.decisions[1].decision == "ignore"
    assert [gated.candidate_id for gated in result.injected] == ["c1"]
    # acceptance output shape: {decision, reason, candidate_id} (+ payload fields)
    payload: dict[str, Any] = top.to_dict()
    assert {
        "candidate_id",
        "decision",
        "reason",
        "text",
        "needs_verification",
    } == set(payload)
    # the batched prompt carried the turn text and every candidate
    assert model.calls == 1
    assert "which theme should I use?" in model.prompts[0]
    assert '"candidate_id": "c1"' in model.prompts[0]
    assert '"candidate_id": "c2"' in model.prompts[0]


@pytest.mark.unit
def test_verify_decision_for_medium_trust() -> None:
    """Medium-trust candidate: 'verify' -> injected flagged, claims budget intact.

    Plan-strict budget rule: only ``use`` consumes claims. With
    budget_claims=1 the verify entry AND the use entry are both injected
    while the third use decision is dropped — proving verify did not take a
    claim slot.
    """
    model = _CountingModel(_decision_list(c1="verify", c2="use", c3="use"))
    gate = CBDCGate(gate_model=model, budget_claims=1)
    candidates = [
        _cand("c1", "Team standup moved to 9:15", trust=0.55, score=3.0),
        _cand("c2", "User prefers pytest for tests", trust=0.95, score=2.0),
        _cand("c3", "Weekly sync notes from the offsite", trust=0.95, score=1.0),
    ]

    result = gate.gate(candidates, turn_text="when is the standup now?")

    verify_entry = next(d for d in result.decisions if d.candidate_id == "c1")
    assert verify_entry.decision == "verify"
    assert verify_entry.needs_verification is True
    assert verify_entry.reason == "because verify"
    injected_ids = [gated.candidate_id for gated in result.injected]
    assert injected_ids == ["c1", "c2"]  # verify kept, third use dropped
    dropped = result.decisions[2]
    assert dropped.candidate_id == "c3"
    assert dropped.decision == "ignore"
    assert dropped.reason == "budget: claims budget exhausted"
    assert [gated.decision for gated in result.injected].count("use") == 1


@pytest.mark.unit
def test_ignore_decision_for_irrelevant() -> None:
    """An irrelevant candidate gets 'ignore' and never reaches the payload."""
    model = _CountingModel(_decision_list(c1="ignore", c2="use"))
    gate = CBDCGate(gate_model=model)
    candidates = [
        _cand("c1", "Random gossip about reality TV", trust=0.5, score=4.0),
        _cand("c2", "Deploy pipeline uses GitHub Actions", trust=0.95, score=1.0),
    ]

    result = gate.gate(candidates, turn_text="how do deploys work?")

    ignored = result.decisions[0]
    assert ignored.candidate_id == "c1"
    assert ignored.decision == "ignore"
    assert ignored.needs_verification is False
    assert ignored.reason == "because ignore"
    assert [gated.candidate_id for gated in result.injected] == ["c2"]


@pytest.mark.unit
def test_budget_claims_enforced() -> None:
    """More 'use' decisions than budget_claims: overflow downgraded to ignore."""
    model = _CountingModel(_decision_list(c1="use", c2="use", c3="use"))
    gate = CBDCGate(gate_model=model, budget_claims=2)
    candidates = [
        _cand("c1", "fact one", trust=0.95, score=3.0),
        _cand("c2", "fact two", trust=0.95, score=2.0),
        _cand("c3", "fact three", trust=0.95, score=1.0),
    ]

    result = gate.gate(candidates, turn_text="turn")

    assert [gated.candidate_id for gated in result.injected] == ["c1", "c2"]
    overflow = result.decisions[2]
    assert overflow.decision == "ignore"
    assert overflow.reason == "budget: claims budget exhausted"

    # boundary: the plan default budget (3 claims) exactly fits the top-3
    default_gate = CBDCGate(gate_model=_CountingModel(_decision_list(c1="use", c2="use", c3="use")))
    boundary = default_gate.gate(list(candidates), turn_text="turn")
    assert [gated.candidate_id for gated in boundary.injected] == ["c1", "c2", "c3"]


@pytest.mark.unit
def test_budget_chars_enforced() -> None:
    """Cumulative injected text size is capped at budget_chars (1200 default)."""
    model = _CountingModel(_decision_list(c1="use", c2="use"))
    gate = CBDCGate(gate_model=model, budget_chars=1000)
    candidates = [
        _cand("c1", "a" * 600, trust=0.95, score=2.0),
        _cand("c2", "b" * 600, trust=0.95, score=1.0),
    ]

    result = gate.gate(candidates, turn_text="turn")

    assert [gated.candidate_id for gated in result.injected] == ["c1"]
    overflow = result.decisions[1]
    assert overflow.decision == "ignore"
    assert overflow.reason == "budget: character budget exhausted"
    assert sum(len(gated.text) for gated in result.injected) <= 1000


@pytest.mark.unit
def test_single_batched_call_per_turn() -> None:
    """One model call classifies ALL top-3 candidates (never 3 sequential)."""
    model = _CountingModel(_decision_list(c1="use", c2="verify", c3="ignore"))
    gate = CBDCGate(gate_model=model)
    candidates = [
        _cand("c1", "fact one", trust=0.95, score=3.0),
        _cand("c2", "fact two", trust=0.6, score=2.0),
        _cand("c3", "fact three", trust=0.4, score=1.0),
    ]

    result = gate.gate(candidates, turn_text="turn text")

    assert model.calls == 1  # exactly ONE batched call for the whole turn
    assert all(
        f'"candidate_id": "{candidate_id}"' in model.prompts[0]
        for candidate_id in ("c1", "c2", "c3")
    )
    assert [d.candidate_id for d in result.decisions] == ["c1", "c2", "c3"]

    # BM25 pre-filter: only the top-3 of a larger candidate pool are gated,
    # still within the same single call for that turn
    bigger_pool = [
        *candidates,
        _cand("c4", "fact four", trust=0.9, score=9.0),
        _cand("c5", "fact five", trust=0.9, score=8.0),
    ]
    calls_before = model.calls
    result2 = gate.gate(bigger_pool, turn_text="turn text")
    assert model.calls - calls_before == 1
    assert len(result2.decisions) == GATE_TOP_K
    assert [d.candidate_id for d in result2.decisions] == ["c4", "c5", "c1"]  # by score


@pytest.mark.unit
def test_budget_boundary_exactly_3_claims_1200_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget boundaries are exact (plan 2.5 defaults: 3 claims / 1200 chars).

    Mutation-gate boundary test (P20.T6, plan Section 4.7): the defaults
    themselves are pinned so a mutated constant dies, and both sides of each
    boundary are exercised:

    - claims: exactly 3 ``use`` decisions at the default ``budget_claims=3``
      are ALL injected; a 4th ``use`` (the gate's top-3 pre-filter widened
      via the module constant so the 4th candidate reaches the deterministic
      budget pass) is downgraded with the claims-exhausted reason;
    - chars: cumulative injected text of EXACTLY 1200 chars (600+600) at the
      default ``budget_chars=1200`` is fully injected; the 1201st char (a
      third 1-char candidate) is downgraded with the chars-exhausted reason.
    """
    # ── claims: exactly 3 use decisions accepted at the default budget ────
    model = _CountingModel(_decision_list(c1="use", c2="use", c3="use"))
    gate = CBDCGate(gate_model=model)  # plan defaults: budget_claims=3
    assert gate._budget_claims == 3
    candidates = [
        _cand("c1", "short fact one", trust=0.95, score=3.0),
        _cand("c2", "short fact two", trust=0.95, score=2.0),
        _cand("c3", "short fact three", trust=0.95, score=1.0),
    ]
    result = gate.gate(candidates, turn_text="turn")
    assert [gated.candidate_id for gated in result.injected] == ["c1", "c2", "c3"]
    assert [gated.decision for gated in result.injected] == ["use", "use", "use"]
    assert sum(1 for gated in result.injected if gated.decision == "use") == 3
    assert all("budget:" not in gated.reason for gated in result.decisions)

    # ── claims: the 4th use decision is rejected ──────────────────────────
    # gate() pre-filters to the plan's top-3; widen the pre-filter constant
    # so a 4th candidate reaches the deterministic budget pass, which must
    # downgrade it (claims_used >= budget_claims).
    monkeypatch.setattr(hermes_memory_gating, "GATE_TOP_K", 4)
    model4 = _CountingModel(_decision_list(c1="use", c2="use", c3="use", c4="use"))
    gate4 = CBDCGate(gate_model=model4)
    four = [
        _cand("c1", "fact one", trust=0.95, score=4.0),
        _cand("c2", "fact two", trust=0.95, score=3.0),
        _cand("c3", "fact three", trust=0.95, score=2.0),
        _cand("c4", "fact four", trust=0.95, score=1.0),
    ]
    result4 = gate4.gate(four, turn_text="turn")
    injected4 = [gated.candidate_id for gated in result4.injected]
    assert injected4 == ["c1", "c2", "c3"]
    overflow = result4.decisions[3]
    assert overflow.candidate_id == "c4"
    assert overflow.decision == "ignore"
    assert overflow.reason == "budget: claims budget exhausted"
    assert sum(1 for gated in result4.injected if gated.decision == "use") == 3

    # ── chars: exactly 1200 accepted, the 1201st char rejected ────────────
    model_chars = _CountingModel(_decision_list(c1="use", c2="use", c3="use"))
    gate_chars = CBDCGate(gate_model=model_chars)  # plan defaults: budget_chars=1200
    assert gate_chars._budget_chars == 1200
    chars_candidates = [
        _cand("c1", "a" * 600, trust=0.95, score=3.0),
        _cand("c2", "b" * 600, trust=0.95, score=2.0),  # cumulative: exactly 1200
        _cand("c3", "c", trust=0.95, score=1.0),  # the 1201st char
    ]
    result_chars = gate_chars.gate(chars_candidates, turn_text="turn")
    injected_chars = [gated.candidate_id for gated in result_chars.injected]
    assert injected_chars == ["c1", "c2"]
    assert sum(len(gated.text) for gated in result_chars.injected) == 1200
    overflow_chars = result_chars.decisions[2]
    assert overflow_chars.candidate_id == "c3"
    assert overflow_chars.decision == "ignore"
    assert overflow_chars.reason == "budget: character budget exhausted"


@pytest.mark.unit
def test_timeout_falls_back_to_trust_only() -> None:
    """Gate timeout -> trust-only fallback (plan 5.5); the slow call is discarded."""
    calls: list[str] = []

    def _slow_model(_prompt: str) -> str:
        calls.append(_prompt)
        time.sleep(1.0)  # far beyond the gate's timeout
        return _decision_list(c1="use", c2="use", c3="use")

    gate = CBDCGate(gate_model=_slow_model, timeout_seconds=0.2)
    candidates = [
        _cand("c1", "high trust fact", trust=0.95, score=5.0),
        _cand("c2", "medium trust fact", trust=0.6, score=4.0),
        _cand("c3", "low trust fact", trust=0.4, score=3.0),
    ]

    start = time.perf_counter()
    result = gate.gate(candidates, turn_text="turn")
    wall = time.perf_counter() - start

    assert result.fallback_used is True
    assert wall < 0.9  # returned on the 0.2s timeout, NOT after the 1s model
    assert result.elapsed_ms < 3500.0  # inside the plan's hard gate budget
    # trust-only gating: inject only trust > 0.7 candidates, up to budget
    assert [gated.candidate_id for gated in result.injected] == ["c1"]
    assert result.decisions[0].decision == "use"
    assert "trust-only fallback" in result.decisions[0].reason
    assert result.decisions[1].decision == "ignore"
    assert "trust-only fallback" in result.decisions[1].reason
    assert result.decisions[2].decision == "ignore"
    # the timed-out model result is discarded: it never shaped the decisions
    assert len(calls) == 1
    assert "because use" not in [d.reason for d in result.decisions]
