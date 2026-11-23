"""P4.T2 property: gate budget invariants for arbitrary candidates and outputs.

For arbitrary candidate sets (0-8 entries, random scores/trust/texts) and
arbitrary gate-model behaviors (valid decision lists covering every action,
missing candidates, unknown ids, duplicates, malformed JSON, JSON objects,
empty output, fenced output), :meth:`CBDCGate.gate` never raises and — on
the derived injection payload (``GateResult.injected``) — always satisfies
the plan P4.T2 budget contract:

- the injected count never exceeds the budget (top-3 gated, 3 claims);
- every injected entry carries a ``use`` decision or is verify-flagged —
  never an ``ignore`` (plan: "Inject only 'use' items within budget"; verify
  entries are injected with the needs-verification marker);
- injected texts cumulatively stay within the character budget;
- the decisions list covers exactly the gated top-3 candidate ids.

The fallback path (plan 5.5, malformed/failed model output) obeys the same
invariants.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes.memory.entry import MemoryEntry
from hermes.memory.gating import GATE_TOP_K, CBDCGate, GateResult
from hermes.memory.retrieval import RetrievedEntry

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

#: Plan defaults exercised here (budget_claims=3, budget_chars=1200).
_BUDGET_CLAIMS = 3
_BUDGET_CHARS = 1200

_COUNT = st.integers(min_value=0, max_value=8)
_SCORE = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_TRUST = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_TEXT = st.one_of(st.text(max_size=400), st.just(""), st.just("x" * 2000))

#: Per-candidate model decision: valid literal, invalid literal or omitted.
_DECISION = st.one_of(
    st.sampled_from(["use", "verify", "ignore"]),
    st.just("bogus-decision"),
    st.none(),
)

#: Global output quirks (fallback triggers: malformed / json-object / empty).
_QUIRKS = st.sampled_from([
    "clean",
    "malformed",
    "json-object",
    "empty",
    "fenced",
    "unknown-id",
    "duplicate",
])


def _candidate_list(count: int, scores: list[float], trusts: list[float], texts: list[str]):
    """Deterministic candidate list of ``count`` unique-id entries."""
    return [
        RetrievedEntry(
            entry=MemoryEntry(
                id=f"prop-{index}",
                type="User",
                text=texts[index % len(texts)],
                created_at=_NOW,
                trust_score=trusts[index % len(trusts)],
            ),
            score=scores[index % len(scores)],
        )
        for index in range(count)
    ]


def _model_for(
    decisions: list[str | None], quirk: str, candidates: list[RetrievedEntry]
) -> Callable[[str], str]:
    """Deterministic fake gate model for the drawn decisions + quirk.

    The fake echoes the candidate ids it would see in the batched prompt
    (input order is fine — the gate re-orders by score). ``None`` omits the
    candidate from the decision list; an invalid literal is emitted as-is.
    """

    def _model(_prompt: str) -> str:
        items: list[dict[str, Any]] = []
        for index, decision in enumerate(decisions):
            if index >= len(candidates) or decision is None:
                continue
            items.append({
                "candidate_id": candidates[index].entry.id,
                "decision": decision,
                "reason": f"model says {decision}",
            })
        if quirk == "unknown-id":
            items.append({"candidate_id": "zz-not-gated", "decision": "use", "reason": "ghost"})
        if quirk == "duplicate" and items:
            items.append(dict(items[0]))
        raw = json.dumps(items)
        if quirk == "fenced":
            return f"```json\n{raw}\n```"
        if quirk == "malformed":
            return "sorry, no JSON for you"
        if quirk == "json-object":
            return json.dumps({"decisions": items})
        if quirk == "empty":
            return ""
        return raw

    return _model


def _expected_top3(candidates: list[RetrievedEntry]) -> list[str]:
    """Recompute the gate's deterministic top-3 (score desc, id asc)."""
    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.entry.id),
    )
    return [candidate.entry.id for candidate in ranked[:GATE_TOP_K]]


def _assert_invariants(result: GateResult, expected_ids: list[str]) -> None:
    """The budget/decision contract every gate run must satisfy."""
    assert [d.candidate_id for d in result.decisions] == expected_ids
    injected = result.injected
    # budget: injected count never exceeds the claims budget (top-3 cap)
    assert len(injected) <= _BUDGET_CLAIMS
    use_count = sum(1 for entry in injected if entry.decision == "use")
    assert use_count <= _BUDGET_CLAIMS
    # chars budget covers everything injected (use AND verify-flagged)
    assert sum(len(entry.text) for entry in injected) <= _BUDGET_CHARS
    # only gated candidates, never 'ignore', marker only on verify
    gated_ids = set(expected_ids)
    for entry in injected:
        assert entry.candidate_id in gated_ids
        assert entry.decision != "ignore"
        assert entry.needs_verification == (entry.decision == "verify")


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    count=_COUNT,
    scores=st.lists(_SCORE, min_size=1, max_size=8),
    trusts=st.lists(_TRUST, min_size=1, max_size=8),
    texts=st.lists(_TEXT, min_size=1, max_size=8),
    decisions=st.lists(_DECISION, min_size=0, max_size=8),
    quirk=_QUIRKS,
)
def test_injected_count_never_exceeds_budget(
    count: int,
    scores: list[float],
    trusts: list[float],
    texts: list[str],
    decisions: list[str | None],
    quirk: str,
) -> None:
    """Arbitrary candidates + arbitrary model outputs stay within the budget."""
    candidates = _candidate_list(count, scores, trusts, texts)
    gate = CBDCGate(
        gate_model=_model_for(decisions, quirk, candidates),
        budget_claims=_BUDGET_CLAIMS,
        budget_chars=_BUDGET_CHARS,
        timeout_seconds=3.5,
    )
    result = gate.gate(candidates, turn_text="property turn")
    _assert_invariants(result, _expected_top3(candidates))


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    count=_COUNT,
    scores=st.lists(_SCORE, min_size=1, max_size=8),
    trusts=st.lists(_TRUST, min_size=1, max_size=8),
    texts=st.lists(_TEXT, min_size=1, max_size=8),
    decisions=st.lists(_DECISION, min_size=0, max_size=8),
    quirk=_QUIRKS,
)
def test_all_injected_have_use_decision(
    count: int,
    scores: list[float],
    trusts: list[float],
    texts: list[str],
    decisions: list[str | None],
    quirk: str,
) -> None:
    """Every injected entry is a 'use' or a verify-flagged entry — no 'ignore'.

    (Task wording: "every injected has decision 'use' (or verify flagged —
    property asserts no 'ignore' is injected and count within budget)".)
    """
    candidates = _candidate_list(count, scores, trusts, texts)
    gate = CBDCGate(
        gate_model=_model_for(decisions, quirk, candidates),
        budget_claims=_BUDGET_CLAIMS,
        budget_chars=_BUDGET_CHARS,
        timeout_seconds=3.5,
    )
    result = gate.gate(candidates, turn_text="property turn")
    injected = result.injected
    assert len(injected) <= _BUDGET_CLAIMS
    for entry in injected:
        assert entry.decision == "use" or entry.needs_verification
    assert sum(len(entry.text) for entry in injected) <= _BUDGET_CHARS
    # fallback runs stay on-contract too (with candidates: a broken model
    # output always degrades to the trust-only fallback; an empty candidate
    # set short-circuits before any model call, so no fallback happened)
    if quirk in {"malformed", "json-object", "empty"} and count > 0:
        assert result.fallback_used is True
        for entry in result.decisions:
            assert "trust-only fallback" in entry.reason or entry.reason.startswith("budget")


# ── P20.T2: the same budget/decision invariants at 1000 examples ─────────────


@pytest.mark.property
@settings(max_examples=1000, deadline=None)
@given(
    count=_COUNT,
    scores=st.lists(_SCORE, min_size=1, max_size=8),
    trusts=st.lists(_TRUST, min_size=1, max_size=8),
    texts=st.lists(_TEXT, min_size=1, max_size=8),
    decisions=st.lists(_DECISION, min_size=0, max_size=8),
    quirk=_QUIRKS,
)
def test_injected_never_exceeds_budget(
    count: int,
    scores: list[float],
    trusts: list[float],
    texts: list[str],
    decisions: list[str | None],
    quirk: str,
) -> None:
    """P20.T2: the injected payload never exceeds the budgets, for any input.

    Same invariant wording as :func:`test_injected_count_never_exceeds_budget`
    (the P4.T2 test), at the plan P20.T2-required 1000-example depth: the
    injection payload covers only gated candidates, carries at most
    ``budget_claims`` entries and stays within ``budget_chars`` — under valid
    model output, malformed output (trust-only fallback) and everything in
    between.
    """
    candidates = _candidate_list(count, scores, trusts, texts)
    gate = CBDCGate(
        gate_model=_model_for(decisions, quirk, candidates),
        budget_claims=_BUDGET_CLAIMS,
        budget_chars=_BUDGET_CHARS,
        timeout_seconds=3.5,
    )
    result = gate.gate(candidates, turn_text="p20 property turn")
    _assert_invariants(result, _expected_top3(candidates))


@pytest.mark.property
@settings(max_examples=1000, deadline=None)
@given(
    count=_COUNT,
    scores=st.lists(_SCORE, min_size=1, max_size=8),
    trusts=st.lists(_TRUST, min_size=1, max_size=8),
    texts=st.lists(_TEXT, min_size=1, max_size=8),
    decisions=st.lists(_DECISION, min_size=0, max_size=8),
    quirk=_QUIRKS,
)
def test_all_injected_have_use_decision_p20(
    count: int,
    scores: list[float],
    trusts: list[float],
    texts: list[str],
    decisions: list[str | None],
    quirk: str,
) -> None:
    """P20.T2: every injected entry is 'use' or verify-flagged — never 'ignore'.

    Same invariant wording as :func:`test_all_injected_have_use_decision`
    (the P4.T2 test), at the plan P20.T2-required 1000-example depth.
    """
    candidates = _candidate_list(count, scores, trusts, texts)
    gate = CBDCGate(
        gate_model=_model_for(decisions, quirk, candidates),
        budget_claims=_BUDGET_CLAIMS,
        budget_chars=_BUDGET_CHARS,
        timeout_seconds=3.5,
    )
    result = gate.gate(candidates, turn_text="p20 property turn")
    injected = result.injected
    assert len(injected) <= _BUDGET_CLAIMS
    for entry in injected:
        assert entry.decision == "use" or entry.needs_verification
        assert entry.candidate_id in set(_expected_top3(candidates))
    assert sum(len(entry.text) for entry in injected) <= _BUDGET_CHARS
