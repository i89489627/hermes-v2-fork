"""P10.T3 acceptance: gate GEPA adapter (plan Phase 10).

``evaluate`` returns the mean per-example ``gating_precision`` over the
"use" decisions plus a feedback digest naming the false-positive firings and
the missed should-use candidates (coverage is precision-only by convention).
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.optimization.adapters.gate_adapter import GateAdapter


def _example(example_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """One labeled gating eval record with the fixture-set record shape."""
    return {
        "id": example_id,
        "input": {
            "turn": f"turn for {example_id}",
            "candidates": candidates,
        },
        "expected_output": {
            "decisions": [
                {"id": candidate["id"], "action": candidate["expected_action"]}
                for candidate in candidates
            ],
            "budget": {"claims_used": len(candidates), "chars_used": 100},
        },
        "context": "gating case",
    }


def _candidate(candidate_id: str, expected_action: str) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "text": f"memory text for {candidate_id}",
        "trust": 0.9,
        "age_days": 3,
        "expected_action": expected_action,
    }


EVAL_SET: list[dict[str, Any]] = [
    _example("gate-001", [_candidate("pref-oat", "use")]),
    _example(
        "gate-002",
        [_candidate("pref-oat", "use"), _candidate("old-job", "ignore")],
    ),
]


@pytest.mark.unit
def test_evaluate_returns_score_and_feedback() -> None:
    """Correct "use" decisions score 1.0; spurious firings cost precision."""
    adapter = GateAdapter(
        lambda prompt, example: {
            "decisions": [
                {"id": decision["id"], "action": decision["action"]}
                for decision in example["expected_output"]["decisions"]
            ]
        }
    )
    score, feedback = adapter.evaluate("gate candidates", EVAL_SET)
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
    assert "gate score 1.000 over 2 examples: 0 failed, 2 passed" in feedback
    assert "Failure patterns" not in feedback

    # An over-eager gate fires "use" on everything: one false positive.
    overeager = GateAdapter(lambda prompt, example: {
        "decisions": [
            {"id": candidate["id"], "action": "use"}
            for candidate in example["input"]["candidates"]
        ]
    })
    score, feedback = overeager.evaluate("gate candidates", EVAL_SET)
    # gate-001: precision 1/1 = 1.0; gate-002: fired on old-job too -> 1/2 = 0.5.
    assert score == pytest.approx(0.75)
    assert "gate score 0.750 over 2 examples: 1 failed, 1 passed" in feedback
    assert "Failure patterns:" in feedback
    assert "- gate-002 (score 0.500)" in feedback
    assert "fired on candidates that should be ignored: old-job" in feedback

    # A gate that never fires is vacuously precise (precision-only convention);
    # the missed should-use candidates are described but not penalized.
    timid = GateAdapter(lambda prompt, example: {"decisions": []})
    score, feedback = timid.evaluate("gate candidates", EVAL_SET)
    assert score == pytest.approx(1.0)
    assert "0 failed, 2 passed" in feedback
