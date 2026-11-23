"""P10.T3 acceptance: reviewer GEPA adapter (plan Phase 10).

``evaluate`` combines action accuracy (plain examples) with conflict-detection
accuracy (examples whose expected verdict is ``conflict``) into one score and
a feedback digest naming every missed conflict / wrong action.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.optimization.adapters.reviewer_adapter import ReviewerAdapter


def _example(example_id: str, action: str, target: str) -> dict[str, Any]:
    """One labeled reviewer eval record with the fixture-set record shape."""
    return {
        "id": example_id,
        "input": {
            "draft_facts": [f"{example_id} draft fact"],
            "existing_entries": [{"id": target, "text": f"stored text for {target}"}],
        },
        "expected_output": {"action": action, "target": target, "reason": "labeled"},
        "context": "review verdict case",
    }


EVAL_SET: list[dict[str, Any]] = [
    _example("rev-001", "add", "entry-new"),
    _example("rev-002", "update", "entry-pref"),
    _example("rev-003", "conflict", "entry-rent"),
    _example("rev-004", "conflict", "entry-job"),
]


@pytest.mark.unit
def test_evaluate_returns_score_and_feedback() -> None:
    """Perfect verdicts score 1.0; missed conflicts and wrong actions drag it down."""
    adapter = ReviewerAdapter(lambda prompt, example: {"action": example["expected_output"]["action"]})
    score, feedback = adapter.evaluate("review drafts", EVAL_SET)
    assert isinstance(score, float)
    # Action accuracy 2/2 = 1.0, conflict accuracy 2/2 = 1.0 -> mean 1.0.
    assert score == pytest.approx(1.0)
    assert "reviewer score 1.000 over 4 examples: 0 failed, 4 passed" in feedback
    assert "Failure patterns" not in feedback

    # A reviewer that never flags conflicts and mislabels one plain action.
    buggy = ReviewerAdapter(lambda prompt, example: {"action": "add"})
    score, feedback = buggy.evaluate("review drafts", EVAL_SET)
    # Plain actions: rev-001 add==add (1.0), rev-002 add!=update (0.0) -> 0.5;
    # conflicts: rev-003/rev-004 predicted add (0.0 each) -> 0.0; mean 0.25.
    assert score == pytest.approx(0.25)
    assert "reviewer score 0.250 over 4 examples: 3 failed, 1 passed" in feedback
    assert "Failure patterns:" in feedback
    assert "- rev-002 (score 0.000)" in feedback
    assert "wrong action: expected 'update', returned 'add'" in feedback
    assert "- rev-003 (score 0.000)" in feedback
    assert "missed conflict" in feedback
    assert "- rev-004 (score 0.000)" in feedback

    # A reviewer that over-flags conflicts takes the penalty on plain actions
    # (action accuracy 0.0) while detecting every real conflict (1.0) -> 0.5.
    overeager = ReviewerAdapter(lambda prompt, example: {"action": "conflict"})
    score, feedback = overeager.evaluate("review drafts", EVAL_SET)
    assert score == pytest.approx(0.5)
    assert "2 failed, 2 passed" in feedback
    assert "false conflict flag" in feedback
