"""Reviewer GEPA adapter (plan P10.T3).

Wraps the reviewer stage for GEPA self-optimization (plan Phase 10). The
reviewer decides what happens to each extracted draft against the existing
memory: ``add`` | ``update`` | ``skip`` | ``conflict`` (flag a contradiction
instead of merging — plan Phase 3).

Runner contract: ``runner(prompt, example) -> {"action": <verdict>, ...}``.
Metric (per plan P10.T3): action accuracy on the plain-action examples — the
predicted action must equal ``expected_output["action"]`` — plus
:func:`hermes.optimization.metrics.conflict_detection_accuracy` over the
subset whose expected action is ``conflict`` (the reviewer must flag the
conflict instead of silently merging). The aggregate is the unweighted mean
of the two sub-metrics; a vacuous subset is perfect by convention (1.0),
matching the metrics module's degenerate-case rule. Per-example scores stay
binary so the reflection filter keys on them directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from hermes.optimization.adapters import BaseAdapter, ExampleResult, _require_mapping
from hermes.optimization.metrics import conflict_detection_accuracy

__all__ = ["ReviewerAdapter"]


class ReviewerAdapter(BaseAdapter):
    """GEPA adapter for the reviewer stage (plan P10.T3)."""

    component_name: ClassVar[str] = "reviewer"

    #: The verdict meaning "contradicts stored memory; needs user confirmation".
    CONFLICT_ACTION: Final[str] = "conflict"

    def score_example(
        self, example: Mapping[str, Any], output: dict[str, Any]
    ) -> float:
        """Binary action match; conflict-expected examples score via conflict flag."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        expected_action = expected.get("action")
        if not isinstance(expected_action, str):
            raise TypeError(
                f"expected_output['action'] must be str, got {type(expected_action).__name__}"
            )
        predicted_action = output.get("action")
        if not isinstance(predicted_action, str):
            raise TypeError(
                f"output['action'] must be str, got {type(predicted_action).__name__}"
            )
        if expected_action == self.CONFLICT_ACTION:
            # Conflict-detection case: ground truth says "flag it"; the score
            # is the (single-case) conflict-detection accuracy.
            predicted_flag = predicted_action == self.CONFLICT_ACTION
            return conflict_detection_accuracy([predicted_flag], [True])
        return 1.0 if predicted_action == expected_action else 0.0

    def aggregate(self, results: Sequence[ExampleResult]) -> float:
        """Mean of action accuracy (plain examples) and conflict-detection accuracy.

        A vacuous subset is perfect by convention (``1.0``), so an eval set
        with only plain-action examples reduces to plain action accuracy and
        vice versa.
        """
        if not results:
            return 1.0
        action_scores: list[float] = []
        conflict_predictions: list[bool] = []
        conflict_truth: list[bool] = []
        for result in results:
            expected = result.expected_output
            expected_action = (
                expected.get("action")
                if isinstance(expected, Mapping)
                else None
            )
            if expected_action == self.CONFLICT_ACTION:
                predicted_action = (
                    result.output.get("action")
                    if isinstance(result.output, Mapping)
                    else None
                )
                conflict_predictions.append(predicted_action == self.CONFLICT_ACTION)
                conflict_truth.append(True)
            else:
                action_scores.append(result.score)
        sub_metrics: list[float] = []
        if action_scores:
            sub_metrics.append(sum(action_scores) / len(action_scores))
        if conflict_truth:
            sub_metrics.append(
                conflict_detection_accuracy(conflict_predictions, conflict_truth)
            )
        if not sub_metrics:
            return 1.0
        return sum(sub_metrics) / len(sub_metrics)

    def describe_example(
        self, example: Mapping[str, Any], output: dict[str, Any], score: float
    ) -> str:
        """Expected vs. returned reviewer verdict for one example."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        expected_action = expected.get("action")
        predicted_action = output.get("action")
        if expected_action == self.CONFLICT_ACTION:
            if predicted_action == self.CONFLICT_ACTION:
                return "conflict correctly flagged"
            return (
                "missed conflict: expected the 'conflict' verdict, returned "
                f"{predicted_action!r}"
            )
        if predicted_action == self.CONFLICT_ACTION:
            return (
                "false conflict flag: expected "
                f"{expected_action!r}, returned 'conflict'"
            )
        if predicted_action == expected_action:
            return f"action {expected_action!r} correct"
        return f"wrong action: expected {expected_action!r}, returned {predicted_action!r}"
