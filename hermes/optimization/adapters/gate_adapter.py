"""Gate GEPA adapter (plan P10.T3).

Wraps the CBDC gating stage (plan Phase 4) for GEPA self-optimization. The
gate sees the turn plus candidate memory entries and decides per candidate:
``use`` (inject into context) | ``ignore`` | ``verify``.

Runner contract: ``runner(prompt, example) -> {"decisions": [{"id": <candidate
id>, "action": "use" | "ignore" | "verify"}, ...]}``.
Metric (per plan P10.T3): :func:`hermes.optimization.metrics.gating_precision`
over the per-candidate "use" decisions — a predicted decision counts as a
"use" attempt when its action is ``use`` and it is correct when the expected
decision for that candidate id is also ``use``. Precision only: a gate that
never fires is vacuously precise (``1.0``) — coverage of should-use candidates
is a recall-style property measured separately (plan Phase 4). Expected
candidates without a predicted decision count as "did not fire"; predicted
"use" decisions for candidate ids absent from the expected list count against
precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final

from hermes.optimization.adapters import BaseAdapter, _require_mapping
from hermes.optimization.metrics import gating_precision

__all__ = ["GateAdapter"]


def _decision_pairs(value: Any, name: str) -> list[tuple[str, str]]:
    """Validate a decisions list (``[{"id": str, "action": str}, ...])``."""
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    pairs: list[tuple[str, str]] = []
    for entry in value:
        record = _require_mapping(entry, f"{name} entry")
        decision_id = record.get("id")
        action = record.get("action")
        if not isinstance(decision_id, str) or not isinstance(action, str):
            raise TypeError(f"{name} entries must carry str 'id' and 'action' fields")
        pairs.append((decision_id, action))
    return pairs


class GateAdapter(BaseAdapter):
    """GEPA adapter for the CBDC gating stage (plan P10.T3)."""

    component_name: ClassVar[str] = "gate"

    #: The decision meaning "inject this candidate into the context".
    USE_ACTION: Final[str] = "use"

    def score_example(
        self, example: Mapping[str, Any], output: dict[str, Any]
    ) -> float:
        """Per-example ``gating_precision`` over the candidate "use" decisions."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        expected_decisions = _decision_pairs(
            expected.get("decisions"), "expected_output['decisions']"
        )
        predicted_decisions = _decision_pairs(output.get("decisions"), "output['decisions']")
        truth_by_id = {
            decision_id: action == self.USE_ACTION
            for decision_id, action in expected_decisions
        }
        predicted_by_id = {
            decision_id: action == self.USE_ACTION
            for decision_id, action in predicted_decisions
        }
        # Expected candidates first (in order), then predicted ids the
        # expected list does not know about (spurious decisions).
        decision_ids = [decision_id for decision_id, _ in expected_decisions]
        decision_ids.extend(
            decision_id
            for decision_id, _ in predicted_decisions
            if decision_id not in truth_by_id and decision_id not in decision_ids
        )
        predictions = [predicted_by_id.get(decision_id, False) for decision_id in decision_ids]
        truth = [truth_by_id.get(decision_id, False) for decision_id in decision_ids]
        return gating_precision(predictions, truth)

    def describe_example(
        self, example: Mapping[str, Any], output: dict[str, Any], score: float
    ) -> str:
        """False-positive "use" decisions + missed should-use candidates."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        expected_decisions = _decision_pairs(
            expected.get("decisions"), "expected_output['decisions']"
        )
        predicted_decisions = _decision_pairs(output.get("decisions"), "output['decisions']")
        truth_by_id = {
            decision_id: action == self.USE_ACTION
            for decision_id, action in expected_decisions
        }
        predicted_by_id = {
            decision_id: action == self.USE_ACTION
            for decision_id, action in predicted_decisions
        }
        false_positives = sorted(
            decision_id
            for decision_id, fired in predicted_by_id.items()
            if fired and not truth_by_id.get(decision_id, False)
        )
        missed_uses = sorted(
            decision_id
            for decision_id, should_use in truth_by_id.items()
            if should_use and not predicted_by_id.get(decision_id, False)
        )
        parts: list[str] = []
        if false_positives:
            parts.append(
                "fired on candidates that should be ignored: "
                + ", ".join(false_positives)
            )
        if missed_uses:
            parts.append(
                "failed to fire on should-use candidates: "
                + ", ".join(missed_uses)
                + " (not counted by precision)"
            )
        if not parts:
            parts.append("gate decisions correct")
        return f"precision {score:.3f}: " + " | ".join(parts)
