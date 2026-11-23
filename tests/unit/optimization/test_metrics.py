"""P10.T1 acceptance: optimization metrics match standard definitions (plan Phase 10).

Every metric must return a float in ``[0.0, 1.0]``, validate its inputs, and
follow the documented degenerate-case conventions (see
``hermes/optimization/metrics.py``).
"""

from __future__ import annotations

import pytest

from hermes.optimization.metrics import (
    conflict_detection_accuracy,
    extraction_f1,
    gating_precision,
    retrieval_recall_at_k,
)


@pytest.mark.unit
def test_extraction_f1_correct() -> None:
    """Set-overlap F1: perfect, partial, normalized, empty and invalid inputs."""
    # Perfect overlap.
    assert extraction_f1(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    # Order and duplicates do not matter (set semantics).
    assert extraction_f1(["b", "a", "a"], ["a", "b"]) == pytest.approx(1.0)
    # Facts are normalized (casefold + whitespace collapse).
    assert extraction_f1(["User  Name"], ["user name"]) == pytest.approx(1.0)
    # Half the facts overlap -> precision 0.5, recall 0.5 -> F1 0.5.
    assert extraction_f1({"alpha", "beta"}, {"beta", "gamma"}) == pytest.approx(0.5)
    # Disjoint sets -> 0.0 (0/0 guard must not divide by zero).
    assert extraction_f1(["x"], ["y"]) == pytest.approx(0.0)
    # Exactly one side empty -> 0.0.
    assert extraction_f1([], ["y"]) == pytest.approx(0.0)
    assert extraction_f1(["x"], []) == pytest.approx(0.0)
    # Both empty -> perfect agreement (documented convention).
    assert extraction_f1([], []) == pytest.approx(1.0)
    # All results stay within [0, 1].
    assert 0.0 <= extraction_f1({"a", "b", "c"}, {"b"}) <= 1.0
    # Non-string facts are rejected.
    with pytest.raises(TypeError):
        extraction_f1([1, 2], ["a"])
    with pytest.raises(TypeError):
        extraction_f1(["a"], [None])
    # A bare string is not a fact collection.
    with pytest.raises(TypeError):
        extraction_f1("abc", ["a"])


@pytest.mark.unit
def test_recall_at_k() -> None:
    """recall@k counts relevant items within the first k ranked results."""
    results = ["d1", "d2", "d3", "d4", "d5", "d6"]
    relevant = {"d3", "d6", "d9"}
    # Top-5 contains only d3 -> 1/3.
    assert retrieval_recall_at_k(results, relevant, 5) == pytest.approx(1 / 3)
    # k beyond the result list -> d3 and d6 -> 2/3.
    assert retrieval_recall_at_k(results, relevant, 10) == pytest.approx(2 / 3)
    # k=2 contains no relevant item -> 0.0.
    assert retrieval_recall_at_k(results, relevant, 2) == pytest.approx(0.0)
    # Perfect recall: all relevant items within the top k.
    assert retrieval_recall_at_k(results, {"d1"}, 1) == pytest.approx(1.0)
    # Nothing relevant to find -> perfect by convention.
    assert retrieval_recall_at_k(results, set(), 3) == pytest.approx(1.0)
    # Every result stays within [0, 1].
    assert 0.0 <= retrieval_recall_at_k(results, relevant, 4) <= 1.0
    # Invalid inputs.
    with pytest.raises(ValueError):
        retrieval_recall_at_k(results, relevant, 0)
    with pytest.raises(TypeError):
        retrieval_recall_at_k(results, relevant, 1.5)
    with pytest.raises(TypeError):
        retrieval_recall_at_k([1, 2], {"a"}, 1)
    with pytest.raises(TypeError):
        retrieval_recall_at_k("d1d2", {"d1"}, 1)


@pytest.mark.unit
def test_gating_precision() -> None:
    """Precision over the gate's "use" decisions only."""
    decisions = [True, True, False, True]
    ground_truth = [True, False, False, False]
    # "use" at indices 0, 1, 3; only index 0 was correct -> 1/3.
    assert gating_precision(decisions, ground_truth) == pytest.approx(1 / 3)
    # Every "use" decision correct -> 1.0.
    assert gating_precision([True, False], [True, True]) == pytest.approx(1.0)
    # No "use" decision at all -> nothing to be wrong about -> 1.0.
    assert gating_precision([False, False], [True, True]) == pytest.approx(1.0)
    # Every "use" decision wrong -> 0.0.
    assert gating_precision([True, True], [False, False]) == pytest.approx(0.0)
    # Result stays in [0, 1] for any input.
    assert 0.0 <= gating_precision([True], [True]) <= 1.0
    # Mismatched lengths and non-bool entries are rejected.
    with pytest.raises(ValueError):
        gating_precision([True], [True, False])
    with pytest.raises(TypeError):
        gating_precision([1, 0], [1, 1])


@pytest.mark.unit
def test_conflict_detection_accuracy() -> None:
    """Accuracy = fraction of per-case conflict verdicts matching ground truth."""
    predictions = [True, False, True, True]
    ground_truth = [True, False, False, True]
    assert conflict_detection_accuracy(predictions, ground_truth) == pytest.approx(0.75)
    # Perfect and worst case.
    assert conflict_detection_accuracy([True, False], [True, False]) == pytest.approx(1.0)
    assert conflict_detection_accuracy([True], [False]) == pytest.approx(0.0)
    # Zero cases -> perfect by convention.
    assert conflict_detection_accuracy([], []) == pytest.approx(1.0)
    # Result stays in [0, 1] for any input.
    assert 0.0 <= conflict_detection_accuracy([True, True], [False, False]) <= 1.0
    # Mismatched lengths and non-bool entries are rejected.
    with pytest.raises(ValueError):
        conflict_detection_accuracy([True], [])
    with pytest.raises(TypeError):
        conflict_detection_accuracy([True, "no"], [True, False])
