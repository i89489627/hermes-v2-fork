"""hermes-v2 optimization metrics (plan P10.T1).

Standard evaluation metrics for the self-optimization loop (plan Section 2.7).
Every metric returns a float clamped to ``[0.0, 1.0]`` and validates its
inputs eagerly (TypeError / ValueError) so a malformed evaluation report
cannot silently produce a misleading score.

Degenerate (0/0) convention
---------------------------
Precision/recall-style metrics are mathematically undefined when the
denominator is zero. This module treats "nothing to be wrong about" as
perfect agreement and returns ``1.0``:

- ``extraction_f1``: empty predictions AND empty ground truth -> ``1.0``
  (exactly one side empty -> ``0.0``);
- ``retrieval_recall_at_k``: empty relevant set -> ``1.0``;
- ``gating_precision``: the gate never chose "use" -> ``1.0``;
- ``conflict_detection_accuracy``: zero cases -> ``1.0``.

``extraction_f1`` definition (set-overlap F1)
---------------------------------------------
Predictions and ground truth are iterables of fact strings. Each fact is
normalized (Unicode casefold, whitespace runs collapsed to one space) and
deduplicated into a set, so duplicate facts neither help nor hurt. With
``P = |pred & truth| / |pred|`` and ``R = |pred & truth| / |truth|`` the score
is ``F1 = 2PR / (P + R)`` (``0.0`` when the sets are disjoint). Token-level
overlap is deliberately not used: v2 memory facts (plan P8) are short
normalized statements, and set-overlap is the well-defined,
order-independent choice for comparing them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

__all__ = [
    "conflict_detection_accuracy",
    "extraction_f1",
    "gating_precision",
    "retrieval_recall_at_k",
]


def _clamp(value: float) -> float:
    """Clamp ``value`` into the closed interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, value))


def _normalize_fact(fact: str) -> str:
    """Normalize a fact string: casefold plus whitespace-run collapse."""
    return " ".join(fact.casefold().split())


def _str_set(values: Iterable[str], arg_name: str) -> set[str]:
    """Validate a string iterable and return its set of normalized elements."""
    if isinstance(values, str):
        raise TypeError(f"{arg_name} must be an iterable of str, not a bare str")
    out: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"{arg_name} must contain only str, got {type(value).__name__}"
            )
        out.add(_normalize_fact(value))
    return out


def _validate_bool_pair(
    first: Sequence[bool],
    second: Sequence[bool],
    first_name: str,
    second_name: str,
    metric: str,
) -> None:
    """Validate paired boolean label sequences (equal length, bool-only)."""
    if len(first) != len(second):
        raise ValueError(
            f"{metric}: {first_name} and {second_name} must have the same "
            f"length ({len(first)} != {len(second)})"
        )
    for name, values in ((first_name, first), (second_name, second)):
        for value in values:
            if not isinstance(value, bool):
                raise TypeError(
                    f"{metric}.{name} must contain only bool, "
                    f"got {type(value).__name__}"
                )


def extraction_f1(predictions: Iterable[str], ground_truth: Iterable[str]) -> float:
    """Set-overlap F1 between predicted and reference fact strings.

    Both sides are normalized (casefold + whitespace collapse) and
    deduplicated, then scored with ``F1 = 2PR / (P + R)`` where ``P`` is the
    precision and ``R`` the recall of the set intersection. Returns ``1.0``
    when both sides are empty (perfect agreement) and ``0.0`` when exactly
    one side is empty or the sets are disjoint. The result is clamped to
    ``[0.0, 1.0]``.
    """
    pred_set = _str_set(predictions, "predictions")
    truth_set = _str_set(ground_truth, "ground_truth")
    if not pred_set and not truth_set:
        return 1.0
    if not pred_set or not truth_set:
        return 0.0
    overlap = len(pred_set & truth_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(truth_set)
    if precision + recall == 0.0:
        # Disjoint sets: the F1 formula degenerates to 0/0 -> no overlap.
        return 0.0
    return _clamp(2.0 * precision * recall / (precision + recall))


def retrieval_recall_at_k(
    results: Sequence[str], relevant: Collection[str], k: int
) -> float:
    """Fraction of ``relevant`` items present in the top-``k`` ranked results.

    ``results`` must be ordered best-first; ``relevant`` is a collection of
    unique identifiers matched exactly (no normalization). ``k`` must be a
    positive int. An empty ``relevant`` set is perfect by convention (there
    is nothing to recall). The result is clamped to ``[0.0, 1.0]``.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if isinstance(results, str):
        raise TypeError("results must be a sequence of str, not a bare str")
    for name, values in (("results", results), ("relevant", relevant)):
        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must contain only str, got {type(value).__name__}"
                )
    if not relevant:
        return 1.0
    top_k = list(results)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return _clamp(hits / len(relevant))


def gating_precision(decisions: Sequence[bool], ground_truth: Sequence[bool]) -> float:
    """Fraction of the gate's "use" decisions that were correct.

    ``decisions[i]`` is the gate's actual choice and ``ground_truth[i]`` the
    correct choice for case ``i`` (``True`` = memory should have been used).
    Precision counts only the cases where the gate said "use": a gate that
    never fires is precise by convention (``1.0``) — its coverage is a
    recall-style property measured separately. Both sequences must have equal
    length and hold only booleans. The result is clamped to ``[0.0, 1.0]``.
    """
    _validate_bool_pair(
        decisions, ground_truth, "decisions", "ground_truth", "gating_precision"
    )
    use_indices = [index for index, used in enumerate(decisions) if used]
    if not use_indices:
        return 1.0
    correct = sum(1 for index in use_indices if ground_truth[index])
    return _clamp(correct / len(use_indices))


def conflict_detection_accuracy(
    predictions: Sequence[bool], ground_truth: Sequence[bool]
) -> float:
    """Fraction of per-case conflict verdicts that match ground truth.

    ``predictions[i]`` and ``ground_truth[i]`` are boolean conflict flags for
    case ``i``; accuracy is the plain mean agreement. Zero cases are perfect
    by convention (``1.0``). Both sequences must have equal length and hold
    only booleans. The result is clamped to ``[0.0, 1.0]``.
    """
    _validate_bool_pair(
        predictions,
        ground_truth,
        "predictions",
        "ground_truth",
        "conflict_detection_accuracy",
    )
    if not predictions:
        return 1.0
    correct = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p == g)
    return _clamp(correct / len(predictions))
