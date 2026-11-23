"""Rewriter (compaction-summary) GEPA adapter (plan P10.T3).

The rewriter is the compaction-summary generator of the Phase 7 session
compactor: it folds a long session transcript into a short summary the next
context window can carry forward. This adapter makes that stage optimizable
by GEPA (plan Phase 10) without coupling it to any model API — the component
logic is injected as a :data:`~hermes.optimization.adapters.Runner` callable.

Runner contract: ``runner(prompt, example) -> {"summary": str}``.

Metric (documented proxy, per plan P10.T3): the per-example score is the
**fraction of expected key points present within the char limit**. When the
produced summary fits ``expected_output["char_limit"]``, the score is the
fraction of ``expected_output["key_points"]`` whose normalized text appears
as a substring of the normalized summary; an over-limit summary is unusable
for context injection regardless of coverage and scores ``0.0``. An empty
``key_points`` list is perfect by convention (``1.0``) when the summary fits
— matching the metrics module's degenerate-case rules. Substring overlap is a
deliberately cheap proxy for semantic key-point coverage: eval-set key points
are short canonical statements, and GEPA's reflection only needs the failure
*pattern* (which points were dropped, by how much the limit was exceeded).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from hermes.optimization.adapters import (
    BaseAdapter,
    _normalize_fact,
    _require_mapping,
    _require_str_list,
)

__all__ = ["RewriterAdapter"]


class RewriterAdapter(BaseAdapter):
    """GEPA adapter for the compaction-summary generator (plan P10.T3)."""

    component_name: ClassVar[str] = "rewriter"

    def score_example(
        self, example: Mapping[str, Any], output: dict[str, Any]
    ) -> float:
        """Key-point coverage within the char limit for one example."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        key_points = _require_str_list(
            expected.get("key_points"), "expected_output['key_points']"
        )
        char_limit = expected.get("char_limit")
        if isinstance(char_limit, bool) or not isinstance(char_limit, int) or char_limit < 1:
            raise ValueError(
                f"expected_output['char_limit'] must be a positive int, got {char_limit!r}"
            )
        summary = output.get("summary")
        if not isinstance(summary, str):
            raise TypeError(f"output['summary'] must be str, got {type(summary).__name__}")
        if len(summary) > char_limit:
            return 0.0
        if not key_points:
            return 1.0
        normalized_summary = _normalize_fact(summary)
        present = sum(
            1
            for point in key_points
            if _normalize_fact(point) in normalized_summary
        )
        return present / len(key_points)

    def describe_example(
        self, example: Mapping[str, Any], output: dict[str, Any], score: float
    ) -> str:
        """Over-limit amount + dropped key points for one example."""
        expected = _require_mapping(example.get("expected_output"), "expected_output")
        key_points = _require_str_list(
            expected.get("key_points"), "expected_output['key_points']"
        )
        char_limit = expected.get("char_limit")
        summary = output.get("summary")
        parts: list[str] = []
        if isinstance(summary, str) and isinstance(char_limit, int):
            over_limit = len(summary) - char_limit
            if over_limit > 0:
                parts.append(f"summary exceeded the char limit by {over_limit} characters")
        if key_points and isinstance(summary, str):
            normalized_summary = _normalize_fact(summary)
            missing = [
                point
                for point in key_points
                if _normalize_fact(point) not in normalized_summary
            ]
            if missing:
                parts.append("missing key points: " + "; ".join(missing))
        if not parts:
            parts.append("summary covers the expected key points within the char limit")
        return f"coverage {score:.3f}: " + " | ".join(parts)
