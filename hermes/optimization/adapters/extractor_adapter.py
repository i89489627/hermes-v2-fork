"""Extractor GEPA adapter (plan P10.T3).

Wraps the extraction stage's draft logic for GEPA self-optimization (plan
Phase 10). The adapter is model-agnostic: the component logic is injected as
a :data:`~hermes.optimization.adapters.Runner` callable; the adapter runs the
eval set, scores each example with
:func:`hermes.optimization.metrics.extraction_f1` between the produced drafts
and the labeled ``expected_output`` facts, and turns failures into feedback
text plus reflective notes (base machinery in
``hermes.optimization.adapters``).

Runner contract: ``runner(prompt, example) -> {"drafts": [fact, ...]}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from hermes.optimization.adapters import BaseAdapter, _normalize_fact, _require_str_list
from hermes.optimization.metrics import extraction_f1

__all__ = ["ExtractorAdapter"]


class ExtractorAdapter(BaseAdapter):
    """GEPA adapter for the extractor's draft stage (plan P10.T3).

    Metric: ``extraction_f1(drafts, expected_output)`` per example (set-overlap
    F1 over normalized facts — see the metrics module for the exact
    definition, including the empty/disjoint conventions). The aggregate is
    the unweighted per-example mean. Feedback lines name the missed ground
    -truth facts and the invented (not-labeled) drafts, which is the failure
    pattern the reflective dataset hands back to GEPA.
    """

    component_name: ClassVar[str] = "extractor"

    def score_example(
        self, example: Mapping[str, Any], output: dict[str, Any]
    ) -> float:
        """``extraction_f1`` between the runner's drafts and the labeled facts."""
        expected = _require_str_list(
            example.get("expected_output"), "expected_output"
        )
        drafts = _require_str_list(output.get("drafts"), "output['drafts']")
        return extraction_f1(drafts, expected)

    def describe_example(
        self, example: Mapping[str, Any], output: dict[str, Any], score: float
    ) -> str:
        """Missed ground-truth facts + invented drafts for one example."""
        expected = {
            _normalize_fact(fact)
            for fact in _require_str_list(
                example.get("expected_output"), "expected_output"
            )
        }
        produced = {
            _normalize_fact(fact)
            for fact in _require_str_list(output.get("drafts"), "output['drafts']")
        }
        missed = sorted(expected - produced)
        invented = sorted(produced - expected)
        parts: list[str] = []
        if missed:
            parts.append("missed facts: " + "; ".join(missed))
        if invented:
            parts.append("invented facts: " + "; ".join(invented))
        if not parts:
            parts.append("drafts match the expected facts")
        return f"F1 {score:.3f}: " + " | ".join(parts)
