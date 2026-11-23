"""hermes-v2 GEPA adapter base (plan P10.T3).

Shared contract + machinery for the per-component GEPA adapters (plan
Phase 10): extractor (``extractor_adapter.py``), reviewer
(``reviewer_adapter.py``), gate (``gate_adapter.py``) and rewriter
(``rewriter_adapter.py`` — the Phase 7 compaction-summary generator).

Mapping to gepa-ai/gepa
-----------------------
The ``gepa`` library (gepa-ai/gepa) drives its optimization loop through a
user-supplied adapter object exposing two methods:

- ``evaluate(batch, candidate, capture_traces) -> EvaluationBatch`` (per-example
  outputs + scores);
- ``make_reflective_dataset(candidate, eval_batch, components_to_update)
  -> dict[str, str]`` (component name -> reflective text).

The v2 adapters keep that shape but collapse it to what P10.T5
(``GEPARunner``) needs, staying model-agnostic and dependency-free. Concretely:

- ``GEPAAdapter.evaluate(candidate_prompt, eval_set)`` runs the whole eval set
  through an injected :data:`Runner` callable (the component's logic — no
  model call happens inside the adapters) and returns the aggregate metric
  score plus a text digest of the failure patterns: the flattened equivalent
  of gepa's ``EvaluationBatch``;
- ``BaseAdapter.evaluate_batch`` exposes the per-example records
  (:class:`ExampleResult`) that correspond 1:1 to gepa's ``EvaluationBatch``
  entries;
- ``make_reflective_dataset(evaluate_results)`` returns
  ``dict[example_id, reflective note]`` filtered to failures only — the
  per-example granularity P10.T5 aggregates into gepa's component-keyed
  reflective text.

No ``gepa`` import at module scope (the optional library import belongs to
the runner task, P10.T5): adapters are plain classes a GEPA loop can drive
later. The ``candidate_prompt`` is never interpreted by the adapters — it is
passed through to the injected runner verbatim, mirroring how gepa feeds the
candidate's prompt text into ``evaluate``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

__all__ = [
    "BaseAdapter",
    "ExampleResult",
    "GEPAAdapter",
    "Runner",
]

#: Component logic injected into an adapter: ``runner(candidate_prompt,
#: example) -> output``. ``example`` is one eval-set record
#: (``{id?, input, expected_output, context}``); ``output`` is the component's
#: raw result for that example (per-adapter contract, documented on each
#: adapter class). The adapters never call a model themselves.
Runner = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ExampleResult:
    """One example's evaluation record (gepa ``EvaluationBatch`` entry).

    ``passed`` is ``score >= failure_threshold``; ``detail`` is the
    human-readable failure pattern used both in the feedback text and in the
    reflective notes.
    """

    example_id: str
    score: float
    passed: bool
    output: dict[str, Any]
    expected_output: Any
    detail: str


class GEPAAdapter(Protocol):
    """The structural interface every component adapter implements.

    A component adapter is usable by the GEPA loop (P10.T5) when it offers
    exactly these two methods — :class:`BaseAdapter` subclasses satisfy the
    protocol by inheritance. ``evaluate`` returns the aggregate score and a
    text feedback digest; ``make_reflective_dataset`` maps failing example
    ids to reflective notes.
    """

    def evaluate(
        self, candidate_prompt: str, eval_set: list[dict[str, Any]]
    ) -> tuple[float, str]:
        """Score ``eval_set`` under ``candidate_prompt``; return (score, feedback)."""
        ...

    def make_reflective_dataset(
        self, evaluate_results: Sequence[ExampleResult]
    ) -> dict[str, str]:
        """Failure-only reflection notes keyed by example id."""
        ...


class BaseAdapter:
    """Shared evaluate / reflect machinery for the component adapters.

    Subclasses declare :attr:`component_name` and implement
    :meth:`score_example` (applies the component's metric from
    :mod:`hermes.optimization.metrics`, validating the example/output shapes
    eagerly) and :meth:`describe_example` (the per-example failure pattern
    line). Aggregation over examples defaults to the unweighted mean; an
    adapter with a composite metric may override :meth:`aggregate` (the
    reviewer does).

    Failure convention: an example is a failure when its score is below
    ``failure_threshold`` (default ``1.0`` — any deviation from the labeled
    expected output is a reflection candidate). An empty eval set aggregates
    to ``1.0`` ("nothing to be wrong about"), matching the degenerate-case
    rule documented in :mod:`hermes.optimization.metrics`.
    """

    #: Human-readable component label used in feedback lines (plan Phase 10).
    component_name: ClassVar[str]

    def __init__(self, runner: Runner, failure_threshold: float = 1.0) -> None:
        if not callable(runner):
            raise TypeError("runner must be a callable (prompt, example) -> dict")
        if isinstance(failure_threshold, bool) or not isinstance(
            failure_threshold, int | float
        ):
            raise TypeError(
                f"failure_threshold must be a number, got {type(failure_threshold).__name__}"
            )
        threshold = float(failure_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"failure_threshold must be a finite float in [0.0, 1.0], got {failure_threshold!r}"
            )
        self._runner: Runner = runner
        self.failure_threshold: float = threshold

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self, candidate_prompt: str, eval_set: list[dict[str, Any]]
    ) -> tuple[float, str]:
        """Run every example through the runner; return (score, feedback digest).

        ``score`` is :meth:`aggregate` over the per-example metric values;
        ``feedback`` summarizes the failure patterns (one line per failure)
        so the optimization loop (and the reflective dataset) can key on
        them.
        """
        results = self.evaluate_batch(candidate_prompt, eval_set)
        return self.aggregate(results), self._build_feedback(results)

    def evaluate_batch(
        self, candidate_prompt: str, eval_set: list[dict[str, Any]]
    ) -> list[ExampleResult]:
        """Per-example records — the 1:1 equivalent of gepa's ``EvaluationBatch``.

        Each example runs through ``runner(candidate_prompt, example)``; the
        output is scored with the adapter's metric and described with its
        failure pattern. Raises ``TypeError``/``ValueError`` eagerly on
        malformed eval-set records or runner outputs.
        """
        if not isinstance(candidate_prompt, str):
            raise TypeError(
                f"candidate_prompt must be str, got {type(candidate_prompt).__name__}"
            )
        results: list[ExampleResult] = []
        for index, example in enumerate(eval_set):
            if not isinstance(example, Mapping):
                raise TypeError(
                    f"eval_set[{index}] must be a mapping, got {type(example).__name__}"
                )
            example_id = str(example.get("id", f"example-{index:03d}"))
            output = self._runner(candidate_prompt, dict(example))
            if not isinstance(output, Mapping):
                raise TypeError(
                    f"{example_id}: runner must return a mapping, got {type(output).__name__}"
                )
            output_dict = dict(output)
            score = float(self.score_example(example, output_dict))
            results.append(
                ExampleResult(
                    example_id=example_id,
                    score=score,
                    passed=score >= self.failure_threshold,
                    output=output_dict,
                    expected_output=example.get("expected_output"),
                    detail=self.describe_example(example, output_dict, score),
                )
            )
        return results

    def aggregate(self, results: Sequence[ExampleResult]) -> float:
        """Overall score from per-example records (default: unweighted mean)."""
        if not results:
            return 1.0  # empty eval set: "nothing to be wrong about" convention
        return sum(result.score for result in results) / len(results)

    # ── reflection ────────────────────────────────────────────────────────

    def make_reflective_dataset(
        self, evaluate_results: Sequence[ExampleResult]
    ) -> dict[str, str]:
        """Failure-only reflection notes keyed by example id.

        Maps each failing record (score below the failure threshold) to a
        short note describing the failure pattern and the prompt change it
        calls for. Passing examples are filtered out — reflection is only
        fed failures, per plan P10.T3.
        """
        if isinstance(evaluate_results, ExampleResult):
            raise TypeError(
                "evaluate_results must be a sequence of ExampleResult, not a single one"
            )
        dataset: dict[str, str] = {}
        for result in evaluate_results:
            if not isinstance(result, ExampleResult):
                raise TypeError(
                    f"evaluate_results must contain only ExampleResult, "
                    f"got {type(result).__name__}"
                )
            if result.passed:
                continue
            dataset[result.example_id] = self._reflective_note(result)
        return dataset

    # ── subclass hooks ────────────────────────────────────────────────────

    def score_example(
        self, example: Mapping[str, Any], output: dict[str, Any]
    ) -> float:
        """Metric value for one example (raise TypeError/ValueError on bad shapes)."""
        raise NotImplementedError

    def describe_example(
        self, example: Mapping[str, Any], output: dict[str, Any], score: float
    ) -> str:
        """One-line failure pattern for one example (feedback + reflection text)."""
        raise NotImplementedError

    # ── internals ─────────────────────────────────────────────────────────

    def _build_feedback(self, results: Sequence[ExampleResult]) -> str:
        """Text digest of the run: aggregate outcome + one line per failure."""
        failed = [result for result in results if not result.passed]
        lines = [
            f"{self.component_name} score {self.aggregate(results):.3f} over "
            f"{len(results)} examples: {len(failed)} failed, "
            f"{len(results) - len(failed)} passed (failure threshold "
            f"{self.failure_threshold:.2f})."
        ]
        if not results:
            lines.append("Eval set was empty: nothing to reflect on.")
        elif failed:
            lines.append("Failure patterns:")
            lines.extend(
                f"- {result.example_id} (score {result.score:.3f}): {result.detail}"
                for result in failed
            )
        else:
            lines.append("All examples passed: no failure patterns to reflect on.")
        return "\n".join(lines)

    def _reflective_note(self, result: ExampleResult) -> str:
        """Reflection note for one failing record."""
        return (
            f"Failure (score {result.score:.3f}, threshold "
            f"{self.failure_threshold:.2f}): {result.detail} Revise the "
            f"{self.component_name} prompt so this example produces the "
            "expected output."
        )


# ── shared validation helpers (used by all four adapters) ───────────────────


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that ``value`` is a mapping and return it."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_str_list(value: Any, name: str) -> list[str]:
    """Validate that ``value`` is a list/tuple of str and return it as a list."""
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a list of str, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{name} must contain only str, got {type(item).__name__}")
    return list(value)


def _normalize_fact(fact: str) -> str:
    """Casefold + whitespace-run collapse (mirrors the metrics module)."""
    return " ".join(fact.casefold().split())
