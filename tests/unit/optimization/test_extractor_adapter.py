"""P10.T3 acceptance: extractor GEPA adapter (plan Phase 10).

``evaluate`` returns the aggregate extraction-F1 score plus a text feedback
digest naming the failure patterns; ``make_reflective_dataset`` maps only the
failing examples (score below the failure threshold) to reflective notes.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.optimization.adapters import ExampleResult
from hermes.optimization.adapters.extractor_adapter import ExtractorAdapter

EVAL_SET: list[dict[str, Any]] = [
    {
        "id": "ext-001",
        "input": "I switched my coffee to oat milk last month.",
        "expected_output": ["User prefers oat milk in coffee"],
        "context": "casual chat; corrects an older stored preference",
    },
    {
        "id": "ext-002",
        "input": "My rent is 1600 starting January.",
        "expected_output": ["User's rent is 1600 per month", "Rent starts in January"],
        "context": "fact update on a stored number",
    },
]


def _perfect_runner(prompt: str, example: dict[str, Any]) -> dict[str, Any]:
    """The ideal extractor: drafts exactly the labeled facts."""
    return {"drafts": list(example["expected_output"])}


def _sloppy_runner(prompt: str, example: dict[str, Any]) -> dict[str, Any]:
    """A buggy extractor: drops the last labeled fact and invents one."""
    drafts = list(example["expected_output"][:-1]) + ["User loves pineapple pizza"]
    return {"drafts": drafts}


@pytest.mark.unit
def test_evaluate_returns_score_and_feedback() -> None:
    """evaluate returns (float in [0, 1], feedback text) and names failures."""
    adapter = ExtractorAdapter(_perfect_runner)
    score, feedback = adapter.evaluate("extract memory drafts", EVAL_SET)
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
    assert isinstance(feedback, str)
    assert feedback
    assert "extractor score 1.000 over 2 examples" in feedback
    assert "0 failed, 2 passed" in feedback
    assert "Failure patterns" not in feedback

    sloppy = ExtractorAdapter(_sloppy_runner)
    score, feedback = sloppy.evaluate("extract memory drafts", EVAL_SET)
    # ext-001: invented fact only -> disjoint sets -> F1 0.0;
    # ext-002: one of two facts kept + one invented -> P = R = 0.5 -> F1 0.5.
    assert score == pytest.approx(0.25)
    assert "extractor score 0.250 over 2 examples: 2 failed, 0 passed" in feedback
    assert "Failure patterns:" in feedback
    assert "- ext-001 (score 0.000)" in feedback
    assert "- ext-002 (score 0.500)" in feedback
    assert "missed facts" in feedback
    assert "invented facts" in feedback

    # Empty eval set: the "nothing to be wrong about" convention (metrics module).
    empty_score, empty_feedback = adapter.evaluate("p", [])
    assert empty_score == pytest.approx(1.0)
    assert "nothing to reflect on" in empty_feedback


@pytest.mark.unit
def test_make_reflective_dataset_filters_failures() -> None:
    """Reflection notes are keyed by failing example id only; passes are filtered."""
    adapter = ExtractorAdapter(_sloppy_runner)
    results: list[ExampleResult] = adapter.evaluate_batch("p", EVAL_SET)
    dataset = adapter.make_reflective_dataset(results)
    assert set(dataset) == {"ext-001", "ext-002"}
    for note in dataset.values():
        assert isinstance(note, str) and note
        assert "Failure (score" in note
        assert "missed facts" in note or "invented facts" in note

    # A mixed run keeps only the failing example's note.
    def mixed(prompt: str, example: dict[str, Any]) -> dict[str, Any]:
        if example["id"] == "ext-001":
            return _perfect_runner(prompt, example)
        return _sloppy_runner(prompt, example)

    mixed_adapter = ExtractorAdapter(mixed)
    dataset = mixed_adapter.make_reflective_dataset(
        mixed_adapter.evaluate_batch("p", EVAL_SET)
    )
    assert set(dataset) == {"ext-002"}

    # An all-passing run reflects nothing.
    perfect = ExtractorAdapter(_perfect_runner)
    assert (
        perfect.make_reflective_dataset(perfect.evaluate_batch("p", EVAL_SET)) == {}
    )
    # And an empty result list yields an empty dataset.
    assert adapter.make_reflective_dataset([]) == {}
