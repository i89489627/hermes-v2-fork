"""P13.T6 acceptance: BundleSuggester (plan Phase 13).

Pairwise co-occurrence over per-session skill sets: pairs reaching the
threshold form edges, connected components (size >= 2) become bundles with
the weakest pairwise link as their strength, and ``propose_bundles`` filters
out bundles the user already has.
"""

from __future__ import annotations

import pytest

from hermes.skills.bundle_suggester import BundleSuggester, BundleSuggestion


@pytest.mark.unit
def test_detects_co_occurrence() -> None:
    """Pairs co-occurring >= threshold times group into bundles; weak pairs do not."""
    suggester = BundleSuggester(threshold=3)
    ledger = [
        ["code-review", "git-commit"],
        ["code-review", "git-commit", "tests"],
        ["code-review", "git-commit"],
        ["tests", "docs"],  # tests+docs co-occur once: below threshold
    ]
    suggestions = suggester.analyze(ledger)
    assert suggestions == [
        BundleSuggestion(skills=("code-review", "git-commit"), co_occurrences=3)
    ]

    # A tight 3-skill clique collapses into ONE bundle (connected component);
    # the reported strength is the weakest pairwise link (3 for all pairs).
    clique_ledger = [["plan", "implement", "review"] for _ in range(3)]
    assert suggester.analyze(clique_ledger) == [
        BundleSuggestion(skills=("implement", "plan", "review"), co_occurrences=3)
    ]

    # Weakest-link reporting with uneven pair strengths (a-b: 3, a-c: 4, b-c: 3).
    uneven = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], ["a", "c"]]
    assert suggester.analyze(uneven) == [
        BundleSuggestion(skills=("a", "b", "c"), co_occurrences=3)
    ]

    # Duplicates within one session count once; empty input yields no bundles.
    assert suggester.analyze([["x", "x", "y"], ["x", "y"], ["x", "y"]]) == [
        BundleSuggestion(skills=("x", "y"), co_occurrences=3)
    ]
    assert suggester.analyze([]) == []


@pytest.mark.unit
def test_proposes_bundle() -> None:
    """propose_bundles returns analyzed bundles minus the ones the user has."""
    suggester = BundleSuggester()  # default threshold 3
    ledger = [
        ["deploy", "changelog"],
        ["deploy", "changelog"],
        ["deploy", "changelog", "notes"],
    ]
    suggestions = suggester.analyze(ledger)
    assert suggestions == [
        BundleSuggestion(skills=("changelog", "deploy"), co_occurrences=3)
    ]

    # The bundle the user already has is filtered out of the proposals.
    assert suggester.propose_bundles(ledger, [frozenset({"deploy", "changelog"})]) == []
    # Unknown / non-matching knowledge passes everything through unchanged.
    assert suggester.propose_bundles(ledger, []) == suggestions
    assert suggester.propose_bundles(ledger, [frozenset({"deploy"})]) == suggestions

    # Eager validation of the threshold.
    with pytest.raises(TypeError):
        BundleSuggester(threshold="3")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BundleSuggester(threshold=0)
