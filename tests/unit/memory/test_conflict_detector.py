"""P8.T1 acceptance tests: ConflictDetector (plan Phase 8).

Covers the spec behaviours: semantic-duplicate detection with best-match
selection among decoy entries (plan acceptance: conflicts detected in 90%+
of test cases), the no-conflict path for unrelated texts (false-positive
rate <10%), and the contradiction classification (negation asymmetry /
antonym pairs) returning the special ``contradiction`` kind. The default
TfidfTrigramEmbedder backend is exercised throughout (see the module
docstring deviation note in ``hermes/memory/conflict_detector.py``).
"""

from __future__ import annotations

import pytest

from hermes.memory.conflict_detector import (
    ConflictDetector,
    ConflictResult,
    SentenceTransformerEmbedder,
    TfidfTrigramEmbedder,
)

DECOYS: list[tuple[str, str]] = [
    (
        "decoy-budget",
        "The quarterly budget review meeting is scheduled in the main conference room.",
    ),
    (
        "decoy-garden",
        "The gardener waters the tomato plants every second morning in summer.",
    ),
]

#: (new text, existing near-duplicate) pairs the detector must flag. Every
#: pair restates the same fact with word reordering, morphology or a small
#: insertion — the near-duplicate distribution a memory store actually sees.
DUPLICATE_CASES: list[tuple[str, str]] = [
    (
        "The deploy pipeline runs on Fridays at noon.",
        "The deploy pipeline runs on Friday at noon.",
    ),
    ("User prefers dark mode in the editor.", "User prefers dark mode in the editor."),
    (
        "Memory entries are stored as markdown files.",
        "Memory entries are stored as markdown files.",
    ),
    ("The main model is GLM-5.3-Flash.", "The main model is GLM-5.3-Flash."),
    ("The gateway restarts at 3am daily.", "The gateway restarts daily at 3am."),
    (
        "Always run tests before pushing to main.",
        "Always run tests before pushing to main branch.",
    ),
    ("Deploys happen at noon on Friday.", "Deploys happen at noon on Fridays."),
    ("The office closes at 5pm.", "Office closes at 5pm."),
    ("Cron jobs run in the UTC timezone.", "Cron jobs run in UTC timezone."),
    ("The report exports to PDF by default.", "The report exports to PDF format by default."),
]

#: (new text, unrelated existing text) pairs the detector must NOT flag.
UNRELATED_CASES: list[tuple[str, str]] = [
    ("The deploy pipeline runs on Fridays at noon.", "User birthday is March 3rd."),
    ("Ruff must pass before every commit.", "The cat is fed twice a day."),
    ("Memory entries are stored as markdown files.", "Flight to Berlin departs at gate C7."),
    ("The main model is GLM-5.3-Flash.", "Parking lot B is reserved for visitors."),
    ("Use pnpm for installing dependencies.", "Grandma recipe uses two eggs."),
    ("Database backups run every Sunday night.", "The office wifi password is hunter2."),
    ("Gateway restarts at 3am daily.", "Doctor appointment on Tuesday at 9."),
    ("Always run tests before pushing to main.", "The car needs an oil change in June."),
    ("CI pipeline deploys every Friday.", "Book club meets on the first Monday."),
    ("User prefers dark mode in the editor.", "Sprint retro scheduled for Thursday."),
]


@pytest.mark.unit
def test_detects_semantic_duplicate() -> None:
    """Near-duplicate restatements are flagged, best match picked, kind=duplicate.

    Plan acceptance: conflicts detected in 90%+ of test cases (asserted as a
    rate over the case matrix) and the result carries the conflicting entry's
    id among decoy entries.
    """
    detector = ConflictDetector()
    detected = 0
    for index, (new_text, existing_text) in enumerate(DUPLICATE_CASES):
        existing = [(f"dup-{index}", existing_text), *DECOYS]
        result = detector.check_conflict(new_text, existing)
        if result.is_conflict:
            detected += 1
            assert result.conflicting_entry_id == f"dup-{index}"
            assert result.kind == "duplicate"
            assert result.similarity > 0.85
    detection_rate = detected / len(DUPLICATE_CASES)
    assert detection_rate >= 0.90, (
        f"duplicate detection rate {detection_rate:.0%} below the 90% "
        f"acceptance bar ({detected}/{len(DUPLICATE_CASES)} detected)"
    )

    # Empty candidate list: never a conflict, zero similarity reported.
    empty = detector.check_conflict("anything at all", [])
    assert isinstance(empty, ConflictResult)
    assert empty.is_conflict is False
    assert empty.similarity == 0.0
    assert empty.conflicting_entry_id is None
    assert empty.kind is None


@pytest.mark.unit
def test_no_conflict_for_unrelated() -> None:
    """Unrelated texts stay below the threshold; false positives stay under 10%."""
    detector = ConflictDetector()
    false_positives = 0
    for new_text, existing_text in UNRELATED_CASES:
        result = detector.check_conflict(
            new_text, [("unrelated-1", existing_text), *DECOYS]
        )
        if result.is_conflict:
            false_positives += 1
        assert result.is_conflict is False
        assert result.conflicting_entry_id is None
        assert result.kind is None
        assert result.similarity <= 0.85
    fp_rate = false_positives / len(UNRELATED_CASES)
    assert fp_rate < 0.10, (
        f"false-positive rate {fp_rate:.0%} at or above the 10% ceiling "
        f"({false_positives}/{len(UNRELATED_CASES)} flagged)"
    )


@pytest.mark.unit
def test_contradiction_returns_special_result() -> None:
    """High-similarity reversed facts return kind='contradiction' (plan P8.T1).

    The heuristic is the plan-5.5 local/no-LLM fallback (main-model
    classification lands with the reviewer pipeline, P8.T2): negation cues
    appearing asymmetrically, or bundled antonym pairs, classify the conflict
    as ``contradiction``; symmetric negation on both sides stays ``duplicate``.
    """
    detector = ConflictDetector()

    # Negation asymmetry: "not" present on exactly one side.
    negation = detector.check_conflict(
        "Deploy on Fridays is allowed.",
        [("entry-negation", "Deploy on Fridays is not allowed."), *DECOYS],
    )
    assert negation.is_conflict is True
    assert negation.kind == "contradiction"
    assert negation.conflicting_entry_id == "entry-negation"
    assert negation.similarity > 0.85

    # Antonym pair from the bundled table (pnpm/npm replacement).
    antonym = detector.check_conflict(
        "Use pnpm for package installs.",
        [("entry-antonym", "Use npm for package installs."), *DECOYS],
    )
    assert antonym.is_conflict is True
    assert antonym.kind == "contradiction"
    assert antonym.conflicting_entry_id == "entry-antonym"

    # Symmetric negation is a restatement, not a reversal.
    symmetric = detector.check_conflict(
        "Cache is not enabled.",
        [("entry-same", "Cache is not enabled.")],
    )
    assert symmetric.is_conflict is True
    assert symmetric.kind == "duplicate"
    assert symmetric.conflicting_entry_id == "entry-same"


@pytest.mark.unit
def test_custom_embedder_and_threshold_validation() -> None:
    """Default is the TF-IDF trigram embedder; thresholds are validated.

    Extra guard for the P8.T1 deviation design: the injectable backend
    protocol accepts any fit/embed implementation, and the optional
    sentence-transformers backend stays lazily importable (never loaded at
    construction time).
    """
    detector = ConflictDetector()
    assert isinstance(detector._embedder, TfidfTrigramEmbedder)  # noqa: SLF001

    class LowercaseEmbedder:
        """Minimal protocol implementation: bag-of-characters vector."""

        def fit(self, texts: object) -> None:
            return None

        def embed(self, text: str) -> dict[str, float]:
            counts: dict[str, float] = {}
            for char in text.casefold():
                if char != " ":
                    counts[char] = counts.get(char, 0.0) + 1.0
            return counts

    custom = ConflictDetector(embedder=LowercaseEmbedder(), threshold=0.5)  # type: ignore[arg-type]
    result = custom.check_conflict("deploy on friday", [("x", "deploy on friday")])
    assert result.is_conflict is True

    with pytest.raises(ValueError, match="threshold"):
        ConflictDetector(threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        ConflictDetector(threshold=0.0)

    # The optional backend must not import sentence_transformers on construct.
    lazy = SentenceTransformerEmbedder()
    lazy.fit([])
    assert lazy._model is None  # noqa: SLF001
