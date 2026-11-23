"""P20.T2 property: ConflictDetector symmetry and self-conflict exclusion.

- **Symmetry** (plan P20.T2: "A conflicts B iff B conflicts A"): for
  arbitrary texts A and B, ``check_conflict(A, [(id-b, B)])`` and
  ``check_conflict(B, [(id-a, A)])`` agree on ``is_conflict`` and ``kind``,
  and their similarities match (the TF-IDF trigram embedder is fitted on
  the same two-document corpus in both directions, so the cosine is
  symmetric; a 1e-9 tolerance absorbs float-rounding noise). High-similarity
  pairs (negation cues / antonym pairs from the bundled tables) are pinned
  as examples so the conflict-firing branch is always exercised.

- **No self-conflict** (honest formalization for the detector's signature
  ``check_conflict(new_entry, existing: list[tuple[str, str]])``): the
  detector is a pure function of the passed-in candidate list — it never
  fabricates an id. Therefore ``conflicting_entry_id`` is ALWAYS one of the
  provided candidate ids; with the entry's own id excluded (what the
  reviewer/pipeline do — they pass stored OTHER entries), an entry can
  never be reported as conflicting with itself. The test also pins the
  caller's duty: comparing a text against ITS OWN pair reports
  ``similarity == 1.0`` with kind ``duplicate`` (a verbatim restate IS a
  duplicate) — which is exactly why the caller must exclude the own id.
"""

from __future__ import annotations

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.conflict_detector import ConflictDetector, _trigrams

_OWN_ID = "own-entry-0"

#: Arbitrary texts plus pinned high-similarity pairs (negation cue /
#: antonym pairs from the bundled tables) so the conflict branch fires.
_TEXT = st.one_of(
    st.text(max_size=120),
    st.just(""),
    st.just("   "),
    st.just("Deployments to production are frozen"),
    st.just("Deployments to production are not frozen"),
    st.just("The acme deploy pipeline uses npm"),
    st.just("The acme deploy pipeline uses pnpm"),
    st.just("The repo uses pnpm for installs"),
    st.just("The repo uses npm for installs"),
    st.just("turn with 🌍 unicode and 控 character"),
)

_OTHER_TEXT = st.one_of(
    st.text(max_size=120),
    st.just("The acme deploy pipeline uses npm"),
    st.just("Deployments to production are frozen"),
)


@pytest.mark.property
@given(text_a=_TEXT, text_b=_TEXT)
@example(text_a="Deployments to production are not frozen", text_b="Deployments to production are frozen")
@example(text_a="The acme deploy pipeline uses pnpm", text_b="The acme deploy pipeline uses npm")
@example(text_a="The repo uses pnpm for installs", text_b="The repo uses npm for installs")
@example(text_a="", text_b="")
@settings(max_examples=1000, deadline=None)
def test_symmetric(text_a: str, text_b: str) -> None:
    """A conflicts B iff B conflicts A (same verdict, kind and similarity)."""
    detector = ConflictDetector()
    forward = detector.check_conflict(text_a, [("id-b", text_b)])
    backward = detector.check_conflict(text_b, [("id-a", text_a)])

    assert forward.is_conflict == backward.is_conflict
    assert forward.kind == backward.kind
    if forward.is_conflict:
        # the reported target is exactly the other side's candidate id
        assert forward.conflicting_entry_id == "id-b"
        assert backward.conflicting_entry_id == "id-a"
    else:
        assert forward.conflicting_entry_id is None
        assert backward.conflicting_entry_id is None
    assert abs(forward.similarity - backward.similarity) <= 1e-9


@pytest.mark.property
@given(entry_text=_TEXT, others=st.lists(st.tuples(st.text(min_size=1, max_size=10), _OTHER_TEXT), max_size=5))
@example(
    entry_text="Deployments to production are not frozen",
    others=[("other-a", "Deployments to production are frozen")],
)  # a real conflict fires — and targets the OTHER entry, never the own id
@example(entry_text="The acme deploy pipeline uses pnpm", others=[])
@settings(max_examples=1000, deadline=None)
def test_no_self_conflict(entry_text: str, others: list[tuple[str, str]]) -> None:
    """An entry is never reported as conflicting with itself.

    With the entry's own id excluded from the candidate list (the
    reviewer/pipeline convention), the result's ``conflicting_entry_id`` —
    if any — is always one of the provided OTHER ids, never the excluded
    own id and never a fabricated id.
    """
    detector = ConflictDetector()
    candidate_ids = [other_id for other_id, _text in others if other_id != _OWN_ID]
    others = [(other_id, text) for other_id, text in others if other_id != _OWN_ID]

    result = detector.check_conflict(entry_text, others)

    assert result.conflicting_entry_id != _OWN_ID
    if result.conflicting_entry_id is not None:
        assert result.conflicting_entry_id in candidate_ids  # never fabricated

    # the caller's duty, pinned: comparing the text against ITS OWN pair
    # reports a verbatim duplicate (similarity 1.0, kind "duplicate") — so
    # excluding the own id is what guarantees the no-self-conflict property.
    # Texts without any [a-z0-9] content carry no trigram embedding at all
    # (similarity 0.0, no conflict) — the pin applies exactly when they do.
    self_result = detector.check_conflict(entry_text, [(_OWN_ID, entry_text)])
    if _trigrams(entry_text):
        assert self_result.is_conflict
        assert self_result.kind == "duplicate"
        assert self_result.conflicting_entry_id == _OWN_ID
        assert self_result.similarity > 0.999
    else:
        assert not self_result.is_conflict
        assert self_result.similarity == 0.0
