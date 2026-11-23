"""P7.T5 property: the compaction summary NEVER exceeds its char limit.

Hypothesis over turn lists (empty / unicode / huge) and char limits:
:func:`hermes.memory.compaction.summary_text` always returns at most
``char_limit`` characters and never raises — for the deterministic extractive
fallback AND for any injected summarizer (whose output is clamped to the same
limit; a RAISING summarizer degrades to the fallback). This is the plan P7.T5
budget contract that keeps compaction summaries bounded no matter what the
summarizer (or the turns) look like.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes.memory.compaction import summary_text

#: Turns: (turn_id, turn_text) pairs; unicode-safe, sizes up to "huge".
_turns = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=12),
        st.text(alphabet=st.characters(codec="utf-8"), max_size=3000),
    ),
    max_size=12,
)

_limits = st.integers(min_value=1, max_value=3000)


def _booming_summarizer(_transcript: str) -> str:
    """A summarizer that ALWAYS fails (degradation path)."""
    raise RuntimeError("summarizer unavailable")


@given(turns=_turns, char_limit=_limits)
@settings(max_examples=200, deadline=None)
def test_summary_never_exceeds_char_limit(
    turns: list[tuple[str, str]], char_limit: int
) -> None:
    """summary_text(turns, limit) fits the limit for ANY turns and summarizer."""
    # deterministic extractive fallback
    summary = summary_text(turns, char_limit)
    assert len(summary) <= char_limit

    # an injected summarizer whose output is way over budget is clamped
    huge = summary_text(
        turns, char_limit, summarizer=lambda _transcript: "x" * (3 * char_limit + 7)
    )
    assert len(huge) <= char_limit

    # a RAISING summarizer degrades to the fallback — never raises, fits
    degraded = summary_text(turns, char_limit, summarizer=_booming_summarizer)
    assert len(degraded) <= char_limit


@pytest.mark.parametrize("char_limit", [1, 2, 3, 4])
def test_summary_clamp_holds_for_tiny_limits(char_limit: int) -> None:
    """Even a 1-char limit never overflows and never raises (edge clamps)."""
    turns: Sequence[tuple[str, str]] = [
        ("t1", "A very long first sentence. And a second one."),
        ("t2", "第二回合讨论部署流水线。所有部署仅限主分支。" * 10),
    ]
    assert len(summary_text(turns, char_limit)) <= char_limit
    assert len(summary_text(turns, char_limit, summarizer=lambda _t: "¡unicode!  übersummary")) <= (
        char_limit
    )
