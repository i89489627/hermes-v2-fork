"""P20.T2 property: the forgetting curve is monotone and vanishes at infinity.

For arbitrary (score, last_verified_at, now) triples with aware UTC
datetimes, :func:`hermes.memory.decay.compute_decay`:

- never increases when time moves forward: for ``now2 > now1``,
  ``compute_decay(score, t0, now2) <= compute_decay(score, t0, now1)`` —
  the formula ``s * exp(-0.01 * days)`` is strictly decreasing in the
  elapsed days (plan Phase 2);
- always stays within [0, 1] (clamped);
- effectively vanishes as time goes to infinity: after 10 years
  (3650 days) even a score of 1.0 has decayed below 1e-6
  (``exp(-36.5) ~= 1.4e-16``) — the plan's "decays to zero" acceptance,
  interpreted as effectively zero since ``exp`` never reaches 0.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.decay import DECAY_RATE, compute_decay

_SCORE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_DAYS = st.floats(min_value=0.0, max_value=3_650.0, allow_nan=False, allow_infinity=False)

_BASE = st.builds(
    datetime,
    year=st.just(2026),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
    hour=st.integers(min_value=0, max_value=23),
    minute=st.integers(min_value=0, max_value=59),
    second=st.integers(min_value=0, max_value=59),
    tzinfo=st.just(UTC),
)


def _at(base: datetime, days: float) -> datetime:
    """``base`` advanced by ``days`` (fractional days allowed)."""
    return base + timedelta(days=days)


@pytest.mark.property
@given(score=_SCORE, base=_BASE, days1=_DAYS, days2=_DAYS)
@example(score=1.0, base=datetime(2026, 1, 1, tzinfo=UTC), days1=0.0, days2=0.0)
@example(score=1.0, base=datetime(2026, 1, 1, tzinfo=UTC), days1=0.0, days2=3650.0)
@example(score=0.0, base=datetime(2026, 1, 1, tzinfo=UTC), days1=10.0, days2=20.0)
@settings(max_examples=1000, deadline=None)
def test_decay_never_increases(score: float, base: datetime, days1: float, days2: float) -> None:
    """compute_decay is monotone non-increasing in time (and stays in [0, 1])."""
    now1 = _at(base, min(days1, days2))
    now2 = _at(base, max(days1, days2))
    decay1 = compute_decay(score, base, now1)
    decay2 = compute_decay(score, base, now2)

    assert 0.0 <= decay1 <= 1.0
    assert 0.0 <= decay2 <= 1.0
    assert decay2 <= decay1 + 1e-12  # monotone; 1e-12 absorbs float rounding


@pytest.mark.property
@given(score=_SCORE, base=_BASE)
@example(score=1.0, base=datetime(2026, 1, 1, tzinfo=UTC))
@settings(max_examples=1000, deadline=None)
def test_decay_zero_at_infinity(score: float, base: datetime) -> None:
    """After 10 years the score has effectively decayed to zero (< 1e-6)."""
    ten_years = _at(base, 3650.0)
    decayed = compute_decay(score, base, ten_years)

    assert decayed < 1e-6
    # and the guarantee only strengthens with more time
    decayed_more = compute_decay(score, base, _at(base, 3650.0 + 3650.0))
    assert decayed_more <= decayed
    assert DECAY_RATE == 0.01  # the plan's forgetting-curve rate (Phase 2)
