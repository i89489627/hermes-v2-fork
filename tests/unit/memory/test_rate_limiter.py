"""P17.T2 acceptance: rate limiting on memory writes.

Covers :class:`hermes.memory.rate_limiter.RateLimiter` with an injected
clock: rolling session-hour and user-day budgets, denial without budget
consumption, window recovery, and burst detection (10 writes / 1 minute
flagged as suspicious but allowed).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hermes.memory.rate_limiter import RateDecision, RateLimiter


class _FakeClock:
    """Injectable clock with explicit test-controlled advancement."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _limiter(**overrides: object) -> tuple[RateLimiter, _FakeClock]:
    clock = _FakeClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))
    return RateLimiter(clock=clock, **overrides), clock  # type: ignore[arg-type]


@pytest.mark.unit
def test_allows_under_limit() -> None:
    """Writes up to each budget pass — plan defaults, burst only flags."""
    limiter = RateLimiter()
    assert limiter.max_per_session_per_hour == 20
    assert limiter.max_per_day_per_user == 100
    assert limiter.burst_max == 10
    assert limiter.burst_window_seconds == 60

    # the first 9 writes for one session / one user are clean
    for _ in range(9):
        decision = limiter.allow("session-a", "user-1")
        assert decision == RateDecision(allowed=True, reason="ok")

    # writes 10..20 all pass (session budget is inclusive, 20/hour); from
    # the 10th on they are flagged as a burst, still allowed
    for _ in range(11):
        decision = limiter.allow("session-a", "user-1")
        assert decision.allowed is True
        assert decision.reason == "burst"

    # a different session, same user: fresh session budget, no burst
    assert limiter.allow("session-b", "user-1") == RateDecision(
        allowed=True, reason="ok"
    )
    # a different user, same (fresh) session: user budget is independent
    assert limiter.allow("session-b", "user-2") == RateDecision(
        allowed=True, reason="ok"
    )

    # the injected-clock path agrees with the explicit-now path
    later = datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC)
    assert limiter.allow("session-c", "user-1", now=later).reason == "ok"


@pytest.mark.unit
def test_blocks_over_limit() -> None:
    """Exceeding the session-hour (then user-day) budget denies the write."""
    limiter, clock = _limiter(
        max_per_session_per_hour=5,
        max_per_day_per_user=8,
    )

    for _ in range(5):  # session budget: exactly filled
        assert limiter.allow("session-a", "user-1").allowed is True

    denied = limiter.allow("session-a", "user-1")
    assert denied == RateDecision(allowed=False, reason="session-hour-limit")

    # another session shares only the user budget: 3 more writes pass
    for _ in range(3):
        assert limiter.allow("session-b", "user-1").allowed is True
    assert limiter.allow("session-b", "user-1") == RateDecision(
        allowed=False, reason="user-day-limit"
    )
    # a fresh user with a fresh session has a fresh day budget
    assert limiter.allow("session-c", "user-2").allowed is True

    # session-hour recovery is independent of the user-day window: user-3's
    # day budget (5 writes) is not exhausted, so once the hour rolls off,
    # the same session is allowed again (limits reset; denied writes
    # consumed nothing)
    for _ in range(5):
        assert limiter.allow("session-d", "user-3").allowed is True
    clock.advance(3601)
    assert limiter.allow("session-d", "user-3").reason == "ok"

    # user-1's day budget, however, still holds all 8 writes until 24h pass
    assert limiter.allow("session-a", "user-1").reason == "user-day-limit"
    clock.advance(24 * 3600)
    assert limiter.allow("session-a", "user-1").reason == "ok"


@pytest.mark.unit
def test_burst_detection() -> None:
    """burst_max writes within the burst window are allowed but flagged."""
    limiter, clock = _limiter(max_per_session_per_hour=100, burst_max=10)

    for _ in range(9):  # under the burst threshold
        assert limiter.allow("session-a", "user-1") == RateDecision(
            allowed=True, reason="ok"
        )

    # the 10th write inside one minute crosses the threshold: suspicious
    assert limiter.allow("session-a", "user-1") == RateDecision(
        allowed=True, reason="burst"
    )
    # writes beyond the threshold stay flagged while the window is full
    assert limiter.allow("session-a", "user-1").reason == "burst"

    # burst is tracked per session: another session is not flagged
    assert limiter.allow("session-b", "user-1").reason == "ok"

    # once the window slides past the early writes, the session calms down
    clock.advance(61)
    assert limiter.allow("session-a", "user-1").reason == "ok"

    # invalid configuration is rejected
    with pytest.raises(ValueError, match="burst_max"):
        RateLimiter(burst_max=0)


@pytest.mark.unit
def test_burst_limit_boundary_10_in_60s() -> None:
    """Burst boundary is exact (P20.T6): 10 writes in 60s flagged, 9 clean.

    Mutation-gate boundary test for the plan defaults (burst_max=10,
    burst_window_seconds=60): writes 1..9 are clean, the 10th write inside
    the 60-second window is exactly the flagged one, and the next write is
    blocked by the hard session-hour limit configured to the same threshold
    (a burst FLAG never blocks by itself — proven against the default 20/hour
    budget, where writes beyond the burst threshold stay allowed).
    """
    limiter, clock = _limiter(max_per_session_per_hour=10, burst_max=10)

    # writes 1..9: inside the 60s window, under the burst threshold — clean
    for index in range(9):
        decision = limiter.allow("session-a", "user-1")
        assert decision == RateDecision(allowed=True, reason="ok"), index

    # the 10th write within 60 seconds: exactly at the boundary — allowed
    # but flagged as a burst (suspicious, not denied)
    tenth = limiter.allow("session-a", "user-1")
    assert tenth == RateDecision(allowed=True, reason="burst")

    # the 11th write is blocked — by the hard session-hour limit (10/hour
    # configured to coincide), not by the burst flag
    clock.advance(1)  # still inside the same 60s burst window
    blocked = limiter.allow("session-a", "user-1")
    assert blocked == RateDecision(allowed=False, reason="session-hour-limit")
    # denied writes are not recorded: nothing consumed by the rejection
    clock.advance(3601)  # hour window rolls off
    assert limiter.allow("session-a", "user-1").reason == "ok"

    # burst is a flag, NOT a block: with the plan's default 20/hour budget,
    # writes 10 and 11 are both allowed (flagged) inside the window
    default_limiter = RateLimiter()
    for _ in range(9):
        assert default_limiter.allow("session-b", "user-1").reason == "ok"
    assert default_limiter.allow("session-b", "user-1") == RateDecision(
        allowed=True, reason="burst"
    )
    assert default_limiter.allow("session-b", "user-1") == RateDecision(
        allowed=True, reason="burst"
    )
