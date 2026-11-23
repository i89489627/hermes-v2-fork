"""Rate limiting on memory writes (plan P17.T2).

Flood protection for the memory pipeline, per plan Phase 17: at most 20
writes per session per hour and 100 writes per user per day (both rolling
windows), plus burst detection — ``burst_max`` writes (default 10) within
``burst_window_seconds`` (default 60) are flagged as suspicious but still
allowed, surfacing through :attr:`RateDecision.reason`.

Windows are rolling and per key: ``session_id`` keys the hour and burst
buckets, ``user_id`` the day bucket. Buckets are pruned lazily on access,
so stale sessions cost no more than one empty deque entry each. The clock
is injectable for deterministic tests, and :meth:`RateLimiter.allow` also
accepts an explicit ``now``; timestamps must be timezone-aware.

Denials do not consume budget — the store raises
:class:`RateLimitExceeded` *before* any file write or Git commit, so a
rejected write leaves the repository untouched.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hermes.memory.types import to_utc

__all__ = ["RateDecision", "RateLimitExceeded", "RateLimiter"]


@dataclass(frozen=True)
class RateDecision:
    """Outcome of one rate-limit check (plan P17.T2).

    ``allowed=False`` means the write must be rejected. ``reason`` is a
    stable token: ``"ok"``, ``"burst"`` (allowed but suspicious — the burst
    threshold was just crossed), ``"session-hour-limit"`` or
    ``"user-day-limit"`` (denied).
    """

    allowed: bool
    reason: str


class RateLimitExceeded(RuntimeError):  # noqa: N818 - name mandated by plan P17.T2
    """A memory write was denied by the rate limiter (plan P17.T2).

    Carries the :class:`RateDecision` that caused the denial.
    """

    def __init__(self, decision: RateDecision) -> None:
        self.decision = decision
        super().__init__(f"memory write rate-limited: {decision.reason}")


class RateLimiter:
    """Rolling-window + burst rate limiter for memory writes (plan P17.T2).

    Defaults mirror the plan: 20 writes per session per hour, 100 writes
    per user per day, burst alert at 10 writes within 60 seconds. All
    thresholds are constructor-configurable (the plan requires
    ``config.yaml`` configurability — callers map config values onto these
    kwargs) and the clock is injectable.
    """

    def __init__(
        self,
        max_per_session_per_hour: int = 20,
        max_per_day_per_user: int = 100,
        burst_window_seconds: float = 60.0,
        burst_max: int = 10,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_per_session_per_hour < 1:
            raise ValueError("max_per_session_per_hour must be >= 1")
        if max_per_day_per_user < 1:
            raise ValueError("max_per_day_per_user must be >= 1")
        if burst_window_seconds <= 0:
            raise ValueError("burst_window_seconds must be > 0")
        if burst_max < 1:
            raise ValueError("burst_max must be >= 1")
        self.max_per_session_per_hour = max_per_session_per_hour
        self.max_per_day_per_user = max_per_day_per_user
        self.burst_window_seconds = burst_window_seconds
        self.burst_max = burst_max
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._session_hour: dict[str, deque[datetime]] = {}
        self._user_day: dict[str, deque[datetime]] = {}
        self._session_burst: dict[str, deque[datetime]] = {}

    @staticmethod
    def _prune(bucket: deque[datetime], cutoff: datetime) -> deque[datetime]:
        """Drop timestamps at or before ``cutoff`` (rolling-window pruning)."""
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        return bucket

    def allow(
        self,
        session_id: str,
        user_id: str,
        now: datetime | None = None,
    ) -> RateDecision:
        """Check (and, if allowed, record) one write for ``session``/``user``.

        ``now=None`` stamps the injected clock. Hard limits are checked
        first (session-hour, then user-day); a write that passes them but
        just crossed the burst threshold is allowed yet flagged with
        ``reason="burst"``. Denied writes are not recorded.
        """
        moment = to_utc(now, "now") if now is not None else to_utc(self._clock(), "clock()")

        session_hour = self._prune(
            self._session_hour.setdefault(session_id, deque()),
            moment - timedelta(hours=1),
        )
        if len(session_hour) >= self.max_per_session_per_hour:
            return RateDecision(
                allowed=False,
                reason="session-hour-limit",
            )

        user_day = self._prune(
            self._user_day.setdefault(user_id, deque()),
            moment - timedelta(days=1),
        )
        if len(user_day) >= self.max_per_day_per_user:
            return RateDecision(allowed=False, reason="user-day-limit")

        session_burst = self._prune(
            self._session_burst.setdefault(session_id, deque()),
            moment - timedelta(seconds=self.burst_window_seconds),
        )
        burst = len(session_burst) + 1 >= self.burst_max

        session_hour.append(moment)
        user_day.append(moment)
        session_burst.append(moment)
        if burst:
            return RateDecision(allowed=True, reason="burst")
        return RateDecision(allowed=True, reason="ok")
