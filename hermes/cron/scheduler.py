"""Minimal cron scheduler wired to the inactivity watchdog (plan P21.T5, issue #156160).

Upstream bug shape: a cron job's agent run hung in ``running`` for hours,
later ticks were *silently* skipped, and nothing was logged — the user only
saw the wedge through missing outputs.

The v2 :class:`CronScheduler` closes that loop:

* :meth:`CronScheduler.tick` dispatches every due job, and if a tick is late
  it is still executed (never dropped) with the number of fully-skipped
  periods recorded on the run and logged at WARNING — ticks are never
  silently skipped;
* each dispatched run moves ``queued -> running -> done`` (or ``failed``) in
  a dedicated worker thread, and the run wrapper heartbeats the watchdog so
  progress is visible;
* the :class:`~hermes.cron.watchdog.Watchdog` force-fails any run that stops
  making progress, records an incident, and alerts through its notifier seam
  (diagnostics when deployed, gateway-message fallback otherwise).

Job bodies that run long can emit their own progress by declaring a
``heartbeat`` keyword parameter — the wrapper passes a zero-argument
progress reporter when the signature accepts one. A body that never reports
progress is exactly the stuck-run signature the watchdog exists to catch.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from hermes.cron.watchdog import (
    DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    CronJobRun,
    Watchdog,
    create_run,
)

__all__ = ["DEFAULT_HEARTBEAT_PARAM", "CronScheduler", "parse_schedule_seconds"]

_LOGGER = logging.getLogger("hermes.cron.scheduler")

#: Keyword parameter job bodies may declare to receive the progress reporter.
DEFAULT_HEARTBEAT_PARAM: Final[str] = "heartbeat"

_MONOTONIC_CLOCK: Final[Callable[[], float]] = time.monotonic

_INTERVAL_SUFFIXES: Final[dict[str, float]] = {
    "": 1.0,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def parse_schedule_seconds(schedule: float | str) -> float:
    """Parse a schedule into an interval in seconds.

    Accepts a positive number of seconds (``float``/``int``) or a string of
    the form ``"<number><unit>"`` with unit ``s``/``m``/``h`` (seconds default
    when the unit is omitted), e.g. ``"30s"``, ``"5m"``, ``"1h"``.
    """
    if isinstance(schedule, (int, float)):
        seconds = float(schedule)
        origin = "interval"
    else:
        text = schedule.strip().lower()
        digits = text.rstrip("smh")
        suffix = text[len(digits):]
        if not digits or suffix not in _INTERVAL_SUFFIXES:
            raise ValueError(
                f"invalid schedule {schedule!r}; expected seconds or '<n>s'/'<n>m'/'<n>h'"
            )
        try:
            seconds = float(digits) * _INTERVAL_SUFFIXES[suffix]
        except ValueError as exc:
            raise ValueError(f"invalid schedule {schedule!r}: {exc}") from exc
        origin = f"schedule {schedule!r}"
    if seconds <= 0:
        raise ValueError(f"{origin} must be > 0 seconds, got {seconds}")
    return seconds


@dataclass
class _RegisteredJob:
    """Internal bookkeeping for a registered cron job."""

    job_id: str
    interval_seconds: float
    fn: Callable[..., object]
    accepts_heartbeat: bool
    next_due: float


@dataclass
class _DispatchedRun:
    """Pairing of a dispatched run with its worker bookkeeping."""

    run: CronJobRun


class CronScheduler:
    """Cron scheduler that records watchdog-backed runs for every tick.

    Args:
        clock: injectable clock shared with the watchdog (tests pass a fake).
        watchdog: externally supplied watchdog; when ``None`` one is created
            with ``inactivity_timeout_seconds`` and the same clock.
        inactivity_timeout_seconds: watchdog timeout used when creating the
            internal watchdog (default 30 minutes, per plan P21.T5).
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        watchdog: Watchdog | None = None,
        inactivity_timeout_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    ) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else _MONOTONIC_CLOCK
        self._watchdog = watchdog if watchdog is not None else Watchdog(
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            clock=self._clock,
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, _RegisteredJob] = {}
        self._dispatched: dict[str, _DispatchedRun] = {}

    # ── registration ───────────────────────────────────────────────────────

    def register(self, job_id: str, schedule: float | str, fn: Callable[..., object]) -> None:
        """Register ``job_id`` to fire every ``schedule`` interval.

        The job is due on the next :meth:`tick` after registration. Repeated
        registration for the same id replaces the previous entry (and resets
        its due time).
        """
        interval = parse_schedule_seconds(schedule)
        with self._lock:
            self._jobs[job_id] = _RegisteredJob(
                job_id=job_id,
                interval_seconds=interval,
                fn=fn,
                accepts_heartbeat=_accepts_heartbeat(fn),
                next_due=self._clock(),
            )

    # ── ticking ────────────────────────────────────────────────────────────

    def tick(self, now: float | None = None) -> list[CronJobRun]:
        """Execute every job due at ``now`` (default: the injected clock).

        Returns the runs dispatched by this tick. A late tick still executes
        the job — it is never silently skipped — and records how many full
        schedule periods were missed on the run plus a WARNING log line.
        """
        current = self._clock() if now is None else now
        dispatched: list[CronJobRun] = []
        with self._lock:
            for job in list(self._jobs.values()):
                if current < job.next_due:
                    continue
                missed = int((current - job.next_due) / job.interval_seconds)
                if missed > 0:
                    _LOGGER.warning(
                        "cron job %s ticked %.3fs late: %d full tick(s) skipped — "
                        "executing now, not silently skipped",
                        job.job_id,
                        current - job.next_due,
                        missed,
                    )
                run = create_run(job.job_id, clock=self._clock, missed_ticks=missed)
                self._watchdog.register_run(run)
                entry = _DispatchedRun(run=run)
                self._dispatched[run.run_id] = entry
                job.next_due = current + job.interval_seconds
                thread = threading.Thread(
                    target=self._run_wrapper,
                    args=(job, entry),
                    name=f"hermes-cron-{job.job_id}-{run.run_id[:8]}",
                    daemon=True,
                )
                thread.start()
                dispatched.append(run)
        return dispatched

    # ── run access / waiting ───────────────────────────────────────────────

    def get_run(self, run_id: str) -> CronJobRun | None:
        """Return a dispatched run by id, or ``None``."""
        with self._lock:
            entry = self._dispatched.get(run_id)
        return None if entry is None else self._watchdog.get_run(run_id)

    def runs(self) -> list[CronJobRun]:
        """Snapshot of every run dispatched so far."""
        with self._lock:
            run_ids = list(self._dispatched)
        return [
            run
            for run_id in run_ids
            if (run := self._watchdog.get_run(run_id)) is not None
        ]

    def wait_for_state(self, run_id: str, state: str, timeout: float = 5.0) -> bool:
        """Wait until ``run_id`` reaches ``state``; ``False`` on timeout.

        The worker thread flips a run to ``running`` asynchronously, so
        callers (and tests) that must observe a specific state poll instead
        of racing the dispatch.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.get_run(run_id)
            if run is not None and run.state == state:
                return True
            time.sleep(0.005)
        return False

    @property
    def watchdog(self) -> Watchdog:
        """The watchdog every dispatched run is registered with."""
        return self._watchdog

    # ── execution ──────────────────────────────────────────────────────────

    def _run_wrapper(self, job: _RegisteredJob, entry: _DispatchedRun) -> None:
        """Worker body: queued -> running -> done/failed with heartbeats."""
        run = entry.run
        watchdog = self._watchdog

        def heartbeat() -> None:
            watchdog.heartbeat(run.run_id)

        try:
            watchdog.mark_running(run.run_id)
            heartbeat()
            if job.accepts_heartbeat:
                job.fn(heartbeat=heartbeat)
            else:
                job.fn()
        except Exception as exc:
            _LOGGER.exception("cron job %s run %s failed", job.job_id, run.run_id)
            watchdog.mark_failed(run.run_id, error=repr(exc))
        else:
            watchdog.mark_done(run.run_id)
            heartbeat()


def _accepts_heartbeat(fn: Callable[..., object]) -> bool:
    """True when ``fn`` declares a ``heartbeat`` parameter the wrapper can pass."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(DEFAULT_HEARTBEAT_PARAM)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
