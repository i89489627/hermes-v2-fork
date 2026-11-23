"""Cron run inactivity watchdog (plan P21.T5, issue #156160).

Upstream bug: a cron job's agent run stayed in the ``running`` state for
hours — no watchdog fired, subsequent ticks were silently skipped, and no
incident was ever logged, so the user discovered the wedge only by noticing
missing outputs.

The v2 fix is an explicit inactivity watchdog:

* every run carries ``last_progress_at``, refreshed by :meth:`Watchdog.heartbeat`
  calls from the run wrapper (and optionally by long-running job bodies
  themselves);
* :meth:`Watchdog.check_once` force-fails any run in the ``running`` state
  whose ``last_progress_at`` is older than the configurable inactivity
  timeout (default 30 minutes), records a :class:`CronIncident`, logs it at
  ERROR level, and alerts via the notifier seam;
* alerting follows the single-path preference "watchdog DETECTS, diagnostics
  REPORTS": callers bind the notifier to the DiagnosticsSystem when it is
  deployed, or to a direct gateway-message fallback when it is not. The
  watchdog itself never owns a transport.

Force-fail is a one-way door: a run already ``force_failed`` can never be
flipped back to ``done``/``failed`` by a late worker thread, so a stuck run
that eventually completes cannot launder its incident away.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

__all__ = [
    "ALERT_CRON_JOB_STUCK",
    "DEFAULT_INACTIVITY_TIMEOUT_SECONDS",
    "CronIncident",
    "CronJobRun",
    "Watchdog",
    "create_run",
]

#: Default inactivity timeout: 30 minutes without progress => force-fail.
DEFAULT_INACTIVITY_TIMEOUT_SECONDS: Final[float] = 1800.0

#: Alert name surfaced through the notifier seam (diagnostics or gateway
#: fallback). Callers route ``cron_job_stuck`` to their monitoring of choice.
ALERT_CRON_JOB_STUCK: Final[str] = "cron_job_stuck"

#: Run lifecycle states (plan P21.T5 design).
RUN_STATE_QUEUED: Final[str] = "queued"
RUN_STATE_RUNNING: Final[str] = "running"
RUN_STATE_DONE: Final[str] = "done"
RUN_STATE_FAILED: Final[str] = "failed"
RUN_STATE_FORCE_FAILED: Final[str] = "force_failed"

_VALID_STATES: Final[frozenset[str]] = frozenset({
    RUN_STATE_QUEUED,
    RUN_STATE_RUNNING,
    RUN_STATE_DONE,
    RUN_STATE_FAILED,
    RUN_STATE_FORCE_FAILED,
})

_LOGGER = logging.getLogger("hermes.cron.watchdog")

#: Monotonic clock used unless the caller injects one (tests inject).
_MONOTONIC_CLOCK: Final[Callable[[], float]] = time.monotonic


@dataclass
class CronIncident:
    """Incident record for a run the watchdog had to force-fail.

    Kept in :attr:`Watchdog.incidents` and on :attr:`CronJobRun.incident` so
    operators (and the diagnostics reporter) can answer "why did this cron
    job die?" from data instead of archaeology.
    """

    incident_id: str
    kind: str
    job_id: str
    run_id: str
    reason: str
    detected_at: float
    inactivity_seconds: float
    timeout_seconds: float


@dataclass
class CronJobRun:
    """One execution of a cron job, tracked by the watchdog.

    ``started_at`` / ``last_progress_at`` are seconds on the watchdog's clock
    (monotonic by default; tests inject a fake clock). ``error`` carries the
    failure text for ``failed`` runs; ``incident`` is set only on
    ``force_failed`` runs.
    """

    job_id: str
    run_id: str
    state: str = RUN_STATE_QUEUED
    started_at: float = 0.0
    last_progress_at: float = 0.0
    missed_ticks: int = 0
    error: str | None = None
    incident: CronIncident | None = None

    def __post_init__(self) -> None:
        if self.state not in _VALID_STATES:
            allowed = sorted(_VALID_STATES)
            raise ValueError(f"invalid run state {self.state!r}; expected one of {allowed}")


def create_run(
    job_id: str,
    *,
    clock: Callable[[], float] = _MONOTONIC_CLOCK,
    missed_ticks: int = 0,
) -> CronJobRun:
    """Build a fresh run in the ``queued`` state, stamped from ``clock``."""
    now = clock()
    return CronJobRun(
        job_id=job_id,
        run_id=uuid.uuid4().hex,
        state=RUN_STATE_QUEUED,
        started_at=now,
        last_progress_at=now,
        missed_ticks=missed_ticks,
    )


class Watchdog:
    """Inactivity watchdog for cron job runs (plan P21.T5, issue #156160).

    The watchdog DETECTS stuck runs and REPORTS through the injected seams:
    ``on_stuck`` (in-process reaction, e.g. scheduler bookkeeping) and
    ``notifier`` (alert transport — DiagnosticsSystem when deployed, direct
    gateway-message fallback otherwise), called as
    ``notifier(ALERT_CRON_JOB_STUCK, message)``.

    Args:
        inactivity_timeout_seconds: a ``running`` run with no heartbeat for
            longer than this is force-failed (default 30 minutes).
        check_interval_seconds: sleep between checks in the background
            :meth:`start` loop.
        on_stuck: optional callback invoked with the force-failed run.
        notifier: optional alert sink ``Callable[[alert_name, message], None]``.
        clock: injectable clock (default ``time.monotonic``); tests pass a
            fake so timeout behaviour is deterministic.
    """

    def __init__(
        self,
        *,
        inactivity_timeout_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
        check_interval_seconds: float = 1.0,
        on_stuck: Callable[[CronJobRun], None] | None = None,
        notifier: Callable[[str, str], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if inactivity_timeout_seconds <= 0:
            raise ValueError("inactivity_timeout_seconds must be > 0")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._check_interval_seconds = check_interval_seconds
        self._on_stuck = on_stuck
        self._notifier = notifier
        self._clock: Callable[[], float] = clock if clock is not None else _MONOTONIC_CLOCK
        self._lock = threading.Lock()
        self._runs: dict[str, CronJobRun] = {}
        self._incidents: list[CronIncident] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── registration / progress ────────────────────────────────────────────

    def register_run(self, run: CronJobRun) -> None:
        """Start tracking ``run`` (typically in the ``queued`` state)."""
        with self._lock:
            self._runs[run.run_id] = run

    def heartbeat(self, run_id: str) -> None:
        """Refresh ``last_progress_at`` for ``run_id`` to the current clock.

        Unknown run ids are logged (never silently ignored) and skipped, so a
        racing unregister cannot crash the job body that heartbeats.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                _LOGGER.warning("cron watchdog heartbeat for unknown run %s", run_id)
                return
            run.last_progress_at = self._clock()

    # ── state transitions ──────────────────────────────────────────────────

    def mark_running(self, run_id: str) -> None:
        """Move a queued run to ``running`` and stamp its progress clock."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                _LOGGER.warning("cron watchdog mark_running for unknown run %s", run_id)
                return
            if run.state not in (RUN_STATE_QUEUED, RUN_STATE_RUNNING):
                _LOGGER.warning(
                    "cron watchdog mark_running ignored for run %s in state %s",
                    run_id,
                    run.state,
                )
                return
            now = self._clock()
            run.state = RUN_STATE_RUNNING
            run.started_at = now
            run.last_progress_at = now

    def mark_done(self, run_id: str) -> None:
        """Mark ``run_id`` ``done`` unless the watchdog already force-failed it."""
        self._mark_terminal(run_id, RUN_STATE_DONE, None)

    def mark_failed(self, run_id: str, *, error: str | None = None) -> None:
        """Mark ``run_id`` ``failed`` unless the watchdog already force-failed it."""
        self._mark_terminal(run_id, RUN_STATE_FAILED, error)

    def _mark_terminal(self, run_id: str, state: str, error: str | None) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                _LOGGER.warning("cron watchdog %s for unknown run %s", state, run_id)
                return
            if run.state not in (RUN_STATE_QUEUED, RUN_STATE_RUNNING):
                # Already force_failed (or terminal): the watchdog's verdict
                # stands — a late worker must not overwrite it (#156160).
                _LOGGER.warning(
                    "cron watchdog %s ignored for run %s already in state %s",
                    state,
                    run_id,
                    run.state,
                )
                return
            run.state = state
            if error is not None:
                run.error = error

    # ── inspection ─────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> CronJobRun | None:
        """Return the tracked run with ``run_id``, or ``None``."""
        with self._lock:
            return self._runs.get(run_id)

    @property
    def runs(self) -> dict[str, CronJobRun]:
        """Snapshot of all tracked runs keyed by run id."""
        with self._lock:
            return dict(self._runs)

    @property
    def incidents(self) -> list[CronIncident]:
        """Copy of every incident the watchdog has recorded."""
        with self._lock:
            return list(self._incidents)

    # ── the check itself ───────────────────────────────────────────────────

    def check_once(self) -> list[CronJobRun]:
        """Force-fail every ``running`` run past the inactivity timeout.

        Returns the runs force-failed by *this* call (empty when all runs are
        healthy). Force-failing records a :class:`CronIncident` on the run and
        in :attr:`incidents`, logs it at ERROR level, then invokes ``on_stuck``
        and the notifier — both outside the lock, both failure-isolated so a
        broken callback can never wedge the watchdog itself.
        """
        now = self._clock()
        candidates: list[tuple[CronJobRun, float]] = []
        with self._lock:
            for run in self._runs.values():
                if run.state != RUN_STATE_RUNNING:
                    continue
                inactivity = now - run.last_progress_at
                if inactivity > self._inactivity_timeout_seconds:
                    candidates.append((run, inactivity))
        forced: list[CronJobRun] = []
        for run, inactivity in candidates:
            if self._force_fail(run, inactivity, now):
                forced.append(run)
        return forced

    def _force_fail(self, run: CronJobRun, inactivity: float, now: float) -> bool:
        """Force-fail ``run``; ``False`` means it left the running state first."""
        reason = (
            f"no progress for {inactivity:.3f}s "
            f"(inactivity timeout {self._inactivity_timeout_seconds:.3f}s)"
        )
        with self._lock:
            if run.state != RUN_STATE_RUNNING:
                return False  # heartbeat / completion landed first — healthy
            run.state = RUN_STATE_FORCE_FAILED
            incident = CronIncident(
                incident_id=uuid.uuid4().hex,
                kind=ALERT_CRON_JOB_STUCK,
                job_id=run.job_id,
                run_id=run.run_id,
                reason=reason,
                detected_at=now,
                inactivity_seconds=inactivity,
                timeout_seconds=self._inactivity_timeout_seconds,
            )
            run.incident = incident
            self._incidents.append(incident)
        _LOGGER.error(
            "cron_job_stuck: job %s run %s force-failed: %s (incident %s)",
            run.job_id,
            run.run_id,
            reason,
            incident.incident_id,
        )
        self._notify_stuck(run, incident)
        return True

    def _notify_stuck(self, run: CronJobRun, incident: CronIncident) -> None:
        """Fan out to the on_stuck callback and the notifier seam."""
        if self._on_stuck is not None:
            try:
                self._on_stuck(run)
            except Exception:
                _LOGGER.exception(
                    "cron watchdog on_stuck callback failed for run %s", run.run_id
                )
        if self._notifier is not None:
            message = (
                f"cron job '{run.job_id}' run {run.run_id} stuck and force-failed: "
                f"{incident.reason}. Incident {incident.incident_id} recorded."
            )
            try:
                self._notifier(ALERT_CRON_JOB_STUCK, message)
            except Exception:
                _LOGGER.exception(
                    "cron watchdog notifier failed for run %s", run.run_id
                )

    # ── background thread ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background check loop (daemon thread, idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._loop, name="hermes-cron-watchdog", daemon=True
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the background loop to stop and join it."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _loop(self) -> None:
        interval = self._check_interval_seconds
        while not self._stop_event.wait(interval):
            try:
                self.check_once()
            except Exception:
                # The watchdog must never die silently (#156160): every check
                # failure is logged, and the loop keeps going.
                _LOGGER.exception("cron watchdog check loop failed")
