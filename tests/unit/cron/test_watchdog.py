"""Unit tests: cron inactivity watchdog (plan P21.T5, issue #156160).

Upstream bug: cron agent runs sat in ``running`` for hours with no watchdog
firing and no incident logged. These tests pin the watchdog contract on an
injected fake clock: a run that stops heartbeating is force-failed exactly
once after the inactivity timeout, while a run that keeps making progress is
never touched.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from hermes.cron.watchdog import (
    ALERT_CRON_JOB_STUCK,
    RUN_STATE_FORCE_FAILED,
    RUN_STATE_RUNNING,
    CronJobRun,
    Watchdog,
    create_run,
)


class FakeClock:
    """Thread-safe injectable clock so timeout behaviour is deterministic."""

    def __init__(self, start: float = 100.0) -> None:
        self._lock = threading.Lock()
        self._now = start

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


def _running_run(job_id: str, clock: Callable[[], float]) -> CronJobRun:
    """Create a run and move it into the ``running`` state on ``clock``."""
    run = create_run(job_id, clock=clock)
    return run


@pytest.mark.unit
def test_timeout_after_inactivity() -> None:
    """A run past the inactivity timeout is force-failed; a heartbeating run is not.

    Two runs go ``running`` at t=100. Run A never reports progress again;
    run B heartbeats at t=120. At t=131 only run A exceeds the 30s timeout.
    """
    clock = FakeClock(start=100.0)
    watchdog = Watchdog(inactivity_timeout_seconds=30.0, clock=clock)

    stale = _running_run("nightly-report", clock)
    fresh = _running_run("quick-sync", clock)
    watchdog.register_run(stale)
    watchdog.register_run(fresh)
    watchdog.mark_running(stale.run_id)
    watchdog.mark_running(fresh.run_id)
    assert stale.state == RUN_STATE_RUNNING
    assert stale.started_at == pytest.approx(100.0)

    clock.advance(20.0)
    watchdog.heartbeat(fresh.run_id)  # last progress at t=120

    clock.advance(11.0)  # t=131: stale silent for 31s, fresh for 11s
    forced = watchdog.check_once()

    assert [run.run_id for run in forced] == [stale.run_id]
    assert stale.state == RUN_STATE_FORCE_FAILED
    assert fresh.state == RUN_STATE_RUNNING
    incident = stale.incident
    assert incident is not None
    assert incident.kind == ALERT_CRON_JOB_STUCK
    assert incident.job_id == "nightly-report"
    assert incident.run_id == stale.run_id
    assert incident.timeout_seconds == pytest.approx(30.0)
    assert incident.inactivity_seconds == pytest.approx(31.0)
    assert incident.detected_at == pytest.approx(131.0)
    assert watchdog.incidents == [incident]

    # Idempotent: the run is already force-failed, a second check finds nothing.
    assert watchdog.check_once() == []


@pytest.mark.unit
def test_force_fail_stuck_run() -> None:
    """Force-fail records the incident, fires on_stuck, alerts, and is one-way."""
    clock = FakeClock(start=0.0)
    stuck_seen: list[CronJobRun] = []
    alerts: list[tuple[str, str]] = []
    watchdog = Watchdog(
        inactivity_timeout_seconds=0.2,
        clock=clock,
        on_stuck=stuck_seen.append,
        notifier=lambda name, message: alerts.append((name, message)),
    )

    run = create_run("nightly-report", clock=clock)
    watchdog.register_run(run)
    watchdog.mark_running(run.run_id)
    clock.advance(0.5)

    forced = watchdog.check_once()

    assert forced == [run]
    assert run.state == RUN_STATE_FORCE_FAILED
    assert run.incident is not None
    assert len(watchdog.incidents) == 1
    assert stuck_seen == [run]

    assert len(alerts) == 1
    alert_name, alert_message = alerts[0]
    assert alert_name == ALERT_CRON_JOB_STUCK
    assert "nightly-report" in alert_message
    assert run.run_id in alert_message

    # One-way door: a late worker thread finishing after the force-fail must
    # not flip the run back to done, and the incident must survive.
    watchdog.mark_done(run.run_id)
    assert run.state == RUN_STATE_FORCE_FAILED
    assert len(watchdog.incidents) == 1
    assert watchdog.check_once() == []


@pytest.mark.unit
def test_inactivity_timeout_exactly_at_boundary() -> None:
    """The inactivity comparison is strict (P20.T6): exactly-at-timeout lives.

    Mutation-gate boundary test for the Tier-1 watchdog logic: a run whose
    inactivity equals the timeout exactly is NOT force-failed; the check
    needs inactivity strictly greater than the timeout. A heartbeat that
    lands exactly at the check instant also keeps the run alive (the
    ``last_progress_at`` refresh beats the timeout at equality).
    """
    clock = FakeClock(start=100.0)
    watchdog = Watchdog(inactivity_timeout_seconds=30.0, clock=clock)

    run = _running_run("nightly-report", clock)
    watchdog.register_run(run)
    watchdog.mark_running(run.run_id)  # last progress at t=100

    # t=130: inactivity is exactly 30.0s == timeout -> still healthy
    clock.advance(30.0)
    assert watchdog.check_once() == []
    assert run.state == RUN_STATE_RUNNING
    assert run.incident is None

    # a heartbeat exactly at the check instant refreshes the window
    watchdog.heartbeat(run.run_id)  # last progress now t=130

    # t=160: inactivity exactly 30.0s again -> still healthy
    clock.advance(30.0)
    assert watchdog.check_once() == []
    assert run.state == RUN_STATE_RUNNING

    # t=160.5: inactivity 30.5s > 30.0s -> force-failed, with the exact
    # inactivity duration recorded in the incident
    clock.advance(0.5)
    forced = watchdog.check_once()
    assert [stale.run_id for stale in forced] == [run.run_id]
    assert run.state == RUN_STATE_FORCE_FAILED
    incident = run.incident
    assert incident is not None
    assert incident.inactivity_seconds == pytest.approx(30.5)
    assert incident.timeout_seconds == pytest.approx(30.0)
