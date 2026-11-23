"""Integration tests: cron scheduler + inactivity watchdog (plan P21.T5, issue #156160).

Exercises the real dispatch path — :class:`hermes.cron.scheduler.CronScheduler`
running a hanging job on a real worker thread while the watchdog watches it on
an injected clock — and asserts the two observability guarantees the upstream
bug violated: the incident is logged, and a ``cron_job_stuck`` alert is sent
through the notifier seam (diagnostics when deployed, gateway-message
fallback otherwise).
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from hermes.cron.scheduler import CronScheduler
from hermes.cron.watchdog import (
    ALERT_CRON_JOB_STUCK,
    RUN_STATE_DONE,
    RUN_STATE_FORCE_FAILED,
    RUN_STATE_RUNNING,
    Watchdog,
)


class FakeClock:
    """Injectable clock shared by scheduler and watchdog (thread-safe)."""

    def __init__(self, start: float = 1000.0) -> None:
        self._lock = threading.Lock()
        self._now = start

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


def _hung_job() -> None:
    """Job body that hangs beyond the scaled watchdog timeout."""
    time.sleep(0.8)


@pytest.mark.integration
def test_incident_logged_on_stuck(caplog: pytest.LogCaptureFixture) -> None:
    """A hung run is force-failed, stays force-failed, and logs a cron_job_stuck incident."""
    clock = FakeClock()
    scheduler = CronScheduler(clock=clock, inactivity_timeout_seconds=0.2)
    scheduler.register("hung-job", "60s", _hung_job)

    [run] = scheduler.tick()
    assert run.job_id == "hung-job"
    assert scheduler.wait_for_state(run.run_id, RUN_STATE_RUNNING), (
        "worker thread must move the run to running"
    )

    with caplog.at_level(logging.ERROR, logger="hermes.cron.watchdog"):
        clock.advance(1.0)  # 1s of silence against a 0.2s timeout
        forced = scheduler.watchdog.check_once()

    assert [stuck.run_id for stuck in forced] == [run.run_id]
    assert run.state == RUN_STATE_FORCE_FAILED

    incidents = scheduler.watchdog.incidents
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident.kind == ALERT_CRON_JOB_STUCK
    assert incident.job_id == "hung-job"
    assert incident.run_id == run.run_id
    assert incident.inactivity_seconds == pytest.approx(1.0)

    # The incident is logged, not just recorded (#156160: "no incident logged").
    assert "cron_job_stuck" in caplog.text
    assert "hung-job" in caplog.text

    # One-way door across the real thread boundary: when the hung body finally
    # returns, its mark_done must not overwrite the watchdog's verdict.
    time.sleep(1.0)  # _hung_job sleeps 0.8s — its thread is finished by now
    assert run.state == RUN_STATE_FORCE_FAILED


@pytest.mark.integration
def test_alert_sent_on_stuck() -> None:
    """A stuck run surfaces exactly one cron_job_stuck alert; healthy runs never alert."""
    clock = FakeClock()
    alerts: list[tuple[str, str]] = []
    watchdog = Watchdog(
        inactivity_timeout_seconds=0.2,
        clock=clock,
        notifier=lambda name, message: alerts.append((name, message)),
    )
    scheduler = CronScheduler(clock=clock, watchdog=watchdog)
    scheduler.register("hung-job", "60s", _hung_job)

    [run] = scheduler.tick()
    assert scheduler.wait_for_state(run.run_id, RUN_STATE_RUNNING)

    clock.advance(1.0)
    [forced] = scheduler.watchdog.check_once()
    assert forced.run_id == run.run_id

    # check_once invokes the notifier inline, so the alert is already recorded.
    assert len(alerts) == 1
    alert_name, alert_message = alerts[0]
    assert alert_name == ALERT_CRON_JOB_STUCK
    assert "hung-job" in alert_message
    assert run.run_id in alert_message
    assert "force-failed" in alert_message

    # A healthy job on the same scheduler completes done and sends no alert.
    scheduler.register("clean-job", "60s", lambda: None)
    [clean] = scheduler.tick()
    assert scheduler.wait_for_state(clean.run_id, RUN_STATE_DONE)
    assert clean.state == RUN_STATE_DONE
    assert len(alerts) == 1  # still just the stuck job's alert
