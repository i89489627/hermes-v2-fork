"""Chaos test: a long-running cron job that reports progress never sticks
(plan P21.T5, issue #156160).

Upstream bug shape: cron runs were treated as binary (fire and forget), so a
job that legitimately ran for a long time was indistinguishable from a wedged
one — and wedged runs were never reaped, with ticks silently skipped. Here a
2-second job that heartbeats every 100ms survives a much tighter 0.5s
inactivity timeout, while a silently-hung job on the same scheduler is
force-failed and alerted — the watchdog must tell those two worlds apart.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from hermes.cron.scheduler import CronScheduler
from hermes.cron.watchdog import (
    ALERT_CRON_JOB_STUCK,
    RUN_STATE_DONE,
    RUN_STATE_FORCE_FAILED,
    RUN_STATE_RUNNING,
    Watchdog,
)

#: Long job duration / heartbeat cadence (chaos scale: seconds, not minutes).
_LONG_JOB_SECONDS = 2.0
_HEARTBEAT_EVERY = 0.1
#: Inactivity timeout the long job must survive via heartbeats.
_TIMEOUT_SECONDS = 0.5
_WATCHDOG_POLL = 0.05


def _long_job(heartbeat: Callable[[], None]) -> None:
    """A legitimately slow job: 2s of work, progress reported every 100ms."""
    beats = int(_LONG_JOB_SECONDS / _HEARTBEAT_EVERY)
    for _ in range(beats):
        time.sleep(_HEARTBEAT_EVERY)
        heartbeat()


def _hung_job() -> None:
    """A wedged job: grabs the running state and never reports progress."""
    time.sleep(3.0)


@pytest.mark.chaos
def test_long_running_does_not_stick() -> None:
    """Heartbeats keep a slow job alive; a silent hang is force-failed and alerted."""
    alerts: list[tuple[str, str]] = []
    watchdog = Watchdog(
        inactivity_timeout_seconds=_TIMEOUT_SECONDS,
        check_interval_seconds=_WATCHDOG_POLL,
        notifier=lambda name, message: alerts.append((name, message)),
    )
    scheduler = CronScheduler(watchdog=watchdog)
    scheduler.register("long-runner", "1h", _long_job)
    scheduler.register("silent-hang", "1h", _hung_job)

    watchdog.start()
    try:
        dispatched = scheduler.tick()
        assert {run.job_id for run in dispatched} == {"long-runner", "silent-hang"}
        by_job = {run.job_id: run for run in dispatched}
        long_run = by_job["long-runner"]
        hung_run = by_job["silent-hang"]

        assert scheduler.wait_for_state(hung_run.run_id, RUN_STATE_RUNNING), (
            "hung job must reach running before the watchdog judges it"
        )
        assert scheduler.wait_for_state(long_run.run_id, RUN_STATE_RUNNING)

        # The hung job trips the watchdog while the long job is still working.
        assert scheduler.wait_for_state(hung_run.run_id, RUN_STATE_FORCE_FAILED, timeout=5.0), (
            "a run with no progress for 0.5s must be force-failed"
        )
        assert hung_run.state == RUN_STATE_FORCE_FAILED
        assert hung_run.incident is not None
        assert hung_run.incident.kind == ALERT_CRON_JOB_STUCK
        assert hung_run.incident.job_id == "silent-hang"

        # The long job runs the full 2s — 4x the inactivity timeout — and
        # heartbeats carry it through to a normal done, untouched by the
        # watchdog (chaos: long-running is NOT the same as stuck).
        assert scheduler.wait_for_state(long_run.run_id, RUN_STATE_DONE, timeout=10.0)
        assert long_run.state == RUN_STATE_DONE
        assert long_run.incident is None
    finally:
        watchdog.stop()

    # Exactly one alert: for the silent hang, never for the long runner.
    stuck_alerts = [(name, msg) for name, msg in alerts if name == ALERT_CRON_JOB_STUCK]
    assert len(stuck_alerts) == 1
    assert "silent-hang" in stuck_alerts[0][1]
    assert "long-runner" not in stuck_alerts[0][1]
