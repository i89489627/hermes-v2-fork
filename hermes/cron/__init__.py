"""hermes-v2 cron package (plan P21.T5, issue #156160).

Inactivity watchdog + minimal scheduler for v2 cron runs. Fixes the upstream
bug where a cron job's agent run stayed ``running`` for hours with no
watchdog firing, ticks silently skipped, and no incident logged: every run
carries a ``last_progress_at`` stamp refreshed by heartbeats, and the
:class:`hermes.cron.watchdog.Watchdog` force-fails runs that stop making
progress, records an incident, and alerts through a notifier seam
(diagnostics when deployed, direct gateway-message fallback otherwise).
"""

from __future__ import annotations

from hermes.cron.scheduler import CronScheduler
from hermes.cron.watchdog import (
    ALERT_CRON_JOB_STUCK,
    CronIncident,
    CronJobRun,
    Watchdog,
    create_run,
)

__all__ = [
    "ALERT_CRON_JOB_STUCK",
    "CronIncident",
    "CronJobRun",
    "CronScheduler",
    "Watchdog",
    "create_run",
]
