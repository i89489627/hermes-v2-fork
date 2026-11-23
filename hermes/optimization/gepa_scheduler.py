"""Weekly GEPA cron (plan P10.T6).

Wires :meth:`~hermes.optimization.gepa_runner.GEPARunner.run_weekly` into the
:class:`~hermes.cron.scheduler.CronScheduler` so prompt optimization runs
automatically once a week (the plan Phase 10 acceptance "cron runs weekly
automatically"; the plan text places the run on Sundays at 03:00 — the
CronScheduler is an interval scheduler without wall-clock anchoring, so the
weekly cadence is expressed as a 7-day interval and the deployment wiring
registers the job at startup; the plan's Sunday-03:00 placement is the
production launch time of the interval, not a property of this module).

Cadence: the job is registered with schedule :data:`GEPA_SCHEDULE`
(``"168h"``). :func:`~hermes.cron.scheduler.parse_schedule_seconds` accepts
second/minute/hour strings or plain seconds but has no ``"d"`` unit, so the
weekly period is spelled ``"168h"`` (= 7 * 86400 s). The job is due on the
first :meth:`~hermes.cron.scheduler.CronScheduler.tick` after registration,
then re-fires every 7 days; late ticks still execute (never silently
skipped) and land in the watchdog's run records like any other cron job.

Manual trigger: :meth:`GepaScheduler.run_once` is the entry point the
``/optimization run-gepa`` command calls (plan P10.T6 acceptance "manual
trigger via /optimization run-gepa"). The command wiring itself lives in the
command layer — this module owns the schedule and returns the runner's
:class:`~hermes.optimization.gepa_runner.OptimizationResult` (candidate
counts and deployments) for the command to report.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from hermes.cron.scheduler import CronScheduler
from hermes.optimization.gepa_runner import GEPARunner, OptimizationResult

__all__ = ["GEPA_JOB_ID", "GEPA_SCHEDULE", "GepaScheduler"]

#: Cron job id the weekly GEPA run registers under (plan P10.T6).
GEPA_JOB_ID: str = "gepa-weekly"

#: Weekly cadence in CronScheduler schedule format: ``"168h"`` — the parser
#: supports ``s``/``m``/``h`` suffixes (no ``d``), so 7 days == 604800 s.
GEPA_SCHEDULE: str = "168h"


class GepaScheduler:
    """Registers the weekly GEPA optimization run as a cron job (plan P10.T6).

    Args:
        scheduler: the :class:`~hermes.cron.scheduler.CronScheduler` to
            register the weekly job with (its watchdog tracks every run).
        runner: the :class:`~hermes.optimization.gepa_runner.GEPARunner`
            whose ``run_weekly`` is the job body.
        components: optional selection narrow-down for the weekly run —
            ``None`` (default) runs every configured component; a single
            component name or a collection narrows the run (same grammar as
            :meth:`GEPARunner.run_weekly`).
    """

    def __init__(
        self,
        scheduler: CronScheduler,
        runner: GEPARunner,
        components: str | Collection[str] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._runner = runner
        #: The weekly selection, kept verbatim (``None`` / str / collection).
        self._components: str | Collection[str] | None = components

    def register(self) -> None:
        """Register the weekly job on the scheduler.

        The job fires on the next
        :meth:`~hermes.cron.scheduler.CronScheduler.tick` and then every
        :data:`GEPA_SCHEDULE` (168h); re-registering replaces the previous
        entry and resets its due time (scheduler semantics).
        """
        self._scheduler.register(GEPA_JOB_ID, GEPA_SCHEDULE, self._run_job)

    def _run_job(self) -> OptimizationResult:
        """The cron body: one weekly run over the wired selection."""
        return self.run_once()

    def run_once(
        self,
        components: str | Collection[str] | None = None,
        now: datetime | None = None,
    ) -> OptimizationResult:
        """One manual GEPA run — the ``/optimization run-gepa`` entry point.

        ``components`` overrides the wired selection for this run (``None``
        keeps the wired selection, which defaults to every configured
        component); ``now`` pins the run id and trace timestamps exactly as
        :meth:`GEPARunner.run_weekly` does (an aware datetime, or naive
        interpreted as UTC). Returns the run's
        :class:`~hermes.optimization.gepa_runner.OptimizationResult` —
        per-component candidate counts and the deployed prompts — for the
        command to report.
        """
        selection = self._components if components is None else components
        return self._runner.run_weekly(components=selection, now=now)
