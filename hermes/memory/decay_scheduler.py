"""Daily decay cron job (plan P2.T4).

Wires the forgetting-curve pass (:mod:`hermes.memory.decay`) into the
:class:`~hermes.cron.scheduler.CronScheduler` (plan P21.T5) so trust decay +
stale archiving run automatically once a day — the plan Phase 2 acceptance
"runs automatically daily".

Cadence: the job is registered with schedule :data:`DECAY_SCHEDULE`
(``"24h"``). :func:`~hermes.cron.scheduler.parse_schedule_seconds` accepts
second/m/hour strings (``"30s"``/``"5m"``/``"1h"``) or plain seconds but has
no ``"d"`` unit, so the daily period is spelled ``"24h"`` (= 86400 s). The
job is due on the first :meth:`~hermes.cron.scheduler.CronScheduler.tick`
after registration, then re-fires every 24h; a late tick still executes
(never silently skipped) and lands in the watchdog's run records like any
other cron job.

Manual trigger: :meth:`DecayScheduler.run_once` is the entry point the
``/diagnostics run-decay`` command calls (plan P2.T4 acceptance "manual
trigger via /diagnostics run-decay"). The command wiring itself lives in the
diagnostics/command layer — this module only owns the pass semantics and
returns a :class:`DecayRunResult` (decayed + archived counts) for the
command to report.

Pass ordering (why archive-before-decay): :meth:`run_once` runs
:func:`~hermes.memory.decay.archive_stale` BEFORE
:func:`~hermes.memory.decay.apply_decay`. An entry that is already stale at
pass start (trust < 0.2 AND >1 year unverified, the plan P8.T6 double rule)
archives with its current score in a single commit. The other order would
let :func:`~hermes.memory.decay.apply_decay` persist a decayed score via
:meth:`~hermes.memory.store.MemoryStore.update`, which re-anchors
``last_verified_at`` to now — the entry would then fail the ">1 year
unverified" half of the double rule on this pass (and every archive pass
would also burn a wasted decay commit on a file about to move). Survivors
of the archive sweep then get the forgetting-curve pass.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from hermes.cron.scheduler import CronScheduler
from hermes.memory.decay import apply_decay, archive_stale
from hermes.memory.store import MemoryStore

__all__ = ["DECAY_JOB_ID", "DECAY_SCHEDULE", "DecayRunResult", "DecayScheduler"]

#: Cron job id the daily decay pass registers under (plan P2.T4).
DECAY_JOB_ID: str = "memory-decay"

#: Daily cadence in CronScheduler schedule format: ``"24h"`` — the parser
#: supports ``s``/``m``/``h`` suffixes (no ``d``), so 1 day == 86400 s.
DECAY_SCHEDULE: str = "24h"


@dataclass(frozen=True)
class DecayRunResult:
    """Outcome of one decay pass (the ``/diagnostics run-decay`` payload).

    ``decayed`` — live entries whose trust score was lowered and persisted by
    :func:`~hermes.memory.decay.apply_decay`; ``archived`` — stale entries
    moved to ``memory-repo/archive/`` by
    :func:`~hermes.memory.decay.archive_stale` (plan P8.T6 double rule).
    """

    decayed: int
    archived: int


class DecayScheduler:
    """Registers the daily decay+archive pass as a cron job (plan P2.T4).

    Args:
        scheduler: the :class:`~hermes.cron.scheduler.CronScheduler` to
            register the daily job with (its watchdog tracks every run).
        store: the memory store the job decays and archives.
        clock: the injectable clock shared with ``scheduler`` (tests pass the
            same fake to both). Cadence decisions belong to the scheduler's
            tick loop — the clock is kept on the instance for wiring symmetry
            and test introspection only.
    """

    def __init__(
        self,
        scheduler: CronScheduler,
        store: MemoryStore,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

    @property
    def clock(self) -> Callable[[], float]:
        """The clock this wiring shares with its scheduler (injected or monotonic)."""
        return self._clock

    def register(self) -> None:
        """Register the daily job on the scheduler.

        The job fires on the next :meth:`~hermes.cron.scheduler.CronScheduler.tick`
        and then every :data:`DECAY_SCHEDULE` (24h); re-registering replaces
        the previous entry and resets its due time (scheduler semantics).
        """
        self._scheduler.register(DECAY_JOB_ID, DECAY_SCHEDULE, self._run_job)

    def _run_job(self) -> DecayRunResult:
        """The cron body: one decay pass over the wired store."""
        return self.run_once()

    def run_once(
        self,
        store: MemoryStore | None = None,
        now: datetime | None = None,
    ) -> DecayRunResult:
        """One manual decay pass — the ``/diagnostics run-decay`` entry point.

        Archive sweep first, forgetting curve second (module docstring
        explains the ordering). ``store`` defaults to the wired store;
        ``now`` pins the archive staleness clock and the decay curve (tests
        pass a fixed datetime; production leaves it None). The pass is
        idempotent within a day: re-running it immediately reports
        ``(decayed=0, archived=0)`` — recently verified entries are below
        the persist floor and nothing new crosses the staleness rule.
        """
        target = self._store if store is None else store
        archived = archive_stale(target, now=now)
        decayed = apply_decay(target, now=now)
        return DecayRunResult(decayed=decayed, archived=archived)
