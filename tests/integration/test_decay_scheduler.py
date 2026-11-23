"""P2.T4 acceptance: daily decay cron (plan Phase 2).

- :func:`test_cron_runs_daily` — the decay pass is registered on the
  :class:`~hermes.cron.scheduler.CronScheduler` with a 24h period, and a
  fake-clock tick schedule proves the cadence deterministically: the job is
  due on the first tick after registration (scheduler semantics), does NOT
  re-fire on ticks inside the 24h window, and fires again exactly 24h after
  each fire — with the store actually decayed by the dispatched run.
- :func:`test_manual_trigger` — :meth:`DecayScheduler.run_once` (the entry
  point the ``/diagnostics run-decay`` command calls; command wiring lives
  in the diagnostics layer) returns the decayed/archived counts over a
  seeded store and is idempotent on an immediate re-run.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.cron.scheduler import CronScheduler
from hermes.cron.watchdog import RUN_STATE_DONE
from hermes.memory.decay_scheduler import (
    DECAY_JOB_ID,
    DECAY_SCHEDULE,
    DecayScheduler,
)
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse, serialize

HOUR: float = 3600.0


class FakeClock:
    """Injectable monotonic-style clock, shared by scheduler and wiring."""

    def __init__(self, start: float = 1000.0) -> None:
        self._lock = threading.Lock()
        self._now = start

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


def _rewrite_card(store: MemoryStore, entry_id: str, **changes: object) -> None:
    """Edit an entry's trust-card frontmatter in place (test helper).

    Same helper shape as tests/unit/memory/test_decay.py: the store API has
    no "backdate" operation by design, so aging entries rewrites the card
    file directly with the public :func:`parse`/:func:`serialize` pair.
    """
    path = store.root / "global" / "facts" / f"{entry_id}.md"
    card, body = parse(path.read_text(encoding="utf-8"))
    for field, value in changes.items():
        setattr(card, field, value)
    path.write_text(serialize(card, body), encoding="utf-8")


@pytest.mark.integration
def test_cron_runs_daily(tmp_path: Path) -> None:
    """The decay job is registered with a 24h period and fires on that cadence."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    entry = store.add("Deploy note nobody re-checked in a month", type="Project")
    fixed_now = datetime.now(UTC)
    _rewrite_card(store, entry.id, last_verified_at=fixed_now - timedelta(days=30))

    clock = FakeClock()
    scheduler = CronScheduler(clock=clock)
    decay = DecayScheduler(scheduler, store, clock=clock)
    decay.register()

    # registration: the daily period, in the scheduler's registry
    assert DECAY_SCHEDULE == "24h"  # parser has no "d" unit; 24h == 86400 s
    job = scheduler._jobs[DECAY_JOB_ID]
    assert job.interval_seconds == 24 * HOUR == 86400.0

    # due on the first tick after registration: the run decays the entry
    [first] = scheduler.tick()
    assert first.job_id == DECAY_JOB_ID
    assert scheduler.wait_for_state(first.run_id, RUN_STATE_DONE)
    decayed = store.get(entry.id)
    assert decayed is not None
    assert decayed.trust_score < 1.0  # the forgetting curve actually ran

    # ticks inside the 24h window do not re-fire the daily job
    clock.advance(HOUR)
    assert scheduler.tick() == []
    clock.advance(22 * HOUR)  # 23h after the fire
    assert scheduler.tick() == []

    # exactly 24h after the fire, the job is due again
    clock.advance(HOUR)
    [second] = scheduler.tick()
    assert second.job_id == DECAY_JOB_ID
    assert scheduler.wait_for_state(second.run_id, RUN_STATE_DONE)
    assert second.run_id != first.run_id


@pytest.mark.integration
def test_manual_trigger(tmp_path: Path) -> None:
    """run_once — the /diagnostics run-decay entry point — returns real counts."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    fresh = store.add("Fresh fact from this morning's session", type="User")
    aged = store.add("Month-old deploy note nobody verified", type="Project")
    stale = store.add("Year-old claim that never held up", type="Reference")
    fixed_now = datetime.now(UTC)
    _rewrite_card(store, aged.id, last_verified_at=fixed_now - timedelta(days=30))
    _rewrite_card(
        store, stale.id, last_verified_at=fixed_now - timedelta(days=400), trust_score=0.05
    )

    scheduler = CronScheduler()
    decay = DecayScheduler(scheduler, store)

    # manual trigger (what /diagnostics run-decay calls): archive sweep first,
    # forgetting curve second, counts for both
    result = decay.run_once(store)

    assert result.decayed == 1  # the 30-day entry decayed and persisted
    assert result.archived == 1  # trust 0.05 AND >365 days unverified (P8.T6)
    assert (store.root / "archive" / f"{stale.id}.md").is_file()
    assert store.get(stale.id) is None

    updated = store.get(aged.id)
    assert updated is not None
    assert updated.trust_score < 1.0
    untouched = store.get(fresh.id)
    assert untouched is not None
    assert untouched.trust_score == pytest.approx(1.0)

    # the archived entry left the default search surface (plan P8.T6)
    assert stale.id not in [hit.id for hit in store.search("claim never held up")]
    assert [hit.id for hit in store.search("claim never held up", include_archived=True)] == [
        stale.id
    ]

    # idempotent within a day: an immediate re-run is a no-op
    again = decay.run_once()
    assert again.decayed == 0
    assert again.archived == 0
