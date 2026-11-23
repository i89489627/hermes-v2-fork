"""P10.T6 acceptance: weekly GEPA cron (plan Phase 10).

- :func:`test_cron_runs_weekly` — the weekly run is registered on the
  :class:`~hermes.cron.scheduler.CronScheduler` with a 7-day period, and a
  fake-clock day-by-day tick schedule proves the cadence deterministically:
  the job is due on the first tick after registration (scheduler
  semantics), does NOT re-fire on the next six daily ticks, and fires again
  exactly 7 days after each fire — with the runner actually executing and
  deploying (the prompt template file changes).
- :func:`test_manual_trigger` — :meth:`GepaScheduler.run_once` (the entry
  point the ``/optimization run-gepa`` command calls; command wiring lives
  in the command layer) drives a fake-model :class:`GEPARunner` over the
  real P10.T2 eval-set fixture and returns an
  :class:`~hermes.optimization.gepa_runner.OptimizationResult` with
  candidates and a deployed prompt.

Fixture pattern reused from tests/integration/test_gepa_runner.py:
deterministic fake runner (deliberately imperfect extractor output keyed
neither on prompt nor variant — candidate count comes from the loop shape,
not from score spread), real ``extractor_labeled.json`` eval set, and
template/pareto surfaces inside the per-test tmp_path.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hermes.cron.scheduler import CronScheduler
from hermes.cron.watchdog import RUN_STATE_DONE
from hermes.optimization.adapters.extractor_adapter import ExtractorAdapter
from hermes.optimization.gepa_runner import ComponentSpec, GEPARunner
from hermes.optimization.gepa_scheduler import (
    GEPA_JOB_ID,
    GEPA_SCHEDULE,
    GepaScheduler,
)
from hermes.optimization.pareto import ParetoFrontStore

EVAL_SETS = Path(__file__).resolve().parents[1] / "fixtures" / "eval_sets"

HOUR: float = 3600.0
DAY: float = 24 * HOUR


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


def _fake_runner(prompt: str, example: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, deliberately imperfect extractor drafts (partial F1)."""
    expected = example["expected_output"]
    facts = [str(fact) for fact in expected] if isinstance(expected, list) else []
    return {"drafts": facts[:1]}


def _seed(component: str) -> str:
    """Seed prompt template content (the pre-optimization deployment surface)."""
    return (
        f"You are the {component} stage of the hermes-v2 memory pipeline.\n"
        "Produce exactly the labeled output schema for every example."
    )


def _make_runner(tmp_path: Path) -> GEPARunner:
    """A one-component GEPARunner with template + Pareto surfaces in tmp_path."""
    template = tmp_path / "templates" / "extractor.txt"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(_seed("extractor"), encoding="utf-8")
    spec = ComponentSpec(
        component="extractor",
        adapter=ExtractorAdapter(_fake_runner),
        eval_set_path=EVAL_SETS / "extractor_labeled.json",
        prompt_template_path=template,
    )
    return GEPARunner(
        {"extractor": spec},
        pareto=ParetoFrontStore(path=tmp_path / "pareto.json"),
    )


@pytest.mark.integration
def test_cron_runs_weekly(tmp_path: Path) -> None:
    """The GEPA job is registered with a 7d period and fires on that cadence."""
    clock = FakeClock()
    scheduler = CronScheduler(clock=clock)
    runner = _make_runner(tmp_path)
    gepa = GepaScheduler(scheduler, runner)
    gepa.register()

    # registration: the weekly period, in the scheduler's registry
    assert GEPA_SCHEDULE == "168h"  # parser has no "d" unit; 168h == 604800 s
    job = scheduler._jobs[GEPA_JOB_ID]
    assert job.interval_seconds == 7 * DAY == 604800.0

    # due on the first tick after registration: the weekly run executes
    template = runner.components["extractor"].resolve_template_path()
    seed = template.read_text(encoding="utf-8")
    [first] = scheduler.tick()
    assert first.job_id == GEPA_JOB_ID
    assert scheduler.wait_for_state(first.run_id, RUN_STATE_DONE)
    assert template.read_text(encoding="utf-8") != seed  # deployed
    assert (tmp_path / "pareto.json").is_file()  # front populated

    # daily ticks across the following six days do not re-fire the weekly job
    for _day in range(6):
        clock.advance(DAY)
        assert scheduler.tick() == []

    # exactly 7 days after the fire, the job is due again
    clock.advance(DAY)
    [second] = scheduler.tick()
    assert second.job_id == GEPA_JOB_ID
    assert scheduler.wait_for_state(second.run_id, RUN_STATE_DONE)
    assert second.run_id != first.run_id


@pytest.mark.integration
def test_manual_trigger(tmp_path: Path) -> None:
    """run_once — the /optimization run-gepa entry point — runs and deploys."""
    runner = _make_runner(tmp_path)
    scheduler = CronScheduler()
    gepa = GepaScheduler(scheduler, runner)

    result = gepa.run_once()

    assert result.components_run == ["extractor"]
    # acceptance (plan P10.T5 loop shape): seed + 3 generations x 3 variants
    assert result.candidates_per_component["extractor"] == 1 + 3 * 3
    assert result.candidates_per_component["extractor"] >= 3
    assert result.deployed["extractor"].strip()
    assert result.run_id.startswith("gepa-")
    assert result.elapsed_seconds > 0

    # the deployment surface (the pipeline reads it at cron time) was updated
    template = runner.components["extractor"].resolve_template_path()
    assert template.read_text(encoding="utf-8") == result.deployed["extractor"]

    # per-call override + pinned run id (same grammar as GEPARunner.run_weekly)
    pinned = gepa.run_once(
        components="extractor", now=datetime(2026, 11, 15, 3, 0, tzinfo=UTC)
    )
    assert pinned.components_run == ["extractor"]
    assert pinned.run_id == "gepa-20261115T030000Z"
