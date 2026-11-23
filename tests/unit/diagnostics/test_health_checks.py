"""P14.T1 acceptance: the six health checks (plan Section 5.1).

Fast, hermetic single-component tests: every check returns
``{status: pass|fail|warn, details}``, completes well within 2 s (no real
network — the main-model probe is an injected fake), and maps the plan's
status semantics: broken trust cards fail, dirty git warns, an unreachable
main model fails, trace size warns approaching / fails over the limit, a
missing or stale Pareto front warns, a stale backup warns.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.diagnostics.health_checks import (
    HealthCheckContext,
    check_backup_freshness,
    check_git_state,
    check_main_model_reachable,
    check_memory_store_integrity,
    check_pareto_front_freshness,
    check_trace_store_size,
    run_all_checks,
)
from hermes.memory.store import MemoryStore
from hermes.optimization.pareto import ParetoFrontStore


@pytest.mark.unit
def test_memory_store_integrity_pass(tmp_path: Path) -> None:
    """A store whose fact files all parse passes; run_all_checks aggregates."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("Valid entry one", type="Feedback", source="user-explicit")
    store.add("Valid entry two", type="User", source="user-explicit")

    check = check_memory_store_integrity(store.root)

    assert check.name == "memory_store_integrity"
    assert check.status == "pass"
    assert "2" in check.details  # both entries validated
    assert check.auto_repair_attempted is False
    assert check.auto_repair_result is None

    # run_all_checks aggregates all six checks in plan-table order, with
    # injected paths and probe so the whole report stays hermetic and fast
    aggregated = run_all_checks(
        HealthCheckContext(
            store_root=store.root,
            repo_path=store.root,
            probe=lambda: True,
            traces_root=tmp_path / "traces",
            pareto_path=tmp_path / "pareto.json",
            backups_dir=tmp_path / "backups",
        )
    )
    assert [aggregated_check.name for aggregated_check in aggregated] == [
        "memory_store_integrity",
        "git_state",
        "main_model_reachable",
        "trace_store_size",
        "pareto_front_freshness",
        "backup_freshness",
    ]
    assert all(
        aggregated_check.status in {"pass", "fail", "warn"} for aggregated_check in aggregated
    )
    assert aggregated[0].status == "pass"  # the store we just built
    assert aggregated[2].status == "pass"  # the injected probe


@pytest.mark.unit
def test_memory_store_integrity_fail_on_missing_trust_card(tmp_path: Path) -> None:
    """A fact file without valid frontmatter fails the integrity check."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("Well-formed entry", type="Feedback", source="user-explicit")
    rogue = store.root / "global" / "facts" / "rogue-entry.md"
    rogue.write_text("plain markdown without any frontmatter\n", encoding="utf-8")

    check = check_memory_store_integrity(store.root)

    assert check.name == "memory_store_integrity"
    assert check.status == "fail"
    assert "rogue-entry.md" in check.details
    assert "1 of 2" in check.details


@pytest.mark.unit
def test_git_state_warn_when_dirty(tmp_path: Path) -> None:
    """A clean tree passes; modified + untracked files warn with the count."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("Committed entry", type="Feedback", source="user-explicit")

    clean = check_git_state(store.root)
    assert clean.name == "git_state"
    assert clean.status == "pass"
    assert "clean" in clean.details

    # dirty the worktree: one tracked file modified + one untracked file
    index_path = store.root / "MEMORY.md"
    index_path.write_text(
        index_path.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8"
    )
    (store.root / "untracked-note.md").write_text("untracked\n", encoding="utf-8")

    dirty = check_git_state(store.root)
    assert dirty.status == "warn"
    assert "2 file(s)" in dirty.details


@pytest.mark.unit
def test_main_model_reachable_pass(tmp_path: Path) -> None:
    """An injected probe returning True passes the reachability check."""
    check = check_main_model_reachable(probe=lambda: True)

    assert check.name == "main_model_reachable"
    assert check.status == "pass"
    assert "reachable" in check.details.lower()


@pytest.mark.unit
def test_main_model_fail_when_api_down(tmp_path: Path) -> None:
    """A failing (or crashing) probe fails the check, with the error named."""
    down = check_main_model_reachable(probe=lambda: False)

    assert down.name == "main_model_reachable"
    assert down.status == "fail"
    assert "unreachable" in down.details.lower()

    def crashing_probe() -> bool:
        raise ConnectionError("api down")

    crashed = check_main_model_reachable(probe=crashing_probe)
    assert crashed.status == "fail"
    assert "ConnectionError" in crashed.details


@pytest.mark.unit
def test_trace_store_size_pass(tmp_path: Path) -> None:
    """Under the limit passes; >= 80% warns; over the limit fails."""

    def store_with(total_bytes: int) -> Path:
        traces = tmp_path / f"traces-{total_bytes}"
        traces.mkdir()
        (traces / "v1.0").mkdir()
        (traces / "v1.0" / "2026-11-20.jsonl").write_bytes(b"x" * total_bytes)
        return traces

    # max_gb=1e-6 -> limit ~1074 bytes; 80% warn line ~859 bytes
    small = check_trace_store_size(store_with(500), max_gb=1e-6)
    assert small.name == "trace_store_size"
    assert small.status == "pass"
    assert "500" in small.details

    approaching = check_trace_store_size(store_with(1000), max_gb=1e-6)
    assert approaching.status == "warn"
    assert "approaching" in approaching.details.lower()

    over = check_trace_store_size(store_with(2000), max_gb=1e-6)
    assert over.status == "fail"
    assert "over limit" in over.details.lower()


@pytest.mark.unit
def test_pareto_front_freshness_pass(tmp_path: Path) -> None:
    """A freshly written front passes; stale and missing fronts warn."""
    pareto_path = tmp_path / "optimization" / "pareto-front.json"
    pareto_path.parent.mkdir()
    ParetoFrontStore(path=pareto_path).add_candidate(
        "extraction", "prompt v1", {"quality": 0.9}
    )
    now = datetime.now(UTC)

    fresh = check_pareto_front_freshness(pareto_path, max_age_days=8, now=now)
    assert fresh.name == "pareto_front_freshness"
    assert fresh.status == "pass"
    assert "fresh" in fresh.details.lower()

    # stale: last update 9 days ago (limit 8) -> warn, not fail
    old = (now - timedelta(days=9)).timestamp()
    os.utime(pareto_path, (old, old))
    stale = check_pareto_front_freshness(pareto_path, max_age_days=8, now=now)
    assert stale.status == "warn"
    assert "stale" in stale.details.lower()

    # missing front: no GEPA run yet is a legitimate early state -> warn
    missing = check_pareto_front_freshness(tmp_path / "absent.json", now=now)
    assert missing.status == "warn"
    assert "no gepa run" in missing.details.lower()


@pytest.mark.unit
def test_backup_freshness_pass(tmp_path: Path) -> None:
    """A recent YYYY-MM-DD.tar.gz passes; stale and never-backed-up warn."""
    backups = tmp_path / "backups"
    backups.mkdir()
    fresh = backups / "2026-11-20.tar.gz"
    fresh.write_bytes(b"archive-bytes")
    now = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

    check = check_backup_freshness(backups, max_age_hours=25, now=now)
    assert check.name == "backup_freshness"
    assert check.status == "pass"
    assert "2026-11-20.tar.gz" in check.details

    # stale: newest archive is 30 h old (limit 25) -> warn
    old = (now - timedelta(hours=30)).timestamp()
    os.utime(fresh, (old, old))
    stale = check_backup_freshness(backups, max_age_hours=25, now=now)
    assert stale.status == "warn"
    assert "30.0 h" in stale.details

    # no backups directory at all -> warn (never backed up)
    never = check_backup_freshness(tmp_path / "empty", now=now)
    assert never.status == "warn"
    assert "never backed up" in never.details.lower()
