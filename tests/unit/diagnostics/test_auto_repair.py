"""P14.T4 acceptance: the plan 5.4 top-5 auto-repairs.

Each YAML-named test drives one repair against a real tmp tree — a dirty Git
memory repo (via GitBackend/MemoryStore), a fact file missing its trust card,
old trace day files, a stale Pareto front (mtime faked via os.utime) and a
stale backup set — and asserts the repair outcome, the repairs.log JSON line
and (where applicable) that the injected expensive action fired. Escalation of
failed repairs through the on_failure notifier seam gets its own guard test.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.diagnostics.auto_repair import (
    AUTO_REPAIR_COMMIT_MESSAGE,
    AUTO_REPAIR_FAILED,
    AutoRepair,
    RepairContext,
)
from hermes.memory.git_backend import GitBackend
from hermes.memory.trust_card import parse
from hermes.memory.store import MemoryStore
from hermes.optimization.pareto import ParetoFrontStore

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)


def _make_repair(tmp_path: Path, **context_kwargs: object) -> tuple[AutoRepair, Path]:
    """AutoRepair with the fixed clock and a repairs.log under tmp_path."""
    log_path = tmp_path / "diagnostics" / "repairs.log"
    context = RepairContext(**context_kwargs)  # type: ignore[arg-type]
    repairer = AutoRepair(context, log_path=log_path, now=lambda: _NOW)
    return repairer, log_path


@pytest.mark.unit
def test_repair_git_commits_dirty_state(tmp_path: Path) -> None:
    """A dirty working tree is committed with the plan's exact message."""
    root = tmp_path / "memory-repo"
    store = MemoryStore(root=root)
    first = store.add("Hermes prefers concise answers", type="User", source="user-explicit")
    # dirty the tree: an in-place edit of the tracked entry + an untracked file
    entry_path = root / "global" / "facts" / f"{first.id}.md"
    entry_path.write_text(entry_path.read_text(encoding="utf-8") + "\nedited by hand\n",
                          encoding="utf-8")
    (root / "global" / "facts" / "scratch-note.md").write_text(
        "untracked scratch", encoding="utf-8"
    )
    backend = GitBackend(root)
    assert not backend.is_clean()

    repairer, log_path = _make_repair(
        tmp_path, repo_path=root, store_root=root,
        traces_root=tmp_path / "traces",
        pareto_path=tmp_path / "pareto.json",
        backups_dir=tmp_path / "backups",
    )
    result = repairer.repair_dirty_git()

    assert result.issue == "dirty_git"
    assert result.status == "repaired"
    assert "committed 2 pending path(s)" in result.details
    assert backend.is_clean()
    history = backend.log(limit=5)
    assert history[0].message == AUTO_REPAIR_COMMIT_MESSAGE

    # idempotent second pass: tree is clean now -> skipped
    again = repairer.repair_dirty_git()
    assert again.status == "skipped"

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["status"] for record in records] == ["repaired", "skipped"]
    assert records[0]["ts"] == _NOW.isoformat()


@pytest.mark.unit
def test_repair_missing_trust_cards(tmp_path: Path) -> None:
    """Fact files without trust cards get the migration-default card, body kept."""
    root = tmp_path / "memory-repo"
    store = MemoryStore(root=root)
    valid = store.add("A perfectly valid entry", type="Feedback", source="user-explicit")
    facts_dir = root / "global" / "facts"
    orphan = facts_dir / "orphan-fact.md"
    orphan_body = "The orphan fact body preserved byte-exactly.\n"
    orphan.write_text(orphan_body, encoding="utf-8")  # no frontmatter at all
    broken = facts_dir / "broken-fact.md"
    broken.write_text(
        "---\nid: broken-fact\ntype: [unclosed\n---\n\nBroken frontmatter body.\n",
        encoding="utf-8",
    )

    repairer, log_path = _make_repair(
        tmp_path, repo_path=root, store_root=root,
        traces_root=tmp_path / "traces",
        pareto_path=tmp_path / "pareto.json",
        backups_dir=tmp_path / "backups",
    )
    result = repairer.repair_missing_trust_cards()

    assert result.issue == "missing_trust_cards"
    assert result.status == "repaired"
    assert "2 of 3 entries" in result.details

    for repaired_path in (orphan, broken):
        card, body = parse(repaired_path.read_text(encoding="utf-8"))
        assert card.id == repaired_path.stem
        assert card.trust_score == pytest.approx(0.5)  # migration default
        assert len(card.verification_events) == 1
        event = card.verification_events[0]
        assert event.source == "agent-inferred"
        assert event.value == pytest.approx(0.5)
        assert event.verifier == "auto-repair"
    assert "The orphan fact body preserved byte-exactly.\n" in orphan.read_text(
        encoding="utf-8"
    )
    assert "Broken frontmatter body.\n" in broken.read_text(encoding="utf-8")
    assert broken.read_text(encoding="utf-8").startswith("---\n")

    # the valid entry was untouched and still loads through the store
    assert store.get(valid.id) is not None
    # repaired files were committed
    assert GitBackend(root).is_clean()

    # second pass: everything valid now -> skipped
    again = repairer.repair_missing_trust_cards()
    assert again.status == "skipped"
    records = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert [record["issue"] for record in records] == ["missing_trust_cards"] * 2
    assert [record["status"] for record in records] == ["repaired", "skipped"]


@pytest.mark.unit
def test_repair_trace_store_archives_old(tmp_path: Path) -> None:
    """Trace day files older than 90 days move to traces/archive/; fresh stay."""
    traces_root = tmp_path / "traces"
    version_dir = traces_root / "v1.0"
    version_dir.mkdir(parents=True)
    old_day = (_NOW - timedelta(days=120)).strftime("%Y-%m-%d")
    fresh_day = _NOW.strftime("%Y-%m-%d")
    old_file = version_dir / f"{old_day}.jsonl"
    old_file.write_text('{"trace": "old"}\n', encoding="utf-8")
    fresh_file = version_dir / f"{fresh_day}.jsonl"
    fresh_file.write_text('{"trace": "fresh"}\n', encoding="utf-8")
    # mtime alone must NOT archive a file whose day stamp is fresh
    ancient = (_NOW - timedelta(days=200)).timestamp()
    os.utime(fresh_file, (ancient, ancient))

    repairer, log_path = _make_repair(
        tmp_path, repo_path=tmp_path / "memory-repo",
        store_root=tmp_path / "memory-repo",
        traces_root=traces_root,
        pareto_path=tmp_path / "pareto.json",
        backups_dir=tmp_path / "backups",
    )
    result = repairer.repair_trace_store_full()

    assert result.issue == "trace_store_full"
    assert result.status == "repaired"
    assert f"archived 1 trace day file(s)" in result.details
    assert (traces_root / "archive" / f"{old_day}.jsonl").read_text(
        encoding="utf-8"
    ) == '{"trace": "old"}\n'
    assert not old_file.exists()
    assert fresh_file.exists()  # fresh day stamp wins over its faked old mtime

    again = repairer.repair_trace_store_full()
    assert again.status == "skipped"
    records = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert [record["status"] for record in records] == ["repaired", "skipped"]


@pytest.mark.unit
def test_repair_stale_pareto_triggers_gepa(tmp_path: Path) -> None:
    """Stale/missing Pareto front triggers the injected GEPA run once."""
    pareto_path = tmp_path / "pareto.json"
    ParetoFrontStore(path=pareto_path).add_candidate(
        "extractor", "extract facts concisely", {"accuracy": 0.9, "cost": 0.2}
    )
    old = (_NOW - timedelta(days=10)).timestamp()
    os.utime(pareto_path, (old, old))
    calls: list[int] = []

    repairer, log_path = _make_repair(
        tmp_path, repo_path=tmp_path / "memory-repo",
        store_root=tmp_path / "memory-repo",
        traces_root=tmp_path / "traces",
        pareto_path=pareto_path,
        backups_dir=tmp_path / "backups",
        gepa_trigger=lambda: calls.append(1),
    )
    result = repairer.repair_stale_pareto()

    assert result.issue == "stale_pareto"
    assert result.status == "repaired"
    assert "GEPA run triggered" in result.details
    assert "10.0 days ago" in result.details
    assert len(calls) == 1

    # fresh front -> skipped, trigger not called again
    fresh = _NOW.timestamp()
    os.utime(pareto_path, (fresh, fresh))
    again = repairer.repair_stale_pareto()
    assert again.status == "skipped"
    assert len(calls) == 1

    # missing front counts as stale (GEPA never completed) -> triggers
    pareto_path.unlink()
    missing = repairer.repair_stale_pareto()
    assert missing.status == "repaired"
    assert "no Pareto front yet" in missing.details
    assert len(calls) == 2

    records = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert [record["status"] for record in records] == ["repaired", "skipped", "repaired"]


@pytest.mark.unit
def test_repair_stale_backup_triggers_backup(tmp_path: Path) -> None:
    """Backups older than 25h trigger the injected backup action; fresh skip."""
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    stale_archive = backups_dir / "2026-11-19.tar.gz"
    stale_archive.write_bytes(b"archive-bytes")
    old = (_NOW - timedelta(hours=30)).timestamp()
    os.utime(stale_archive, (old, old))
    created: list[Path] = []

    def backup_trigger() -> Path:
        backups_dir.mkdir(parents=True, exist_ok=True)
        fresh = backups_dir / "2026-11-21.tar.gz"
        fresh.write_bytes(b"fresh-archive")
        created.append(fresh)
        return fresh

    repairer, log_path = _make_repair(
        tmp_path, repo_path=tmp_path / "memory-repo",
        store_root=tmp_path / "memory-repo",
        traces_root=tmp_path / "traces",
        pareto_path=tmp_path / "pareto.json",
        backups_dir=backups_dir,
        backup_trigger=backup_trigger,
    )
    result = repairer.repair_stale_backup()

    assert result.issue == "stale_backup"
    assert result.status == "repaired"
    assert "30.0 h old" in result.details
    assert "backup triggered" in result.details
    assert created == [backups_dir / "2026-11-21.tar.gz"]

    # fresh archive (mtime just now) -> skipped, trigger not called again
    again = repairer.repair_stale_backup()
    assert again.status == "skipped"
    assert len(created) == 1

    # empty backups dir -> "never backed up" -> triggers
    shutil.rmtree(backups_dir)
    never = repairer.repair_stale_backup()
    assert never.status == "repaired"
    assert "never backed up" in never.details
    assert len(created) == 2

    records = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert [record["status"] for record in records] == ["repaired", "skipped", "repaired"]


@pytest.mark.unit
def test_failed_repair_escalates_to_alert(tmp_path: Path) -> None:
    """Failed repairs surface through the on_failure seam and are logged."""
    escalations: list[tuple[str, str]] = []
    log_path = tmp_path / "diagnostics" / "repairs.log"
    context = RepairContext(
        repo_path=tmp_path / "memory-repo",
        store_root=tmp_path / "memory-repo",
        traces_root=tmp_path / "traces",
        pareto_path=tmp_path / "pareto.json",
        backups_dir=tmp_path / "backups",
        backup_trigger=None,  # stale backup with no trigger configured -> failed
    )
    repairer = AutoRepair(context, on_failure=lambda name, msg: escalations.append((name, msg)),
                          log_path=log_path, now=lambda: _NOW)
    result = repairer.repair_stale_backup()

    assert result.status == "failed"
    assert "no backup_trigger configured" in result.details
    assert escalations == [
        (AUTO_REPAIR_FAILED, f"auto-repair failed for 'stale_backup': {result.details}")
    ]
    records = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert records[0]["issue"] == "stale_backup"
    assert records[0]["status"] == "failed"

    # repair_all is failure-isolated: one failed repair must not stop the rest
    results = repairer.repair_all()
    assert [r.issue for r in results] == [
        "dirty_git", "missing_trust_cards", "trace_store_full",
        "stale_pareto", "stale_backup",
    ]
    assert results[0].status == "skipped"  # no repo at repo_path -> skipped
    assert results[3].status == "failed"  # stale pareto, no gepa_trigger
    assert results[4].status == "failed"  # stale backup, no backup_trigger
    assert len(escalations) == 3  # direct call + pareto + backup
    assert {name for name, _msg in escalations} == {AUTO_REPAIR_FAILED}
