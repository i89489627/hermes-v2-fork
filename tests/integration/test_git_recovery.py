"""P18.T3 acceptance: Git repo corruption recovery (plan Phase 18).

Real end-to-end scenarios over real Git repositories (pygit2-created, like
the memory store): corruption detection flags a truncated loose object via
``git fsck``, the repair escalation removes a stale ``index.lock`` and runs
``git gc`` for the repairable class of corruption, and an unrepairable
corruption (missing object bytes) escalates to a restore from the backup
taken before the corruption — the repository is fsck-clean with every entry
intact afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.backup.git_recovery import (
    attempt_repair,
    detect_corruption,
    recover_from_backup,
)
from hermes.backup.manager import BackupManager
from hermes.backup.recovery import Recovery
from hermes.memory.git_backend import GitBackend
from hermes.memory.store import MemoryStore


def _build_repo(repo_path: Path) -> GitBackend:
    """A real repository with two commits (loose objects, never gc'd)."""
    backend = GitBackend.init(repo_path)
    first = repo_path / "first.txt"
    first.write_text("first committed file\n", encoding="utf-8")
    backend.commit("commit one", ["first.txt"])
    second = repo_path / "second.txt"
    second.write_text("second committed file\n", encoding="utf-8")
    backend.commit("commit two", ["second.txt"])
    return backend


def _truncate_first_loose_object(repo_path: Path) -> Path:
    """Corrupt the repository by truncating one loose object file in place.

    Git/libgit2 stamp object files read-only on Windows, so the write bit is
    cleared first. Any truncated object fails ``git fsck`` whether it is
    reachable or not.
    """
    objects = sorted((repo_path / ".git" / "objects").glob("*/*"))
    assert objects, "expected loose objects in a freshly committed repository"
    victim = objects[0]
    data = victim.read_bytes()
    victim.chmod(0o644)  # clear the read-only stamp before truncating
    victim.write_bytes(data[: max(1, len(data) // 2)])
    return victim


@pytest.mark.integration
def test_detects_corruption(tmp_path: Path) -> None:
    """A truncated loose object is flagged by fsck with concrete findings."""
    repo_path = tmp_path / "memory-repo"
    _build_repo(repo_path)

    healthy = detect_corruption(repo_path)
    assert healthy.corrupted is False
    assert healthy.findings == []  # fsck clean, HEAD resolves, no stale locks

    _truncate_first_loose_object(repo_path)
    report = detect_corruption(repo_path)

    assert report.corrupted is True
    assert report.findings, "fsck evidence must be reported as findings"
    evidence = " ".join(report.findings).lower()
    assert "error" in evidence or "missing" in evidence


@pytest.mark.integration
def test_repair_via_gc(tmp_path: Path) -> None:
    """Stale index.lock corruption: lock removal + gc + fsck re-check repairs."""
    repo_path = tmp_path / "memory-repo"
    _build_repo(repo_path)
    lock = repo_path / ".git" / "index.lock"
    lock.write_bytes(b"")  # a crashed writer left its lock behind

    report = detect_corruption(repo_path, lock_stale_seconds=0.0)
    assert report.corrupted is True
    assert any("index.lock" in finding for finding in report.findings)

    outcome = attempt_repair(repo_path, lock_stale_seconds=0.0)

    assert outcome.repaired is True
    assert any("removed stale .git/index.lock" in action for action in outcome.actions)
    assert any("git gc --prune=now" in action for action in outcome.actions)
    assert any("git fsck clean after repair" in action for action in outcome.actions)
    assert not lock.exists()  # the stale lock is gone
    assert detect_corruption(repo_path, lock_stale_seconds=0.0).corrupted is False

    # a FRESH lock (possible live writer) is never flagged nor removed
    lock.write_bytes(b"")
    fresh = attempt_repair(repo_path)  # default staleness: 5-minute cadence
    assert fresh.repaired is True  # with a live-writer lock the repo is still healthy
    assert any("left in place" in action for action in fresh.actions)
    assert lock.exists()


@pytest.mark.integration
def test_restore_from_backup_on_unrepairable(tmp_path: Path) -> None:
    """Missing object bytes: gc fails, the pre-corruption backup is restored."""
    root = tmp_path / "hermes"
    repo_path = root / "memory-repo"
    store = MemoryStore(root=repo_path)
    store.add("Entry saved before corruption", type="Feedback", source="user-explicit")
    store.add("Second entry saved before corruption", type="Feedback", source="user-explicit")
    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup()  # taken BEFORE the corruption
    assert archive.is_file()

    _truncate_first_loose_object(repo_path)
    report = detect_corruption(repo_path)
    assert report.corrupted is True

    outcome = attempt_repair(repo_path)
    assert outcome.repaired is False  # gc cannot regenerate object bytes
    assert any("git gc --prune=now failed" in action for action in outcome.actions)
    assert any("corruption remains after repair" in action for action in outcome.actions)

    result = recover_from_backup(repo_path, root / "backups")

    assert result.restored_from == archive
    assert result.entries_count == 2
    assert result.git_commits >= 2  # the commit history came back with the files

    # the repository is healthy again and every entry is intact
    assert detect_corruption(repo_path).corrupted is False
    revived = MemoryStore(root=repo_path)
    assert sorted(entry.text for entry in revived.list()) == [
        "Entry saved before corruption",
        "Second entry saved before corruption",
    ]
    assert revived.git.log(limit=10)
    assert Recovery(BackupManager(hermes_root=root)).verify_recovered() == []
