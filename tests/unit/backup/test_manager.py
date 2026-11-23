"""P18.T1 acceptance: automatic backups (plan Phase 18).

One ``YYYY-MM-DD.tar.gz`` archive per day under ``<hermes_root>/backups/``
(idempotent per day, refreshed when it already exists); archives older than
30 days are deleted (by name date when parseable, mtime otherwise); ``git
fsck --no-progress`` must reject a corrupted repository before anything is
archived, while a plain non-git tree is archived without the check; the
scheduling helper is due once per local day at 02:00.
"""

from __future__ import annotations

import os
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.backup.manager import (
    BackupIntegrityError,
    BackupManager,
    due_for_backup,
)
from hermes.memory.git_backend import GitBackend

_T0 = datetime(2026, 11, 20, 2, 0, 0)


def _repo_with_commit(root: Path) -> GitBackend:
    """A minimal memory-repo: one committed fact file inside a git repo."""
    repo_dir = root / "memory-repo"
    git = GitBackend.init(repo_dir)
    facts = repo_dir / "global" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    entry = facts / "2026-11-20-test.md"
    entry.write_text("the wheel was a good idea\n", encoding="utf-8")
    git.commit("[memory] add 2026-11-20-test global", [entry])
    return git


@pytest.mark.unit
def test_creates_daily_backup(tmp_path: Path) -> None:
    """Daily archive naming, contents (incl. .git), per-day idempotency."""
    root = tmp_path / "hermes"
    _repo_with_commit(root)
    manager = BackupManager(hermes_root=root)

    archive = manager.create_backup(now=_T0)
    assert archive == root / "backups" / "2026-11-20.tar.gz"
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "global/facts/2026-11-20-test.md" in names
    assert any(name.startswith(".git/") for name in names)  # history is archived

    # idempotent per day: same day refreshes the same archive with new content
    (root / "memory-repo" / "extra.txt").write_text("added later\n", encoding="utf-8")
    refreshed = manager.create_backup(now=_T0 + timedelta(hours=3))
    assert refreshed == archive
    with tarfile.open(archive, "r:gz") as tar:
        assert "extra.txt" in tar.getnames()

    # the next day gets its own archive; both stay on disk (inside retention)
    next_day = manager.create_backup(now=_T0 + timedelta(days=1))
    assert next_day.name == "2026-11-21.tar.gz"
    assert sorted(path.name for path in (root / "backups").glob("*.tar.gz")) == [
        "2026-11-20.tar.gz",
        "2026-11-21.tar.gz",
    ]

    # scheduling: once per local day at 02:00 (default hour)
    assert due_for_backup(None, now=_T0)  # never backed up -> due immediately
    assert not due_for_backup(_T0, now=_T0 + timedelta(hours=1))  # done today
    yesterday = _T0 + timedelta(days=1)
    assert not due_for_backup(_T0, now=yesterday - timedelta(minutes=1))  # before 02:00
    assert due_for_backup(_T0, now=yesterday)  # 02:00 sharp
    assert due_for_backup(_T0, now=yesterday + timedelta(hours=5))


@pytest.mark.unit
def test_retention_enforced(tmp_path: Path) -> None:
    """Archives older than 30 days are deleted; exactly 30 days is kept."""
    root = tmp_path / "hermes"
    manager = BackupManager(hermes_root=root)
    backups = root / "backups"

    t0 = datetime(2026, 11, 1, 2, 0)
    manager.create_backup(now=t0)  # 2026-11-01.tar.gz (no repo -> empty archive)
    manager.create_backup(now=t0 + timedelta(days=10))  # 2026-11-11.tar.gz
    assert (backups / "2026-11-01.tar.gz").is_file()

    # a non-date archive ages by its (backdated) mtime
    adhoc = backups / "adhoc-snapshot.tar.gz"
    adhoc.write_bytes(b"not a real tar")
    backdated = (t0 + timedelta(days=10) - timedelta(days=40)).timestamp()
    os.utime(adhoc, (backdated, backdated))

    # enforce at 2026-12-11: cutoff is 2026-11-11 (30 days) -> 11-01 and adhoc go
    deleted = manager.enforce_retention(now=t0 + timedelta(days=40))
    assert (backups / "2026-11-01.tar.gz") in deleted
    assert adhoc in deleted
    assert not (backups / "2026-11-01.tar.gz").exists()
    assert not adhoc.exists()
    assert (backups / "2026-11-11.tar.gz").is_file()  # exactly 30 days -> kept

    # retention also runs automatically after every create_backup
    manager.create_backup(now=t0 + timedelta(days=50))  # 2026-12-21.tar.gz
    assert (backups / "2026-11-11.tar.gz").exists() is False
    assert (backups / "2026-12-21.tar.gz").is_file()


@pytest.mark.unit
def test_integrity_verified(tmp_path: Path) -> None:
    """git fsck gates archiving; corrupt objects raise BackupIntegrityError."""
    root = tmp_path / "hermes"
    _repo_with_commit(root)
    manager = BackupManager(hermes_root=root)

    # healthy repo: fsck passes silently and the archive lands
    archive = manager.create_backup(now=_T0)
    assert archive.is_file()

    # corrupt one loose object (2-hex dir / 38-hex file) -> fsck must reject
    objects = root / "memory-repo" / ".git" / "objects"
    loose = next(
        path
        for path in objects.glob("*/*")
        if len(path.parent.name) == 2 and len(path.name) == 38
    )
    loose.chmod(0o644)  # libgit2 writes loose objects read-only
    loose.write_bytes(b"corrupted object payload")
    with pytest.raises(BackupIntegrityError):
        manager.create_backup(now=_T0 + timedelta(days=1))
    assert not (root / "backups" / "2026-11-21.tar.gz").exists()  # nothing archived

    # a plain non-git tree is archived without the fsck check
    plain_root = tmp_path / "plain"
    (plain_root / "memory-repo").mkdir(parents=True)
    (plain_root / "memory-repo" / "note.txt").write_text("hello\n", encoding="utf-8")
    plain = BackupManager(hermes_root=plain_root).create_backup(now=_T0)
    with tarfile.open(plain, "r:gz") as tar:
        assert "note.txt" in tar.getnames()
