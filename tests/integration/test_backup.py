"""P18.T1 acceptance: full backup/restore cycle (plan Phase 18).

A real memory store (trust-card entries, one Git commit per write) is backed
up, the repository is destroyed outright, and the archive restores both the
entry files and the Git commit history over the same directory.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.backup.manager import BackupManager
from hermes.memory.git_backend import GitBackend
from hermes.memory.store import MemoryStore

_T0 = datetime(2026, 11, 20, 2, 0, 0, tzinfo=UTC)


def _rmtree_force(dirpath: Path) -> None:
    """rmtree that clears the read-only bit libgit2 stamps on .git objects."""
    for path in dirpath.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    shutil.rmtree(dirpath)


@pytest.mark.integration
def test_full_backup_restore_cycle(tmp_path: Path) -> None:
    """backup -> destroy repo -> restore -> entries and git history present."""
    root = tmp_path / "hermes"
    repo_dir = root / "memory-repo"
    store = MemoryStore(root=repo_dir)
    first = store.add("Hermes prefers concise answers", type="User", source="user-explicit")
    second = store.add("Back up before every update", type="Feedback", source="user-explicit")
    assert first.id != second.id

    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup(now=_T0)
    assert archive.is_file()
    assert archive == root / "backups" / "2026-11-20.tar.gz"

    # destroy the repository outright — entries, index and .git all gone
    _rmtree_force(repo_dir)
    assert not repo_dir.exists()

    restored = manager.restore(archive)
    assert restored == repo_dir

    # entries are back (trust cards parse, text intact, ordering stable)
    revived = MemoryStore(root=repo_dir)
    assert [entry.id for entry in revived.list()] == [first.id, second.id]
    assert revived.get(first.id).text == "Hermes prefers concise answers"
    assert revived.get(second.id).text == "Back up before every update"

    # and the Git history came back with the archive
    history = GitBackend(repo_dir).log(limit=10)
    assert len(history) >= 2
    assert any(first.id in commit.message for commit in history)
    assert any(second.id in commit.message for commit in history)

    # restore can also target an arbitrary directory (not only the live repo)
    other = tmp_path / "elsewhere"
    manager.restore(archive, target_dir=other)
    assert (other / "MEMORY.md").is_file()
    assert any((other / "global" / "facts").glob("*.md"))
