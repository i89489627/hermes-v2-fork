"""P18.T2 acceptance: recovery from backup (plan Phase 18).

The full restore workflow over a real memory store (trust-card entries, one
Git commit per write): safety backup of the corrupted live repo, wipe of the
read-only-object Git tree, archive extraction (entries + ``.git`` history
back), and verification of the recovered state — including the partial
corruption case (some entry files missing) being recoverable.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.backup.manager import BackupManager
from hermes.backup.recovery import Recovery, force_rmtree
from hermes.memory.git_backend import GitBackend
from hermes.memory.store import MemoryStore

_T0 = datetime(2026, 11, 20, 2, 0, 0, tzinfo=UTC)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_store(root: Path, texts: list[str]) -> MemoryStore:
    """A real memory store at ``root/memory-repo`` with one entry per text."""
    store = MemoryStore(root=root / "memory-repo")
    for text in texts:
        store.add(text, type="Feedback", source="user-explicit")
    return store


@pytest.mark.integration
def test_restore_from_backup(tmp_path: Path) -> None:
    """hermes-restore CLI: corrupted live repo replaced by the archived state."""
    root = tmp_path / "hermes"
    repo_dir = root / "memory-repo"
    _build_store(root, ["Entry one survives recovery", "Entry two survives recovery"])
    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup(now=_T0)
    assert archive.is_file()

    # the live repo drifted/corrupted after the backup: an extra entry is present
    store = MemoryStore(root=repo_dir)
    store.add("Post-backup corruption entry", type="Feedback", source="user-explicit")
    assert len(MemoryStore(root=repo_dir).list()) == 3

    script = _REPO_ROOT / "scripts" / "hermes-restore.py"
    result = subprocess.run(
        [sys.executable, str(script), str(archive), "--hermes-root", str(root)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "entries" in result.stdout
    assert "verification" in result.stdout

    # restore is replace-not-overlay: the post-backup entry is gone again
    revived = MemoryStore(root=repo_dir)
    entries = revived.list()
    assert len(entries) == 2
    assert {entry.text for entry in entries} == {
        "Entry one survives recovery",
        "Entry two survives recovery",
    }
    assert revived.git.log(limit=10)  # commit history came back with the files
    # the pre-restore safety backup of the corrupted state is kept
    assert (root / "memory-repo-pre-restore").is_dir()


@pytest.mark.integration
def test_restore_preserves_all_entries(tmp_path: Path) -> None:
    """Every entry of the archive is back, byte-faithful, after total loss."""
    root = tmp_path / "hermes"
    texts = [f"Preserved entry number {index}" for index in range(1, 6)]
    store = _build_store(root, texts)
    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup(now=_T0)

    # total loss: the fact directory is wiped after the backup
    force_rmtree(store.root / "global" / "facts")
    assert MemoryStore(root=store.root).list() == []

    recovery = Recovery(manager)
    result = recovery.restore(archive)

    assert result.restored_from == archive
    assert result.entries_count == 5
    assert result.git_commits == 5
    assert result.stopped_note  # documents the manual stop/restart step

    revived = MemoryStore(root=store.root)
    assert sorted(entry.text for entry in revived.list()) == sorted(texts)
    assert all(entry.trust_score == 1.0 for entry in revived.list())
    assert recovery.verify_recovered() == []


@pytest.mark.integration
def test_restore_preserves_git_history(tmp_path: Path) -> None:
    """Commit shas and messages survive a full destroy + restore cycle."""
    root = tmp_path / "hermes"
    store = _build_store(root, ["History anchor one", "History anchor two", "History anchor three"])
    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup(now=_T0)
    pre_commits = store.git.log(limit=100)
    assert len(pre_commits) == 3

    # destroy the repository outright — entries, index and .git all gone;
    # the safety-backup step skips cleanly when there is no repo left
    force_rmtree(store.root)
    assert not store.root.exists()

    result = Recovery(manager).restore(archive)

    assert result.entries_count == 3
    assert result.git_commits == 3
    post_commits = GitBackend(store.root).log(limit=100)
    assert [commit.sha for commit in post_commits] == [commit.sha for commit in pre_commits]
    assert [commit.message for commit in post_commits] == [
        commit.message for commit in pre_commits
    ]
    assert Recovery(manager).verify_recovered() == []


@pytest.mark.integration
def test_partial_corruption_recovery(tmp_path: Path) -> None:
    """Some entry files missing is detected by the verifier and fully repaired."""
    root = tmp_path / "hermes"
    texts = ["Alpha fact", "Beta fact", "Gamma fact"]
    store = _build_store(root, texts)
    manager = BackupManager(hermes_root=root)
    archive = manager.create_backup(now=_T0)

    # partial corruption: 2 of the 3 entry files vanish from the live repo
    facts = sorted((store.root / "global" / "facts").glob("*.md"))
    assert len(facts) == 3
    facts[0].unlink()
    facts[1].unlink()

    recovery = Recovery(manager)
    problems = recovery.verify_recovered()
    assert len(problems) == 2
    assert all("missing" in problem.lower() for problem in problems)

    result = recovery.restore(archive)

    assert result.entries_count == 3
    assert result.git_commits == 3
    revived = MemoryStore(root=store.root)
    assert sorted(entry.text for entry in revived.list()) == sorted(texts)
    assert recovery.verify_recovered() == []
