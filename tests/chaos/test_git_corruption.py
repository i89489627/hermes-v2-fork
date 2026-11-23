"""P20.T1 chaos: git repository corruption detection and recovery (P18.T3 stack).

Two corruption classes over the REAL git-backed memory store (plan P20.T1:
"corrupt one memory file -> verify Git recovery"; recovery stack per
plan P18.T3: detect -> attempt repair -> restore from backup):

- ``test_corrupted_index_recoverable`` — a truncated ``.git/index`` (the torn
  write of a crashed writer). ``detect_corruption`` flags the corruption, a
  FRESH store handle fails cleanly (``add`` raises, no partial file, no data
  loss), ``attempt_repair`` honestly reports that ``git gc`` cannot rebuild a
  torn index, and the restore-from-backup (backup taken BEFORE the corruption)
  returns a fully usable store with every entry intact. The stale-lock class
  (a ``.git/index.lock`` left behind by a crashed writer) repairs IN PLACE.
- ``test_missing_object_restored_from_backup`` — truncated object bytes (a
  lost loose object): detected, gc cannot regenerate the bytes, and the
  backup restore recovers the store with no data loss.

Victim choice is deliberate: the truncated object is an ENTRY BLOB, never the
HEAD commit or tree — libgit2 hangs reading a truncated loose commit object
(observed on Windows/pygit2), so the chaos injection must not corrupt an
object the recovery probe itself reads. Reads and writes through the live
store survive a corrupt blob (they never inflate it); ``git fsck`` flags it
honestly.
"""

from __future__ import annotations

import os
import time

import pytest

from hermes.backup.git_recovery import attempt_repair, detect_corruption, recover_from_backup
from hermes.backup.manager import BackupManager
from hermes.memory.store import MemoryStore

_INDEX_CORRUPTION_TEXTS = [
    "index corruption survivor one",
    "index corruption survivor two",
]
_OBJECT_CORRUPTION_TEXTS = [
    "missing object survivor one",
    "missing object survivor two",
]


def _truncate_index(store: MemoryStore) -> None:
    """Corrupt the repository by truncating ``.git/index`` in place."""
    index_path = store.root / ".git" / "index"
    data = index_path.read_bytes()
    index_path.write_bytes(data[: len(data) // 3])


def _truncate_entry_blob(store: MemoryStore, entry_id: str) -> str:
    """Truncate the committed blob of one entry file; returns the blob oid.

    libgit2 stamps loose objects read-only on Windows, so the write bit is
    cleared before the in-place truncation.
    """
    tip = store.git._repo[store.git._repo.head.target]
    oid = str(tip.tree[f"global/facts/{entry_id}.md"].id)
    victim = store.root / ".git" / "objects" / oid[:2] / oid[2:]
    data = victim.read_bytes()
    victim.chmod(0o644)
    victim.write_bytes(data[: len(data) // 2])
    return oid


@pytest.mark.chaos
def test_corrupted_index_recoverable(chaos_store) -> None:
    """Torn .git/index: detected, fresh writes fail cleanly, restore recovers."""
    store = chaos_store
    for text in _INDEX_CORRUPTION_TEXTS:
        store.add(text, type="Feedback", source="user-explicit")

    manager = BackupManager()  # hermes_home(): repo dir == the store root
    archive = manager.create_backup()  # taken BEFORE the corruption
    assert archive.is_file()

    _truncate_index(store)

    # 1) the corruption is detected honestly, with fsck evidence
    report = detect_corruption(store.root)
    assert report.corrupted is True
    assert report.findings

    # 2) a FRESH handle fails cleanly: add raises naming the index, the
    #    file-level change is rolled back, nothing is half-written
    with pytest.raises(Exception, match="index"):
        MemoryStore(root=store.root).add(
            "doomed write under index corruption", type="Feedback"
        )
    facts_dir = store.root / "global" / "facts"
    assert not any(path.name.startswith(".") for path in facts_dir.iterdir())
    assert sorted(entry.text for entry in store.list()) == _INDEX_CORRUPTION_TEXTS

    # 3) the repair escalation is honest: gc cannot rebuild a torn index
    outcome = attempt_repair(store.root)
    assert outcome.repaired is False

    # 4) restore from the pre-corruption backup: usable again, no data loss
    result = recover_from_backup(store.root, manager.backups_dir)
    assert result.restored_from == archive
    assert result.entries_count == 2
    revived = MemoryStore(root=store.root)
    assert sorted(entry.text for entry in revived.list()) == _INDEX_CORRUPTION_TEXTS
    assert detect_corruption(store.root).corrupted is False
    assert revived.git.is_clean()
    post = revived.add("written after index restore", type="Feedback", source="user-explicit")
    assert revived.get(post.id) is not None
    assert revived.git.is_clean()
    assert revived.git.log(path=f"global/facts/{post.id}.md", limit=5)

    # 5) the stale-lock class repairs IN PLACE (remove lock -> clean): the
    #    lock a crashed writer left behind blocks every git write and is
    #    flagged as corruption once past the staleness cadence
    stale_lock = store.root / ".git" / "index.lock"
    stale_lock.write_bytes(b"")
    long_ago = time.time() - 2 * 3600
    os.utime(stale_lock, (long_ago, long_ago))
    locked = detect_corruption(store.root)
    assert locked.corrupted is True
    assert any("index.lock" in finding for finding in locked.findings)
    repaired = attempt_repair(store.root)
    assert repaired.repaired is True
    assert any("removed stale .git/index.lock" in action for action in repaired.actions)
    assert not stale_lock.exists()
    assert detect_corruption(store.root).corrupted is False
    assert revived.git.is_clean()


@pytest.mark.chaos
def test_missing_object_restored_from_backup(chaos_store) -> None:
    """Lost object bytes: flagged, unrepairable, backup restore loses nothing."""
    store = chaos_store
    for text in _OBJECT_CORRUPTION_TEXTS:
        store.add(text, type="Feedback", source="user-explicit")
    entry_ids = [entry.id for entry in store.list()]
    assert len(entry_ids) == 2

    manager = BackupManager()
    archive = manager.create_backup()  # taken BEFORE the corruption
    assert archive.is_file()

    _truncate_entry_blob(store, entry_ids[0])

    # the corruption is detected honestly (fsck names the missing object)
    report = detect_corruption(store.root)
    assert report.corrupted is True
    evidence = " ".join(report.findings).lower()
    assert "missing" in evidence or "corrupt" in evidence

    # reads survive the lost blob: every entry is intact at the file level
    assert sorted(entry.text for entry in store.list()) == _OBJECT_CORRUPTION_TEXTS
    assert all(store.get(entry_id) is not None for entry_id in entry_ids)

    # gc cannot regenerate object bytes — the escalation says so honestly
    outcome = attempt_repair(store.root)
    assert outcome.repaired is False

    # restore from the pre-corruption backup: every entry intact, no data loss
    result = recover_from_backup(store.root, manager.backups_dir)
    assert result.restored_from == archive
    assert result.entries_count == 2
    revived = MemoryStore(root=store.root)
    assert sorted(entry.text for entry in revived.list()) == _OBJECT_CORRUPTION_TEXTS
    assert detect_corruption(store.root).corrupted is False
    assert revived.git.is_clean()
    post = revived.add("written after object restore", type="Feedback", source="user-explicit")
    assert revived.get(post.id) is not None
    assert revived.git.is_clean()
    assert revived.git.log(path=f"global/facts/{post.id}.md", limit=5)
