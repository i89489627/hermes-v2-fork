"""P1.T5 acceptance: cross-platform exclusive file locking.

Covers :class:`hermes.memory.locking.FileLock` (acquire/release, context
manager, reentrancy, bounded timeout) plus the plan's acceptance criterion
that two concurrent writes against the same store serialize on the lock
and leave every entry intact.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes.memory.locking import FileLock, LockTimeoutError
from hermes.memory.store import MemoryStore


@pytest.mark.unit
def test_acquire_release(tmp_path: Path) -> None:
    """acquire creates the lock file and holds it; release lets it go."""
    lock_path = tmp_path / "repo" / ".lock"
    lock = FileLock(lock_path)
    assert not lock.is_held

    lock.acquire()
    assert lock_path.is_file()
    assert lock.is_held

    lock.release()
    assert not lock.is_held

    lock.acquire()  # re-acquire after release works
    lock.release()

    # context manager form acquires and releases symmetrically
    with FileLock(lock_path) as scoped:
        assert scoped.is_held
    assert not scoped.is_held

    # reentrant acquire from the owning thread: depth-counted, one OS lock
    with FileLock(lock_path) as outer:
        with outer:
            assert outer.is_held
        assert outer.is_held  # inner release only unwinds one level
    assert not outer.is_held

    # releasing a lock this instance does not hold is an error
    idle = FileLock(lock_path)
    with pytest.raises(RuntimeError, match="not held"):
        idle.release()

    # invalid configuration rejected eagerly
    with pytest.raises(ValueError, match="timeout"):
        FileLock(lock_path, timeout=-1.0)
    with pytest.raises(ValueError, match="poll_interval"):
        FileLock(lock_path, poll_interval=0.0)


@pytest.mark.unit
def test_concurrent_acquire_blocks(tmp_path: Path) -> None:
    """A second contender blocks until the holder releases; writes stay intact."""
    lock_path = tmp_path / "repo" / ".lock"
    holder = FileLock(lock_path)
    holder.acquire()

    acquired = threading.Event()

    def contender() -> None:
        waiter = FileLock(lock_path)
        waiter.acquire(timeout=5.0)  # blocks while `holder` owns the lock
        acquired.set()
        waiter.release()

    thread = threading.Thread(target=contender)
    thread.start()
    assert not acquired.wait(timeout=0.4), "contender acquired while lock was held"
    holder.release()
    assert acquired.wait(timeout=5.0), "contender did not acquire after release"
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    # Acceptance: concurrent writes serialize on <root>/.lock and no file is
    # corrupted. Two threads share one store so the Git handle is confined
    # to whichever thread holds the lock (cross-handle Git index caching is
    # a Phase 6 concern, not a P1.T5 one).
    store = MemoryStore(root=tmp_path / "memory-repo")

    def write_many(prefix: str) -> None:
        for i in range(3):
            store.add(f"{prefix} fact {i}", type="Project")

    writers = [
        threading.Thread(target=write_many, args=("alpha",)),
        threading.Thread(target=write_many, args=("beta",)),
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)
    assert all(not writer.is_alive() for writer in writers)

    entries = store.list()
    assert len(entries) == 6
    assert sum(entry.text.startswith("alpha") for entry in entries) == 3
    assert sum(entry.text.startswith("beta") for entry in entries) == 3
    for entry in entries:
        assert store.get(entry.id) == entry  # every file parses and round-trips
    index_text = (tmp_path / "memory-repo" / "MEMORY.md").read_text(encoding="utf-8")
    assert all(entry.id in index_text for entry in entries)
    assert store.git.is_clean()  # everything committed; .lock excluded via info/exclude


@pytest.mark.unit
def test_lock_timeout(tmp_path: Path) -> None:
    """A blocked acquire fails cleanly after the configured timeout."""
    lock_path = tmp_path / "repo" / ".lock"
    holder = FileLock(lock_path)
    holder.acquire()

    waiter = FileLock(lock_path)
    start = time.monotonic()
    with pytest.raises(LockTimeoutError, match="0.150"):
        waiter.acquire(timeout=0.15)
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed < 5.0  # bounded wait, then a clean failure
    assert not waiter.is_held
    assert lock_path.is_file()  # the persistent lock file survives timeouts

    with pytest.raises(LockTimeoutError):  # timeout=0 fails without waiting
        waiter.acquire(timeout=0.0)

    # constructor default timeout is configurable per instance
    strict = FileLock(lock_path, timeout=0.05)
    with pytest.raises(LockTimeoutError):
        strict.acquire()

    # once the holder releases, both waiters acquire normally
    holder.release()
    waiter.acquire()
    waiter.release()
    strict.acquire()
    strict.release()
