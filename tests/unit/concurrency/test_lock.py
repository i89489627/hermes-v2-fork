"""P6.T1 acceptance tests: ConcurrencyManager file locking (plan Phase 6).

Covers the manager-level contract over the P1.T5 ``FileLock`` primitives:
clean acquire/release, a second contender blocking until release, and a
bounded wait that degrades into :class:`LockTimeoutError`. The concurrent
test also proves the cross-process guarantee directly: while this process
holds the lock, a spawned foreign process attempting a short non-blocking
acquire must fail fast with the timeout error (the underlying msvcrt/flock
byte-range lock is process-wide).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.concurrency.manager import (
    ConcurrencyManager,
    LockHandle,
    LockTimeoutError,
    default_lock_path,
)

#: Tiny script run in a foreign process: try to take the lock for a moment.
_CHILD_ATTEMPT = """
import sys
sys.path.insert(0, sys.argv[1])
from hermes.memory.locking import FileLock, LockTimeoutError

lock = FileLock(sys.argv[2], timeout=1.0, poll_interval=0.02)
try:
    lock.acquire()
except LockTimeoutError:
    print("LOCK_TIMEOUT")
    raise SystemExit(0)
print("LOCK_ACQUIRED")
raise SystemExit(1)
"""


@pytest.mark.unit
def test_acquire_release(tmp_path: Path) -> None:
    """acquire_lock holds cleanly, release lets go, reentrancy depth-counts."""
    lock_path = tmp_path / "home" / ".lock"
    manager = ConcurrencyManager(lock_path=lock_path)
    assert manager.lock_path == lock_path

    handle = manager.acquire_lock()
    assert isinstance(handle, LockHandle)
    assert lock_path.is_file()  # persistent lock file created on acquire
    assert handle.is_held

    manager.release_lock(handle)
    assert not handle.is_held

    # re-acquire after a clean release
    again = manager.acquire_lock()
    manager.release_lock(again)

    # same-thread reentrancy matches FileLock: depth-counted, one OS lock
    outer = manager.acquire_lock()
    inner = manager.acquire_lock()
    assert inner is not outer  # distinct handles...
    assert inner.lock is outer.lock  # ...over the same underlying lock
    manager.release_lock(inner)  # unwinds one level only
    assert outer.is_held
    manager.release_lock(outer)
    assert not outer.is_held

    # queue_write executes under the lock and returns the operation result
    assert manager.queue_write(lambda: "written") == "written"
    counter = {"n": 0}

    def bump() -> int:
        counter["n"] += 1
        return counter["n"]

    assert manager.queue_write(bump) == 1
    assert counter["n"] == 1
    # reentrant while already held: no deadlock, result still returned
    held = manager.acquire_lock()
    assert manager.queue_write(lambda: 41 + 1) == 42
    manager.release_lock(held)

    # the default lock path derives from hermes_home() (per-test tempdir)
    assert ConcurrencyManager().lock_path == hermes_home() / ".lock"
    assert default_lock_path() == hermes_home() / ".lock"

    # invalid configuration and misuse rejected eagerly
    with pytest.raises(ValueError, match="timeout"):
        ConcurrencyManager(lock_path=lock_path, timeout=-1.0)
    with pytest.raises(TypeError, match="LockHandle"):
        manager.release_lock("not-a-handle")  # type: ignore[arg-type]
    foreign = ConcurrencyManager(lock_path=tmp_path / "elsewhere" / ".lock")
    with pytest.raises(ValueError, match="not issued by this"):
        manager.release_lock(foreign.acquire_lock())


@pytest.mark.unit
def test_concurrent_acquire_blocks(tmp_path: Path) -> None:
    """A second contender blocks until release — threads here, processes via spawn."""
    lock_path = tmp_path / "home" / ".lock"
    manager = ConcurrencyManager(lock_path=lock_path, timeout=5.0)
    handle = manager.acquire_lock()

    acquired = threading.Event()
    release_order: list[str] = []

    def contender() -> None:
        second = ConcurrencyManager(lock_path=lock_path, timeout=5.0)
        got = second.acquire_lock()  # must block until the holder releases
        release_order.append("second")
        acquired.set()
        second.release_lock(got)

    thread = threading.Thread(target=contender)
    thread.start()
    try:
        # while held, the contender must NOT get through
        assert not acquired.wait(timeout=0.4), (
            "second acquire succeeded while the lock was held"
        )
        release_order.append("first")
        manager.release_lock(handle)
        # after release the blocked contender acquires and finishes
        assert acquired.wait(timeout=5.0)
        thread.join(timeout=5.0)
        assert release_order == ["first", "second"]
    finally:
        thread.join(timeout=5.0)

    # Cross-process proof: re-hold the lock, then a foreign process that
    # attempts a 1s non-blocking acquire must fail fast with LockTimeoutError
    # (the byte-range lock is exclusive across processes on both fcntl and
    # msvcrt). Deterministic: the holder is owned by this thread for the
    # whole subprocess run.
    held_again = manager.acquire_lock()
    try:
        repo_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD_ATTEMPT, str(repo_root), str(lock_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        manager.release_lock(held_again)

    assert completed.returncode == 0, (
        f"child output: stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert "LOCK_TIMEOUT" in completed.stdout
    assert "LOCK_ACQUIRED" not in completed.stdout


@pytest.mark.unit
def test_lock_timeout(tmp_path: Path) -> None:
    """A blocked acquire degrades into LockTimeoutError within its budget."""
    lock_path = tmp_path / "home" / ".lock"
    manager = ConcurrencyManager(lock_path=lock_path, timeout=5.0)
    handle = manager.acquire_lock()

    contender = ConcurrencyManager(lock_path=lock_path, timeout=5.0)
    # per-call timeout overrides the constructor default
    with pytest.raises(LockTimeoutError):
        contender.acquire_lock(timeout=0.15)
    # constructor default applies when the call omits the timeout
    fast = ConcurrencyManager(lock_path=lock_path, timeout=0.1, poll_interval=0.01)
    with pytest.raises(LockTimeoutError, match="0\\.1"):
        fast.acquire_lock()

    # a timed-out contender holds nothing and the manager stays usable
    manager.release_lock(handle)
    recovered = contender.acquire_lock(timeout=2.0)
    assert recovered.is_held
    contender.release_lock(recovered)
    # and the original holder can re-acquire too
    handle2 = manager.acquire_lock()
    manager.release_lock(handle2)
