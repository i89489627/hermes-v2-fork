"""Cross-platform exclusive file locking for the memory store (plan P1.T5).

Concurrent writers (multi-agent sessions, plan Phase 6) must not corrupt
the memory repository: every mutating :class:`~hermes.memory.store.MemoryStore`
operation runs while holding an exclusive lock on ``<root>/.lock``.
:class:`FileLock` implements that lock portably — ``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows — polling with a bounded timeout so a stalled
holder degrades into :class:`LockTimeoutError` instead of a deadlock.
Byte-range locks are taken on byte 0 of the (never-written) lock file; the
file itself is persistent (Windows cannot unlink an open file) and the
store keeps it out of Git status via ``.git/info/exclude``.

The lock is reentrant per owning thread (depth-counted, identified by
thread id) so internal helpers can nest mutation paths safely; instances
are cheap and one instance per store is the expected shape.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import TracebackType

__all__ = ["DEFAULT_LOCK_TIMEOUT", "LOCK_NAME", "FileLock", "LockTimeoutError"]

#: Default bound on :meth:`FileLock.acquire` waits.
DEFAULT_LOCK_TIMEOUT: float = 10.0

#: Name of the store's persistent lock file, under the store root.
LOCK_NAME = ".lock"

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        """One non-blocking exclusive byte-range lock attempt (Windows)."""
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the byte-range lock taken by :func:`_try_lock`."""
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        """One non-blocking exclusive ``flock`` attempt (POSIX)."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the ``flock`` taken by :func:`_try_lock`."""
        fcntl.flock(fd, fcntl.LOCK_UN)


class LockTimeoutError(TimeoutError):
    """The lock could not be acquired within the configured timeout."""


class FileLock:
    """Exclusive, timeout-bounded, thread-reentrant lock on one lock file.

    ``acquire(timeout)`` blocks in a poll loop (``poll_interval`` sleeps)
    until the byte-range lock is taken or ``timeout`` elapses — the timeout
    is fully configurable per call and via the constructor default. Holding
    the lock is tracked per instance: re-acquiring from the owning thread
    simply deepens a counter, while another thread (or process) must wait
    for release like any external contender. ``release`` must be called by
    the acquiring thread and balanced with successful acquires.
    """

    def __init__(
        self,
        path: str | Path,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        poll_interval: float = 0.05,
    ) -> None:
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd: int | None = None
        self._depth = 0
        self._owner: int | None = None

    def acquire(self, timeout: float | None = None) -> None:
        """Take the exclusive lock, waiting at most ``timeout`` seconds.

        ``timeout=None`` uses the constructor default. Re-acquiring from the
        owning thread deepens the reentrancy counter instead of touching the
        OS lock. Raises :class:`LockTimeoutError` when the wait budget is
        exhausted (the lock file is left in place; a later acquire may
        succeed as soon as the current holder releases).
        """
        ident = threading.get_ident()
        if self._depth > 0 and self._owner == ident:
            self._depth += 1
            return
        effective = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + effective
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            while not _try_lock(fd):
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(
                        f"could not acquire lock {self.path} within {effective:.3f}s"
                    )
                time.sleep(self.poll_interval)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._depth = 1
        self._owner = ident

    def release(self) -> None:
        """Release one level of the lock (unlocking on the last level).

        Raises :class:`RuntimeError` when the lock is not currently held by
        this instance's owning thread.
        """
        ident = threading.get_ident()
        if self._depth == 0 or self._owner != ident:
            raise RuntimeError(f"lock {self.path} is not held by thread {ident}")
        self._depth -= 1
        if self._depth > 0:
            return
        fd = self._fd
        self._fd = None
        self._owner = None
        if fd is not None:
            try:
                _unlock(fd)
            finally:
                os.close(fd)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    @property
    def is_held(self) -> bool:
        """True while this instance holds the lock (any depth)."""
        return self._depth > 0
