"""ConcurrencyManager: file locking for a shared Hermes home (plan P6.T1).

Multiple agent processes (main + subagents, plan Phase 6) share one Hermes
home; every mutating operation must serialize on a single exclusive lock so
the memory repository cannot corrupt. This module is a thin coordination
layer ON TOP of :class:`~hermes.memory.locking.FileLock` (plan P1.T5) — the
low-level locking primitives are reused, not duplicated:

- :meth:`ConcurrencyManager.acquire_lock` / :meth:`ConcurrencyManager.release_lock`
  wrap one shared :class:`FileLock` instance and return/accept an opaque
  :class:`LockHandle` (proof-of-hold);
- :meth:`ConcurrencyManager.queue_write` executes a write operation while
  holding the lock — the serialized-writer seam the Phase 6 writer process
  (P6.T2) and the "main agent bypass queue for direct writes" path both
  build on.

Guarantees (all inherited from ``FileLock``):

- **Cross-process exclusivity**: the byte-range lock (``fcntl.flock`` on
  POSIX, ``msvcrt.locking`` on Windows) is held on byte 0 of the persistent
  lock file, so a second *process* cannot take the lock while it is held —
  P1.T5 exercises the primitives; P6.T1's tests re-assert the behavior at
  the manager level (threads block; a foreign process fails fast).
- **Reentrancy (same thread)**: re-acquiring from the owning thread
  depth-counts, exactly like ``FileLock``; releases unwind one level.
- **Bounded waiting**: every wait is timeout-bounded and degrades into
  :class:`LockTimeoutError` instead of a deadlock. ``timeout`` is
  configurable per call and via the constructor (default 10s, plan
  Phase 6: ``acquire_lock(timeout=10)``).

Async note: the lock itself is OS-level, so it is safe across event loops
and processes ("async-safety via the lock"); the poll-wait inside
``FileLock.acquire`` blocks the calling thread/event loop for at most
``timeout`` seconds, which is the intended behavior for a writer queue.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from hermes.config import hermes_home
from hermes.memory.locking import (
    DEFAULT_LOCK_TIMEOUT,
    LOCK_NAME,
    FileLock,
    LockTimeoutError,
)

__all__ = [
    "ConcurrencyManager",
    "LockHandle",
    "LockTimeoutError",
    "default_lock_path",
    "get_shared_lock",
]

T = TypeVar("T")

#: Default poll cadence handed to the underlying :class:`FileLock`.
_DEFAULT_POLL_INTERVAL: float = 0.05


def default_lock_path() -> Path:
    """The shared Hermes home lock file: ``<hermes_home()>/.lock`` (plan P6.T1)."""
    return hermes_home() / ".lock"


def get_shared_lock(
    repo_root: Path | str,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> FileLock:
    """A :class:`FileLock` on the shared memory repository's own lock file (plan P6.T3).

    Additive P6.T3 convenience: the SAME ``<repo_root>/.lock`` byte-range
    lock every :class:`~hermes.memory.store.MemoryStore` instance takes
    around its mutations, exposed so components acting OUTSIDE a store
    mutation — the crash-recovery layer's Git durability sweep
    (:mod:`hermes.concurrency.recovery`), a direct (bypass-queue) write
    coordination window, a second agent's check-then-send — serialize
    against every store (and writer flush) on that repository.

    Note: do NOT hold it across ``MemoryStore`` mutations made through a
    store instance that locks the same file — ``FileLock`` reentrancy is
    per-instance, so nesting the shared lock around ``store.add()`` would
    self-deadlock until the timeout.
    """
    return FileLock(
        Path(repo_root) / LOCK_NAME, timeout=timeout, poll_interval=poll_interval
    )


@dataclass(frozen=True)
class LockHandle:
    """Opaque proof-of-hold returned by :meth:`ConcurrencyManager.acquire_lock`.

    Pass it back to :meth:`ConcurrencyManager.release_lock`; ``is_held``
    reports whether the underlying lock is currently held (any depth).
    """

    lock: FileLock

    @property
    def is_held(self) -> bool:
        """True while the lock is held by this manager (any reentrancy depth)."""
        return self.lock.is_held


class ConcurrencyManager:
    """Coordinates exclusive access to the shared Hermes home (plan P6.T1).

    One manager owns exactly one :class:`FileLock` on ``lock_path`` (default
    ``<hermes_home()>/.lock``); ``timeout`` is the default wait budget for
    every acquire made through this manager (overridable per call).
    """

    def __init__(
        self,
        lock_path: Path | None = None,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        self.lock_path: Path = (
            Path(lock_path) if lock_path is not None else default_lock_path()
        )
        self.timeout: float = timeout
        self._lock = FileLock(
            self.lock_path, timeout=timeout, poll_interval=poll_interval
        )

    # ── lock lifecycle ────────────────────────────────────────────────────
    def acquire_lock(self, timeout: float | None = None) -> LockHandle:
        """Take the exclusive lock, waiting at most ``timeout`` seconds.

        ``timeout=None`` uses the constructor default. Re-acquiring from the
        owning thread deepens the reentrancy counter (same-thread behavior
        matches ``FileLock``). Raises :class:`LockTimeoutError` when the
        budget is exhausted.
        """
        self._lock.acquire(timeout)
        return LockHandle(self._lock)

    def release_lock(self, handle: LockHandle) -> None:
        """Release one level of the lock identified by ``handle``.

        Must be called from the acquiring thread and balanced with successful
        acquires. Raises :class:`TypeError` for a non-handle and
        :class:`ValueError` for a handle issued by another manager.
        """
        if not isinstance(handle, LockHandle):
            raise TypeError(
                f"release_lock expects a LockHandle, got {type(handle).__name__}"
            )
        if handle.lock is not self._lock:
            raise ValueError("lock handle was not issued by this ConcurrencyManager")
        handle.lock.release()

    # ── serialized writes ─────────────────────────────────────────────────
    def queue_write(self, operation: Callable[[], T]) -> T:
        """Execute ``operation`` while holding the exclusive lock; returns its result.

        The seam for the Phase 6 writer queue (P6.T2) and direct writes:
        the operation runs under the lock, so concurrent callers — threads
        here, processes via the underlying byte-range lock — serialize.
        Reentrant when the calling thread already holds the lock.
        """
        if not callable(operation):
            raise TypeError(
                f"operation must be callable, got {type(operation).__name__}"
            )
        with self._lock:
            return operation()
