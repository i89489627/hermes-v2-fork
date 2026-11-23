"""hermes-v2 concurrency package (plan P0.T3).

P6.T1 adds the :class:`~hermes.concurrency.manager.ConcurrencyManager`
(file locking over ``<hermes_home()>/.lock``); the writer process / IPC
protocol (P6.T2) and crash recovery (P6.T3) land in later tasks.
"""

from hermes.concurrency.manager import (
    ConcurrencyManager,
    LockHandle,
    LockTimeoutError,
    default_lock_path,
)

__all__ = [
    "ConcurrencyManager",
    "LockHandle",
    "LockTimeoutError",
    "default_lock_path",
]
