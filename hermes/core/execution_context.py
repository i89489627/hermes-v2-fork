"""hermes-v2 per-execution interrupt state (plan P21.T1).

Fixes upstream issue #35309: the fork-base interrupt registry was keyed by
**thread ID** while :class:`concurrent.futures.ThreadPoolExecutor` recycles
worker threads. When a worker raised (in particular a ``BaseException``
subclass) after an interrupt was requested, the thread's entry stayed
"poisoned" and a *later*, unrelated tool execution that recycled the same
thread ID observed ``is_interrupted() == True`` and aborted silently.

The v2 fix tracks interrupt state per **execution**, not per thread:

- each tool run gets an :class:`ExecutionContext` carrying its own
  :class:`threading.Event`;
- a module-level registry (a :class:`threading.local`) publishes the
  ExecutionContext of the run currently executing on *this* thread. It is set
  at execution start and **cleared in a ``finally`` block**, so a recycled
  thread ID always starts clean regardless of how its predecessor died.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "ExecutionContext",
    "clear_current_execution",
    "current_execution",
    "request_interrupt",
    "set_current_execution",
]


@dataclass
class ExecutionContext:
    """Interrupt + identity state of exactly ONE tool execution.

    Instance-scoped by construction: ``_interrupted`` is a per-instance
    :class:`threading.Event`, so two executions can never observe each
    other's interrupt requests even when they run on the same recycled
    worker thread (issue #35309).
    """

    tool_name: str = ""
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    _interrupted: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )

    def request_interrupt(self) -> None:
        """Request that THIS execution stops; affects no other execution."""
        self._interrupted.set()

    def is_interrupted(self) -> bool:
        """True only if an interrupt was requested for this very execution."""
        return self._interrupted.is_set()


#: Per-thread slot holding the ExecutionContext that is currently running on
#: this OS thread. ``threading.local`` guarantees one slot per thread, so
#: concurrent executions never collide; the ToolExecutor clears the slot in a
#: ``finally`` block so recycled thread IDs always start clean.
_CURRENT: Final[threading.local] = threading.local()


def set_current_execution(execution: ExecutionContext) -> None:
    """Publish ``execution`` as the one currently running on this thread."""
    _CURRENT.execution = execution


def clear_current_execution() -> None:
    """Drop the current-execution slot for this thread (idempotent)."""
    _CURRENT.execution = None


def current_execution() -> ExecutionContext | None:
    """The ExecutionContext running on this thread, or ``None``.

    Returns ``None`` outside any execution and on a freshly recycled worker
    thread — a stale predecessor can never leak through (issue #35309).
    """
    value = getattr(_CURRENT, "execution", None)
    return value if isinstance(value, ExecutionContext) else None


def request_interrupt(execution: ExecutionContext) -> None:
    """Set only ``execution``'s interrupt event; siblings are untouched."""
    execution.request_interrupt()
