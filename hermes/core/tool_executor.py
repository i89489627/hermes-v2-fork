"""hermes-v2 tool executor with crash-safe interrupt tracking (plan P21.T1).

Fixes upstream issue #35309. The fork base recorded "this thread was
interrupted" in a registry keyed by worker-thread ID; because
:class:`concurrent.futures.ThreadPoolExecutor` recycles threads, a worker
that died from an exception after an interrupt request left a stale entry
behind, and the *next* tool run scheduled onto the recycled thread saw
``is_interrupted() == True`` and silently aborted.

:class:`ToolExecutor` here keys interrupt state by :class:`ExecutionContext`
instead: every ``submit()`` creates a fresh execution whose event is set only
by ``request_interrupt()`` on that very execution, and the thread-local
current-execution slot is cleared in a ``finally`` block — so thread-ID
recycling can never leak interrupt state across executions, no matter how
the previous run on that thread terminated (including ``BaseException``).

``concurrent.futures``' worker loop funnels ``BaseException`` raised by a
tool into the returned future (``_WorkItem.run`` catches ``BaseException``
since Python 3.8), so a crashing tool neither kills the pool nor poisons
its successor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Final

from hermes.core.execution_context import (
    ExecutionContext,
    clear_current_execution,
    set_current_execution,
)

__all__ = ["DEFAULT_MAX_WORKERS", "ExecutionHandle", "ToolExecutor"]

#: Default worker-thread count for :class:`ToolExecutor` (plan P21.T1).
DEFAULT_MAX_WORKERS: Final[int] = 8

_LOGGER = logging.getLogger("hermes.core.tool_executor")


class ExecutionHandle:
    """Future-like handle returned by :meth:`ToolExecutor.submit`.

    Exposes ``.execution`` (the run's :class:`ExecutionContext`) alongside the
    usual future surface, so a caller can request an interrupt of a specific
    in-flight execution instead of a whole thread.
    """

    def __init__(self, execution: ExecutionContext, future: Future[Any]) -> None:
        self.execution = execution
        self._future = future

    def result(self, timeout: float | None = None) -> Any:
        """Block until done and return the tool's result; re-raises on crash."""
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Return the exception the execution crashed with (``None`` = success)."""
        return self._future.exception(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def cancel(self) -> bool:
        return self._future.cancel()


class ToolExecutor:
    """ThreadPool-based tool runner whose interrupt state is per-execution.

    Usage::

        executor = ToolExecutor(max_workers=8)
        handle = executor.submit("search_web", search_fn, "query")
        handle.execution.request_interrupt()   # stops only THIS execution
        result = handle.result(timeout=30)

    Each submitted function runs with the thread-local current-execution slot
    bound to its own :class:`ExecutionContext`; the slot is cleared in a
    ``finally`` block so a recycled thread always starts clean (issue #35309).
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="hermes-tool"
        )

    def submit(
        self, tool_name: str, fn: Callable[..., Any], *args: Any
    ) -> ExecutionHandle:
        """Schedule ``fn(*args)`` as one named tool execution.

        Creates a fresh :class:`ExecutionContext` for the run; the returned
        handle exposes it via ``.execution`` and the outcome via ``.result()``
        (which re-raises any exception the tool raised, including
        ``BaseException`` subclasses).
        """
        execution = ExecutionContext(tool_name=tool_name)
        future: Future[Any] = self._pool.submit(self._run, execution, fn, args)
        _LOGGER.debug(
            "submitted execution %s of tool %r", execution.execution_id, tool_name
        )
        return ExecutionHandle(execution, future)

    @staticmethod
    def _run(
        execution: ExecutionContext, fn: Callable[..., Any], args: tuple[Any, ...]
    ) -> Any:
        """Worker wrapper: bind the thread-local slot, run, ALWAYS unbind.

        The ``finally`` clear is the load-bearing line for issue #35309: it
        runs even when ``fn`` raises ``BaseException``, so the next task on
        this recycled thread can never observe this execution's state.
        """
        set_current_execution(execution)
        try:
            _LOGGER.debug(
                "execution %s of tool %r starting", execution.execution_id, execution.tool_name
            )
            return fn(*args)
        finally:
            clear_current_execution()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shut the underlying pool down; existing handles stay queryable."""
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> ToolExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown(wait=True)
