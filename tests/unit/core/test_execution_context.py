"""Unit tests for per-execution interrupt state (plan P21.T1, issue #35309).

Upstream bug: the interrupt registry was keyed by worker-thread ID while
``ThreadPoolExecutor`` recycles threads, so a crashed (interrupted) worker
poisoned every later tool run that recycled its thread. The v2 fix keys
interrupt state on an :class:`~hermes.core.execution_context.ExecutionContext`
and clears the thread-local current-execution slot in a ``finally`` block.
"""

from __future__ import annotations

import threading

import pytest

from hermes.core.execution_context import (
    ExecutionContext,
    clear_current_execution,
    current_execution,
    request_interrupt,
    set_current_execution,
)
from hermes.core.tool_executor import ToolExecutor


@pytest.mark.unit
def test_interrupt_state_per_execution() -> None:
    """``request_interrupt()`` flips exactly one execution's flag.

    Two executions — even conceptually "on the same worker thread" — carry
    independent Events; interrupting one must never leak into the other.
    """
    first = ExecutionContext(tool_name="search_web")
    second = ExecutionContext(tool_name="fetch_url")

    # Identity metadata: unique ids, sane start stamps.
    assert first.execution_id != second.execution_id
    assert first.started_at > 0
    assert second.started_at > 0
    assert first.tool_name == "search_web"
    assert second.tool_name == "fetch_url"

    assert first.is_interrupted() is False
    assert second.is_interrupted() is False

    request_interrupt(first)
    assert first.is_interrupted() is True
    assert second.is_interrupted() is False

    second.request_interrupt()
    assert second.is_interrupted() is True
    assert first.is_interrupted() is True  # both set, fully independent

    # Module-level helper only ever touches the execution it was given.
    fresh = ExecutionContext(tool_name="unused")
    assert fresh.is_interrupted() is False


@pytest.mark.unit
def test_thread_id_recycle_safe() -> None:
    """A recycled worker thread must never observe a predecessor's interrupt.

    Mirrors the issue-#35309 recycle scenario twice: once directly against
    the thread-local registry, once through the real ToolExecutor with a
    single worker (forcing the second task onto the same recycled thread).
    """
    # 1. Registry-level simulation of thread-ID recycling on ONE thread.
    poisoned = ExecutionContext(tool_name="old-tool")
    recycled = ExecutionContext(tool_name="new-tool")

    set_current_execution(poisoned)
    try:
        request_interrupt(poisoned)
        assert poisoned.is_interrupted() is True
        assert current_execution() is poisoned
    finally:
        # The executor's finally-clear in the face of any crash, simulated.
        clear_current_execution()

    # Same physical thread now "recycled" for the next execution.
    set_current_execution(recycled)
    try:
        assert current_execution() is recycled
        assert recycled.is_interrupted() is False
    finally:
        clear_current_execution()
    assert current_execution() is None

    # 2. Real pool with a single worker: both tasks run on the same thread,
    #    the first one self-interrupts and crashes, the second starts clean.
    executor = ToolExecutor(max_workers=1)
    seen_idents: list[int] = []

    def crash_and_interrupt() -> None:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        assert ctx is not None, "worker must see its own ExecutionContext"
        ctx.request_interrupt()
        raise RuntimeError("first execution crashed")

    crashing = executor.submit("tool-a", crash_and_interrupt)
    with pytest.raises(RuntimeError, match="first execution crashed"):
        crashing.result(timeout=10)
    assert crashing.execution.is_interrupted() is True

    def verify_clean() -> bool:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        return ctx is not None and not ctx.is_interrupted()

    followup = executor.submit("tool-b", verify_clean)
    assert followup.result(timeout=10) is True
    assert followup.execution.is_interrupted() is False
    executor.shutdown(wait=True)

    # Same worker thread was recycled — proving the poisoning scenario shape.
    assert len(seen_idents) == 2
    assert seen_idents[0] == seen_idents[1]
