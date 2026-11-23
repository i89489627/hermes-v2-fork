"""Integration tests: ToolExecutor crash isolation (plan P21.T1, issue #35309).

Exercises the real thread pool: a ``BaseException``-derived crash in an
interrupted execution must neither kill the pool nor poison the next call
scheduled onto the recycled worker thread.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes.core.execution_context import ExecutionContext, current_execution
from hermes.core.tool_executor import ToolExecutor


class _WorkerAbort(BaseException):
    """BaseException-derived crash, like the upstream mid-run worker abort."""


@pytest.mark.integration
def test_crash_does_not_poison_next_call() -> None:
    """An interrupted execution that dies from a BaseException-derived error
    mid-run must leave the pool clean for the next call on the same thread.

    ``max_workers=1`` forces the follow-up task onto the recycled thread, so
    the scenario reproduces the issue-#35309 thread-ID-recycle shape exactly.
    """
    executor = ToolExecutor(max_workers=1)
    seen_idents: list[int] = []
    crashed_execution: list[ExecutionContext] = []

    def crashing_tool() -> int:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        assert ctx is not None, "tool must see its own ExecutionContext"
        ctx.request_interrupt()  # mark interrupted, then die mid-run
        crashed_execution.append(ctx)
        time.sleep(0.05)  # "mid-run": die after the interrupt was registered
        raise _WorkerAbort("worker aborted mid-run")

    crashing = executor.submit("flaky-tool", crashing_tool)
    with pytest.raises(_WorkerAbort):
        crashing.result(timeout=10)

    # The crashed execution keeps its own (correct) state...
    assert crashed_execution[0].is_interrupted() is True
    assert crashing.execution is crashed_execution[0]
    # ...but the recycled thread must start clean.
    assert seen_idents, "crashing tool did run"

    def healthy_tool() -> bool:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        return ctx is not None and not ctx.is_interrupted()

    followup = executor.submit("healthy-tool", healthy_tool)
    assert followup.result(timeout=10) is True
    assert followup.execution.is_interrupted() is False
    assert seen_idents[0] == seen_idents[1], "worker thread was recycled"

    # One more call for good measure: the pool keeps serving cleanly.
    assert executor.submit("healthy-tool-2", healthy_tool).result(timeout=10) is True
    executor.shutdown(wait=True)

    # The main thread never executed a tool, so its slot stays untouched.
    assert current_execution() is None


@pytest.mark.integration
def test_is_interrupted_returns_correct_state() -> None:
    """``is_interrupted()`` reflects only the execution it was requested on,
    before, during and after the run, across concurrent sibling executions."""
    executor = ToolExecutor(max_workers=4)
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, bool] = {}

    def blocking_tool() -> None:
        ctx = current_execution()
        started.set()
        release.wait(timeout=10)
        observed["blocker"] = ctx is not None and ctx.is_interrupted()

    blocked = executor.submit("blocking-tool", blocking_tool)
    assert started.wait(timeout=10), "blocking tool never started"

    # Before the interrupt request: correct False, on the live execution.
    assert blocked.execution.is_interrupted() is False

    blocked.execution.request_interrupt()
    assert blocked.execution.is_interrupted() is True

    # A sibling execution running concurrently must NOT observe the
    # blocker's interrupt (per-execution, not per-thread / global).
    def probe_tool() -> bool:
        ctx = current_execution()
        return ctx is not None and not ctx.is_interrupted()

    probe = executor.submit("probe-tool", probe_tool)
    assert probe.result(timeout=10) is True
    assert probe.execution.is_interrupted() is False

    # Unblocking lets the interrupted tool report its own state: True.
    release.set()
    assert blocked.result(timeout=10) is None
    assert observed["blocker"] is True

    # A post-hoc third execution is clean again.
    third = executor.submit("third-tool", probe_tool)
    assert third.result(timeout=10) is True
    assert third.execution.is_interrupted() is False
    executor.shutdown(wait=True)
