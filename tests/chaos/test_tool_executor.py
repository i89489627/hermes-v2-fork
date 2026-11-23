"""Chaos test: concurrent crashes must not poison sibling executions
(plan P21.T1, issue #35309).

20 concurrent tasks on an 8-worker pool: the even-indexed tasks request an
interrupt and then crash; the odd-indexed tasks assert they started clean.
Afterwards every execution's flag must reflect only its own request, and the
pool must still serve subsequent clean tasks.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hermes.core.execution_context import current_execution
from hermes.core.tool_executor import ToolExecutor

_TOTAL_TASKS = 20
_FOLLOWUP_TASKS = 10
_TIMEOUT = 60


def _probe() -> bool:
    """Clean-state probe: a fresh execution must never be pre-interrupted."""
    ctx = current_execution()
    return ctx is not None and not ctx.is_interrupted()


def _make_task(index: int) -> Callable[[], bool]:
    """Build the task body: odd tasks assert clean start; even ones crash."""

    def run() -> bool:
        ctx = current_execution()
        assert ctx is not None, "task must see its own ExecutionContext"
        clean_at_start = not ctx.is_interrupted()
        if index % 2 == 0:
            ctx.request_interrupt()  # interrupt THIS execution only...
            raise RuntimeError(f"task {index} crashed mid-run")  # ...then die
        return clean_at_start

    return run


@pytest.mark.chaos
def test_concurrent_crashes_no_poisoning() -> None:
    """Half the pool crashes while interrupted; nobody else gets poisoned."""
    executor = ToolExecutor(max_workers=8)
    handles = [
        executor.submit(f"tool-{index}", _make_task(index))
        for index in range(_TOTAL_TASKS)
    ]

    for index, handle in enumerate(handles):
        if index % 2 == 0:
            with pytest.raises(RuntimeError, match=f"task {index} crashed"):
                handle.result(timeout=_TIMEOUT)
        else:
            # Odd tasks observed a clean execution even though ~half the
            # pool was crashing and self-interrupting around them.
            assert handle.result(timeout=_TIMEOUT) is True

    # Each execution's flag reflects only its own request: even -> True
    # (it asked), odd -> False (nobody else may have set it).
    for index, handle in enumerate(handles):
        assert handle.execution.is_interrupted() == (index % 2 == 0), (
            f"execution {index} has cross-poisoned interrupt state"
        )

    # The pool still serves subsequent tasks, all starting clean.
    followups = [
        executor.submit(f"post-crash-{index}", _probe)
        for index in range(_FOLLOWUP_TASKS)
    ]
    assert all(handle.result(timeout=_TIMEOUT) is True for handle in followups)
    assert all(not handle.execution.is_interrupted() for handle in followups)
    executor.shutdown(wait=True)
