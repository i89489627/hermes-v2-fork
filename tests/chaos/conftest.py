"""Shared chaos-test fixtures (plan P20.T1: chaos testing framework).

The chaos suite injects failures into the hermes-v2 memory stack and asserts
the system recovers gracefully with NO DATA LOSS (plan P20.T1 acceptance).
This conftest provides the shared failure-injection primitives the P20.T1
test modules build on:

- ``chaos_store`` — a fresh, hermetic :class:`~hermes.memory.store.MemoryStore`
  under the per-test ``HERMES_HOME`` tempdir (the suite-wide conftest already
  redirects it; the store's default path derives from
  :func:`hermes.config.hermes_home`).
- ``failing_model`` — a factory for main-model callables that raise
  :class:`ModelUnavailableError` (the provider-unreachable failure), either
  on every call or only after ``fail_after`` successful completions — the
  API-failure-mid-request shape (the session worked, then the provider went
  away).
- ``scripted_model`` — a factory for always-available model callables that
  return one scripted valid completion (the "provider recovered" end of a
  chaos scenario).
- ``deferred_ops`` — a :class:`DeferredOpQueue`: the offline queue an agent
  keeps while the main-model provider is unreachable. Ops are deferred (never
  lost) and flushed through the real store IN ORDER on recovery (the plan
  Section 5.5 "deferred queue" behind the network-failure chaos scenario).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore

__all__ = [
    "DeferredOpQueue",
    "FailingModel",
    "ModelUnavailableError",
    "ScriptedModel",
]


class ModelUnavailableError(RuntimeError):
    """The main-model provider is unreachable (chaos injection)."""


class FailingModel:
    """Injectable main-model callable that fails like an unreachable provider.

    ``fail_after=None`` raises :class:`ModelUnavailableError` on EVERY call
    (provider unreachable at startup). ``fail_after=N`` serves the first
    ``N`` calls from ``success_response`` (a scripted valid completion) and
    raises afterwards — the API-failure-mid-request shape. Every received
    prompt is recorded so tests can assert how often the (un)available
    provider was contacted, and ``failures`` counts the refused calls.
    """

    def __init__(
        self,
        *,
        fail_after: int | None = None,
        success_response: str = "",
        error: str = "main model provider unreachable (chaos)",
    ) -> None:
        if fail_after is not None and fail_after < 0:
            raise ValueError(f"fail_after must be >= 0 or None, got {fail_after}")
        self.fail_after = fail_after
        self.success_response = success_response
        self.error = error
        self.prompts: list[str] = []
        self.failures = 0

    @property
    def calls(self) -> int:
        """Total model calls received (successful + failed)."""
        return len(self.prompts)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail_after is None or len(self.prompts) > self.fail_after:
            self.failures += 1
            raise ModelUnavailableError(self.error)
        return self.success_response


class ScriptedModel:
    """Always-available model returning one scripted completion (recovery)."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        """Total model calls received."""
        return len(self.prompts)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


class DeferredOpQueue:
    """Offline queue of memory ops collected while the provider is down.

    ``defer`` records an op (never loses it); ``flush`` applies every
    deferred op through the real :class:`MemoryStore` IN DEFER ORDER once
    the provider recovered — the plan Section 5.5 deferred-queue contract
    the network-failure chaos scenario verifies.
    """

    def __init__(self) -> None:
        self._ops: list[dict[str, Any]] = []
        self._deferred_at: list[float] = []
        self.flushed: list[MemoryEntry] = []

    def defer(self, op: dict[str, Any]) -> int:
        """Queue one memory op; returns the queue length after the append."""
        self._ops.append(dict(op))
        self._deferred_at.append(time.monotonic())
        return len(self._ops)

    @property
    def pending(self) -> int:
        """Ops still waiting to be flushed."""
        return len(self._ops)

    @property
    def ops(self) -> list[dict[str, Any]]:
        """A copy of the deferred ops, in defer order."""
        return [dict(op) for op in self._ops]

    def flush(self, store: MemoryStore) -> list[MemoryEntry]:
        """Apply every deferred op through ``store.add``, in order.

        Raises :class:`RuntimeError` on a second flush — each op must be
        applied exactly once (the no-data-loss / no-duplication contract).
        The queue drains on success: ``pending`` drops to 0.
        """
        if self.flushed:
            raise RuntimeError("deferred ops were already flushed once")
        applied: list[MemoryEntry] = []
        for op in self._ops:
            applied.append(
                store.add(
                    str(op["text"]),
                    type=str(op.get("type", "Feedback")),
                    context=str(op.get("context", "global")),
                    source=str(op.get("source", "agent-inferred")),
                )
            )
        self._ops.clear()
        self._deferred_at.clear()
        self.flushed = applied
        return applied


@pytest.fixture()
def chaos_store() -> Iterator[MemoryStore]:
    """A hermetic MemoryStore for chaos scenarios (per-test HERMES_HOME)."""
    store = MemoryStore()
    yield store


@pytest.fixture()
def failing_model() -> Callable[..., FailingModel]:
    """Factory for provider-unreachable model callables (see FailingModel)."""

    def make(**kwargs: Any) -> FailingModel:
        return FailingModel(**kwargs)

    return make


@pytest.fixture()
def scripted_model() -> Callable[[str], ScriptedModel]:
    """Factory for always-available scripted models (see ScriptedModel)."""

    def make(response: str) -> ScriptedModel:
        return ScriptedModel(response)

    return make


@pytest.fixture()
def deferred_ops() -> DeferredOpQueue:
    """The offline deferred-op queue (network-failure chaos scenarios)."""
    return DeferredOpQueue()
