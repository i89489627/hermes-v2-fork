"""P3.T5 acceptance: memory pipeline hooked into the Stop event (plan Phase 3).

Covered: the post-turn hook built by :mod:`hermes.memory.hooks.stop_hook`
(registered via :func:`hermes.core.agent.install_memory_pipeline`) fires the
:class:`~hermes.memory.pipeline.MemoryPipeline` ASYNC through
:meth:`hermes.core.agent.V2Agent.run_post_turn_hooks` — the turn completes,
the entry is committed by the background run (awaited through the hook's
future seam with a bounded timeout) — and a failing pipeline component is
logged without crashing the session: the hook completes, the pipeline
degrades (plan 5.5), and subsequently-registered hooks still run.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from hermes.core.agent import V2Agent, install_memory_pipeline
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.hooks.stop_hook import StopHook
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore

_PIPELINE_LOGGER = "hermes.memory.pipeline"
_HOOK_LOGGER = "hermes.memory.hooks.stop_hook"

_TURN_TEXT = "From now on I prefer dark mode in all my IDEs"
_DRAFT_TEXT = "User prefers dark mode IDE themes"


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


def _add_pipeline(store: MemoryStore, *, extractor_model: Any) -> MemoryPipeline:
    """Pipeline with deterministic fakes over the tmp-rooted store."""
    extractor = ExtractorAgent(model=extractor_model)
    reviewer = ReviewerAgent(
        model=lambda _prompt: json.dumps({
            "action": "add",
            "target_id": None,
            "rationale": "new durable user preference",
            "merged_text": None,
        }),
        search_fn=lambda _query: [],
    )
    return MemoryPipeline(extractor, reviewer, store)


def _fake_agent() -> V2Agent:
    """V2Agent without the fork-base machinery (hooks are instance-local).

    ``V2Agent.__init__`` is a pure passthrough to the untyped fork base;
    only the hook lists are v2 state, so bypassing ``__init__`` keeps the
    test hermetic (no model/session wiring) while exercising the real hook
    registration + dispatch code (same pattern as tests/integration/
    test_anchors.py).
    """
    agent = V2Agent.__new__(V2Agent)
    agent._pre_step_hooks = []  # type: ignore[attr-defined]
    agent._post_turn_hooks = []  # type: ignore[attr-defined]
    agent._command_execution_hooks = []  # type: ignore[attr-defined]
    return agent


@pytest.mark.integration
def test_stop_triggers_pipeline(tmp_path: Path) -> None:
    """Stop event -> async pipeline run -> entry committed; bounded wait."""
    store = _make_store(tmp_path)
    pipeline = _add_pipeline(store, extractor_model=lambda _prompt: json.dumps([
        {
            "type": "User",
            "text": _DRAFT_TEXT,
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ]))
    agent = _fake_agent()

    hook = install_memory_pipeline(agent, pipeline)
    assert isinstance(hook, StopHook)
    assert agent._post_turn_hooks == [hook]  # registered on the agent

    payload: dict[str, Any] = {
        "turn_id": "t1",
        "turn_text": _TURN_TEXT,
        "session_id": "s1",
        "context": "global",
    }
    agent.run_post_turn_hooks(payload)  # returns immediately (async dispatch)

    # async dispatch: the hook recorded a Future, not a blocking result
    assert len(hook.futures) == 1
    future = hook.futures[-1]
    assert isinstance(future, Future)

    # bounded wait on the background run, then the entry is committed
    result = future.result(timeout=10)
    assert result.turn_id == "t1"
    assert [draft.text for draft in result.drafts] == [_DRAFT_TEXT]
    assert [action.action for action in result.actions] == ["add"]
    assert len(result.applied) == 1
    entry = store.get(result.applied[0].entry_id)
    assert entry is not None
    assert entry.text == _DRAFT_TEXT
    assert entry.verification_events[0].evidence.startswith("extractor draft ")


@pytest.mark.integration
def test_pipeline_failure_does_not_crash_session(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing extractor: logged (plan 5.5), hook completes, next hooks run."""

    def _exploding_model(_prompt: str) -> str:
        raise RuntimeError("extractor model exploded")

    store = _make_store(tmp_path)
    pipeline = _add_pipeline(store, extractor_model=_exploding_model)
    agent = _fake_agent()
    hook = install_memory_pipeline(agent, pipeline)

    subsequent: list[dict[str, Any]] = []
    agent.register_post_turn_hook(lambda payload: subsequent.append(payload))

    payload: dict[str, Any] = {
        "turn_id": "t1",
        "turn_text": _TURN_TEXT,
        "session_id": "s1",
        "context": "global",
    }
    with caplog.at_level(logging.WARNING):
        agent.run_post_turn_hooks(payload)  # must not raise
        result = hook.futures[-1].result(timeout=10)

    # the pipeline degraded gracefully (extraction skipped for this turn)
    assert result.turn_id == "t1"
    assert result.drafts == []
    assert result.applied == []
    # ...and the failure was logged, not swallowed silently
    assert "extraction failed" in caplog.text
    assert "extractor model exploded" in caplog.text

    # the agent loop continued: the hook registered after the stop hook ran
    assert subsequent == [payload]

    # the hook's own never-raise contract: even a synchronous dispatch
    # failure is contained (logged with the hook's exception log) and a
    # minimal payload missing every documented key is tolerated
    def _exploding_dispatch(*args: object, **kwargs: object) -> object:
        raise RuntimeError("dispatch exploded")

    monkeypatch.setattr(pipeline, "run_after_turn", _exploding_dispatch)
    with caplog.at_level(logging.ERROR, logger=_HOOK_LOGGER):
        hook({})  # direct call: never raises, even with no payload keys
    assert "stop hook: memory pipeline dispatch failed" in caplog.text
    assert "dispatch exploded" in caplog.text
