"""P4.T4 acceptance: UserPromptSubmit pre-step hook (plan Phase 4).

Covered: the pre-step hook built by
:mod:`hermes.memory.hooks.user_prompt_submit_hook` (registered via
:func:`hermes.core.agent.install_prompt_injection`, the install twin of
:func:`hermes.core.agent.install_memory_pipeline`) runs the Phase 4
injection chain over a REAL tmp store + injector with a scripted fake gate
model and returns ``additionalContext`` carrying the gated claims; an empty
store yields NO injection (the hook returns ``{}`` — the prompt is not
polluted); a gate model slower than the hook budget times out into an empty
contribution that is logged and never crashes the turn (a later-registered
hook still runs).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pytest

from hermes.core.agent import V2Agent, install_prompt_injection
from hermes.memory.context import ContextManager
from hermes.memory.gating import CBDCGate
from hermes.memory.hooks.user_prompt_submit_hook import PromptSubmitHook, build_prompt_submit_hook
from hermes.memory.injector import ContextInjector
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore

_HOOK_LOGGER = "hermes.memory.hooks.user_prompt_submit_hook"

_DEPLOY_FACT = "acme deploy pipeline runs via GitHub Actions"
_TURN_TEXT = "how do I deploy the acme service today?"
#: Filler corpus so the target terms keep positive BM25 idf (see
#: tests/unit/memory/test_injector.py — a term hitting the only document of
#: a tiny corpus scores 0 and is dropped by the P4.T1 engine).
_FILLER_FACTS = (
    "quarterly budget review spreadsheet numbers",
    "gardening log tomatoes basil balcony",
    "weekly team sync offsite notes",
    "grandma apple pie recipe cinnamon nutmeg",
)


class _ScriptedGateModel:
    """Fake gate model: scripted decision per candidate id found in the prompt."""

    def __init__(self, decisions: dict[str, str], delay_seconds: float = 0.0) -> None:
        self._decisions = decisions
        self._delay_seconds = delay_seconds
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        # the gate prompt embeds its own instruction example ("<id>") — skip it
        found = [
            candidate_id
            for candidate_id in re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if candidate_id != "<id>"
        ]
        return json.dumps([
            {
                "candidate_id": candidate_id,
                "decision": self._decisions.get(candidate_id, "ignore"),
                "reason": "scripted decision",
            }
            for candidate_id in found
        ])


def _seed_store(tmp_path: Path, *facts: str) -> tuple[MemoryStore, ContextManager]:
    """Tmp-rooted store seeded with ``facts`` plus the idf filler corpus."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    for fact in facts:
        store.add(fact, type="Project")
    for filler in _FILLER_FACTS:
        store.add(filler, type="Reference")
    return store, ContextManager(repo_root=tmp_path / "memory-repo")


def _make_injector(
    store: MemoryStore,
    manager: ContextManager,
    decisions: dict[str, str],
    gate_model: _ScriptedGateModel | None = None,
) -> tuple[ContextInjector, _ScriptedGateModel]:
    """Injector over the given store/manager with a scripted fake gate model."""
    model = gate_model if gate_model is not None else _ScriptedGateModel(decisions)
    injector = ContextInjector(
        retrieval=RetrievalEngine(store),
        gate=CBDCGate(gate_model=model),
        context_manager=manager,
    )
    return injector, model


def _fake_agent() -> V2Agent:
    """V2Agent without the fork-base machinery (hooks are instance-local)."""
    agent = V2Agent.__new__(V2Agent)
    agent._pre_step_hooks = []  # type: ignore[attr-defined]
    agent._post_turn_hooks = []  # type: ignore[attr-defined]
    agent._command_execution_hooks = []  # type: ignore[attr-defined]
    return agent


@pytest.mark.integration
def test_pre_step_hook_injects_context(tmp_path: Path) -> None:
    """Relevant facts in the store -> additionalContext carries the claim."""
    store, manager = _seed_store(tmp_path, _DEPLOY_FACT)
    deploy_id = store.list(context="global")[0].id
    injector, gate_model = _make_injector(store, manager, {deploy_id: "use"})
    agent = _fake_agent()

    hook = install_prompt_injection(agent, injector)
    assert isinstance(hook, PromptSubmitHook)
    assert agent._pre_step_hooks == [hook]  # registered on the agent

    payload: dict[str, Any] = {
        "user_prompt": _TURN_TEXT,
        "session_metadata": {"session_id": "s1", "cwd": str(tmp_path)},
        "session_id": "s1",
        "context": "global",
    }
    merged = agent.run_pre_step_hooks(payload)

    assert "additionalContext" in merged
    additional_context = merged["additionalContext"]
    assert _DEPLOY_FACT in additional_context
    assert "[hermes-v2 injected memory context]" in additional_context
    # the gate ran exactly once (single batched call per turn, plan Phase 4)
    assert gate_model.calls == 1
    # the injected entry really is in the store this hook read from
    assert store.get(deploy_id) is not None


@pytest.mark.integration
def test_no_injection_when_no_relevant_facts(tmp_path: Path) -> None:
    """Empty store -> hook returns {} (falsy); the prompt is not polluted."""
    store, manager = _seed_store(tmp_path)  # filler only — nothing relevant
    injector, gate_model = _make_injector(store, manager, {})
    agent = _fake_agent()
    hook = install_prompt_injection(agent, injector)

    payload: dict[str, Any] = {
        "user_prompt": _TURN_TEXT,
        "session_metadata": {"session_id": "s1"},
        "session_id": "s1",
        "context": "global",
    }
    result = hook(payload)

    assert result == {}  # empty additionalContext — no context pollution
    assert not result  # falsy: run_pre_step_hooks merges nothing
    assert agent.run_pre_step_hooks(payload) == {}
    # no candidates -> the gate short-circuits without a model call
    assert gate_model.calls == 0

    # tolerant of a completely missing payload: never raises, still {}
    assert hook({}) == {}


@pytest.mark.integration
def test_hook_timeout_handled_gracefully(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Gate model slower than the budget -> {} fast, warning logged, no crash."""
    store, manager = _seed_store(tmp_path, _DEPLOY_FACT)
    deploy_id = store.list(context="global")[0].id
    # slow gate model: 0.3s >> the 0.05s hook budget below
    slow_model = _ScriptedGateModel({deploy_id: "use"}, delay_seconds=0.3)
    injector, _unused_model = _make_injector(store, manager, {deploy_id: "use"}, slow_model)
    agent = _fake_agent()
    hook = build_prompt_submit_hook(injector, timeout_seconds=0.05)
    agent.register_pre_step_hook(hook)

    subsequent: list[dict[str, Any]] = []
    agent.register_pre_step_hook(lambda payload: subsequent.append(payload))

    payload: dict[str, Any] = {
        "user_prompt": _TURN_TEXT,
        "session_metadata": {"session_id": "s1"},
        "session_id": "s1",
        "context": "global",
    }
    started = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger=_HOOK_LOGGER):
        merged = agent.run_pre_step_hooks(payload)  # must not raise
    elapsed = time.perf_counter() - started

    # returned BEFORE the slow gate finished (0.3s) — the budget fired
    assert elapsed < 0.25, f"hook blocked {elapsed:.3f}s instead of timing out"
    assert merged == {}  # empty additionalContext, no crash
    assert "timed out" in caplog.text
    # the agent loop continued: the hook registered after the injection hook ran
    assert subsequent == [payload]
