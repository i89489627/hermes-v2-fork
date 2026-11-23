"""Integration: hot memory injected before tool calls (plan P21.T2, #26568).

Issue #26568 reproduction, injection half: entries added to the memory store
DURING the current session ("hot memory") must reach the model before each
tool call. The flow exercised here is the real wiring: collect the session's
entries from the store (:func:`collect_hot_entries`), build the
additionalContext payload (:meth:`ContextInjector.build_dynamic_context` —
hot entries bypass retrieval, join the one batched CBDC gate call with top
priority, land under ``dynamic.hot_memory``), format it and hand it to a
guarded tool invocation — the tool then acts on facts it could not have
retrieved mid-session before the fix.

Also covers the tool half in an injection context: a completely disabled
toolset is invisible to the registry and its invoker is refused.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from hermes.core.tool_registry import ToolDef, ToolNotAvailableError, ToolRegistry
from hermes.memory.gating import CBDCGate
from hermes.memory.injector import ContextInjector, collect_hot_entries, format_additional_context
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore


class _UseEverythingGateModel:
    """Fake gate model: ``use`` for every candidate; records the last prompt."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        found = [
            candidate_id
            for candidate_id in re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if candidate_id != "<id>"  # the gate prompt embeds its own example
        ]
        return json.dumps([
            {"candidate_id": candidate_id, "decision": "use", "reason": "relevant"}
            for candidate_id in found
        ])


def _injector(store: MemoryStore) -> tuple[ContextInjector, _UseEverythingGateModel]:
    """Injector over ``store`` with a scripted all-``use`` gate model."""
    gate_model = _UseEverythingGateModel()
    injector = ContextInjector(
        retrieval=RetrievalEngine(store), gate=CBDCGate(gate_model=gate_model)
    )
    return injector, gate_model


@pytest.mark.integration
def test_hot_memory_injected_before_tool_call() -> None:
    """Session-added entries reach the guarded tool call (issue #26568)."""
    store = MemoryStore()  # default root derives from hermes.config.hermes_home()

    # pre-session corpus: two entries matching the turn's tokens (retrievable)
    # plus a decoy sharing none, so BM25 idf stays positive
    checklist_id = store.add(
        "staging deploy checklist lives in the ops wiki", type="Reference"
    ).id
    approval_id = store.add(
        "deploy to staging requires approval from ops", type="Reference"
    ).id
    store.add("grandma apple pie recipe cinnamon nutmeg", type="Reference")

    # the session starts NOW; everything added after is hot
    session_started_at = datetime.now(UTC)
    hot_token = store.add(
        "staging deploy token rotated to glpat-hot-abc123", type="Project"
    )
    hot_host = store.add("staging target host changed to staging-02", type="User")

    hot = collect_hot_entries(store, session_started_at)
    assert [entry.id for entry in hot] == [hot_token.id, hot_host.id]  # pre-session excluded

    injector, gate_model = _injector(store)
    turn = "deploy the service to staging"
    payload = injector.build_dynamic_context(turn, "global", hot_entries=hot)

    dynamic = payload["dynamic"]
    # both hot entries injected (newest first), through the ONE batched gate call
    assert gate_model.calls == 1
    assert [item["candidate_id"] for item in dynamic["hot_memory"]] == [
        hot_host.id,
        hot_token.id,
    ]
    assert dynamic["hot_memory"][0]["needs_verification"] is False
    # retrieved facts fill the remaining gate slot; the hot entries never
    # entered the gate twice (deduplicated out of the retrieval results)
    assert gate_model.last_prompt.count(hot_token.id) == 1
    assert len(dynamic["claims"]) == 1
    assert dynamic["claims"][0]["candidate_id"] in {checklist_id, approval_id}
    # shared budget accounting: 2 hot + 1 retrieved use decisions
    assert dynamic["budget_used"]["claims"] == 3
    assert dynamic["fallback_used"] is False
    # the decoy (no token overlap, pre-session) is nowhere in the payload
    rendered = format_additional_context(payload)
    assert "apple pie" not in rendered

    # the fix contract: this payload is what accompanies the NEXT tool call
    tool_saw: list[str] = []

    def deploy_service(*args: Any, **kwargs: Any) -> str:
        tool_saw.append(str(kwargs.get("injected_context", "")))
        return "deployed"

    registry = ToolRegistry(
        toolset_map={"deploy": [ToolDef("deploy_service", "deploy", deploy_service, "deploy")]}
    )
    guarded = registry.wrap_invoker(
        lambda name, arguments=None: registry.get(name).callable(injected_context=arguments)
    )
    guarded("deploy_service", rendered)

    assert len(tool_saw) == 1
    assert "## Hot memory (this session)" in tool_saw[0]
    assert "glpat-hot-abc123" in tool_saw[0]  # hot fact visible to the tool call
    assert "staging-02" in tool_saw[0]


@pytest.mark.integration
def test_disabled_toolset_completely_ignored() -> None:
    """A disabled toolset is invisible and its invoker is refused (#26568)."""
    invocations: list[str] = []

    def web_search(*args: Any, **kwargs: Any) -> str:
        invocations.append("web_search")
        return "search results"

    def memory_search(*args: Any, **kwargs: Any) -> str:
        invocations.append("memory_search")
        return "memory hits"

    registry = ToolRegistry(
        config={"disabled_toolsets": ["browser"]},
        toolset_map={
            "browser": [ToolDef("web_search", "browser", web_search, "web search")],
            "memory": [ToolDef("memory_search", "memory", memory_search, "memory search")],
        },
    )
    guarded = registry.wrap_invoker(lambda name, arguments=None: registry.get(name).callable())

    # completely ignored: not registered, not listed, not available
    assert registry.get("web_search") is None
    assert registry.is_available("web_search") is False
    assert "web_search" not in [tool.name for tool in registry.list_tools()]

    # the model requesting it is refused without executing the tool
    with pytest.raises(ToolNotAvailableError):
        guarded("web_search", {"query": "deploy docs"})
    assert invocations == []

    # enabled toolsets keep working through the same guarded path
    assert guarded("memory_search", {"query": "deploy docs"}) == "memory hits"
    assert invocations == ["memory_search"]
