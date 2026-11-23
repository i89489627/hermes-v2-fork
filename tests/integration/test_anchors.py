"""P2.T5 acceptance: full anchor flow through the agent command hook.

End-to-end: a store with three entries (one tagged with the test module
slug, one cited by the session, one unrelated), the command-execution hook
built by :func:`hermes.memory.anchors.build_command_hook` registered on
:class:`hermes.core.agent.V2Agent` (via the shared hook plumbing), one green
pytest result fired through the agent -> exactly the two related entries
gain a ``test-green`` event, the unrelated entry is untouched, trust scores
update. Also covers the red path and the non-test-command no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hermes.core.agent import V2Agent
from hermes.memory.anchors import build_command_hook
from hermes.memory.store import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


def _fake_agent() -> V2Agent:
    """V2Agent without the fork-base machinery (hooks are instance-local).

    ``V2Agent.__init__`` is a pure passthrough to the untyped fork base;
    only the hook lists are v2 state, so bypassing ``__init__`` keeps the
    unit hermetic (no model/session wiring) while exercising the real hook
    registration + dispatch code.
    """
    agent = V2Agent.__new__(V2Agent)
    agent._pre_step_hooks = []  # type: ignore[attr-defined]
    agent._post_turn_hooks = []  # type: ignore[attr-defined]
    agent._command_execution_hooks = []  # type: ignore[attr-defined]
    return agent


@pytest.mark.integration
def test_full_run_updates_trust_scores(tmp_path: Path) -> None:
    """Green pytest run: exactly the 2 related entries get events; scores update."""
    store = _make_store(tmp_path)
    tagged = store.add(
        "greeter skill spec lives in skills/greeter", type="Project", tags=["test_greeter"]
    )
    cited = store.add("session decided on a BM25 prefilter for injection", type="User")
    unrelated = store.add("user prefers dark theme in the TUI", type="User")
    unrelated_events_before = list(unrelated.verification_events)

    agent = _fake_agent()
    hook = build_command_hook(store, session_cited_entry_ids=[cited.id])
    agent.register_command_execution_hook(hook)

    cmd_result: dict[str, Any] = {
        "command": "pytest tests/unit/memory/test_greeter.py",
        "exit_code": 0,
        "changed_files": [],
    }
    outcomes = agent.on_command_execution(cmd_result)

    assert len(outcomes) == 1
    result = outcomes[0]
    assert result.events_added == 2
    assert sorted(result.related_ids) == sorted([tagged.id, cited.id])

    for entry_id in (tagged.id, cited.id):
        updated = store.get(entry_id)
        assert updated is not None
        assert updated.verification_events[-1].source == "test-green"
        assert updated.verification_events[-1].weight == 1.0
        assert updated.verification_events[-1].value == 1.0
        assert updated.trust_score == 1.0

    still = store.get(unrelated.id)
    assert still is not None
    assert len(still.verification_events) == len(unrelated_events_before)
    assert still.trust_score == 1.0  # unchanged (only its seeded event remains)

    # Non-test commands are a no-op: nothing recorded, hook reports None.
    assert agent.on_command_execution({"command": "ls -la", "exit_code": 0}) == []
    fresh = store.get(unrelated.id)
    assert fresh is not None
    assert len(fresh.verification_events) == len(unrelated_events_before)


@pytest.mark.integration
def test_red_run_downweights_and_retries_green(tmp_path: Path) -> None:
    """Red hook result downweights related entries; a later green run restores."""
    store = _make_store(tmp_path)
    entry = store.add("hermes/memory/store.py holds the CRUD facade", type="Project")
    hook = build_command_hook(store)
    agent = _fake_agent()
    agent.register_command_execution_hook(hook)

    red_outcomes = agent.on_command_execution(
        {
            "command": "python -m pytest tests/unit/memory/test_store.py",
            "exit_code": 1,
            "changed_files": ["hermes/memory/store.py"],
        }
    )
    assert len(red_outcomes) == 1
    assert red_outcomes[0].events_added == 1
    red_state = store.get(entry.id)
    assert red_state is not None
    red_event = red_state.verification_events[-1]
    assert red_event.source == "tool-output"
    assert red_event.value == 0.0
    assert red_event.evidence.startswith("test-red:")
    assert red_state.trust_score == pytest.approx(0.5 / 1.35)

    green_outcomes = agent.on_command_execution(
        {
            "command": "python -m pytest tests/unit/memory/test_store.py",
            "exit_code": 0,
            "changed_files": ["hermes/memory/store.py"],
        }
    )
    assert len(green_outcomes) == 1
    assert green_outcomes[0].events_added == 1
    green_state = store.get(entry.id)
    assert green_state is not None
    assert green_state.verification_events[-1].source == "test-green"
    # Weighted avg over agent-inferred (0.5*1) + red (0.85*0) + green (1.0*1).
    assert green_state.trust_score == pytest.approx(1.5 / 2.35)
