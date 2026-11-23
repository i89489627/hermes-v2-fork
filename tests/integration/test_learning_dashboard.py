"""P13.T5 acceptance: /learning dashboard (plan Phase 13).

Drives the subcommands through ``hermes.core.agent.handle_learning_command``
with real v2 components: SkillLifecycle + SkillRegistry over the shared
per-test registry file, a real TraceStore (plan P11.T1) with appended
traces, and an AutoDeprecator writing its pending-deprecations file. The
history subcommand is exercised against a real ParetoFrontStore (the
documented GEPA-history proxy until P10.T5's runner exists).

The ``agent`` argument of the handler is the reserved harness context and is
unused by the dispatch, so the tests pass ``None``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.core.agent import handle_learning_command
from hermes.optimization.pareto import ParetoFrontStore
from hermes.skills.auto_deprecator import AutoDeprecator
from hermes.skills.lifecycle import SkillLifecycle, SkillState
from hermes.skills.registry import SkillRegistry
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)


def _learning(tmp_path: Path) -> tuple[SkillLifecycle, Path]:
    """Lifecycle over the per-test registry file (registry path returned)."""
    store_path = tmp_path / "memory-repo" / "skills" / "_skill_registry.json"
    return SkillLifecycle(store_path=store_path), store_path


@pytest.mark.integration
def test_stats_shows_counts(tmp_path: Path) -> None:
    """stats: total/active/deprecated skills, pending approvals, traces count."""
    lifecycle, registry_path = _learning(tmp_path)
    proposed = lifecycle.propose("Draft Helper", "name: draft-helper\n", "drafts replies")
    approved_skill = lifecycle.propose("Greeter Skill", "name: greeter\n", "greets users")
    lifecycle.approve(approved_skill.skill_id)
    retired = lifecycle.propose("Old Skill", "name: old-skill\n", "legacy")
    lifecycle.approve(retired.skill_id)
    lifecycle.deprecate(retired.skill_id, "superseded")
    assert lifecycle.get(proposed.skill_id).state is SkillState.PROPOSED

    registry = SkillRegistry(store_path=registry_path)  # built AFTER mutations: fresh cache
    trace_store = TraceStore(root=tmp_path / "traces")
    for index in range(3):
        trace_store.append(
            Trace(
                turn_id=f"t{index}",
                session_id="s1",
                timestamp=_NOW.timestamp() + index,
                user_prompt=f"prompt {index}",
                quality_label="good" if index % 2 == 0 else "acceptable",
            )
        )

    out = handle_learning_command(
        None,
        ["stats"],
        lifecycle=lifecycle,
        registry=registry,
        trace_store=trace_store,
    )
    assert "skills: total=3 active=1 deprecated=1" in out
    assert "proposed=1" in out
    assert "registry: 3 skill record(s)" in out
    assert "pending approvals: 1" in out
    assert "traces: 3" in out

    # The GEPA-history proxy renders real pareto fronts (documented deviation).
    pareto = ParetoFrontStore(path=tmp_path / "pareto-front.json")
    pareto.add_candidate("extraction", "prompt v1", {"quality": 0.7, "cost": 0.9})
    pareto.add_candidate("extraction", "prompt v2", {"quality": 0.9, "cost": 0.8})
    out = handle_learning_command(None, ["history"], pareto=pareto)
    assert "extraction" in out and "front=2" in out


@pytest.mark.integration
def test_traces_recent(tmp_path: Path) -> None:
    """traces: the 5 most recent traces, newest first, with quality labels."""
    trace_store = TraceStore(root=tmp_path / "traces")
    base = _NOW.timestamp()
    appended_ids = []
    for index in range(7):  # 7 traces; the dashboard shows only the 5 newest
        appended_ids.append(
            trace_store.append(
                Trace(
                    turn_id=f"t{index}",
                    session_id="s1",
                    timestamp=base + index * 60,
                    user_prompt=f"number {index} question",
                    quality_label=("good" if index % 2 == 0 else "bad"),
                )
            )
        )

    out = handle_learning_command(None, ["traces"], trace_store=trace_store)
    assert "recent traces (up to 5 newest" in out
    newest_id = appended_ids[-1]
    assert newest_id[:12] in out  # newest trace shown first
    newest_line = out.splitlines()[1]
    assert newest_id[:12] in newest_line  # newest is the FIRST listed row
    assert "good" in out and "bad" in out  # quality labels visible
    # Older traces (outside the 5 newest) are not shown.
    assert appended_ids[0][:12] not in out
    assert appended_ids[1][:12] not in out

    # Empty store renders the placeholder, not an exception.
    empty = TraceStore(root=tmp_path / "traces-empty")
    assert "no traces recorded yet" in handle_learning_command(None, ["traces"], trace_store=empty)


@pytest.mark.integration
def test_pending_approvals(tmp_path: Path) -> None:
    """pending: proposed skills + auto-deprecator proposals, both listed."""
    lifecycle, _registry_path = _learning(tmp_path)
    pending_one = lifecycle.propose("Draft Helper", "name: draft-helper\n", "drafts replies")
    pending_two = lifecycle.propose("Summarizer", "name: summarizer\n", "summarizes threads")
    active_skill = lifecycle.propose("Greeter Skill", "name: greeter\n", "greets users")
    lifecycle.approve(active_skill.skill_id)

    # Auto-deprecation: an active skill unused since long before the fake now.
    deprecator = AutoDeprecator(
        lifecycle,
        last_used=lambda _skill_id: None,
        now=lambda: datetime(2027, 1, 5, tzinfo=UTC),
        threshold_days=30,
        pending_path=tmp_path / "memory-repo" / "skills" / "pending_deprecations.json",
    )
    proposed_for_deprecation = deprecator.scan()
    assert [skill.skill_id for skill in proposed_for_deprecation] == [active_skill.skill_id]

    out = handle_learning_command(
        None, ["pending"], lifecycle=lifecycle, auto_deprecator=deprecator
    )
    assert "pending approvals:" in out
    assert pending_one.skill_id in out and pending_two.skill_id in out
    assert "awaiting /skills approve" in out
    assert active_skill.skill_id in out
    assert "awaiting /skills deprecate" in out
    assert "last_used_at=None" in out  # never used -> None in the proposal

    # Approving and confirming clears the pending views.
    lifecycle.approve(pending_one.skill_id)
    deprecator.confirm(active_skill.skill_id)
    out = handle_learning_command(
        None, ["pending"], lifecycle=lifecycle, auto_deprecator=deprecator
    )
    assert pending_one.skill_id not in out
    assert active_skill.skill_id not in out
    assert pending_two.skill_id in out  # still proposed
