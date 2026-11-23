"""P13.T4 acceptance: /skills command workflow (plan Phase 13).

Drives the full lifecycle through ``hermes.core.agent.handle_skills_command``
plus SkillLifecycle + SkillRegistry: propose (agent-driven, P13.T1) ->
approve via command -> use (usage ledger, P13.T3) -> deprecate via command,
rollback via command asserting the content is restored as a NEW version, and
a list that shows every lifecycle state with version + last_used_at.

The ``agent`` argument of the handler is the reserved harness context and is
unused by the dispatch, so the tests pass ``None``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.core.agent import handle_skills_command
from hermes.skills.auto_deprecator import UsageLedger
from hermes.skills.lifecycle import SkillLifecycle, SkillState
from hermes.skills.registry import SkillRegistry

NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
YAML_V1 = "name: greeter\ndescription: Greets the user.\n"
YAML_V2 = "name: greeter\ndescription: Greets the user warmly.\n"


def _workflow(tmp_path: Path) -> tuple[SkillLifecycle, SkillRegistry]:
    """Lifecycle + registry sharing the per-test _skill_registry.json store."""
    store = tmp_path / "memory-repo" / "skills" / "_skill_registry.json"
    return SkillLifecycle(store_path=store), SkillRegistry(store_path=store)


@pytest.mark.integration
def test_propose_approve_use_deprecate(tmp_path: Path) -> None:
    """propose (agent) -> approve -> use -> deprecate, all visible via /skills."""
    lifecycle, registry = _workflow(tmp_path)
    # The agent proposes autonomously (plan P13.T1); the user governs below.
    skill = lifecycle.propose("Greeter Skill", YAML_V1, "greets users on request")
    assert skill.state is SkillState.PROPOSED

    usage = UsageLedger(
        path=tmp_path / "memory-repo" / "skills" / "skills_usage.json"
    )

    out = handle_skills_command(None, ["list"], lifecycle, registry, last_used=None)
    assert skill.skill_id in out
    assert "proposed" in out
    assert "last_used_at=never" in out

    # approve <id>: proposed -> active.
    out = handle_skills_command(
        None, ["approve", skill.skill_id], lifecycle, registry
    )
    assert "active" in out
    assert lifecycle.get(skill.skill_id).state is SkillState.ACTIVE

    # "use" = record usage in the ledger; list/inspect surface it.
    used_at = usage.record_usage(skill.skill_id, NOW)
    last_used = {skill.skill_id: usage.last_used(skill.skill_id)}
    assert last_used[skill.skill_id] == used_at

    out = handle_skills_command(
        None, ["list"], lifecycle, registry, last_used=last_used
    )
    assert skill.skill_id in out
    assert "active" in out
    assert f"last_used_at={used_at.isoformat()}" in out

    out = handle_skills_command(
        None,
        ["inspect", "Greeter Skill"],  # display name resolves to the id
        lifecycle,
        registry,
        last_used=last_used,
    )
    assert skill.skill_id in out
    assert used_at.isoformat() in out
    assert "active v1" in out

    # deprecate <name> [reason]: active -> deprecated with the recorded reason.
    out = handle_skills_command(
        None,
        ["deprecate", "greeter-skill", "no longer helpful"],
        lifecycle,
        registry,
    )
    assert "deprecated" in out
    deprecated = lifecycle.get(skill.skill_id)
    assert deprecated.state is SkillState.DEPRECATED
    assert deprecated.deprecation_reason == "no longer helpful"

    # The state machine rejected nothing here, but illegal transitions still
    # come back as error text instead of crashing the harness.
    out = handle_skills_command(
        None, ["approve", skill.skill_id], lifecycle, registry
    )
    assert out.startswith("error:")
    out = handle_skills_command(None, ["approve", "ghost"], lifecycle, registry)
    assert out.startswith("error:")


@pytest.mark.integration
def test_rollback_to_previous_version(tmp_path: Path) -> None:
    """rollback <name> <version> restores content as a NEW version, everywhere."""
    lifecycle, registry = _workflow(tmp_path)
    skill = lifecycle.propose("Greeter Skill", YAML_V1, "greets users")
    handle_skills_command(None, ["approve", skill.skill_id], lifecycle, registry)

    # An agent-driven content bump: v2 rewrites the greeting.
    lifecycle.version(skill.skill_id, YAML_V2)
    assert lifecycle.get(skill.skill_id).yaml == YAML_V2

    out = handle_skills_command(
        None, ["rollback", "Greeter Skill", "1"], lifecycle, registry
    )
    assert "rolled back" in out
    assert "version 3" in out

    record = lifecycle.get(skill.skill_id)
    assert record.version == 3
    assert record.yaml == YAML_V1  # content restored through the lifecycle view

    # The registry retains everything: the rollback is itself a new version.
    fresh = SkillRegistry(store_path=registry.path)
    assert fresh.get_latest_version(skill.skill_id).version == 3
    assert fresh.get_latest_version(skill.skill_id).yaml == YAML_V1
    assert [entry.version for entry in fresh.list_versions(skill.skill_id)] == [1, 2, 3]

    # Rolling back to the live version is a no-op, reported as such.
    out = handle_skills_command(
        None, ["rollback", "greeter-skill", "3"], lifecycle, registry
    )
    assert "already at version 3" in out
    assert lifecycle.get(skill.skill_id).version == 3

    # Bad inputs are reported, not raised.
    out = handle_skills_command(
        None, ["rollback", "greeter-skill", "x"], lifecycle, registry
    )
    assert out.startswith("error:")
    out = handle_skills_command(None, ["rollback", "ghost", "1"], lifecycle, registry)
    assert out.startswith("error:")


@pytest.mark.integration
def test_list_shows_all_states(tmp_path: Path) -> None:
    """list shows every lifecycle state with version and last_used_at."""
    lifecycle, registry = _workflow(tmp_path)
    fixtures = {
        SkillState.PROPOSED: lifecycle.propose("Draft Helper", YAML_V1, "drafts replies"),
        SkillState.APPROVED: lifecycle.propose("Approved Skill", YAML_V1, "intermediate"),
        SkillState.ACTIVE: lifecycle.propose("Greeter Skill", YAML_V1, "greets users"),
        SkillState.DEPRECATED: lifecycle.propose("Old Skill", YAML_V1, "superseded"),
        SkillState.ARCHIVED: lifecycle.propose("Ancient Skill", YAML_V1, "long gone"),
    }
    lifecycle.mark_approved(fixtures[SkillState.APPROVED].skill_id)
    lifecycle.approve(fixtures[SkillState.ACTIVE].skill_id)
    lifecycle.approve(fixtures[SkillState.DEPRECATED].skill_id)
    lifecycle.deprecate(fixtures[SkillState.DEPRECATED].skill_id, "superseded")
    lifecycle.approve(fixtures[SkillState.ARCHIVED].skill_id)
    lifecycle.deprecate(fixtures[SkillState.ARCHIVED].skill_id, "long gone")
    lifecycle.archive(fixtures[SkillState.ARCHIVED].skill_id)

    out = handle_skills_command(None, ["list"], lifecycle, registry, last_used=None)
    assert "no skills registered" not in out
    lines = out.splitlines()
    for state, skill in fixtures.items():
        line = next(line for line in lines if skill.skill_id in line)
        assert state.value in line
        assert f"v{skill.version}" in line
        assert "last_used_at=never" in line

    # Empty registry: friendly message instead of an empty table.
    empty_lifecycle, empty_registry = _workflow(tmp_path / "empty")
    out = handle_skills_command(None, ["list"], empty_lifecycle, empty_registry)
    assert "no skills registered" in out
