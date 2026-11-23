"""P13.T1 acceptance tests: SkillLifecycle state machine (plan Phase 13).

Covers: propose -> proposed (v1 recorded), approve -> active (with the
intermediate ``approved`` state reachable via ``mark_approved``/``activate``),
version bumps retaining history, deprecate -> archive, and invalid
transitions raising ``InvalidTransitionError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.skills.lifecycle import (
    InvalidTransitionError,
    SkillLifecycle,
    SkillState,
)

YAML_V1 = "name: greeter\ndescription: Greets the user.\n"
YAML_V2 = "name: greeter\ndescription: Greets the user warmly.\n"


def _registry(tmp_path: Path) -> SkillLifecycle:
    """Registry at an explicit per-test store path."""
    return SkillLifecycle(
        store_path=tmp_path / "memory-repo" / "skills" / "_skill_registry.json"
    )


@pytest.mark.unit
def test_propose_creates_in_proposed_state(tmp_path: Path) -> None:
    store = tmp_path / "skills.json"
    registry = SkillLifecycle(store_path=store)
    skill = registry.propose("Greeter Skill", YAML_V1, "greets users on request")

    assert skill.skill_id == "greeter-skill"
    assert skill.name == "Greeter Skill"
    assert skill.state is SkillState.PROPOSED
    assert skill.version == 1
    assert skill.rationale == "greets users on request"
    assert [entry.version for entry in skill.versions] == [1]
    assert skill.versions[0].yaml == YAML_V1
    assert skill.created_at
    assert skill.updated_at
    assert registry.get("greeter-skill") is skill

    # Default storage path derives from hermes_home (per-test tempdir).
    default_registry = SkillLifecycle()
    assert default_registry.path == (
        hermes_home() / "memory-repo" / "skills" / "_skill_registry.json"
    )

    # Persisted after the mutation and reloaded on next construction.
    doc = json.loads(store.read_text(encoding="utf-8"))
    assert doc["skills"]["greeter-skill"]["state"] == "proposed"
    assert doc["skills"]["greeter-skill"]["versions"][0]["version"] == 1
    reloaded = SkillLifecycle(store_path=store)
    loaded = reloaded.get("greeter-skill")
    assert loaded is not None
    assert loaded.state is SkillState.PROPOSED
    assert loaded.yaml == YAML_V1
    assert [entry.version for entry in loaded.versions] == [1]


@pytest.mark.unit
def test_approve_moves_to_active(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    skill = registry.propose("Greeter Skill", YAML_V1, "greets users on request")

    approved = registry.approve(skill.skill_id)
    assert approved.state is SkillState.ACTIVE

    # The intermediate state from the five-state diagram stays reachable:
    # mark_approved() then activate() (or approve() from `approved`).
    second = registry.propose("Farewell Skill", YAML_V1, "says goodbye")
    marked = registry.mark_approved(second.skill_id)
    assert marked.state is SkillState.APPROVED
    activated = registry.activate(second.skill_id)
    assert activated.state is SkillState.ACTIVE

    persisted = SkillLifecycle(store_path=registry.path).get(skill.skill_id)
    assert persisted is not None
    assert persisted.state is SkillState.ACTIVE


@pytest.mark.unit
def test_version_creates_new_version(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    proposed = registry.propose("Greeter Skill", YAML_V1, "greets users on request")
    active = registry.approve(proposed.skill_id)

    bumped = registry.version(active.skill_id, YAML_V2)
    assert bumped.version == 2
    assert bumped.yaml == YAML_V2
    assert bumped.state is SkillState.ACTIVE  # active -> active via version()
    assert [entry.version for entry in bumped.versions] == [1, 2]
    assert bumped.versions[0].yaml == YAML_V1  # full history retained
    assert bumped.versions[1].yaml == YAML_V2
    assert bumped.versions[1].changed_at

    # version() also works from the intermediate approved state, preserving it.
    other = registry.propose("Farewell Skill", YAML_V1, "says goodbye")
    registry.mark_approved(other.skill_id)
    bumped_approved = registry.version(other.skill_id, YAML_V2)
    assert bumped_approved.state is SkillState.APPROVED
    assert bumped_approved.version == 2

    persisted = SkillLifecycle(store_path=registry.path).get(proposed.skill_id)
    assert persisted is not None
    assert persisted.version == 2
    assert [entry.version for entry in persisted.versions] == [1, 2]


@pytest.mark.unit
def test_deprecate_moves_to_deprecated(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    proposed = registry.propose("Greeter Skill", YAML_V1, "greets users on request")
    active = registry.approve(proposed.skill_id)

    deprecated = registry.deprecate(active.skill_id, "superseded by greeter v2")
    assert deprecated.state is SkillState.DEPRECATED
    assert deprecated.deprecation_reason == "superseded by greeter v2"

    archived = registry.archive(deprecated.skill_id)
    assert archived.state is SkillState.ARCHIVED

    persisted = SkillLifecycle(store_path=registry.path).get(proposed.skill_id)
    assert persisted is not None
    assert persisted.state is SkillState.ARCHIVED
    assert persisted.deprecation_reason == "superseded by greeter v2"


@pytest.mark.unit
def test_invalid_transition_raises(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    skill = registry.propose("Greeter Skill", YAML_V1, "greets users on request")

    # proposed: only approve()/mark_approved() are legal.
    with pytest.raises(InvalidTransitionError):
        registry.deprecate(skill.skill_id, "not active yet")
    with pytest.raises(InvalidTransitionError):
        registry.archive(skill.skill_id)
    with pytest.raises(InvalidTransitionError):
        registry.version(skill.skill_id, YAML_V2)
    with pytest.raises(InvalidTransitionError):
        registry.activate(skill.skill_id)

    active = registry.approve(skill.skill_id)
    # active: only version()/deprecate() are legal; approve() cannot loop
    # (active -> active happens only via version()).
    with pytest.raises(InvalidTransitionError):
        registry.approve(active.skill_id)
    with pytest.raises(InvalidTransitionError):
        registry.activate(active.skill_id)
    with pytest.raises(InvalidTransitionError):
        registry.mark_approved(active.skill_id)

    deprecated = registry.deprecate(active.skill_id, "superseded")
    with pytest.raises(InvalidTransitionError):
        registry.version(deprecated.skill_id, YAML_V2)

    archived = registry.archive(deprecated.skill_id)
    with pytest.raises(InvalidTransitionError):
        registry.archive(archived.skill_id)
    with pytest.raises(InvalidTransitionError):
        registry.deprecate(archived.skill_id, "already archived")
    with pytest.raises(InvalidTransitionError):
        registry.approve(archived.skill_id)

    # Unknown ids never pretend to transition.
    with pytest.raises(KeyError):
        registry.approve("no-such-skill")
