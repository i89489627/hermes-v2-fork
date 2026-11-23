"""P13.T2 acceptance tests: SkillRegistry version store (plan Phase 13).

Covers: registration (version 1 + persistence), full version history
retention with ascending ``list_versions``, latest-version queries, and
non-destructive rollback to any prior version (the rolled-back content
re-enters as a NEW latest version; nothing is deleted). Also checks the
shared-file compatibility contract with ``SkillLifecycle`` (plan P13.T1):
each reader can load the file the other one wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.skills.lifecycle import SkillLifecycle, SkillState
from hermes.skills.registry import SkillRegistry

YAML_V1 = "name: greeter\ndescription: Greets the user.\n"
YAML_V2 = "name: greeter\ndescription: Greets the user warmly.\n"
YAML_V3 = "name: greeter\ndescription: Greets the user warmly, by name.\n"


def _registry(tmp_path: Path) -> SkillRegistry:
    """Registry at an explicit per-test store path."""
    return SkillRegistry(store_path=tmp_path / "memory-repo" / "skills" / "_skill_registry.json")


@pytest.mark.unit
def test_register_adds_skill(tmp_path: Path) -> None:
    store = tmp_path / "skills.json"
    registry = SkillRegistry(store_path=store)
    assert registry.list_skills() == []  # empty store starts clean

    entry = registry.register("greeter", "Greeter Skill", YAML_V1, "greets on request")
    assert entry.version == 1
    assert entry.yaml == YAML_V1
    assert entry.rationale == "greets on request"
    assert entry.origin == "register"
    assert entry.changed_at

    assert registry.list_skills() == ["greeter"]
    assert registry.get_latest_version("greeter").version == 1

    # Persisted to JSON at the store path and reloaded on next construction.
    doc = json.loads(store.read_text(encoding="utf-8"))
    assert doc["skills"]["greeter"]["version"] == 1
    assert doc["skills"]["greeter"]["versions"][0]["version"] == 1
    reloaded = SkillRegistry(store_path=store)
    assert reloaded.list_skills() == ["greeter"]
    assert reloaded.get_latest_version("greeter").yaml == YAML_V1

    # Duplicates and empty ids are rejected.
    with pytest.raises(ValueError):
        registry.register("greeter", "Greeter Again", YAML_V2)
    with pytest.raises(ValueError):
        registry.register("", "No Id", YAML_V1)

    # Default storage path derives from hermes_home (per-test tempdir) and
    # matches the lifecycle's file (shared-version-store contract).
    default_registry = SkillRegistry()
    assert default_registry.path == (
        hermes_home() / "memory-repo" / "skills" / "_skill_registry.json"
    )

    # A registry record is loadable by the lifecycle (superset shape).
    lifecycle_view = SkillLifecycle(store_path=store)
    skill = lifecycle_view.get("greeter")
    assert skill is not None
    assert skill.state is SkillState.PROPOSED
    assert [v.version for v in skill.versions] == [1]


@pytest.mark.unit
def test_list_returns_all_versions(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.register("greeter", "Greeter Skill", YAML_V1, "initial")

    second = registry.add_version("greeter", YAML_V2, "warmer greeting")
    third = registry.add_version("greeter", YAML_V3, "personalized greeting")
    assert second.version == 2
    assert third.version == 3

    versions = registry.list_versions("greeter")
    assert [entry.version for entry in versions] == [1, 2, 3]  # ascending
    assert [entry.yaml for entry in versions] == [YAML_V1, YAML_V2, YAML_V3]
    assert [entry.rationale for entry in versions] == [
        "initial",
        "warmer greeting",
        "personalized greeting",
    ]
    assert [entry.origin for entry in versions] == [
        "register",
        "add_version",
        "add_version",
    ]
    assert all(entry.changed_at for entry in versions)

    # Unknown skills are a KeyError, not a silent empty list.
    with pytest.raises(KeyError):
        registry.list_versions("no-such-skill")

    # Full history survives a restart via the persisted JSON.
    reloaded = _registry(tmp_path)
    assert [entry.version for entry in reloaded.list_versions("greeter")] == [1, 2, 3]


@pytest.mark.unit
def test_get_latest_version(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.register("greeter", "Greeter Skill", YAML_V1)
    registry.register("farewell", "Farewell Skill", "name: farewell\n")

    latest = registry.get_latest_version("greeter")
    assert latest.version == 1
    assert latest.yaml == YAML_V1

    registry.add_version("greeter", YAML_V2)
    registry.add_version("greeter", YAML_V3)
    latest = registry.get_latest_version("greeter")
    assert latest.version == 3
    assert latest.yaml == YAML_V3

    # The other skill's latest is unaffected (per-skill versioning).
    assert registry.get_latest_version("farewell").version == 1

    # get_version addresses every retained version; unknown ones raise.
    assert registry.get_version("greeter", 1).yaml == YAML_V1
    assert registry.get_version("greeter", 2).yaml == YAML_V2
    with pytest.raises(KeyError):
        registry.get_version("greeter", 99)
    with pytest.raises(KeyError):
        registry.get_latest_version("no-such-skill")


@pytest.mark.unit
def test_rollback_to_previous_version(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.register("greeter", "Greeter Skill", YAML_V1, "initial")
    registry.add_version("greeter", YAML_V2, "warmer")
    registry.add_version("greeter", YAML_V3, "too chatty")

    # Roll back from v3 to v1 (any prior version, not just the previous one).
    rolled = registry.rollback("greeter", 1)
    assert rolled.version == 4  # the restored content becomes a NEW latest version
    assert rolled.yaml == YAML_V1
    assert rolled.origin == "rollback"
    assert rolled.changed_at
    assert registry.get_latest_version("greeter") is rolled

    # Rollback never deletes: the full history (v1..v4) stays addressable.
    assert [entry.version for entry in registry.list_versions("greeter")] == [1, 2, 3, 4]
    assert registry.get_version("greeter", 3).yaml == YAML_V3  # superseded, retained
    assert registry.get_version("greeter", 1).yaml == YAML_V1

    # Rolling back again, this time to the middle version.
    registry.rollback("greeter", 2)
    latest = registry.get_latest_version("greeter")
    assert latest.version == 5
    assert latest.yaml == YAML_V2
    assert [entry.version for entry in registry.list_versions("greeter")] == [1, 2, 3, 4, 5]

    # Rolling back to the already-live version is a no-op (no new entry).
    unchanged = registry.rollback("greeter", 5)
    assert unchanged.version == 5
    assert len(registry.list_versions("greeter")) == 5

    # Unknown skill or version raises KeyError.
    with pytest.raises(KeyError):
        registry.rollback("no-such-skill", 1)
    with pytest.raises(KeyError):
        registry.rollback("greeter", 42)

    # Everything (including the rollback markers) persists across a restart.
    reloaded = _registry(tmp_path)
    assert reloaded.get_latest_version("greeter").yaml == YAML_V2
    assert [entry.version for entry in reloaded.list_versions("greeter")] == [1, 2, 3, 4, 5]
    origins = {entry.version: entry.origin for entry in reloaded.list_versions("greeter")}
    assert origins[4] == "rollback"
    assert origins[5] == "rollback"
