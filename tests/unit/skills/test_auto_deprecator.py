"""P13.T3 acceptance: AutoDeprecator (plan Phase 13).

An active skill unused for >= 30 days appears in the scan candidates and in
the ``pending_deprecations.json`` projection file, while the lifecycle state
machine stays untouched (the state only changes on ``confirm``). Recently
used skills (via the usage ledger) are never proposed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.skills.auto_deprecator import AutoDeprecator, UsageLedger
from hermes.skills.lifecycle import SkillLifecycle, SkillState

YAML = "name: greeter\ndescription: Greets the user.\n"
NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)


def _lifecycle(tmp_path: Path) -> SkillLifecycle:
    return SkillLifecycle(
        store_path=tmp_path / "memory-repo" / "skills" / "_skill_registry.json"
    )


@pytest.mark.unit
def test_proposes_after_30_days_unused(tmp_path: Path) -> None:
    """Active skill idle 31 days -> candidate + pending file; state only moves on confirm."""
    lifecycle = _lifecycle(tmp_path)
    skill = lifecycle.propose("Greeter Skill", YAML, "greets users on request")
    lifecycle.approve(skill.skill_id)

    pending_path = tmp_path / "memory-repo" / "skills" / "pending_deprecations.json"
    last_used_at = NOW - timedelta(days=31)
    deprecator = AutoDeprecator(
        lifecycle,
        last_used=lambda _skill_id: last_used_at,
        now=lambda: NOW,
        pending_path=pending_path,
    )

    candidates = deprecator.scan()
    assert [candidate.skill_id for candidate in candidates] == ["greeter-skill"]
    assert candidates[0].state is SkillState.ACTIVE  # proposal only: state untouched

    doc = json.loads(pending_path.read_text(encoding="utf-8"))
    assert doc["version"] == 1
    assert doc["proposals"] == [
        {
            "skill_id": "greeter-skill",
            "last_used_at": last_used_at.isoformat(),
            "proposed_at": NOW.isoformat(),
        }
    ]

    # Default pending path derives from hermes_home (per-test tempdir).
    default_deprecator = AutoDeprecator(lifecycle)
    assert default_deprecator.pending_path == (
        hermes_home() / "memory-repo" / "skills" / "pending_deprecations.json"
    )
    assert default_deprecator.pending() == []  # no file there yet

    # Boundary: exactly threshold_days idle is proposed too (>= semantics).
    edge = AutoDeprecator(
        lifecycle,
        last_used=lambda _skill_id: NOW - timedelta(days=30),
        now=lambda: NOW,
        pending_path=pending_path,
    )
    assert [candidate.skill_id for candidate in edge.scan()] == ["greeter-skill"]

    # The lifecycle state machine was never touched by the scans.
    assert lifecycle.get("greeter-skill").state is SkillState.ACTIVE

    # confirm() performs the real retirement (active -> deprecated) and
    # clears the pending entry.
    deprecated = deprecator.confirm("greeter-skill")
    assert deprecated.state is SkillState.DEPRECATED
    assert deprecated.deprecation_reason is not None
    assert "auto-deprecation" in deprecated.deprecation_reason
    assert ">= 30 days" in deprecated.deprecation_reason  # reason names the threshold
    assert deprecator.pending() == []
    reloaded = json.loads(pending_path.read_text(encoding="utf-8"))
    assert reloaded["proposals"] == []

    # Confirming without a pending proposal is rejected.
    with pytest.raises(ValueError):
        deprecator.confirm("greeter-skill")


@pytest.mark.unit
def test_recently_used_not_proposed(tmp_path: Path) -> None:
    """A skill used yesterday stays out of the candidates and the pending file."""
    lifecycle = _lifecycle(tmp_path)
    skill = lifecycle.propose("Greeter Skill", YAML, "greets users on request")
    lifecycle.approve(skill.skill_id)

    ledger = UsageLedger(path=tmp_path / "memory-repo" / "skills" / "skills_usage.json")
    used_at = ledger.record_usage("greeter-skill", NOW - timedelta(days=1))
    assert ledger.last_used("greeter-skill") == used_at

    pending_path = tmp_path / "memory-repo" / "skills" / "pending_deprecations.json"
    deprecator = AutoDeprecator(
        lifecycle,
        last_used=ledger.last_used,
        now=lambda: NOW,
        pending_path=pending_path,
    )

    assert deprecator.scan() == []
    doc = json.loads(pending_path.read_text(encoding="utf-8"))
    assert doc["proposals"] == []
    assert lifecycle.get("greeter-skill").state is SkillState.ACTIVE

    # A never-used (freshly approved) skill falls back to created_at as the
    # unused-since baseline — created seconds ago, so it is not stale either.
    fresh = lifecycle.propose("Fresh Skill", YAML, "brand new")
    lifecycle.approve(fresh.skill_id)
    assert [candidate.skill_id for candidate in deprecator.scan()] == []
    assert doc["proposals"] == json.loads(pending_path.read_text(encoding="utf-8"))["proposals"] == []
