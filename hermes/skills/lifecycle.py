"""SkillLifecycle state machine (plan P13.T1).

Phase 13 lets the agent extend its own skill library under user governance:
the agent proposes a skill (name + SKILL.md-style YAML + rationale), the
user approves it, and the skill joins the usable library. This module owns
the lifecycle state machine plus its minimal persistence; P13.T2 builds the
full skill registry on top of it.

States (plan Phase 13): proposed | approved | active | deprecated | archived.

Transition table (deliberately minimal, single source of truth in
``_TRANSITIONS``; everything not listed raises :class:`InvalidTransitionError`):

- ``propose(name, yaml, rationale)`` creates a new skill in ``proposed``
  state (version 1, with v1 recorded in the version history).
- ``approve(skill_id)`` maps proposed -> active. DEVIATION NOTE: the
  five-state diagram draws proposed -> approved -> active, but the P13.T1
  acceptance criterion is "approve moves the skill to active", so the
  primary approval flow lands directly on ``active``. The intermediate
  ``approved`` state stays reachable via the explicit
  ``mark_approved()`` + ``activate()`` pair (and ``approve()`` also
  accepts a skill already sitting in ``approved``).
- ``mark_approved(skill_id)`` maps proposed -> approved (intermediate
  state, kept for completeness of the five-state diagram).
- ``activate(skill_id)`` maps approved -> active.
- ``version(skill_id, new_yaml)`` maps approved -> approved and active ->
  active: a content bump that preserves the current state, increments the
  version and appends the revision to the retained history. A re-proposal
  transition (approved -> proposed) is deliberately NOT modelled, keeping
  the machine minimal.
- ``deprecate(skill_id, reason)`` maps active -> deprecated and records
  the reason on the skill.
- ``archive(skill_id)`` maps deprecated -> archived (terminal state).

An unknown ``skill_id`` raises ``KeyError``; proposing a duplicate skill
id raises ``ValueError``.

Persistence: the whole registry is rewritten atomically (tmp file +
``os.replace``) after every mutation to ``store_path`` (default
``<hermes_home>/memory-repo/skills/_skill_registry.json``) and reloaded
on construction when the file exists.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from hermes.config import hermes_home

__all__ = [
    "InvalidTransitionError",
    "Skill",
    "SkillLifecycle",
    "SkillState",
    "SkillVersion",
]


class SkillState(StrEnum):
    """Lifecycle states of a governed skill (plan Phase 13)."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class InvalidTransitionError(ValueError):
    """A lifecycle action is not legal for the skill's current state."""


@dataclass
class SkillVersion:
    """One immutable entry of a skill's version history (plan P13.T1)."""

    version: int
    yaml: str
    changed_at: str


@dataclass
class Skill:
    """A governed skill record (plan P13.T1).

    ``versions`` retains every YAML revision: entry ``n`` holds the content
    as of version ``n.version``. State changes must go through
    :class:`SkillLifecycle` so the transition table stays authoritative.
    """

    skill_id: str
    name: str
    version: int
    state: SkillState
    yaml: str
    rationale: str
    created_at: str
    updated_at: str
    deprecation_reason: str | None = None
    versions: list[SkillVersion] = field(default_factory=list)


#: Action -> states the action is legal from. This table IS the state
#: machine: every mutating method validates against it before touching a
#: skill, so no transition can bypass the documented lifecycle.
_TRANSITIONS: dict[str, frozenset[SkillState]] = {
    "approve": frozenset({SkillState.PROPOSED, SkillState.APPROVED}),
    "mark_approved": frozenset({SkillState.PROPOSED}),
    "activate": frozenset({SkillState.APPROVED}),
    "version": frozenset({SkillState.APPROVED, SkillState.ACTIVE}),
    "deprecate": frozenset({SkillState.ACTIVE}),
    "archive": frozenset({SkillState.DEPRECATED}),
}


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _slugify(name: str) -> str:
    """Derive a skill id from a display name (``"Greeter Skill"`` -> ``greeter-skill``)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "skill"


def _default_store_path() -> Path:
    """Default registry location under the Hermes home (plan P13.T1)."""
    return hermes_home() / "memory-repo" / "skills" / "_skill_registry.json"


def _skill_to_dict(skill: Skill) -> dict[str, Any]:
    """Serialize a skill (including its version history) to a JSON-ready dict."""
    doc: dict[str, Any] = asdict(skill)
    doc["state"] = skill.state.value
    return doc


def _skill_from_dict(raw: dict[str, Any]) -> Skill:
    """Rebuild a skill from its persisted form (inverse of ``_skill_to_dict``)."""
    versions = [
        SkillVersion(
            version=int(entry["version"]),
            yaml=str(entry["yaml"]),
            changed_at=str(entry["changed_at"]),
        )
        for entry in raw.get("versions", [])
    ]
    reason = raw.get("deprecation_reason")
    return Skill(
        skill_id=str(raw["skill_id"]),
        name=str(raw["name"]),
        version=int(raw["version"]),
        state=SkillState(raw["state"]),
        yaml=str(raw["yaml"]),
        rationale=str(raw["rationale"]),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        deprecation_reason=None if reason is None else str(reason),
        versions=versions,
    )


class SkillLifecycle:
    """Governed skill lifecycle: state machine + atomic JSON persistence.

    All state changes flow through the ``_TRANSITIONS`` table; queries
    (``get`` / ``list_skills``) never mutate. The default store lives at
    ``<hermes_home>/memory-repo/skills/_skill_registry.json`` (plan P13.T2
    layers the full skill registry on top of this file format).
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.path: Path = store_path if store_path is not None else _default_store_path()
        self._skills: dict[str, Skill] = {}
        self._load()

    # -- queries -----------------------------------------------------------

    def get(self, skill_id: str) -> Skill | None:
        """Return the skill with ``skill_id``, or ``None`` when unknown."""
        return self._skills.get(skill_id)

    def list_skills(self) -> list[Skill]:
        """All registered skills, in proposal (insertion) order."""
        return list(self._skills.values())

    # -- transitions ---------------------------------------------------------

    def propose(self, name: str, yaml: str, rationale: str) -> Skill:
        """Register a new skill in ``proposed`` state (version 1).

        The skill id is the slugified ``name``; a duplicate id raises
        ``ValueError`` so an existing skill can never be silently replaced.
        """
        skill_id = _slugify(name)
        if skill_id in self._skills:
            raise ValueError(f"skill {skill_id!r} already exists in the registry")
        now = _now()
        skill = Skill(
            skill_id=skill_id,
            name=name,
            version=1,
            state=SkillState.PROPOSED,
            yaml=yaml,
            rationale=rationale,
            created_at=now,
            updated_at=now,
            versions=[SkillVersion(version=1, yaml=yaml, changed_at=now)],
        )
        self._skills[skill_id] = skill
        self._persist()
        return skill

    def approve(self, skill_id: str) -> Skill:
        """User approval: proposed -> active (accepted from ``approved`` too)."""
        skill = self._checked(skill_id, "approve")
        skill.state = SkillState.ACTIVE
        return self._touch_and_save(skill)

    def mark_approved(self, skill_id: str) -> Skill:
        """Intermediate five-state-diagram transition: proposed -> approved."""
        skill = self._checked(skill_id, "mark_approved")
        skill.state = SkillState.APPROVED
        return self._touch_and_save(skill)

    def activate(self, skill_id: str) -> Skill:
        """Promote an intermediate-approved skill: approved -> active."""
        skill = self._checked(skill_id, "activate")
        skill.state = SkillState.ACTIVE
        return self._touch_and_save(skill)

    def version(self, skill_id: str, new_yaml: str) -> Skill:
        """Bump the skill content: increments the version, keeps the state
        (approved -> approved, active -> active) and appends the revision
        to the retained version history."""
        skill = self._checked(skill_id, "version")
        skill.version += 1
        skill.yaml = new_yaml
        skill.versions.append(
            SkillVersion(version=skill.version, yaml=new_yaml, changed_at=_now())
        )
        return self._touch_and_save(skill)

    def deprecate(self, skill_id: str, reason: str) -> Skill:
        """Retire an active skill: active -> deprecated, recording ``reason``."""
        skill = self._checked(skill_id, "deprecate")
        skill.state = SkillState.DEPRECATED
        skill.deprecation_reason = reason
        return self._touch_and_save(skill)

    def archive(self, skill_id: str) -> Skill:
        """Terminal transition: deprecated -> archived."""
        skill = self._checked(skill_id, "archive")
        skill.state = SkillState.ARCHIVED
        return self._touch_and_save(skill)

    # -- internals -----------------------------------------------------------

    def _checked(self, skill_id: str, action: str) -> Skill:
        """Look up the skill and validate that ``action`` is legal right now."""
        skill = self._skills.get(skill_id)
        if skill is None:
            raise KeyError(f"unknown skill_id {skill_id!r} (cannot {action})")
        allowed = _TRANSITIONS[action]
        if skill.state not in allowed:
            legal_from = ", ".join(s.value for s in sorted(allowed))
            raise InvalidTransitionError(
                f"cannot {action} skill {skill.skill_id!r} while in state "
                f"{skill.state.value!r} (legal from: {legal_from})"
            )
        return skill

    def _touch_and_save(self, skill: Skill) -> Skill:
        """Stamp ``updated_at`` and persist the registry atomically."""
        skill.updated_at = _now()
        self._persist()
        return skill

    def _load(self) -> None:
        """Load the registry from disk when the store file exists."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            doc: dict[str, Any] = json.load(fh)
        skills_raw: dict[str, Any] = doc.get("skills", {})
        self._skills = {
            str(skill_id): _skill_from_dict(raw)
            for skill_id, raw in skills_raw.items()
        }

    def _persist(self) -> None:
        """Write the whole registry atomically (tmp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "skills": {
                skill_id: _skill_to_dict(skill)
                for skill_id, skill in self._skills.items()
            },
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
