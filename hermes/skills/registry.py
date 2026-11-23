"""SkillRegistry version store (plan P13.T2).

Complements :class:`~hermes.skills.lifecycle.SkillLifecycle` (plan P13.T1):
the lifecycle remains the STATE MACHINE (proposed/approved/active/... with
its transition table), while the registry is the VERSION STORE that owns the
versioning metadata of every skill — full history retention, latest-version
queries, and rollback.

Rollback semantics (deliberately non-destructive)
-------------------------------------------------
``rollback(skill_id, version)`` re-activates the content of a prior version
as a NEW latest version: the target version's YAML is appended as a fresh
entry with ``origin="rollback"`` (and a rationale noting the rollback), and
``get_latest_version`` returns that new entry from then on. History is never
deleted or rewritten — every prior version stays addressable via
``get_version`` — so a rollback is always itself reversible.

Persistence
-----------
Same atomic JSON file the lifecycle uses
(``<hermes_home>/memory-repo/skills/_skill_registry.json`` by default,
overridable via ``store_path``). The on-disk shape is a shared-compatible
superset of the lifecycle's: the envelope (``{"version": 1, "skills":
{...}}``) and every lifecycle-required field (``skill_id``, ``name``,
``version``, ``state``, ``yaml``, ``rationale``, ``created_at``,
``updated_at``, ``versions[].{version,yaml,changed_at}``) are preserved so
:class:`SkillLifecycle` can load a registry-written file unchanged; the
registry additionally stores per-version ``rationale``/``origin`` and an
``active_version`` pointer, which the lifecycle's loader simply ignores.
Version entries created by the lifecycle itself are read back with
``rationale=""``/``origin="register"`` defaults, keeping the registry
self-consistent with lifecycle-written stores.

The registry deliberately does NOT touch the lifecycle ``state`` field: state
transitions belong to the state machine, versioning belongs here. Records
created by the registry carry ``state="proposed"`` (the neutral lifecycle
entry state) so a mixed file stays valid for both readers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes.config import hermes_home

__all__ = ["SkillRegistry", "SkillRegistryRecord", "SkillRegistryVersion"]

#: Neutral lifecycle state stamped on registry-created records (the state
#: machine owns every real transition; see module docstring).
_NEUTRAL_STATE = "proposed"


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _default_store_path() -> Path:
    """Default registry location (the lifecycle's file, plan P13.T1/T2)."""
    return hermes_home() / "memory-repo" / "skills" / "_skill_registry.json"


@dataclass
class SkillRegistryVersion:
    """One immutable version entry of a skill (plan P13.T2).

    ``origin`` records how the entry came into being: ``register`` (initial
    ``register()``), ``add_version`` (explicit content bump) or ``rollback``
    (content restored from a prior version as a new latest version).
    """

    version: int
    yaml: str
    changed_at: str
    rationale: str = ""
    origin: str = "register"


@dataclass
class SkillRegistryRecord:
    """Version-store record of one skill (plan P13.T2).

    ``version`` / ``yaml`` mirror the lifecycle's "current" fields and always
    equal the latest entry of ``versions``; ``active_version`` names the
    version whose content is live (equal to ``version`` — the registry never
    forks the pointer from the history it owns).
    """

    skill_id: str
    name: str
    version: int
    yaml: str
    rationale: str
    created_at: str
    updated_at: str
    state: str = _NEUTRAL_STATE
    versions: list[SkillRegistryVersion] = field(default_factory=list)

    @property
    def active_version(self) -> int:
        """Version number whose content is live (the latest entry)."""
        return self.version


class SkillRegistry:
    """Version store for skills: full history + rollback, atomic JSON persistence.

    Queries (``list_skills`` / ``list_versions`` / ``get_version`` /
    ``get_latest_version``) never mutate; every mutation persists atomically
    (tmp file + ``os.replace``). Not thread-safe, mirroring
    :class:`~hermes.skills.lifecycle.SkillLifecycle`.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.path: Path = store_path if store_path is not None else _default_store_path()
        self._skills: dict[str, SkillRegistryRecord] = {}
        self._load()

    # -- queries -----------------------------------------------------------

    def list_skills(self) -> list[str]:
        """All skill ids in registration (insertion) order."""
        return list(self._skills)

    def list_versions(self, skill_id: str) -> list[SkillRegistryVersion]:
        """Version history of ``skill_id``, ascending by version number.

        Raises ``KeyError`` for an unknown skill id.
        """
        return list(self._record(skill_id).versions)

    def get_version(self, skill_id: str, version: int) -> SkillRegistryVersion:
        """One specific version entry; ``KeyError`` when skill or version is unknown."""
        record = self._record(skill_id)
        for entry in record.versions:
            if entry.version == version:
                return entry
        raise KeyError(
            f"skill {skill_id!r} has no version {version} "
            f"(known: {[entry.version for entry in record.versions]})"
        )

    def get_latest_version(self, skill_id: str) -> SkillRegistryVersion:
        """The newest version entry of ``skill_id``; ``KeyError`` when unknown."""
        record = self._record(skill_id)
        return record.versions[-1]

    # -- mutations -----------------------------------------------------------

    def register(
        self,
        skill_id: str,
        name: str,
        yaml: str,
        rationale: str = "",
    ) -> SkillRegistryVersion:
        """Add a new skill with version 1; returns the created version entry.

        A duplicate ``skill_id`` raises ``ValueError`` (an existing skill is
        never silently replaced); empty ids raise ``ValueError`` too.
        """
        if not skill_id:
            raise ValueError("skill_id must be a non-empty string")
        if not name:
            raise ValueError("name must be a non-empty string")
        if skill_id in self._skills:
            raise ValueError(f"skill {skill_id!r} already exists in the registry")
        now = _now()
        entry = SkillRegistryVersion(
            version=1, yaml=yaml, changed_at=now, rationale=rationale, origin="register"
        )
        record = SkillRegistryRecord(
            skill_id=skill_id,
            name=name,
            version=1,
            yaml=yaml,
            rationale=rationale,
            created_at=now,
            updated_at=now,
            versions=[entry],
        )
        self._skills[skill_id] = record
        self._persist()
        return entry

    def add_version(
        self,
        skill_id: str,
        yaml: str,
        rationale: str = "",
    ) -> SkillRegistryVersion:
        """Append a new version (latest + 1); every prior version is retained."""
        record = self._record(skill_id)
        entry = SkillRegistryVersion(
            version=record.version + 1,
            yaml=yaml,
            changed_at=_now(),
            rationale=rationale,
            origin="add_version",
        )
        record.versions.append(entry)
        record.version = entry.version
        record.yaml = yaml
        record.updated_at = entry.changed_at
        self._persist()
        return entry

    def rollback(self, skill_id: str, version: int) -> SkillRegistryVersion:
        """Re-activate a prior version's content as a NEW latest version.

        The target version's YAML is appended as ``latest + 1`` with
        ``origin="rollback"``; nothing is deleted, so the full history
        (including the superseded newer versions) remains addressable via
        :meth:`get_version`. Rolling back to the current latest version is a
        no-op that returns that entry. Unknown skill or version raises
        ``KeyError``.
        """
        record = self._record(skill_id)
        target = self.get_version(skill_id, version)
        if target is record.versions[-1]:
            return target  # already the live version; nothing to append
        suffix = f": {target.rationale}" if target.rationale else ""
        entry = SkillRegistryVersion(
            version=record.version + 1,
            yaml=target.yaml,
            changed_at=_now(),
            rationale=f"rollback to version {version}{suffix}",
            origin="rollback",
        )
        record.versions.append(entry)
        record.version = entry.version
        record.yaml = entry.yaml
        record.updated_at = entry.changed_at
        self._persist()
        return entry

    # -- internals -----------------------------------------------------------

    def _record(self, skill_id: str) -> SkillRegistryRecord:
        """Look up a skill record; ``KeyError`` when unknown."""
        record = self._skills.get(skill_id)
        if record is None:
            raise KeyError(f"unknown skill_id {skill_id!r}")
        return record

    def _load(self) -> None:
        """Load the store when the file exists (lifecycle-compatible shape)."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            doc: dict[str, Any] = json.load(fh)
        skills_raw: dict[str, Any] = doc.get("skills", {})
        for skill_id, raw in skills_raw.items():
            versions = [
                SkillRegistryVersion(
                    version=int(entry["version"]),
                    yaml=str(entry["yaml"]),
                    changed_at=str(entry["changed_at"]),
                    rationale=str(entry.get("rationale", "")),
                    origin=str(entry.get("origin", "register")),
                )
                for entry in raw.get("versions", [])
            ]
            versions.sort(key=lambda entry: entry.version)
            self._skills[str(skill_id)] = SkillRegistryRecord(
                skill_id=str(raw["skill_id"]),
                name=str(raw["name"]),
                version=int(raw["version"]),
                yaml=str(raw["yaml"]),
                rationale=str(raw["rationale"]),
                created_at=str(raw["created_at"]),
                updated_at=str(raw["updated_at"]),
                state=str(raw.get("state", _NEUTRAL_STATE)),
                versions=versions,
            )

    def _persist(self) -> None:
        """Write the whole store atomically (tmp file + ``os.replace``).

        The emitted per-skill shape is a superset of the lifecycle's loader
        contract, so the same file stays readable by ``SkillLifecycle``.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "skills": {
                record.skill_id: {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "version": record.version,
                    "state": record.state,
                    "yaml": record.yaml,
                    "rationale": record.rationale,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "versions": [
                        {
                            "version": entry.version,
                            "yaml": entry.yaml,
                            "changed_at": entry.changed_at,
                            "rationale": entry.rationale,
                            "origin": entry.origin,
                        }
                        for entry in record.versions
                    ],
                }
                for record in self._skills.values()
            },
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
