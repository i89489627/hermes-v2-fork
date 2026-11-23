"""Auto-deprecator for unused skills (plan P13.T3).

Phase 13 keeps the agent's skill library under user governance: skills that
sit unused for 30+ days are *proposed* for deprecation, and the user approves
the retirement through the ``/skills`` command (plan P13.T4). This module
owns the proposal side only. The lifecycle state machine
(:class:`~hermes.skills.lifecycle.SkillLifecycle` and its ``_TRANSITIONS``
table) stays untouched: no state transition happens here — a proposal is a
record in ``pending_deprecations.json``, and only :meth:`AutoDeprecator.confirm`
(user-approved) calls ``SkillLifecycle.deprecate``.

Mechanics
---------
- Usage source: a JSON usage ledger — ``{skill_id: last-used ISO timestamp}``
  persisted at ``<hermes_home>/memory-repo/skills/skills_usage.json``
  (:class:`UsageLedger`, ``record_usage`` stamps it after each use). Callers
  and tests may inject any ``last_used`` provider callable instead.
- ``scan()`` walks the lifecycle's **active** skills only, computes how long
  each has been idle and returns the candidates whose idle time reached
  ``threshold_days`` (>= semantics — a skill idle exactly 30 days is
  proposed). A skill with no ledger entry is treated as unused since birth:
  the baseline falls back to the skill's ``created_at``. The scan also writes
  ``pending_deprecations.json`` next to the registry as the projection of
  this run: ``{"version": 1, "proposals": [{"skill_id", "last_used_at",
  "proposed_at"}]}`` (``last_used_at`` is ``null`` for never-used skills). A
  skill that gets used again simply drops out of the next scan's file — the
  file is a projection of the latest scan, not an accumulating log.
- ``confirm(skill_id)`` performs the real retirement via
  ``SkillLifecycle.deprecate`` (active -> deprecated, with an
  auto-deprecation reason) and removes the entry from the pending file. Only
  skills with a pending proposal can be confirmed, and only an ``active``
  skill can take the ``deprecate`` transition (enforced by the lifecycle
  itself).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from hermes.config import hermes_home
from hermes.skills.lifecycle import Skill, SkillLifecycle, SkillState

__all__ = [
    "AutoDeprecator",
    "UsageLedger",
    "default_pending_deprecations_path",
    "default_usage_ledger_path",
]

#: Format version stamped on the pending-deprecations.json envelope.
_PENDING_FORMAT_VERSION: Final[int] = 1


def default_usage_ledger_path() -> Path:
    """Usage-ledger location under ``hermes_home()`` (plan P13.T3)."""
    return hermes_home() / "memory-repo" / "skills" / "skills_usage.json"


def default_pending_deprecations_path() -> Path:
    """Pending-deprecation proposal file location (plan P13.T3)."""
    return hermes_home() / "memory-repo" / "skills" / "pending_deprecations.json"


def _normalize_utc(value: datetime) -> datetime:
    """Interpret naive datetimes as UTC; pass aware ones through unchanged."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _parse_timestamp(raw: str, where: str) -> datetime:
    """Parse an ISO-8601 timestamp from persisted JSON; naive counts as UTC."""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{where}: not an ISO-8601 timestamp: {raw!r}") from exc
    return _normalize_utc(parsed)


class UsageLedger:
    """JSON usage ledger: skill id -> last-used timestamp (plan P13.T3).

    The file is read on every access (no in-memory cache) so concurrent
    writers — the pipeline stamps usage after each turn — are always seen.
    Writes are atomic (tmp file + ``os.replace``), mirroring the other
    ``memory-repo`` stores.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = path if path is not None else default_usage_ledger_path()

    def record_usage(self, skill_id: str, now: datetime | None = None) -> datetime:
        """Stamp ``skill_id`` as used at ``now`` (default: current UTC time).

        Returns the recorded (UTC-normalized) timestamp. Naive ``now`` values
        are interpreted as UTC.
        """
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        if now is None:
            stamp = datetime.now(UTC)
        elif isinstance(now, datetime):
            stamp = _normalize_utc(now)
        else:
            raise TypeError(f"now must be a datetime or None, got {type(now).__name__}")
        doc = self._read()
        doc[skill_id] = stamp.isoformat()
        self._write(doc)
        return stamp

    def last_used(self, skill_id: str) -> datetime | None:
        """Last recorded usage of ``skill_id``; ``None`` when never recorded."""
        raw = self._read().get(skill_id)
        if raw is None:
            return None
        return _parse_timestamp(raw, f"usage ledger entry {skill_id!r}")

    # ── internals ─────────────────────────────────────────────────────────

    def _read(self) -> dict[str, str]:
        """Load the ledger (missing file -> empty mapping); validate its shape."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8") as fh:
                doc: Any = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt usage ledger: {self.path}") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"usage ledger must be a JSON object: {self.path}")
        for key, value in doc.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"usage ledger must map skill ids to ISO timestamp strings: {self.path}"
                )
        return {str(key): str(value) for key, value in doc.items()}

    def _write(self, doc: dict[str, str]) -> None:
        """Persist the ledger atomically (tmp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


class AutoDeprecator:
    """Proposes active skills idle >= ``threshold_days`` for deprecation.

    The deprecation proposal itself is file-level only (``pending_deprecations.json``);
    the lifecycle state machine is untouched until the user confirms, at which
    point :meth:`confirm` drives ``SkillLifecycle.deprecate`` — the one
    lifecycle mutator this module ever calls.
    """

    def __init__(
        self,
        lifecycle: SkillLifecycle,
        last_used: Callable[[str], datetime | None] | None = None,
        now: Callable[[], datetime] | None = None,
        threshold_days: int = 30,
        pending_path: Path | None = None,
    ) -> None:
        if not isinstance(lifecycle, SkillLifecycle):
            raise TypeError("lifecycle must be a SkillLifecycle")
        if isinstance(threshold_days, bool) or not isinstance(threshold_days, int):
            raise TypeError(
                f"threshold_days must be an int, got {type(threshold_days).__name__}"
            )
        if threshold_days < 1:
            raise ValueError(f"threshold_days must be >= 1, got {threshold_days}")
        if last_used is not None and not callable(last_used):
            raise TypeError("last_used must be a callable (skill_id) -> datetime | None")
        if now is not None and not callable(now):
            raise TypeError("now must be a callable returning a datetime")
        self.lifecycle = lifecycle
        if last_used is None:
            ledger = UsageLedger()
            self.last_used: Callable[[str], datetime | None] = ledger.last_used
        else:
            self.last_used = last_used
        if now is None:
            self.now: Callable[[], datetime] = lambda: datetime.now(UTC)
        else:
            self.now = now
        self.threshold_days: int = threshold_days
        self.pending_path: Path = (
            pending_path if pending_path is not None else default_pending_deprecations_path()
        )

    # ── scanning ──────────────────────────────────────────────────────────

    def scan(self) -> list[Skill]:
        """Return the active skills unused >= ``threshold_days``; record proposals.

        Only ``active`` skills are considered (proposed/approved skills have
        never had a chance to be used; deprecated/archived are already
        retired). Each candidate contributes a ``pending_deprecations.json``
        proposal entry; the file is rewritten as the projection of this scan.
        No lifecycle transition is performed.
        """
        now = _normalize_utc(self.now())
        candidates: list[Skill] = []
        proposals: list[dict[str, str | None]] = []
        for skill in self.lifecycle.list_skills():
            if skill.state is not SkillState.ACTIVE:
                continue
            last_used = self.last_used(skill.skill_id)
            baseline = (
                last_used
                if last_used is not None
                else _parse_timestamp(skill.created_at, f"skill {skill.skill_id!r} created_at")
            )
            unused_days = (now - _normalize_utc(baseline)).total_seconds() / 86400.0
            if unused_days >= self.threshold_days:
                candidates.append(skill)
                proposals.append(
                    {
                        "skill_id": skill.skill_id,
                        "last_used_at": None if last_used is None else last_used.isoformat(),
                        "proposed_at": now.isoformat(),
                    }
                )
        candidates.sort(key=lambda skill: skill.skill_id)
        proposals.sort(key=lambda proposal: str(proposal["skill_id"]))
        self._write_pending(proposals)
        return candidates

    def pending(self) -> list[dict[str, Any]]:
        """The pending deprecation proposals (projection of the last scan).

        Missing file -> no proposals. Each entry carries ``skill_id``,
        ``last_used_at`` (ISO string or ``None``) and ``proposed_at``.
        """
        if not self.pending_path.exists():
            return []
        try:
            with open(self.pending_path, encoding="utf-8") as fh:
                doc: Any = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt pending-deprecations file: {self.pending_path}") from exc
        if not isinstance(doc, dict):
            raise ValueError(
                f"pending-deprecations file must be a JSON object: {self.pending_path}"
            )
        proposals = doc.get("proposals", [])
        if not isinstance(proposals, list):
            raise ValueError(
                f"pending-deprecations 'proposals' must be a list: {self.pending_path}"
            )
        return [dict(entry) for entry in proposals if isinstance(entry, dict)]

    # ── user-confirmed retirement ─────────────────────────────────────────

    def confirm(self, skill_id: str) -> Skill:
        """User-approved retirement: ``SkillLifecycle.deprecate`` + clear proposal.

        Requires a pending proposal for ``skill_id`` (``ValueError`` otherwise)
        and an ``active`` lifecycle record (the lifecycle raises its own
        ``InvalidTransitionError`` otherwise). Removes the entry from the
        pending file on success.
        """
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        proposals = self.pending()
        entry = next(
            (proposal for proposal in proposals if proposal.get("skill_id") == skill_id),
            None,
        )
        if entry is None:
            raise ValueError(f"no pending deprecation proposal for skill {skill_id!r}")
        last_used_at = entry.get("last_used_at")
        reason = (
            f"auto-deprecation: unused for >= {self.threshold_days} days (last used {last_used_at})"
            if last_used_at is not None
            else f"auto-deprecation: unused for >= {self.threshold_days} days (never used)"
        )
        skill = self.lifecycle.deprecate(skill_id, reason)
        remaining = [
            proposal for proposal in proposals if proposal.get("skill_id") != skill_id
        ]
        self._write_pending(remaining)
        return skill

    # ── internals ─────────────────────────────────────────────────────────

    def _write_pending(self, proposals: list[dict[str, str | None]]) -> None:
        """Persist the pending file atomically (tmp file + ``os.replace``)."""
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"version": _PENDING_FORMAT_VERSION, "proposals": proposals}
        tmp = self.pending_path.with_name(self.pending_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.pending_path)
