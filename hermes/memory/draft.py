"""DraftFact and ReviewAction records for the P3 review pipeline (plan P3.T1).

The extraction stage proposes candidate memories as :class:`DraftFact`
objects; the reviewer consumes them and emits :class:`ReviewAction`
decisions — ``add`` | ``update`` | ``skip`` | ``conflict``, exactly the
``hermes.memory.types.ReviewAction`` literal set. Both records round-trip
through JSON-safe dicts so they can cross the pipeline boundary (plan P3.T2
consumes them; P3.T6 extends ``DraftFact`` with multi-turn provenance via
``source_turn_ids``, reserved here).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast, get_args

from hermes.memory.types import (
    ENTRY_TYPES,
    SOURCE_WEIGHTS,
    EntryType,
    VerificationSource,
    clamp01,
    to_utc,
)
from hermes.memory.types import (
    ReviewAction as ReviewActionLiteral,
)

#: Runtime set of valid reviewer actions, derived from the literal.
REVIEW_ACTIONS: frozenset[str] = frozenset(get_args(ReviewActionLiteral))

#: Word-token pattern for :func:`normalize_fact_text` (unicode-aware, no _).
_FACT_TOKEN_RE = re.compile(r"[^\W_]+")


def normalize_fact_text(text: str) -> str:
    """Normalize a fact text for exact-match duplicate suppression (P3.T6).

    Casefolds, keeps only word characters (unicode-aware, underscores
    stripped) and collapses punctuation and whitespace runs to single
    spaces, so ``"The migration tool is Alembic!"`` and ``"the migration
    tool is alembic"`` normalize to the same string. Deliberately LEXICAL:
    semantic near-duplicates are the reviewer's job (P3.T3); this helper
    only powers the extractor's deterministic re-extraction suppression
    against confirmed facts (plan P3.T6).
    """
    return " ".join(_FACT_TOKEN_RE.findall(text.casefold()))


def _iso_utc(value: datetime) -> str:
    """ISO 8601 UTC string with a ``Z`` suffix (roundtrip-stable)."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, field_name: str) -> datetime:
    """Parse a dict value into an aware UTC datetime (ISO 8601 strings ok)."""
    if isinstance(value, datetime):
        return to_utc(value, field_name)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name}: {value!r} is not an ISO 8601 timestamp") from exc
        return to_utc(parsed, field_name)
    raise ValueError(f"{field_name}: cannot parse {value!r} as a datetime")


@dataclass
class DraftFact:
    """A candidate memory proposed by the extraction stage (plan P3.T1).

    ``type`` and ``proposed_source`` must be valid :data:`EntryType` /
    :class:`VerificationSource` values (the draft carries the verification
    metadata it *would* have if accepted); ``confidence`` is the extractor's
    certainty, clamped into [0, 1]; ``created_at`` must be timezone-aware and
    is normalized to UTC. Field order matches the plan's P3.T1 table, so
    positional construction follows it.
    """

    draft_id: str
    text: str
    type: EntryType
    proposed_source: VerificationSource
    confidence: float
    context_hint: str = "global"
    rationale: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.draft_id:
            raise ValueError("draft_id must be a non-empty string")
        if not self.text:
            raise ValueError("text must be a non-empty string")
        if self.type not in ENTRY_TYPES:
            raise ValueError(f"unknown entry type: {self.type!r}")
        if self.proposed_source not in SOURCE_WEIGHTS:
            raise ValueError(f"unknown verification source: {self.proposed_source!r}")
        self.confidence = clamp01(self.confidence)
        self.created_at = to_utc(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping of every field (datetime as ISO 8601 UTC)."""
        return {
            "draft_id": self.draft_id,
            "text": self.text,
            "type": self.type,
            "proposed_source": self.proposed_source,
            "confidence": self.confidence,
            "context_hint": self.context_hint,
            "rationale": self.rationale,
            "source_turn_ids": list(self.source_turn_ids),
            "created_at": _iso_utc(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DraftFact:
        """Rebuild a :class:`DraftFact` from :meth:`to_dict` output."""
        try:
            return cls(
                draft_id=str(data["draft_id"]),
                text=str(data["text"]),
                type=cast(EntryType, data["type"]),
                proposed_source=cast(VerificationSource, data["proposed_source"]),
                confidence=float(data["confidence"]),
                context_hint=str(data.get("context_hint", "global")),
                rationale=str(data.get("rationale", "")),
                source_turn_ids=[str(turn) for turn in data.get("source_turn_ids") or []],
                created_at=_parse_utc(data["created_at"], "created_at"),
            )
        except KeyError as exc:
            raise ValueError(f"draft fact missing field {exc}") from exc


@dataclass
class ReviewAction:
    """A reviewer decision on one :class:`DraftFact` (plan P3.T1).

    ``action`` must be one of the ``hermes.memory.types.ReviewAction`` literal
    values (``add`` | ``update`` | ``skip`` | ``conflict``) — anything else
    raises :class:`ValueError` at construction. ``target_id`` points at the
    existing memory entry for update/conflict decisions; ``merged_text``
    carries the reviewer-edited body for ``update`` decisions.
    """

    action: ReviewActionLiteral
    target_id: str | None
    rationale: str
    merged_text: str | None = None

    def __post_init__(self) -> None:
        if self.action not in REVIEW_ACTIONS:
            allowed = sorted(REVIEW_ACTIONS)
            raise ValueError(f"invalid review action: {self.action!r}; expected one of {allowed}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping of every field."""
        return {
            "action": self.action,
            "target_id": self.target_id,
            "rationale": self.rationale,
            "merged_text": self.merged_text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewAction:
        """Rebuild a :class:`ReviewAction` from :meth:`to_dict` output."""
        try:
            target_id = data["target_id"]
            merged_text = data.get("merged_text")
            return cls(
                action=cast(ReviewActionLiteral, data["action"]),
                target_id=None if target_id is None else str(target_id),
                rationale=str(data["rationale"]),
                merged_text=None if merged_text is None else str(merged_text),
            )
        except KeyError as exc:
            raise ValueError(f"review action missing field {exc}") from exc
