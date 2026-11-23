"""TrustCard YAML frontmatter parse/serialize + trust scoring (plan P1.T1).

A v2 memory entry is stored as one markdown file: a YAML frontmatter block
(the *trust card* — identity, timestamps and verification history) followed
by the entry text as the markdown body (plan Section 2.4). ``parse`` and
``serialize`` are inverse operations; datetimes round-trip as ISO 8601 UTC
with a ``Z`` suffix. ``compute_trust_score`` is the plan's weighted average
``sum(w_i * v_i) / sum(w_i)`` clamped to [0, 1]; an empty (unverified)
history returns 0.5, the documented migration default for entries imported
without verification evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, cast

from frontmatter.default_handlers import YAMLHandler
from yaml import YAMLError

from hermes.memory.types import (
    ENTRY_TYPES,
    WEIGHT_FROM_SOURCE,
    EntryType,
    VerificationEvent,
    VerificationSource,
    clamp01,
    to_utc,
)


@dataclass
class TrustCard:
    """Frontmatter metadata of one memory entry (the body carries the text).

    Field invariants mirror :class:`hermes.memory.entry.MemoryEntry`.
    """

    id: str
    type: EntryType
    created_at: datetime
    last_verified_at: datetime | None = None
    trust_score: float = 0.5
    verification_events: list[VerificationEvent] = field(default_factory=list)
    context: str = "global"
    scope: list[str] = field(default_factory=list)
    ttl_days: int | None = None
    tags: list[str] = field(default_factory=list)
    pii_checked: bool = False
    attachments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must be a non-empty string")
        if self.type not in ENTRY_TYPES:
            raise ValueError(f"unknown entry type: {self.type!r}")
        self.created_at = to_utc(self.created_at, "created_at")
        if self.last_verified_at is not None:
            self.last_verified_at = to_utc(self.last_verified_at, "last_verified_at")
        self.trust_score = clamp01(self.trust_score)


def compute_trust_score(events: Sequence[VerificationEvent]) -> float:
    """Weighted mean of event values, clamped to [0, 1] (plan Section 2.4).

    ``sum(w_i * v_i) / sum(w_i)`` where ``w_i`` is each event's weight and
    ``v_i`` its value (a ``test-red`` negative anchor contributes ``v_i=0``).
    An empty (unverified) history returns ``0.5`` — the documented migration
    default for entries imported without verification evidence; the same
    default applies to the degenerate all-zero-weight case.
    """
    if not events:
        return 0.5
    total_weight = sum(event.weight for event in events)
    if total_weight <= 0.0:
        return 0.5
    score = sum(event.weight * event.value for event in events) / total_weight
    return clamp01(score)


def add_verification_event(card: TrustCard, event: VerificationEvent) -> TrustCard:
    """Append ``event`` to ``card`` and recompute its trust score (plan P2.T1).

    Mutates and returns ``card``: the event joins ``verification_events``,
    ``trust_score`` becomes :func:`compute_trust_score` over the full history,
    and ``last_verified_at`` moves to the event's timestamp. The caller is
    responsible for persisting the card (see
    :meth:`hermes.memory.store.MemoryStore.record_event`).
    """
    card.verification_events.append(event)
    card.trust_score = compute_trust_score(card.verification_events)
    card.last_verified_at = event.timestamp
    return card


def _iso_z(value: datetime) -> str:
    """Format ``value`` as ISO 8601 UTC with a ``Z`` suffix (roundtrip-stable)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _coerce_datetime(value: Any, field_name: str) -> datetime:
    """Convert frontmatter ``value`` to an aware UTC datetime.

    Accepts what YAML can produce: aware/naive datetimes (PyYAML resolves
    unquoted timestamps), ISO 8601 strings and plain dates. Naive values are
    interpreted as UTC so hand-authored cards parse leniently.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name}: {value!r} is not an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError(f"{field_name}: cannot parse {value!r} as a datetime")


def _optional_datetime(meta: Mapping[str, Any], key: str) -> datetime | None:
    raw = meta.get(key)
    return None if raw is None else _coerce_datetime(raw, key)


def _require(meta: Mapping[str, Any], key: str) -> Any:
    if key not in meta:
        raise ValueError(f"trust card frontmatter missing required field {key!r}")
    return meta[key]


def _str_list(meta: Mapping[str, Any], key: str) -> list[str]:
    raw = meta.get(key) or []
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list")
    return [str(item) for item in raw]


def _event_list(meta: Mapping[str, Any]) -> list[Any]:
    raw = meta.get("verification_events") or []
    if not isinstance(raw, list):
        raise ValueError("verification_events must be a list")
    return raw


def _event_from(raw: Any) -> VerificationEvent:
    """Build a validated :class:`VerificationEvent` from frontmatter data."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"verification event must be a mapping, got {raw!r}")
    try:
        return VerificationEvent(
            event_id=str(raw["event_id"]),
            timestamp=_coerce_datetime(raw["timestamp"], "verification_events[].timestamp"),
            source=cast(VerificationSource, raw["source"]),
            verifier=str(raw.get("verifier", "unknown")),
            weight=float(raw["weight"]) if raw.get("weight") is not None else WEIGHT_FROM_SOURCE,
            evidence=str(raw.get("evidence", "")),
            value=float(raw.get("value", 1.0)),
        )
    except KeyError as exc:
        raise ValueError(f"verification event missing field {exc}") from exc


#: Shared python-frontmatter YAML handler (SafeLoader/SafeDumper based).
_YAML_HANDLER = YAMLHandler()


def _body_after_delimiter(content: str) -> str:
    """Drop the single blank line serialize() emits after the ``---`` block.

    Unlike ``frontmatter.loads`` (which strips surrounding whitespace), the
    body round-trips byte-exactly.
    """
    return content[1:] if content.startswith("\n") else content


def _split_frontmatter(markdown_content: str) -> tuple[dict[str, Any], str]:
    """Split ``markdown_content`` into (frontmatter mapping, exact body).

    Uses python-frontmatter's :class:`YAMLHandler` for detection, splitting
    and YAML loading, but keeps the body unstripped. Text without a ``---``
    block yields an empty mapping (``parse`` then rejects it for missing
    required fields).
    """
    text = markdown_content.lstrip("\ufeff").lstrip()
    if not _YAML_HANDLER.detect(text):
        return {}, text
    fm, content = _YAML_HANDLER.split(text)
    try:
        data = _YAML_HANDLER.load(fm)
    except YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        return {}, content
    return cast(dict[str, Any], data), _body_after_delimiter(content)


def parse(markdown_content: str) -> tuple[TrustCard, str]:
    """Split a memory markdown file into ``(TrustCard, body)``.

    Raises :class:`ValueError` when the frontmatter is missing required
    fields (``id``, ``type``, ``created_at``) or carries invalid values.
    """
    meta, body = _split_frontmatter(markdown_content)

    entry_type = _require(meta, "type")
    if entry_type not in ENTRY_TYPES:
        raise ValueError(f"unknown entry type: {entry_type!r}")
    raw_score = meta.get("trust_score")
    raw_ttl = meta.get("ttl_days")

    card = TrustCard(
        id=str(_require(meta, "id")),
        type=cast(EntryType, entry_type),
        created_at=_coerce_datetime(_require(meta, "created_at"), "created_at"),
        last_verified_at=_optional_datetime(meta, "last_verified_at"),
        trust_score=clamp01(float(raw_score)) if raw_score is not None else 0.5,
        verification_events=[_event_from(raw) for raw in _event_list(meta)],
        context=str(meta.get("context", "global")),
        scope=_str_list(meta, "scope"),
        ttl_days=None if raw_ttl is None else int(raw_ttl),
        tags=_str_list(meta, "tags"),
        pii_checked=bool(meta.get("pii_checked", False)),
        attachments=_str_list(meta, "attachments"),
    )
    return card, body


def _card_to_meta(card: TrustCard) -> dict[str, Any]:
    """Flatten ``card`` into the frontmatter mapping (datetimes as ISO-Z)."""
    return {
        "id": card.id,
        "type": card.type,
        "created_at": _iso_z(card.created_at),
        "last_verified_at": (
            None if card.last_verified_at is None else _iso_z(card.last_verified_at)
        ),
        "trust_score": card.trust_score,
        "verification_events": [
            {
                "event_id": event.event_id,
                "timestamp": _iso_z(event.timestamp),
                "source": event.source,
                "verifier": event.verifier,
                "weight": event.weight,
                "evidence": event.evidence,
                "value": event.value,
            }
            for event in card.verification_events
        ],
        "context": card.context,
        "scope": list(card.scope),
        "ttl_days": card.ttl_days,
        "tags": list(card.tags),
        "pii_checked": card.pii_checked,
        "attachments": list(card.attachments),
    }


def serialize(card: TrustCard, body: str) -> str:
    """Render ``card`` + ``body`` back to markdown with YAML frontmatter.

    The body is embedded byte-exactly (no whitespace stripping), so
    ``parse(serialize(card, body)) == (card, body)``.
    """
    yaml_block = _YAML_HANDLER.export(_card_to_meta(card)).strip()
    return f"---\n{yaml_block}\n---\n\n{body}"
