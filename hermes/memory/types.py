"""Shared value types for the hermes-v2 memory layer (plan P1.T1).

Defines the entry taxonomy, the verification-source weight table and the
:class:`VerificationEvent` record from plan Section 2.4. Later phases build
on these: the P2 reviewer emits :data:`ReviewAction` decisions and the P4
CBDC gate reads trust scores derived from verification events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

#: Kind of a stored memory entry (plan Section 2.4).
EntryType = Literal["User", "Feedback", "Project", "Reference"]

#: Runtime set of valid entry types, derived from :data:`EntryType`.
ENTRY_TYPES: frozenset[str] = frozenset(get_args(EntryType))

#: Provenance of a verification event (plan Section 2.4).
VerificationSource = Literal[
    "user-explicit",
    "agent-inferred",
    "file-read",
    "tool-output",
    "test-green",
    "code-review-human",
    "llm",
]

#: Evidence weight per verification source (plan Section 2.4 table).
SOURCE_WEIGHTS: dict[VerificationSource, float] = {
    "user-explicit": 1.0,
    "test-green": 1.0,
    "code-review-human": 0.95,
    "tool-output": 0.85,
    "file-read": 0.7,
    "agent-inferred": 0.5,
    "llm": 0.3,
}

#: Reviewer decision on a candidate memory draft (consumed by P2/P3).
ReviewAction = Literal["add", "update", "skip", "conflict"]

#: Sentinel weight meaning "derive from ``SOURCE_WEIGHTS[source]``".
WEIGHT_FROM_SOURCE: float = -1.0


def _validate_source_weights() -> None:
    """Import-time guard: the weight table must match the plan exactly.

    Ensures every ``VerificationSource`` literal has exactly one weight, no
    extra keys exist, and all weights lie in ``(0, 1]`` so that trust scores
    derived from them stay meaningful.
    """
    sources = frozenset(get_args(VerificationSource))
    if frozenset(SOURCE_WEIGHTS) != sources:
        missing = sorted(sources - frozenset(SOURCE_WEIGHTS))
        extra = sorted(frozenset(SOURCE_WEIGHTS) - sources)
        raise ValueError(
            "SOURCE_WEIGHTS must have exactly one weight per VerificationSource "
            f"(missing: {missing}, extra: {extra})"
        )
    out_of_range = {s: w for s, w in SOURCE_WEIGHTS.items() if not 0.0 < w <= 1.0}
    if out_of_range:
        raise ValueError(f"SOURCE_WEIGHTS values must be in (0, 1], got {out_of_range}")


_validate_source_weights()


def to_utc(value: datetime, field_name: str) -> datetime:
    """Validate ``value`` is timezone-aware and normalize it to UTC."""
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime")
    return value.astimezone(UTC)


def clamp01(score: float) -> float:
    """Clamp ``score`` to the closed interval [0, 1] (trust_score invariant)."""
    return min(1.0, max(0.0, float(score)))


@dataclass
class VerificationEvent:
    """One piece of evidence for (or against) a memory entry (plan Section 2.4).

    ``weight`` defaults to :data:`SOURCE_WEIGHTS` for the event's ``source``
    (the :data:`WEIGHT_FROM_SOURCE` sentinel triggers the derivation).
    ``value`` is the verdict: ``1.0`` confirms the entry, ``0.0`` is a
    negative anchor — e.g. a red test (``test-red``) that disproves the claim.
    """

    event_id: str
    timestamp: datetime
    source: VerificationSource
    verifier: str
    weight: float = WEIGHT_FROM_SOURCE
    evidence: str = ""
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.source not in SOURCE_WEIGHTS:
            raise ValueError(f"unknown verification source: {self.source!r}")
        self.timestamp = to_utc(self.timestamp, "timestamp")
        if self.weight == WEIGHT_FROM_SOURCE:
            self.weight = SOURCE_WEIGHTS[self.source]
        elif not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {self.value}")


# ── Context (appended additively by plan P5.T1; P1/P2 definitions untouched) ─


@dataclass
class Context:
    """A named memory context (plan P5.T1).

    Contexts scope the memory graph: ``global`` plus named contexts under
    ``<memory-repo>/contexts/<name>/``. ``parent`` names the context whose
    facts are inherited when this context has no override (the root context
    chain always reaches ``global`` — see the P5.T1 acceptance "contexts can
    have parent (inherit global facts)"); ``path`` is the directory backing
    the context (``<root>/global`` for the implicit root); ``active_sessions``
    lists the session ids currently attached to it. Validation is kept
    minimal on purpose — slug rules and existence are enforced by
    :class:`hermes.memory.context.ContextManager`, which owns the metadata.
    """

    name: str
    parent: str | None
    path: Path
    active_sessions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        self.path = Path(self.path)
