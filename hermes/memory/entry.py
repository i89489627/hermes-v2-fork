"""MemoryEntry dataclass: one node of the v2 memory graph (plan P1.T1).

Invariants enforced at construction (plan Section 2.4): ``type`` is one of
:data:`hermes.memory.types.ENTRY_TYPES`, timestamps are timezone-aware and
normalized to UTC, and ``trust_score`` always lies in the closed interval
[0, 1] (values outside are clamped, never stored raw).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hermes.memory.types import ENTRY_TYPES, EntryType, VerificationEvent, clamp01, to_utc


@dataclass
class MemoryEntry:
    """A single stored memory (plan Section 2.4).

    ``text`` is the entry payload; ``verification_events`` hold the evidence
    trail whose weighted average (see
    :func:`hermes.memory.trust_card.compute_trust_score`) yields
    ``trust_score``. ``scope``/``context``/``ttl_days`` drive later retrieval
    and expiry phases. ``attachments`` lists multimodal assets as
    repo-relative paths under ``memory-repo/assets/`` (see
    :class:`hermes.memory.multimodal.MultimodalAttachment`, plan P8.T3);
    the elements stay plain strings so trust-card frontmatter round-trips.
    """

    id: str
    type: EntryType
    text: str
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

    def attachment_asset_ids(self) -> list[str]:
        """Asset ids of this entry's multimodal attachments, in order (plan P8.T3).

        Each ``attachments`` element is the asset's repo-relative path
        (``assets/<kind-dir>/<asset_id><ext>``); this additive helper returns
        just the asset-id stems, so callers can pass them to
        :meth:`hermes.memory.multimodal.MultimodalAttachment.detach`.
        """
        return [Path(reference).stem for reference in self.attachments]
