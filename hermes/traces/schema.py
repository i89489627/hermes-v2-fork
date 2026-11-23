"""hermes-v2 trace schema (plan P11.T1).

The :class:`Trace` dataclass is the single record type shared by the v2
pipeline stages and :class:`~hermes.traces.store.TraceStore`. One trace
captures one turn: the raw user prompt, which models served it, the artifact
each v2 stage produced (``extraction`` / ``review`` / ``gating`` /
``injection`` payloads) and the quality label used to curate the GEPA
training set (plan P12). ``timestamp`` is epoch seconds (UTC); consumers
derive the JSONL day file from it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final

__all__ = ["QUALITY_LABELS", "Trace"]

#: Allowed values for :attr:`Trace.quality_label`.
QUALITY_LABELS: Final[frozenset[str]] = frozenset(
    {"good", "acceptable", "bad", "needs-review"}
)


def _stage_payload(raw: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    """Extract a stage payload; anything but a mapping degrades to ``None``."""
    value = raw.get(key)
    return value if isinstance(value, dict) else None


@dataclass
class Trace:
    """One turn's pipeline record (one JSONL line / one SQLite row)."""

    trace_id: str = ""
    turn_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    user_prompt: str = ""
    models_used: list[str] = field(default_factory=list)
    extraction: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    gating: dict[str, Any] | None = None
    injection: dict[str, Any] | None = None
    quality_label: str | None = None
    pii_flagged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Deep JSON-serializable projection (as written to JSONL)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Trace:
        """Rebuild a trace from :meth:`to_dict` output; unknown keys ignored.

        Missing fields fall back to dataclass defaults; a non-mapping stage
        payload becomes ``None`` and an unknown quality label becomes ``None``,
        so a corrupted line can never crash a reader.
        """
        models_raw = raw.get("models_used")
        label_raw = raw.get("quality_label")
        label: str | None = None
        if isinstance(label_raw, str) and label_raw in QUALITY_LABELS:
            label = label_raw
        return cls(
            trace_id=str(raw.get("trace_id", "")),
            turn_id=str(raw.get("turn_id", "")),
            session_id=str(raw.get("session_id", "")),
            timestamp=float(raw.get("timestamp", 0.0)),
            user_prompt=str(raw.get("user_prompt", "")),
            models_used=(
                [str(item) for item in models_raw]
                if isinstance(models_raw, list)
                else []
            ),
            extraction=_stage_payload(raw, "extraction"),
            review=_stage_payload(raw, "review"),
            gating=_stage_payload(raw, "gating"),
            injection=_stage_payload(raw, "injection"),
            quality_label=label,
            pii_flagged=bool(raw.get("pii_flagged", False)),
        )
