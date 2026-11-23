"""Trust score anomaly detection (plan P17.T3).

:class:`TrustAnomalyDetector` watches verification events for unusual trust
patterns and holds suspicious ones out of the applied history until a human
reviews them:

- **rapid increase** — one event raising the score by
  :data:`RAPID_INCREASE_THRESHOLD` (0.3) or more (plan: "0.3 -> 0.9 in one
  event");
- **verifier flooding** — :data:`VERIFIER_FLOOD_MAX_EVENTS` (5) or more
  events from the SAME verifier within :data:`VERIFIER_FLOOD_WINDOW_MINUTES`
  (60). History comes either from the caller (``recent_events=`` — e.g.
  :func:`hermes.memory.verification.recent_events` over an entry) or from the
  detector's internal per-verifier ledger fed by :meth:`record_event`;
- **fake-verification pattern** — the event's ``evidence`` text matching a
  small local trust-manipulation phrasing set. Deliberately NOT imported from
  :mod:`hermes.memory.adversarial_detector` to keep the two detectors
  decoupled (the P17 modules deploy on different surfaces: write gate vs.
  trust ledger); the duplication is the documented trade.

Flagged events are appended to the human review queue
(``<hermes_home>/memory-repo/review_queue.json``, one JSON object per flagged
change: ``{"entry_id", "reasons", "flagged_at", "status": "pending"}``).
Guidance contract for callers (e.g. the P2.1 apply pipeline): a flagged
result means HOLD the event pending review — :attr:`AnomalyResult.should_hold`
makes that explicit. :meth:`approve_review` marks the queued item approved so
the caller may re-apply; :meth:`reject_review` marks it rejected — revert
guidance is to recompute the entry's score from its event history WITHOUT
the rejected event (which is automatic when the event was never applied).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from hermes.config import hermes_home

__all__ = [
    "FAKE_VERIFICATION_PATTERNS",
    "RAPID_INCREASE_THRESHOLD",
    "REVIEW_STATUS_APPROVED",
    "REVIEW_STATUS_PENDING",
    "REVIEW_STATUS_REJECTED",
    "VERIFIER_FLOOD_MAX_EVENTS",
    "VERIFIER_FLOOD_WINDOW_MINUTES",
    "AnomalyResult",
    "TrustAnomalyDetector",
    "default_review_queue_path",
]

#: One event may not raise the trust score by this much or more (plan P17.T3).
RAPID_INCREASE_THRESHOLD: Final[float] = 0.3

#: Same-verifier event count within the window that constitutes flooding.
VERIFIER_FLOOD_MAX_EVENTS: Final[int] = 5

#: Flooding window in minutes (plan P17.T3: "short time").
VERIFIER_FLOOD_WINDOW_MINUTES: Final[int] = 60

#: Human review queue item statuses.
REVIEW_STATUS_PENDING: Final[str] = "pending"
REVIEW_STATUS_APPROVED: Final[str] = "approved"
REVIEW_STATUS_REJECTED: Final[str] = "rejected"

#: Small trust-manipulation phrasing set for fake-verification evidence
#: (duplicated from the adversarial pattern family on purpose — see module
#: docstring; verification.py must not grow a cross-detector dependency).
FAKE_VERIFICATION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("this fact is verified", re.compile(r"this (?:fact|entry|claim) is verified", re.IGNORECASE)),
    ("verified by user", re.compile(r"verified by (?:the\s+)?user", re.IGNORECASE)),
    ("verified by admin", re.compile(r"verified by (?:the\s+)?(?:admin|owner)", re.IGNORECASE)),
    ("trust_score=1", re.compile(r"trust[_\s]*score\s*[=:]\s*1(?:\.0+)?", re.IGNORECASE)),
    ("mark as verified", re.compile(r"\bmark(?:ed)?\s+as\s+verified\b", re.IGNORECASE)),
    ("verified: true", re.compile(r"\bverified\s*:\s*true\b", re.IGNORECASE)),
)

_MINUTES_PER_SECOND: Final[float] = 60.0


def default_review_queue_path() -> Path:
    """Default queue: ``<hermes_home>/memory-repo/review_queue.json``."""
    path: Path = hermes_home() / "memory-repo" / "review_queue.json"
    return path


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class _VerifierStamped(Protocol):
    """Minimal shape :meth:`check_event` needs from history events."""

    verifier: str
    timestamp: datetime


@dataclass
class AnomalyResult:
    """Outcome of one :meth:`TrustAnomalyDetector.check_event` call.

    ``flagged`` events must be HELD pending human review (see
    :attr:`should_hold`); ``reasons`` lists every triggered anomaly in
    human-readable form.
    """

    flagged: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def should_hold(self) -> bool:
        """Caller guidance: hold the event out of the applied history."""
        return self.flagged


class TrustAnomalyDetector:
    """Flags anomalous trust changes and queues them for human review.

    Args:
        rapid_increase_threshold: per-event score jump that flags.
        verifier_flood_window_minutes: flooding window length.
        verifier_flood_max_events: same-verifier events within the window
            that flag (including the event under check).
        review_queue_path: JSON queue file (default
            ``<hermes_home>/memory-repo/review_queue.json``).
        now: injectable clock (aware UTC) used for queue stamps and as the
            default flooding reference point.
    """

    def __init__(
        self,
        rapid_increase_threshold: float = RAPID_INCREASE_THRESHOLD,
        verifier_flood_window_minutes: int = VERIFIER_FLOOD_WINDOW_MINUTES,
        verifier_flood_max_events: int = VERIFIER_FLOOD_MAX_EVENTS,
        review_queue_path: Path | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.rapid_increase_threshold = rapid_increase_threshold
        self.verifier_flood_window_minutes = verifier_flood_window_minutes
        self.verifier_flood_max_events = verifier_flood_max_events
        self.review_queue_path = (
            review_queue_path if review_queue_path is not None else default_review_queue_path()
        )
        self._now = now if now is not None else (lambda: datetime.now(UTC))
        #: internal per-verifier ledger: verifier -> [(entry_id, timestamp)]
        self._ledger: dict[str, list[tuple[str, datetime]]] = {}
        self._lock = threading.Lock()

    # ── ledger feeding (internal history, plan P17.T3 design) ────────────────
    def record_event(self, entry_id: str, event: _VerifierStamped) -> None:
        """Feed the internal per-verifier ledger (call when an event applies)."""
        with self._lock:
            self._ledger.setdefault(event.verifier, []).append(
                (entry_id, _as_utc(event.timestamp))
            )

    # ── the check ─────────────────────────────────────────────────────────────
    def check_event(
        self,
        entry_id: str,
        event: Any,
        current_score: float,
        previous_score: float,
        now: datetime | None = None,
        *,
        recent_events: Sequence[_VerifierStamped] | None = None,
    ) -> AnomalyResult:
        """Evaluate one pending verification event (plan P17.T3).

        ``event`` is a :class:`~hermes.memory.types.VerificationEvent`-like
        object (``verifier``/``timestamp``/``evidence`` attributes). History
        for the flooding check comes from ``recent_events`` when given,
        otherwise from the internal ledger fed by :meth:`record_event`. The
        event under check counts toward the flooding total. Flagged events
        are enqueued for human review as a side effect.
        """
        reasons: list[str] = []
        jump = current_score - previous_score
        if jump >= self.rapid_increase_threshold:
            reasons.append(
                f"rapid trust increase: +{jump:.3f} in one event "
                f"(threshold {self.rapid_increase_threshold:g})"
            )

        reference = _as_utc(now) if now is not None else _as_utc(event.timestamp)
        window = timedelta(minutes=self.verifier_flood_window_minutes)
        same_verifier_recent = self._recent_from_verifier(
            event.verifier, reference, window, recent_events
        )
        total_from_verifier = len(same_verifier_recent) + 1  # + the event under check
        if total_from_verifier >= self.verifier_flood_max_events:
            reasons.append(
                f"verifier flooding: {total_from_verifier} events from "
                f"'{event.verifier}' within {self.verifier_flood_window_minutes} min "
                f"(threshold {self.verifier_flood_max_events})"
            )

        fake_labels = [
            label
            for label, pattern in FAKE_VERIFICATION_PATTERNS
            if pattern.search(str(getattr(event, "evidence", "") or ""))
        ]
        if fake_labels:
            reasons.append(
                f"fake-verification pattern in evidence: {', '.join(fake_labels)}"
            )

        result = AnomalyResult(flagged=bool(reasons), reasons=reasons)
        if result.flagged:
            self._enqueue(entry_id, reasons)
        return result

    def _recent_from_verifier(
        self,
        verifier: str,
        reference: datetime,
        window: timedelta,
        recent_events: Sequence[_VerifierStamped] | None,
    ) -> list[str]:
        """History events from ``verifier`` within ``window`` of ``reference``.

        Returns the matching history entries as verifier labels (the caller
        only counts them; the event under check is added on top).
        """
        if recent_events is not None:
            candidates: list[tuple[str, datetime]] = [
                (str(getattr(event, "verifier", "")), _as_utc(event.timestamp))
                for event in recent_events
            ]
        else:
            with self._lock:
                # ledger entries under this verifier are (entry_id, timestamp);
                # normalize to the same (verifier, timestamp) shape as above
                candidates = [
                    (verifier, timestamp)
                    for _entry_id, timestamp in self._ledger.get(verifier, [])
                ]
        return [
            candidate_verifier
            for candidate_verifier, timestamp in candidates
            if candidate_verifier == verifier
            and reference - window <= timestamp <= reference
        ]

    # ── human review queue ────────────────────────────────────────────────────
    def _enqueue(self, entry_id: str, reasons: list[str]) -> dict[str, Any]:
        """Append one pending review item to the queue file (atomic rewrite)."""
        item: dict[str, Any] = {
            "entry_id": entry_id,
            "reasons": list(reasons),
            "flagged_at": _as_utc(self._now()).isoformat(),
            "status": REVIEW_STATUS_PENDING,
        }
        with self._lock:
            queue = self._read_queue()
            queue.append(item)
            self._write_queue(queue)
        return item

    def _read_queue(self) -> list[dict[str, Any]]:
        """Parsed queue items (oldest first); a missing file is an empty queue."""
        if not self.review_queue_path.exists():
            return []
        document = json.loads(self.review_queue_path.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            raise ValueError(
                f"review queue must be a JSON list: {self.review_queue_path}"
            )
        return [item for item in document if isinstance(item, dict)]

    def _write_queue(self, queue: list[dict[str, Any]]) -> None:
        """Atomically rewrite the queue file (tmp + os.replace)."""
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.review_queue_path.with_name(
            f".{self.review_queue_path.name}.{os.getpid()}.tmp"
        )
        try:
            tmp.write_text(
                json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self.review_queue_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def pending_reviews(self) -> list[dict[str, Any]]:
        """All queue items still in ``pending`` status (human work list)."""
        with self._lock:
            return [item for item in self._read_queue()
                    if item.get("status") == REVIEW_STATUS_PENDING]

    def approve_review(self, entry_id: str) -> int:
        """Approve every pending item for ``entry_id``; returns items updated.

        Guidance after approval: the caller may re-apply the held event(s) —
        the anomaly check is a gate, not a ledger mutation.
        """
        return self._set_status(entry_id, REVIEW_STATUS_APPROVED)

    def reject_review(self, entry_id: str) -> int:
        """Reject every pending item for ``entry_id``; returns items updated.

        Revert guidance: recompute the entry's score from its event history
        WITHOUT the rejected event — automatic when the event was never
        applied (the hold contract), else drop it from the history and
        recompute via :func:`hermes.memory.trust_card.compute_trust_score`.
        """
        return self._set_status(entry_id, REVIEW_STATUS_REJECTED)

    def _set_status(self, entry_id: str, status: str) -> int:
        """Move every pending item of ``entry_id`` to ``status``."""
        updated = 0
        with self._lock:
            queue = self._read_queue()
            for item in queue:
                if item.get("entry_id") == entry_id and item.get("status") == REVIEW_STATUS_PENDING:
                    item["status"] = status
                    updated += 1
            if updated:
                self._write_queue(queue)
        return updated
