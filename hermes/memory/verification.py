"""Verification event tracking for memory entries (plan P2.T1).

Every memory operation leaves evidence behind as a
:class:`~hermes.memory.types.VerificationEvent`, and an entry's
``trust_score`` is the weighted average of its event history — weights per
the plan Section 2.4 table (:data:`hermes.memory.types.SOURCE_WEIGHTS`:
``user-explicit`` 1.0, ``test-green`` 1.0, ``code-review-human`` 0.95,
``tool-output`` 0.85, ``file-read`` 0.7, ``agent-inferred`` 0.5, ``llm``
0.3). :func:`add_verification_event` appends one event and recomputes the
score via :func:`hermes.memory.trust_card.compute_trust_score`; it accepts
either a :class:`~hermes.memory.store.MemoryStore` (the event is persisted —
entry file rewritten atomically, exactly one Git commit) or a single
:class:`~hermes.memory.entry.MemoryEntry` (in-memory update only). The
guarantee that every entry carries at least one verification event is
established at creation time: :meth:`MemoryStore.add` always seeds the
history with the event built from its ``source``/``verifier``/``evidence``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, TypeAlias
from uuid import uuid4

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import compute_trust_score
from hermes.memory.types import VerificationEvent, VerificationSource, clamp01

__all__ = [
    "ANCHOR_VERIFIER",
    "TrustAnomalyError",
    "add_verification_event",
    "build_event",
    "build_test_green_event",
    "build_test_red_event",
    "clear_anomaly_hook",
    "get_anomaly_hook",
    "recent_events",
    "set_anomaly_hook",
]

#: Verifier stamped on automatic ground-truth anchor events (plan P2.T5):
#: the rule-based test/build hook, never an LLM.
ANCHOR_VERIFIER: Final[str] = "test-hook"

#: :func:`add_verification_event` accepts either an in-memory entry or a store.
VerificationTarget: TypeAlias = MemoryEntry | MemoryStore


# ── P17.T3 anomaly hook point (ADDITIVE — default behaviour unchanged) ───────
#
# The trust-anomaly detector (hermes/memory/trust_anomaly.py, plan P17.T3)
# owns ALL anomaly logic; this module only exposes the seam the pipeline
# wires it into. ``set_anomaly_hook`` registers a callable consulted by
# :func:`add_verification_event` BEFORE an event is applied:
#
#     hook(entry_id, event, previous_score, prospective_score) -> flagged
#
# A ``True`` return raises :class:`TrustAnomalyError` and the event is NOT
# applied (held pending human review; see TrustAnomalyDetector.approve_review
# / reject_review). With no hook registered (the default) behaviour is
# exactly as before — the hook is strictly optional.
AnomalyHook = Callable[[str, VerificationEvent, float, float], bool]

_anomaly_hook: AnomalyHook | None = None
_anomaly_hook_lock = threading.Lock()


def set_anomaly_hook(hook: AnomalyHook | None) -> None:
    """Register (or, with ``None``, remove) the trust-anomaly hook (P17.T3)."""
    global _anomaly_hook
    with _anomaly_hook_lock:
        _anomaly_hook = hook


def get_anomaly_hook() -> AnomalyHook | None:
    """The currently registered anomaly hook (``None`` when unset)."""
    with _anomaly_hook_lock:
        return _anomaly_hook


def clear_anomaly_hook() -> None:
    """Remove the anomaly hook (restores pre-P17.T3 behaviour exactly)."""
    set_anomaly_hook(None)


class TrustAnomalyError(RuntimeError):
    """Raised when the anomaly hook flags an event before it is applied.

    Attributes ``entry_id`` and ``event`` identify the held change; the
    caller routes it to the human review queue (TrustAnomalyDetector).
    """

    def __init__(self, entry_id: str, event: VerificationEvent) -> None:
        self.entry_id = entry_id
        self.event = event
        super().__init__(
            f"trust anomaly flagged for entry {entry_id!r}: event {event.event_id} "
            "(verifier " + event.verifier + ") held pending human review"
        )


def recent_events(entry: MemoryEntry, limit: int = 20) -> list[VerificationEvent]:
    """The entry's most recent verification events (chronological tail).

    Trivial feed helper for the P17.T3 detector: pass the result as
    ``TrustAnomalyDetector.check_event(..., recent_events=...)`` so the
    flooding check can run without the detector maintaining its own ledger.
    ``limit <= 0`` returns an empty list.
    """
    if limit <= 0:
        return []
    return list(entry.verification_events[-limit:])


def _check_anomaly_hook(
    entry_id: str, event: VerificationEvent, source: MemoryEntry | None
) -> None:
    """Consult the registered anomaly hook BEFORE an event is applied.

    ``source`` is the entry the event would be applied to (``None`` when the
    id is unknown — :meth:`MemoryStore.record_event` raises ``KeyError`` as
    before). A flagged verdict raises :class:`TrustAnomalyError` and leaves
    the target untouched.
    """
    hook = get_anomaly_hook()
    if hook is None or source is None:
        return
    previous_score = source.trust_score
    prospective_score = compute_trust_score([*source.verification_events, event])
    if hook(entry_id, event, previous_score, prospective_score):
        raise TrustAnomalyError(entry_id, event)


def build_event(
    source: VerificationSource,
    verifier: str,
    evidence: str = "",
    value: float = 1.0,
    weight: float | None = None,
    timestamp: datetime | None = None,
) -> VerificationEvent:
    """Build a :class:`VerificationEvent` with store-conventional defaults.

    ``weight=None`` derives the event weight from the plan Section 2.4 table
    for ``source`` (the :data:`hermes.memory.types.WEIGHT_FROM_SOURCE`
    sentinel path); ``timestamp=None`` stamps the current UTC time; ``value``
    is clamped into [0, 1].
    """
    return VerificationEvent(
        event_id=f"ev-{uuid4().hex[:12]}",
        timestamp=timestamp if timestamp is not None else datetime.now(UTC),
        source=source,
        verifier=verifier,
        weight=weight if weight is not None else -1.0,  # sentinel: derive from source
        evidence=evidence,
        value=clamp01(value),
    )


def _apply_to_entry(entry: MemoryEntry, event: VerificationEvent) -> MemoryEntry:
    """Append ``event`` to ``entry`` in memory and recompute the trust score."""
    entry.verification_events.append(event)
    entry.trust_score = compute_trust_score(entry.verification_events)
    entry.last_verified_at = event.timestamp
    return entry


def add_verification_event(  # noqa: PLR0913 - explicit event fields beat a kwargs bag
    target: VerificationTarget,
    entry_id: str | None = None,
    *,
    source: VerificationSource = "user-explicit",
    verifier: str = "user",
    evidence: str = "",
    value: float = 1.0,
    weight: float | None = None,
    timestamp: datetime | None = None,
) -> MemoryEntry:
    """Append one verification event and recompute the trust score (plan P2.T1).

    ``target`` selects the mode:

    - **store** (a :class:`~hermes.memory.store.MemoryStore`): ``entry_id`` is
      required; the event is appended to the persisted entry's trust card via
      :meth:`MemoryStore.record_event` — atomic rewrite plus one Git commit.
      Unknown ids raise :class:`KeyError`.
    - **entry** (a :class:`~hermes.memory.entry.MemoryEntry`): the event is
      appended in memory and the score recomputed; persistence is the
      caller's concern. Passing an ``entry_id`` that does not match the
      entry's id raises :class:`ValueError`.

    The resulting score is :func:`compute_trust_score` over the full event
    history (weighted average per the plan Section 2.4 table); ``value`` and
    an explicit ``weight`` are clamped/validated by :class:`VerificationEvent`.

    P17.T3 hook point: when an anomaly hook is registered
    (:func:`set_anomaly_hook`), it is consulted BEFORE anything is applied —
    a flagged event raises :class:`TrustAnomalyError` and the target is left
    untouched (the event is held pending human review). With no hook, or when
    the hook does not flag, behaviour is unchanged.
    """
    event = build_event(
        source=source,
        verifier=verifier,
        evidence=evidence,
        value=value,
        weight=weight,
        timestamp=timestamp,
    )
    if isinstance(target, MemoryEntry):
        if entry_id is not None and entry_id != target.id:
            raise ValueError(
                f"entry_id {entry_id!r} does not match target entry id {target.id!r}"
            )
        _check_anomaly_hook(target.id, event, target)
        return _apply_to_entry(target, event)
    if entry_id is None:
        raise ValueError("entry_id is required when target is a MemoryStore")
    _check_anomaly_hook(entry_id, event, target.get(entry_id))
    return target.record_event(entry_id, event)


# ── ground-truth anchor events (appended additively by plan P2.T5) ──────────
#
# The plan's VerificationSource literal set has no "test-red" member, so the
# failing-run mapping below is the documented deviation for the P2.T5 anchor
# hook: red is recorded as the P1.T1 NEGATIVE anchor (source "tool-output",
# weight 0.85, value 0.0 — plan Section 2.4 already defines value=0.0 as the
# negative-anchor verdict) with the event evidence prefixed "test-red:" so
# the origin stays machine-readable in the trust card. Green uses the native
# "test-green" source (weight 1.0, value 1.0) unchanged.


def build_test_green_event(
    evidence: str,
    *,
    verifier: str = ANCHOR_VERIFIER,
    timestamp: datetime | None = None,
) -> VerificationEvent:
    """Build the GREEN ground-truth anchor event (plan P2.T5).

    Source ``test-green`` (weight 1.0 per the plan Section 2.4 table),
    ``value=1.0`` — a passing test/build run confirms the related entries.
    ``verifier`` defaults to :data:`ANCHOR_VERIFIER` (the rule-based hook).
    """
    return build_event(
        source="test-green",
        verifier=verifier,
        evidence=evidence,
        value=1.0,
        timestamp=timestamp,
    )


def build_test_red_event(
    evidence: str,
    *,
    verifier: str = ANCHOR_VERIFIER,
    timestamp: datetime | None = None,
) -> VerificationEvent:
    """Build the RED (failing-run) ground-truth anchor event (plan P2.T5).

    DEVIATION NOTE — the plan's literal source set lacks ``test-red``, so a
    red anchor is recorded as the P1.T1 negative anchor: source
    ``tool-output`` with explicit weight 0.85 and ``value=0.0``; the
    ``evidence`` carries a ``test-red:`` prefix (idempotent — an evidence
    string already prefixed is left untouched) so the trust card still names
    the failing-run origin.
    """
    prefixed = evidence if evidence.startswith("test-red:") else f"test-red: {evidence}"
    return build_event(
        source="tool-output",
        verifier=verifier,
        evidence=prefixed,
        value=0.0,
        weight=0.85,
        timestamp=timestamp,
    )
