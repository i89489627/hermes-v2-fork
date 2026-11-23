"""Virtual session boundaries for long gateway chats (plan P7.T4).

Phase 7: long messaging sessions never close the gateway chat, so memory
extraction needs its own boundaries: "On topic shift, create virtual
session. Memory extraction scoped to virtual session." Every N messages the
:class:`~hermes.gateway.topic_detector.TopicDetector` classifies the
conversation topic against the active virtual session's topic; on a shift
the active virtual session is CLOSED (durable boundary record + one
pipeline extraction scoped to exactly its turns) and a new virtual session
starts under the detected topic. The user can force the same boundary any
time via ``/session new`` (:meth:`VirtualSessionManager.force_new_session`;
the slash-command wiring lives in the commands wave — the reason string
``"user command /session new"`` is the documented contract the command
passes).

Topic lifecycle (the "compare vs previous topic" contract)
------------------------------------------------------------------------------
A fresh virtual session has NO topic: its first :meth:`feed` calls the
detector once with ``prev_topic=None`` (the detector's session-start case —
no shift is possible) to ESTABLISH the topic from the opening turn.
Afterwards the detector runs every ``every_n_messages`` messages (the
default 5 matches the plan's "every N messages" cadence) with the
established topic as ``prev_topic``; a no-shift decision keeps the topic, a
shift closes the session and the NEW session adopts the detected topic.

Boundary marker via the pipeline (plan P7.T4)
------------------------------------------------------------------------------
Closing a virtual session produces the memory boundary through
:meth:`hermes.memory.pipeline.MemoryPipeline.extract_session_scope` — the
extract/review/apply flow run over EXACTLY the closed session's turns, so
extraction never sees turns from before or after the boundary (clean
virtual-session boundary). Persistence happens FIRST (durable boundary
record even if the extraction's store write fails), and an extraction
failure is logged, never propagated — memory must not break the gateway
turn (plan Section 5.5 graceful degradation).

Persistence (``SessionStore``)
------------------------------------------------------------------------------
With a :class:`~hermes.sessions.store.SessionStore` injected, every closed
virtual session is persisted under its virtual id
(``<parent>-v<ordinal>``; ids that do not satisfy the store's session-id
pattern are sanitized deterministically) as a JSON state dict:
``topic``, ``turn_ids``, ``closed_at`` (+ ``close_reason``, ``started_at``,
``parent_session_id``, ``virtual: True``). Without a store, boundaries
exist in-memory only (tests / offline runs).

Compaction summaries (plan P7.T5, additive)
------------------------------------------------------------------------------
With a :class:`~hermes.memory.compaction.SessionCompactor` injected, EVERY
close path (topic shift, ``/session new`` via
:meth:`VirtualSessionManager.force_new_session`, and the end-of-session
close :meth:`VirtualSessionManager.close_current`) compacts the closed
session's turns into a Reference-type memory entry whose trace-range
provenance is built from the session's recorded turn ids
(``"<first_turn_id>..<last_turn_id>"``). Compaction runs AFTER the boundary
persistence and BEFORE the scoped extraction, so the boundary extraction
(:meth:`MemoryPipeline.extract_session_scope` for the SAME session id) finds
the fresh summary in the store and leads its context with it — the DSH
pattern "extraction uses compaction summaries as primary input, raw turns
secondary". Compaction failures are logged and never break the boundary
(plan Section 5.5 — memory must not break the gateway turn).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from hermes.gateway.topic_detector import TopicDetector
    from hermes.memory.compaction import SessionCompactor
    from hermes.memory.pipeline import MemoryPipeline
    from hermes.sessions.store import SessionStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EVERY_N_MESSAGES",
    "SESSION_CLOSE_REASON",
    "SESSION_NEW_REASON",
    "TOPIC_SHIFT_REASON",
    "ShiftDecision",
    "VirtualSession",
    "VirtualSessionManager",
]

#: Plan Phase 7 default cadence: "Topic detector: every N messages".
DEFAULT_EVERY_N_MESSAGES: Final[int] = 5

#: Close reason recorded when the detector reported a topic shift.
TOPIC_SHIFT_REASON: Final[str] = "topic shift"

#: Close reason contract for the ``/session new`` command (commands wave).
SESSION_NEW_REASON: Final[str] = "user command /session new"

#: Close reason of the end-of-session close
#: (:meth:`VirtualSessionManager.close_current`, plan P7.T5).
SESSION_CLOSE_REASON: Final[str] = "session closed"

#: Characters not allowed by the SessionStore session-id pattern.
_UNSAFE_ID_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class VirtualSession:
    """One virtual session of a (never-closing) gateway chat (plan P7.T4).

    ``session_id`` is the virtual session's OWN id (``<parent>-v<ordinal>``)
    — the id under which its closed boundary is persisted and its scoped
    extraction is traced; ``topic`` is the established conversation topic;
    ``turn_ids`` / ``turn_texts`` are the session's turns, oldest first
    (index-aligned).
    """

    session_id: str
    topic: str
    started_at: datetime
    turn_ids: list[str] = field(default_factory=list)
    turn_texts: list[str] = field(default_factory=list)

    def turns(self) -> list[tuple[str, str]]:
        """The session's ``(turn_id, turn_text)`` pairs, oldest first."""
        return list(zip(self.turn_ids, self.turn_texts, strict=True))


@dataclass(frozen=True)
class ShiftDecision:
    """Outcome of one :meth:`VirtualSessionManager.feed` call.

    ``shifted`` / ``topic`` / ``confidence`` mirror the detector's decision
    (``shifted=False`` at confidence 1.0 when no detection ran this feed);
    ``closed_session`` carries the virtual session that was closed when this
    feed created a boundary (topic shift or explicit override), else
    ``None``.
    """

    shifted: bool
    topic: str
    confidence: float
    closed_session: VirtualSession | None = None


def _virtual_session_id(parent_session_id: str, ordinal: int) -> str:
    """Deterministic persisted id ``<parent>-v<ordinal>`` (store-pattern safe).

    Characters outside the :data:`hermes.sessions.store.SESSION_ID_PATTERN`
    alphabet are replaced with ``_``, leading non-alphanumerics are stripped
    and the parent part is capped so the composite id always satisfies the
    pattern (ids become file names in the store).
    """
    base = _UNSAFE_ID_CHARS.sub("_", parent_session_id).strip("._-")
    base = re.sub(r"^[^A-Za-z0-9]+", "", base) or "session"
    return f"{base[:120]}-v{ordinal:03d}"


class VirtualSessionManager:
    """Topic-shift driven virtual sessions over one gateway chat (plan P7.T4).

    ``topic_detector`` supplies the classification decisions,
    ``pipeline`` the scoped boundary extraction, and the optional
    ``session_store`` the durable boundary records. All collaborators are
    injected; the manager holds per-parent-session state (active virtual
    session, detection cadence counters, virtual id ordinals).
    """

    def __init__(
        self,
        topic_detector: TopicDetector,
        pipeline: MemoryPipeline,
        session_store: SessionStore | None = None,
        compactor: SessionCompactor | None = None,
    ) -> None:
        self._topic_detector = topic_detector
        self._pipeline = pipeline
        self._session_store = session_store
        self._compactor = compactor
        self._active: dict[str, VirtualSession] = {}
        self._since_check: dict[str, int] = {}
        self._ordinals: dict[str, int] = {}

    # ── session access ─────────────────────────────────────────────────────
    def active_session(self, session_id: str) -> VirtualSession:
        """The active virtual session of ``session_id`` (created on demand).

        A first access creates an empty, topic-less virtual session — its
        topic is established by the next :meth:`feed` (session-start
        detection).
        """
        session = self._active.get(session_id)
        if session is None:
            session = self._start_session(session_id)
        return session

    # ── per-turn feed ──────────────────────────────────────────────────────
    def feed(
        self,
        session_id: str,
        turn_id: str,
        turn_text: str,
        every_n_messages: int = DEFAULT_EVERY_N_MESSAGES,
    ) -> ShiftDecision:
        """Record one turn; every N messages run the topic-shift check.

        The turn is appended to the active virtual session. On a detected
        shift the active session is closed (persisted boundary + scoped
        extraction over exactly its turns) and a new virtual session starts
        under the detected topic; the returned :class:`ShiftDecision`
        carries the closed session. ``every_n_messages`` counts messages
        since the last detection (the establishing call included); values
        below 1 raise :class:`ValueError`.
        """
        if every_n_messages < 1:
            raise ValueError(
                f"every_n_messages must be >= 1, got {every_n_messages}"
            )
        session = self.active_session(session_id)
        session.turn_ids.append(turn_id)
        session.turn_texts.append(turn_text)

        if not session.topic:
            # Session start (fresh or after a boundary): establish the topic
            # from the opening turn — the detector's no-prev-topic case.
            result = self._topic_detector.detect_shift(
                list(session.turn_texts), prev_topic=None
            )
            session.topic = result.topic
            self._since_check[session_id] = 0
            return ShiftDecision(
                shifted=False, topic=session.topic, confidence=result.confidence
            )

        self._since_check[session_id] += 1
        if self._since_check[session_id] < every_n_messages:
            return ShiftDecision(shifted=False, topic=session.topic, confidence=1.0)

        self._since_check[session_id] = 0
        result = self._topic_detector.detect_shift(
            list(session.turn_texts), prev_topic=session.topic or None
        )
        if not result.shifted:
            return ShiftDecision(
                shifted=False, topic=session.topic, confidence=result.confidence
            )
        closed = self._close_session(session, reason=TOPIC_SHIFT_REASON)
        new_session = self._start_session(session_id, topic=result.topic)
        logger.info(
            "session manager: topic shift in %s (%s -> %s); closed virtual "
            "session %s (%d turn(s)) and started %s",
            session_id,
            session.topic,
            result.topic,
            closed.session_id,
            len(closed.turn_ids),
            new_session.session_id,
        )
        return ShiftDecision(
            shifted=True,
            topic=result.topic,
            confidence=result.confidence,
            closed_session=closed,
        )

    # ── explicit override (/session new contract, commands wave wiring) ────
    def force_new_session(
        self, session_id: str, reason: str = SESSION_NEW_REASON
    ) -> VirtualSession | None:
        """Close the active virtual session WITHOUT a topic shift.

        The ``/session new`` contract (the slash-command wiring lives in the
        commands wave): the boundary is created immediately — persisted with
        ``close_reason=reason`` and extracted over exactly the closed
        session's turns — even when the detector would report a
        continuation. Returns the closed session, or ``None`` when the
        parent session had no turns to close (an empty active session is
        ensured either way).
        """
        session = self.active_session(session_id)
        if not session.turn_ids:
            logger.info(
                "session manager: %s on %s has no turns; nothing to close",
                reason,
                session_id,
            )
            return None
        closed = self._close_session(session, reason=reason)
        self._start_session(session_id)  # topic established by the next feed
        logger.info(
            "session manager: explicit boundary in %s (reason: %s); closed "
            "virtual session %s (%d turn(s))",
            session_id,
            reason,
            closed.session_id,
            len(closed.turn_ids),
        )
        return closed

    # ── end-of-session close (plan P7.T5, additive) ─────────────────────────
    def close_current(
        self, session_id: str, reason: str = SESSION_CLOSE_REASON
    ) -> VirtualSession | None:
        """Close the active virtual session WITHOUT starting a replacement.

        The P7.T5 end-of-session close (the gateway chat is ending or the
        session must be compacted as a whole): the boundary is persisted with
        ``close_reason=reason``, the session's turns are compacted into a
        Reference-type summary entry (when a compactor is wired — trace range
        built from the session's recorded turn ids), and the scoped boundary
        extraction runs over exactly the closed session's turns (which, with
        a compactor, now lead with the fresh compaction summary). Returns the
        closed session, or ``None`` when the parent session had no turns to
        close (an empty active session is ensured either way). Never raises
        for memory-side failures (plan Section 5.5); the boundary stays
        closed even when compaction/extraction fail.
        """
        session = self.active_session(session_id)
        if not session.turn_ids:
            logger.info(
                "session manager: %s on %s has no turns; nothing to close",
                reason,
                session_id,
            )
            return None
        closed = self._close_session(session, reason=reason)
        logger.info(
            "session manager: end-of-session close in %s (reason: %s); closed "
            "virtual session %s (%d turn(s)); next turn starts a fresh session",
            session_id,
            reason,
            closed.session_id,
            len(closed.turn_ids),
        )
        return closed

    # ── internals ──────────────────────────────────────────────────────────
    def _start_session(
        self, session_id: str, topic: str = ""
    ) -> VirtualSession:
        """Start a new virtual session for ``session_id`` and make it active."""
        self._ordinals[session_id] = self._ordinals.get(session_id, 0) + 1
        session = VirtualSession(
            session_id=_virtual_session_id(session_id, self._ordinals[session_id]),
            topic=topic,
            started_at=datetime.now(UTC),
        )
        self._active[session_id] = session
        self._since_check[session_id] = 0
        return session

    def _close_session(
        self, session: VirtualSession, *, reason: str
    ) -> VirtualSession:
        """Close ``session``: persist the boundary, compact, then extract.

        Persistence goes FIRST (the boundary record must survive even if the
        memory-side work fails); then — with a compactor wired (plan P7.T5) —
        the session's turns are compacted into a Reference summary entry
        whose trace range is built from the session's recorded turn ids; the
        extraction (the boundary MARKER —
        :meth:`MemoryPipeline.extract_session_scope` over exactly this
        session's turns) runs LAST so its context leads with the fresh
        compaction summary (DSH pattern, module docstring). Both memory-side
        steps are best-effort: a failure is logged and the boundary stays
        closed (plan Section 5.5 — memory never breaks the turn).
        """
        closed_at = datetime.now(UTC)
        if self._session_store is not None:
            self._session_store.save_session(
                session.session_id,
                {
                    "virtual": True,
                    "session_id": session.session_id,
                    "topic": session.topic,
                    "started_at": session.started_at.isoformat(),
                    "closed_at": closed_at.isoformat(),
                    "turn_ids": list(session.turn_ids),
                    "close_reason": reason,
                },
            )
        if session.turn_ids:
            self._compact_session(session)
            try:
                self._pipeline.extract_session_scope(
                    session.session_id, session.turns()
                )
            except Exception:
                logger.warning(
                    "session manager: boundary extraction for %s failed; the "
                    "boundary stays closed (plan 5.5)",
                    session.session_id,
                    exc_info=True,
                )
        return session

    # ── compaction (plan P7.T5, additive) ──────────────────────────────────
    def _compact_session(self, session: VirtualSession) -> None:
        """Compact ``session``'s turns into a Reference summary entry.

        The trace range is built from the session's RECORDED turn ids
        (``"<first>..<last>"`` — the plan P7.T5 provenance the summary points
        to). Never raises: a compaction failure is logged and the close
        continues with the extraction (plan Section 5.5).
        """
        if self._compactor is None:
            return
        trace_range = f"{session.turn_ids[0]}..{session.turn_ids[-1]}"
        try:
            entry = self._compactor.compact(
                session.session_id, session.turns(), trace_range=trace_range
            )
        except Exception:
            logger.warning(
                "session manager: compaction of virtual session %s failed; "
                "closing without a summary (plan 5.5)",
                session.session_id,
                exc_info=True,
            )
            return
        logger.info(
            "session manager: compacted virtual session %s into entry %s "
            "(trace range %s)",
            session.session_id,
            entry.id,
            trace_range,
        )
