"""SessionCompactor: compaction summaries as first-class memory citizens (plan P7.T5).

Long sessions are compacted into a short summary stored as a **Reference-type
memory entry** whose source points to the trace range it summarizes — "memory
points, doesn't mirror" (plan Phase 7 / DSH pattern): the entry is a pointer
with provenance, never a copy of the turns. Compaction summaries are FIRST-CLASS
extraction inputs: :meth:`SessionCompactor.compact` runs when a virtual session
closes (or the session exceeds N turns), and the extraction flow
(:mod:`hermes.memory.pipeline`, additive P7.T5 wiring) puts the session's
compaction summary FIRST in the extractor's context window with the raw turns
marked secondary.

Entry contract (all deterministic, mutation-testable)
----------------------------------------------------------------------
The stored entry's ``text`` BEGINS with
``"Compaction summary for session <id>"`` (the header, then a blank line, then
the summary body) so consumers can recognize it textually as well as by tags.
Its trust card carries the provenance:

- ``tags``: :data:`COMPACTION_TAG` (``"compaction"``),
  ``"session:<session_id>"`` (the lookup key used by
  :func:`find_compaction_summary`), and ``"trace_range:<first>..<last>"``;
- one ``agent-inferred`` verification event with ``verifier
  "session-compactor"`` and evidence
  ``"source points to trace range <range> — memory points, doesn't mirror"``
  (the deliberate low-weight provenance for machine-derived content, plan
  Section 2.4 weight table);
- persistence goes through :meth:`MemoryStore.add` — one atomic markdown file
  plus ONE git commit (plan P1.T3), so the summary is git-committed like any
  other entry.

Summary text (:func:`summary_text`, pure and store-free)
----------------------------------------------------------------------
The injected ``summarizer`` (``Callable[[str], str]``) receives the raw
transcript (turn texts joined with newlines, oldest first) and returns its
summary; a raising summarizer degrades to the deterministic extractive
fallback (plan Section 5.5 — compaction must never break a session close).
The fallback takes the FIRST SENTENCE of each turn and joins them; the result
— and any summarizer's output — is clamped to ``char_limit`` (truncated with
an ellipsis), so ``len(summary) <= char_limit`` holds for ANY input and ANY
summarizer (property-tested, plan P7.T5).

P7.T5 notes: the trace range may be passed explicitly (the virtual session
manager builds ``"<first_turn_id>..<last_turn_id>"`` from the recorded turn
ids) or is derived from ``turns`` when omitted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

from hermes.memory.entry import MemoryEntry
from hermes.memory.types import VerificationSource

if TYPE_CHECKING:
    from hermes.memory.store import MemoryStore

logger = logging.getLogger(__name__)

__all__ = [
    "COMPACTION_TAG",
    "DEFAULT_CHAR_LIMIT",
    "SESSION_TAG_PREFIX",
    "SUMMARY_TEXT_PREFIX",
    "TRACE_RANGE_TAG_PREFIX",
    "SessionCompactor",
    "find_compaction_summary",
    "summary_text",
]

#: Tag marking a compaction summary entry (the extraction lookup filter).
COMPACTION_TAG: Final[str] = "compaction"

#: Tag prefix carrying the summarized session id (``session:<id>``).
SESSION_TAG_PREFIX: Final[str] = "session:"

#: Tag prefix carrying the trace-range provenance (``trace_range:<f>..<l>``).
TRACE_RANGE_TAG_PREFIX: Final[str] = "trace_range:"

#: The stored entry's text begins with this + ``" <session_id>"`` (plan P7.T5).
SUMMARY_TEXT_PREFIX: Final[str] = "Compaction summary for session"

#: Plan-default summary budget in characters (task P7.T5 signature default).
DEFAULT_CHAR_LIMIT: Final[int] = 1500

#: Provenance source of the machine-derived summary (plan Section 2.4).
_SUMMARY_SOURCE: Final[VerificationSource] = "agent-inferred"

#: Verifier stamped on the summary's verification event.
_SUMMARY_VERIFIER: Final[str] = "session-compactor"

#: Evidence line template — the "memory points, doesn't mirror" contract.
_EVIDENCE_TEMPLATE: Final[str] = (
    "source points to trace range {trace_range} — memory points, doesn't mirror"
)

#: Ellipsis appended by the char clamp.
_ELLIPSIS: Final[str] = "..."

#: Sentence enders for the extractive fallback (ASCII + typographic + CJK).
_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?…。！？])\s+")


def _transcript(turns: Sequence[tuple[str, str]]) -> str:
    """Raw transcript handed to the summarizer: turn texts, oldest first."""
    return "\n".join(text for _turn_id, text in turns)


def _first_sentence(text: str) -> str:
    """The first sentence of ``text`` (whole text when no boundary exists)."""
    one_line = " ".join(text.split())
    parts = _SENTENCE_SPLIT_RE.split(one_line, maxsplit=1)
    return parts[0].strip() if parts else one_line.strip()


def _clamp(text: str, char_limit: int) -> str:
    """Truncate ``text`` to at most ``char_limit`` characters (ellipsis)."""
    if len(text) <= char_limit:
        return text
    if char_limit <= len(_ELLIPSIS):
        return text[:char_limit]
    return text[: char_limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def summary_text(
    turns: Sequence[tuple[str, str]],
    char_limit: int = DEFAULT_CHAR_LIMIT,
    summarizer: Callable[[str], str] | None = None,
) -> str:
    """Deterministic compaction summary for ``turns`` (plan P7.T5; pure).

    ``summarizer`` (when given) receives the raw transcript
    (:func:`_transcript` format) and returns the summary; a summarizer that
    raises is logged and replaced by the deterministic extractive fallback
    (plan Section 5.5). The fallback: the first sentence of each turn joined
    with spaces. EVERY output — injected or fallback — is clamped to
    ``char_limit`` (truncated with an ellipsis), so ``len(result) <=
    char_limit`` always holds and the function never raises for any turn list
    (empty lists summarize to ``""``; unicode and huge turns are safe).

    Raises :class:`ValueError` only for an unusable ``char_limit`` (not a
    positive int) — callers may rely on the clamp otherwise.
    """
    if isinstance(char_limit, bool) or not isinstance(char_limit, int) or char_limit < 1:
        raise ValueError(f"char_limit must be a positive int, got {char_limit!r}")
    if summarizer is not None:
        try:
            return _clamp(summarizer(_transcript(turns)), char_limit)
        except Exception:
            logger.warning(
                "compaction: injected summarizer failed; using the deterministic "
                "extractive fallback (plan 5.5)",
                exc_info=True,
            )
    sentences = [_first_sentence(text) for _turn_id, text in turns]
    return _clamp(" ".join(sentence for sentence in sentences if sentence), char_limit)


def find_compaction_summary(store: MemoryStore, session_id: str) -> MemoryEntry | None:
    """The LATEST compaction summary stored for ``session_id``, or ``None``.

    Looks up entries tagged :data:`COMPACTION_TAG` and keeps the one carrying
    ``"session:<session_id>"``; :meth:`MemoryStore.list` returns entries in
    creation order, so the LAST match is the latest compaction (a session may
    be compacted more than once). Never raises for an empty store.
    """
    session_tag = f"{SESSION_TAG_PREFIX}{session_id}"
    latest: MemoryEntry | None = None
    for entry in store.list(tag=COMPACTION_TAG):
        if session_tag in entry.tags:
            latest = entry
    return latest


class SessionCompactor:
    """Compacts a closed session's turns into a Reference memory entry (P7.T5).

    ``store`` persists the summary (git-committed via the store), ``summarizer``
    optionally replaces the deterministic extractive fallback, and
    ``char_limit`` bounds the summary body (default 1500, plan P7.T5 signature).
    """

    def __init__(
        self,
        store: MemoryStore,
        summarizer: Callable[[str], str] | None = None,
        char_limit: int = DEFAULT_CHAR_LIMIT,
    ) -> None:
        if isinstance(char_limit, bool) or not isinstance(char_limit, int) or char_limit < 1:
            raise ValueError(f"char_limit must be a positive int, got {char_limit!r}")
        self._store = store
        self._summarizer = summarizer
        self._char_limit = char_limit

    # ── public API ────────────────────────────────────────────────────────
    def compact(
        self,
        session_id: str,
        turns: list[tuple[str, str]],
        trace_range: str | None = None,
        *,
        context: str = "global",
    ) -> MemoryEntry:
        """Summarize ``turns`` and store the summary as a Reference entry.

        ``turns`` are the session's ``(turn_id, turn_text)`` pairs, oldest
        first (non-empty — there is nothing to compact otherwise, and the
        call raises :class:`ValueError`); ``trace_range`` names the turn range
        the summary points to (``"<first>..<last>"``; derived from ``turns``
        when omitted — the virtual session manager builds it from the
        recorded turn ids). The summary body comes from
        :func:`summary_text` (injected summarizer or the extractive fallback,
        clamped to ``char_limit``) and the entry is stored via
        :meth:`MemoryStore.add` in ``context`` (default ``global``) — one git
        commit, with the ``compaction`` / ``session:<id>`` /
        ``trace_range:<range>`` tags and the trace-range evidence line as the
        trust-card provenance (module docstring).
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
        if not turns:
            raise ValueError("cannot compact an empty session (no turns to summarize)")
        resolved_range = (
            trace_range if trace_range else f"{turns[0][0]}..{turns[-1][0]}"
        )
        summary = summary_text(turns, self._char_limit, self._summarizer)
        entry = self._store.add(
            text=f"{SUMMARY_TEXT_PREFIX} {session_id}\n\n{summary}",
            type="Reference",
            context=context,
            source=_SUMMARY_SOURCE,
            verifier=_SUMMARY_VERIFIER,
            evidence=_EVIDENCE_TEMPLATE.format(trace_range=resolved_range),
            tags=[
                COMPACTION_TAG,
                f"{SESSION_TAG_PREFIX}{session_id}",
                f"{TRACE_RANGE_TAG_PREFIX}{resolved_range}",
            ],
            session_id=session_id,
        )
        logger.info(
            "compaction: stored summary entry %s for session %s (%d turn(s), "
            "trace range %s, %d summary chars)",
            entry.id,
            session_id,
            len(turns),
            resolved_range,
            len(summary),
        )
        return entry
