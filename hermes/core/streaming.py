"""hermes-v2 internal-marker streaming filter (plan P21.T10).

Fixes upstream issue #157239: internal sentinel markers (e.g.
``"<NO_REPLY>"``) leaked into gateway-delivered messages when the sentinel
arrived split across stream chunks or spelled in an unexpected letter case.

This module owns the ONE registry of internal markers
(:data:`INTERNAL_MARKERS`) and the single filtering implementation used
before any user-facing delivery:

- :func:`strip_internal_markers` — filter a complete message;
- :func:`filter_stream` — filter a chunked response stream, catching markers
  that straddle chunk boundaries (or that would be *fused together* by the
  chunk seam) by emitting only a prefix whose tail cannot be completed into
  a marker by future chunks;
- :func:`contains_internal_marker` — the shared case-insensitive test.

Matching is case-insensitive over every documented marker. Cleanup covers
the artifacts removal leaves behind: doubled whitespace runs are compacted
(a run containing a newline becomes one newline, otherwise one space) and
empty bracket leftovers such as ``"<>"/"[]"`` (including whitespace-padded
ones) are removed. Cleanup steps re-run to a fixpoint so they cannot
themselves re-create a marker.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

__all__ = [
    "INTERNAL_MARKERS",
    "MAX_MARKER_LENGTH",
    "contains_internal_marker",
    "filter_stream",
    "strip_internal_markers",
]

#: The documented registry of internal sentinel markers (issue #157239).
#: Exactly these strings (case-insensitively) must never reach a user:
#:
#: - ``"<NO_REPLY>"`` — the model declined to answer; the whole message is
#:   suppressed by the delivery gate.
#: - ``"[NO_REPLY]"`` — bracketed variant emitted by older prompt revisions.
#: - ``"<HERMES_INTERNAL>"`` — wrapper for internal-only annotations.
#: - ``"<<INTERNAL_ERROR>>"`` — crash/telemetry banner meant for logs.
INTERNAL_MARKERS: frozenset[str] = frozenset({
    "<NO_REPLY>",
    "[NO_REPLY]",
    "<HERMES_INTERNAL>",
    "<<INTERNAL_ERROR>>",
})

#: Length of the longest registered marker; :func:`filter_stream` holds back
#: ``MAX_MARKER_LENGTH - 1`` characters between chunks.
MAX_MARKER_LENGTH: int = max(len(marker) for marker in INTERNAL_MARKERS)

# Longest-first alternation so overlapping spellings (e.g. a marker nested in
# doubled angle brackets) match the outermost/full form in one pass.
_MARKER_RE = re.compile(
    "|".join(re.escape(marker) for marker in sorted(INTERNAL_MARKERS, key=len, reverse=True)),
    re.IGNORECASE,
)
_EMPTY_BRACKET_RE = re.compile(r"<\s*>|\[\s*\]|\{\s*\}|\(\s*\)")
_DOUBLED_WS_RE = re.compile(r"\s{2,}")


def _collapse_whitespace(match: re.Match[str]) -> str:
    """One run of doubled whitespace -> one newline if it spans lines, else one space."""
    return "\n" if "\n" in match.group(0) else " "


def _clean_fragment(text: str) -> str:
    """Remove markers and their cleanup artifacts from ``text`` to a fixpoint.

    The three passes re-run until stable: removing an empty bracket or a
    whitespace run can fuse previously separated characters into a marker
    (e.g. ``"<NO_RE[]PLY>"``), and the fixpoint guarantees the returned text
    contains no marker, no empty-bracket leftover and no doubled whitespace.
    Every pass strictly shrinks the text, so the loop terminates.
    """
    cleaned = text
    while True:
        next_pass = _MARKER_RE.sub(" ", cleaned)
        next_pass = _EMPTY_BRACKET_RE.sub("", next_pass)
        next_pass = _DOUBLED_WS_RE.sub(_collapse_whitespace, next_pass)
        if next_pass == cleaned:
            return cleaned
        cleaned = next_pass


def contains_internal_marker(text: str) -> bool:
    """True when ``text`` contains any registered marker, case-insensitively."""
    return _MARKER_RE.search(text) is not None


def strip_internal_markers(text: str) -> str:
    """Remove every internal marker from a complete message (issue #157239).

    Case-insensitive; also compacts doubled whitespace and removes empty
    bracket leftovers. Leading/trailing whitespace is left for the caller
    (the delivery gate trims and suppresses blank results).
    """
    return _clean_fragment(text)


#: Case-insensitive proper prefixes of the registered markers. A buffer tail
#: matching one of these could still be completed into a full marker by a
#: future chunk, so it must never be emitted yet.
_MARKER_PREFIXES: frozenset[str] = frozenset(
    marker.lower()[:cut]
    for marker in INTERNAL_MARKERS
    for cut in range(1, len(marker))
)
_MAX_PREFIX_LEN = MAX_MARKER_LENGTH - 1


def _safe_emit_len(text: str) -> int:
    """Longest prefix of ``text`` whose tail cannot fuse into a marker.

    A marker can be created by *fusion* at an emission seam, not only by
    passing through whole: cleanup steps remove characters (empty brackets),
    so ``"<NO_RE"`` emitted now plus ``"PLY>"`` arriving later would join
    into a fresh ``"<NO_REPLY>"`` in the concatenated output. The emitted
    prefix therefore must not END with any proper marker prefix; shrink the
    emission until that holds.
    """
    emit_len = len(text)
    while emit_len > 0:
        lowered = text[:emit_len].lower()
        longest_bad = 0
        for size in range(1, min(emit_len, _MAX_PREFIX_LEN) + 1):
            if lowered[-size:] in _MARKER_PREFIXES:
                longest_bad = size
        if longest_bad == 0:
            break
        emit_len -= longest_bad
    return emit_len


def filter_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Stream ``chunks`` through the internal-marker filter.

    Applies :func:`strip_internal_markers` semantics across chunk
    boundaries. Each chunk is appended to the (already cleaned) buffer and
    the whole buffer is re-cleaned, so cleanup artifacts — including
    characters fused into a *new* marker by the chunk seam — are caught
    before emission. Only a prefix whose tail cannot be completed into a
    marker by future chunks is emitted (see :func:`_safe_emit_len`); the
    final flush emits whatever is still buffered. A stream made purely of
    marker-prefix text may buffer indefinitely (correctness is preserved; it
    simply cannot yet be proven safe to emit).
    """
    buffer = ""
    for chunk in chunks:
        buffer = _clean_fragment(buffer + chunk)
        emit_len = _safe_emit_len(buffer)
        if emit_len:
            yield buffer[:emit_len]
            buffer = buffer[emit_len:]
    if buffer:
        yield buffer
