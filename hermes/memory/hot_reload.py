"""MemoryHotReloader: segmented system prompt with selective invalidation (plan P7.T1).

Phase 7 fixes the frozen-snapshot breakage on long sessions: memory updates
must become visible MID-session without destroying the per-conversation
prompt prefix cache. The prompt is therefore SEGMENTED — one segment per
memory entry, each wrapped in stable markers::

    <mem-segment id="2026-11-21-deploy-token-a1b2c3d4">entry text</mem-segment>

plus fixed non-memory segments (by default the stable user profile and the
memory index; callers may substitute their own via ``segments``).

The prefix-cache contract (plan Phase 7 trade-off note): only invalidated
segments change between two renders. :meth:`MemoryHotReloader.render_prompt`
returns the text of EVERY segment, but when the caller passes the
invalidated set, non-invalidated segments are served from the last rendered
bytes — byte-identical across renders even if the store changed behind the
reloader's back (the caller re-injects only the invalidated segments, so the
model-visible prompt prefix stays cacheable; plan acceptance: "other
segments unchanged (prefix cache preserved)").

Reading model (deliberately simple, per plan P7.T1):

- :meth:`segment_layout` renders every segment from the CURRENT store state
  and records it as the reference ("last rendered") bytes.
- :meth:`invalidate` marks one entry's segment dirty: it reads the FRESH
  content (never the cache), hashes it against the last rendered bytes and
  returns an :class:`InvalidatedSegment` receipt (old/new sha256). The
  receipt lands in the pending set that
  :meth:`ContextInjector.build_dynamic_context` surfaces to the agent
  harness as ``invalidated_segments`` (see :func:`hermes.memory.injector.
  wire_hot_reload`).
- :meth:`reload_segment` re-reads ONE segment fresh from the
  :class:`~hermes.memory.store.MemoryStore` (``store.get(entry_id)`` — the
  plan acceptance "reload reads fresh content from MemoryStore"; fixed
  segments re-run their renderer, which reads the backing files fresh).
- :meth:`render_prompt` re-renders; invalidated (and newly appeared)
  segments come fresh, everything else from the reference bytes.

The store is only ever READ here — invalidation is driven by callers (the
memory tool wiring lands in P7.T2), keeping this module read-only and free
of write-lock coupling.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import INDEX_NAME, MemoryStore

__all__ = [
    "INDEX_SEGMENT_ID",
    "InvalidatedSegment",
    "MemoryHotReloader",
    "STABLE_PROFILE_SEGMENT_ID",
    "SegmentRenderer",
]

#: Segment id of the default stable user-profile segment (plan Phase 5 step 6
#: layout: ``<memory-repo>/global/user-profile.md``).
STABLE_PROFILE_SEGMENT_ID: Final[str] = "stable_profile"

#: Segment id of the default memory-index segment (the store's
#: :data:`~hermes.memory.store.INDEX_NAME` document).
INDEX_SEGMENT_ID: Final[str] = "index"

#: Stable markers wrapping every per-entry segment (plan Phase 7 step 1).
_SEGMENT_OPEN_TEMPLATE: Final[str] = '<mem-segment id="{segment_id}">'
_SEGMENT_CLOSE: Final[str] = "</mem-segment>"

#: Renderer of one fixed segment: called with the store, returns the text.
SegmentRenderer = Callable[[MemoryStore], str]


def _read_file_text(path: Path) -> str:
    """Best-effort text read; ``""`` when the file does not exist."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class InvalidatedSegment:
    """Receipt of one :meth:`MemoryHotReloader.invalidate` call (plan P7.T1).

    ``old_hash`` is the sha256 of the segment as last rendered (``""`` when
    the segment had never been rendered before — e.g. an entry created after
    the reference snapshot); ``new_hash`` the sha256 of the fresh content.
    Equal hashes mean the update did not change the rendered segment.
    """

    segment_id: str
    old_hash: str
    new_hash: str

    @property
    def changed(self) -> bool:
        """True when the fresh content differs from the last rendered bytes."""
        return self.old_hash != self.new_hash


def _sha256(text: str) -> str:
    """sha256 hex digest of ``text`` (UTF-8) — the segment hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wrap_segment(segment_id: str, text: str) -> str:
    """Wrap ``text`` in the stable per-entry segment markers (plan Phase 7)."""
    return f"{_SEGMENT_OPEN_TEMPLATE.format(segment_id=segment_id)}{text}{_SEGMENT_CLOSE}"


class MemoryHotReloader:
    """Segmented prompt view over a :class:`~hermes.memory.store.MemoryStore`.

    Segments are PER MEMORY ENTRY (``store.list()`` order) plus the fixed
    non-memory segments. ``segments`` maps fixed segment ids to renderers
    (``store -> text``); when ``None`` the defaults
    (:data:`STABLE_PROFILE_SEGMENT_ID` reading
    ``<root>/global/user-profile.md`` and :data:`INDEX_SEGMENT_ID` reading
    ``<root>/MEMORY.md``, both ``""`` when the file is missing) are used.
    Entry segments always take their id and text from the store, so entries
    added mid-session appear automatically on the next full render.

    The reloader keeps the LAST RENDERED bytes per segment as the
    prefix-cache reference; it never caches store content for serving —
    fresh reads always go through :meth:`reload_segment` /
    :meth:`segment_layout`.
    """

    def __init__(
        self,
        store: MemoryStore,
        segments: dict[str, SegmentRenderer] | None = None,
    ) -> None:
        self._store = store
        if segments is None:
            self._fixed_renderers: dict[str, SegmentRenderer] = {
                STABLE_PROFILE_SEGMENT_ID: self._render_stable_profile,
                INDEX_SEGMENT_ID: self._render_index,
            }
        else:
            self._fixed_renderers = dict(segments)
        #: Last rendered bytes per segment id (the prefix-cache reference).
        self._reference: dict[str, str] = {}
        #: Segment ids invalidated since the last :meth:`clear_invalidations`.
        self._pending: set[str] = set()

    # ── default fixed segments ─────────────────────────────────────────────
    @staticmethod
    def _render_stable_profile(store: MemoryStore) -> str:
        """The stable profile segment: the global user profile document."""
        return _read_file_text(store.root / "global" / "user-profile.md")

    @staticmethod
    def _render_index(store: MemoryStore) -> str:
        """The index segment: the store's ``MEMORY.md`` index document."""
        return _read_file_text(store.root / INDEX_NAME)

    # ── segment enumeration ────────────────────────────────────────────────
    def _current_segments(self) -> dict[str, str]:
        """Fresh text of EVERY segment: fixed ids first, then store entries.

        Entry segments come straight from ``store.list()`` (one segment per
        entry, wrapped in the stable markers), so entries added or removed
        mid-session change the segment set on the next full render.
        """
        rendered: dict[str, str] = {}
        for segment_id, renderer in self._fixed_renderers.items():
            rendered[segment_id] = renderer(self._store)
        for entry in self._store.list():
            rendered[entry.id] = _wrap_segment(entry.id, entry.text)
        return rendered

    @staticmethod
    def _segment_hash(text: str) -> str:
        """Public-shape hash of a segment's rendered text."""
        return _sha256(text)

    # ── public API (plan P7.T1) ────────────────────────────────────────────
    def segment_layout(self) -> dict[str, str]:
        """Fresh text for every segment; recorded as the reference bytes.

        Returns a NEW dict mapping segment id -> rendered text (fixed
        segments first, then entries in ``store.list()`` order). The returned
        bytes become the prefix-cache reference the next
        :meth:`render_prompt` / :meth:`invalidate` compares against.
        """
        fresh = self._current_segments()
        self._reference = dict(fresh)
        return dict(fresh)

    def invalidate(self, entry_id: str) -> InvalidatedSegment:
        """Mark the segment of ``entry_id`` dirty; return the hash receipt.

        Reads the FRESH content from the store (never the reference cache),
        hashes it against the last rendered bytes and records the segment in
        the pending set that :meth:`pending_invalidations` surfaces (the
        agent harness re-injects exactly these segments). Unknown segment
        ids raise :class:`KeyError`; a segment never rendered before reports
        ``old_hash=""``. The reference for the segment moves to the fresh
        bytes, so a second :meth:`invalidate` without an intervening update
        yields equal hashes.
        """
        fresh_text = self.reload_segment(entry_id)
        old_text = self._reference.get(entry_id)
        receipt = InvalidatedSegment(
            segment_id=entry_id,
            old_hash=self._segment_hash(old_text) if old_text is not None else "",
            new_hash=self._segment_hash(fresh_text),
        )
        self._reference[entry_id] = fresh_text
        self._pending.add(entry_id)
        return receipt

    def reload_segment(self, segment_id: str) -> str:
        """Re-read ONE segment fresh — bypassing the reference cache.

        Fixed segments re-run their renderer (fresh file reads); entry
        segments go through ``store.get(segment_id)`` and are returned
        wrapped in the stable markers. Unknown ids raise :class:`KeyError`.
        The reference cache is untouched — this is a pure fresh read (plan
        acceptance: "reload reads fresh content from MemoryStore").
        """
        renderer = self._fixed_renderers.get(segment_id)
        if renderer is not None:
            return renderer(self._store)
        entry: MemoryEntry | None = self._store.get(segment_id)
        if entry is None:
            raise KeyError(f"unknown segment id: {segment_id!r}")
        return _wrap_segment(entry.id, entry.text)

    def render_prompt(self, invalidated: set[str] | None = None) -> dict[str, str]:
        """Text for every segment, honoring the prefix-cache contract.

        Without ``invalidated`` every segment is rendered fresh from the
        current store state and becomes the new reference. With an
        ``invalidated`` set, exactly those segments (plus segments that had
        no previous reference — new entries) come fresh; every other segment
        is served from its last rendered bytes, so its text is BYTE-IDENTICAL
        across renders (plan Phase 7 acceptance: "other segments unchanged
        (prefix cache preserved)") — even when the store changed behind the
        reloader's back and the caller has not invalidated the change yet.
        The caller re-injects only the invalidated segments.
        """
        fresh = self._current_segments()
        if not invalidated:
            self._reference = dict(fresh)
            return dict(fresh)
        rendered: dict[str, str] = {}
        for segment_id, fresh_text in fresh.items():
            if segment_id in invalidated or segment_id not in self._reference:
                rendered[segment_id] = fresh_text
            else:
                rendered[segment_id] = self._reference[segment_id]
        self._reference = dict(rendered)
        return rendered

    # ── pending-invalidation surface (injector integration) ───────────────
    def pending_invalidations(self) -> list[str]:
        """Segment ids invalidated but not yet cleared, in stable order."""
        return sorted(self._pending)

    def clear_invalidations(self, segment_ids: set[str] | None = None) -> None:
        """Consume pending invalidations (harness re-injected them).

        ``None`` clears everything; a set clears exactly those ids (unknown
        ids are ignored).
        """
        if segment_ids is None:
            self._pending.clear()
        else:
            self._pending.difference_update(segment_ids)
