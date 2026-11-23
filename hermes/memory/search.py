"""BM25 keyword search over memory entries (plan P1.T4).

Ranks entries against a query with :class:`rank_bm25.BM25Okapi`; the
indexed corpus for each entry is its text tokens **plus** its tags' tokens
(a tag match boosts an entry the same way an extra text occurrence would).
Tokenization is lowercased ``\\w+`` runs — no stemming, so the index is
rebuild-cheap and queries are predictable.

Index caching strategy (plan P1.T4 leaves either option open): the index is
**cached per context key** on the :class:`BM25Search` instance and
**invalidated on every mutating :class:`~hermes.memory.store.MemoryStore`
operation** routed through the owning store (add/update/remove/
record_event). Readers that face external writers (another process
committing to the same repository) should call :meth:`invalidate` before
searching; per-store instances that own all writes never see stale results.
Context-filtered searches build (and cache) one index per distinct
``context`` argument so that filtering happens **before** ranking — a
high-scoring entry from another context can never crowd a relevant entry
out of the top-k.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from hermes.memory.entry import MemoryEntry

if TYPE_CHECKING:
    from hermes.memory.store import MemoryStore

__all__ = ["BM25Search", "tokenize"]

#: Query/text tokenization: lowercase word characters (unicode-aware).
_TOKEN_RE = re.compile(r"\w+")

#: Placeholder token for empty documents, keeping BM25's average document
#: length well-defined; normal ``\w+`` queries can never produce it.
_EMPTY_DOC_TOKEN = "\x00empty"


def tokenize(text: str) -> list[str]:
    """Lowercase ``\\w+`` tokens of ``text`` (empty list for pure punctuation)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Search:
    """Cached BM25Okapi ranking over memory entries (plan P1.T4).

    Construct with ``store`` for live store-backed search (cache invalidates
    on store mutations) or with ``entries`` for a static in-memory corpus;
    with neither, ``search`` always returns an empty list.
    """

    def __init__(
        self,
        entries: Sequence[MemoryEntry] | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self._store = store
        self._static_entries = None if entries is None else list(entries)
        # context key (None = all contexts) -> (entries, index); the index is
        # None for an empty corpus (BM25Okapi cannot average zero documents).
        self._indexes: dict[str | None, tuple[list[MemoryEntry], BM25Okapi | None]] = {}

    def invalidate(self) -> None:
        """Drop every cached index; the next ``search`` rebuilds from source."""
        self._indexes.clear()

    def _entries(self) -> list[MemoryEntry]:
        """Corpus source for this engine: store listing or the static corpus."""
        if self._store is not None:
            return self._store.list()
        if self._static_entries is not None:
            return self._static_entries
        return []

    def _entry_tokens(self, entry: MemoryEntry) -> list[str]:
        """Indexed tokens for ``entry``: text tokens plus tag tokens."""
        tokens = tokenize(entry.text)
        for tag in entry.tags:
            tokens.extend(tokenize(tag))
        return tokens or [_EMPTY_DOC_TOKEN]

    def _ensure_index(
        self, context: str | None
    ) -> tuple[list[MemoryEntry], BM25Okapi | None]:
        """Cached (context-filtered) corpus + index, built on first use."""
        cached = self._indexes.get(context)
        if cached is not None:
            return cached
        entries = [
            entry
            for entry in self._entries()
            if context is None or entry.context == context
        ]
        corpus = [self._entry_tokens(entry) for entry in entries]
        index: BM25Okapi | None = BM25Okapi(corpus) if corpus else None
        self._indexes[context] = (entries, index)
        return entries, index

    def search(
        self,
        query: str,
        top_k: int = 10,
        context: str | None = None,
    ) -> list[MemoryEntry]:
        """Rank entries for ``query``; returns at most ``top_k`` matching entries.

        An entry counts as matching when it shares at least one token with
        the query (its BM25 score is then non-zero — possibly negative when
        BM25Okapi's epsilon floor kicks in for terms present in most
        documents); entries matching no query term score exactly 0.0 and are
        never returned. An empty query (no tokens after tokenization) or
        ``top_k <= 0`` also yields an empty list. Ties are broken by entry
        id for deterministic results.
        """
        tokens = tokenize(query)
        if not tokens or top_k <= 0:
            return []
        entries, index = self._ensure_index(context)
        if index is None or not entries:
            return []
        scores = index.get_scores(tokens)
        ranked = sorted(
            (
                (entry, score)
                for entry, score in zip(entries, scores, strict=True)
                if score != 0.0
            ),
            key=lambda pair: (-pair[1], pair[0].id),
        )
        return [entry for entry, _score in ranked[:top_k]]
