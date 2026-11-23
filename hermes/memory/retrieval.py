"""RetrievalEngine: BM25 (+ optional vector re-rank) candidate search (plan P4.T1).

The plan Phase 4 component ``hermes/memory/retrieval.py`` — top-K candidates
for a query, feeding the CBDC gate (P4.T2) and injector (P4.T3).

Scope rule (plan Phase 4 acceptance: *context-aware — only retrieves from
active context + global*): retrieving for context ``X`` searches the union of
``X`` and ``global`` (deduplicated by entry id) and **never** any other
named context; ``context="global"`` searches global only. Filtering happens
BEFORE ranking (one index per scope), so a high-scoring entry from another
context cannot crowd a relevant one out of the top-k.

Why a direct BM25Okapi index instead of ``MemoryStore.search``
-------------------------------------------------------------
``MemoryStore.search`` (plan P1.T4) returns ranked **entries without
scores**, while :class:`RetrievedEntry` carries the score (needed for the
documented vector re-rank and the CBDC gate's downstream reporting). The
engine therefore builds :class:`rank_bm25.BM25Okapi` directly over the
store's listing, mirroring the P1.T4 corpus rules exactly — tokenization is
:func:`hermes.memory.search.tokenize` (imported, not re-implemented) and the
indexed text is the entry's text tokens plus its tags' tokens, with the same
empty-document placeholder idea. This is the documented deviation from the
P2.T2-style "via ``store.search``" hint in the task design.

Caching and freshness: the per-scope index is cached and re-validated on
every :meth:`retrieve` against a cheap ``os.scandir`` fingerprint (name,
size, mtime_ns per fact file — no file content reads; ~1ms per 1000 files on
NTFS), so results never lag behind store mutations (including other
processes committing to the same repository) while warm queries stay far
under the 100ms/1000-entries budget. The store's own invalidation hook
(``MemoryStore._invalidate_search_cache``) is unreachable from outside
without touching the frozen store module, hence the fingerprint approach.

P22.T2 result cache (additive): an optional
:class:`~hermes.memory.cache.MemoryCache` can be injected via
``RetrievalEngine(cache=...)``. When set, :meth:`retrieve` consults it
BEFORE the BM25 pass (keyed on query + scope + ``top_k``) and stores the
computed result list afterwards, so repeated queries within the TTL skip
the entire ranking pipeline — the P22.T2 "cache hit rate >=30% for repeated
queries" acceptance. The cache does NOT watch the store; the invalidation
signal is the store's ``on_update`` hook (plan P7.T2) pointed at
:meth:`invalidate_cache` — ``store.on_update = engine.invalidate_cache`` —
so every committed memory update clears the result cache and the next
retrieve recomputes from the fresh store. ``cache`` is appended last with a
``None`` default, so every pre-P22.T2 constructor call site (positional or
keyword) is unaffected.

Vector hook: ``vector_backend`` is the documented ``sentence-transformers``
seam (plan Phase 4: *BM25 by default, optional vector via
sentence-transformers*). When provided, it is called once with the query and
once per BM25 candidate with the candidate's text, and the BM25 candidate
pool is re-ranked by cosine similarity (the returned :attr:`RetrievedEntry.
score` then carries the cosine, range [-1, 1]). Tests inject fakes or
``None``; nothing here imports sentence-transformers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from hermes.memory.cache import MemoryCache
from hermes.memory.entry import MemoryEntry
from hermes.memory.search import tokenize
from hermes.memory.store import CONTEXTS_DIR, MemoryStore

__all__ = ["RetrievalEngine", "RetrievedEntry"]

#: Placeholder token for token-less documents, keeping BM25's average
#: document length well-defined (mirrors ``hermes.memory.search``; no query
#: of ``\w+`` tokens can ever produce it).
_EMPTY_DOC_TOKEN = "\x00empty"

#: Context names must be a single safe path component (mirrors
#: ``MemoryStore``'s validation — the engine derives on-disk scope paths).
_CONTEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: Key prefix of the P22.T2 result-cache entries this engine writes, so
#: ``cache.invalidate_prefix(_CACHE_KEY_PREFIX)`` can target exactly the
#: retrieval memos of any engine sharing the cache.
_CACHE_KEY_PREFIX = "retrieve::"


def _cache_key(query: str, top_k: int, context: str) -> str:
    """Result-cache key for one retrieve() call (plan P22.T2).

    ``retrieve::<context>::<top_k>::<sha256(query)>`` — the query is hashed
    to keep cache keys small for long prompts; the scope and ``top_k`` stay
    in plain text so a cache shared across engines/scopes can still be
    prefix-invalidated selectively.
    """
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"{_CACHE_KEY_PREFIX}{context}::{top_k}::{query_digest}"


@dataclass(frozen=True)
class RetrievedEntry:
    """One search hit: the entry plus its ranking score (plan P4.T1).

    ``score`` is the BM25 relevance score, or the cosine similarity when a
    ``vector_backend`` re-ranked the candidate pool.
    """

    entry: MemoryEntry
    score: float


#: Embedding seam (plan Phase 4): string -> vector in a consistent space.
VectorBackend = Callable[[str], list[float]]

#: Cache slot per scope: (fingerprint, entries, BM25 index; None = empty corpus).
_ScopeIndex = tuple[tuple[tuple[str, int, int], ...], list[MemoryEntry], BM25Okapi | None]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1] (0.0 when either vector is all-zero)."""
    if len(a) != len(b):
        raise ValueError(
            f"vector backend returned inconsistent embedding sizes: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _facts_dirs(store: MemoryStore, scope: str) -> tuple[str, ...]:
    """Fact directories backing ``scope`` (``global`` or one named context)."""
    if scope == "global":
        return ("global/facts",)
    return (f"{CONTEXTS_DIR}/{scope}/facts",)


def _scope_fingerprint(store: MemoryStore, scope: str) -> tuple[tuple[str, int, int], ...]:
    """Cheap freshness probe for a scope's fact files (no content reads).

    ``(name, size, mtime_ns)`` of every ``*.md`` file under the scope's fact
    directories, sorted; an empty tuple when the directory does not exist
    yet. ``os.scandir`` carries the stat data for free on Windows and costs
    one syscall per file elsewhere.
    """
    fingerprint: list[tuple[str, int, int]] = []
    for rel_dir in _facts_dirs(store, scope):
        directory = store.root / rel_dir
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith(".md") or not entry.is_file():
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    fingerprint.append((entry.name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    return tuple(sorted(fingerprint))


def _entry_tokens(entry: MemoryEntry) -> list[str]:
    """Indexed tokens for ``entry``: text tokens plus tag tokens (P1.T4 rule)."""
    tokens = tokenize(entry.text)
    for tag in entry.tags:
        tokens.extend(tokenize(tag))
    return tokens or [_EMPTY_DOC_TOKEN]


class RetrievalEngine:
    """BM25 candidate retrieval over a memory store, with a vector seam (P4.T1).

    ``store`` is the live :class:`~hermes.memory.store.MemoryStore`;
    ``vector_backend`` (optional) re-ranks the BM25 candidates by cosine —
    the sentence-transformers hook, injectable for tests. ``cache``
    (optional, plan P22.T2) memoizes retrieve() results — see the module
    docstring for the invalidation contract.
    """

    def __init__(
        self,
        store: MemoryStore,
        vector_backend: VectorBackend | None = None,
        cache: MemoryCache[list[RetrievedEntry]] | None = None,
    ) -> None:
        self._store = store
        self._vector_backend = vector_backend
        self._indexes: dict[str, _ScopeIndex] = {}
        #: P22.T2 result cache (None = uncached, pre-P22.T2 behavior).
        self._cache = cache

    def invalidate(self) -> None:
        """Drop every cached scope index; the next ``retrieve`` rebuilds."""
        self._indexes.clear()

    def invalidate_cache(self, entry_id: str | None = None) -> None:
        """Clear the P22.T2 result cache; the next ``retrieve`` recomputes.

        The signature matches the store's ``on_update`` hook (plan P7.T2),
        which is called with the updated entry's id — the documented wiring
        point for the "cache invalidated when underlying memory entries
        change" acceptance::

            store.on_update = engine.invalidate_cache

        ``entry_id`` is accepted (and ignored): the whole result cache is
        cleared because a memory update can change the ranking of ANY query,
        not just ones mentioning that entry. Without an injected cache this
        is a no-op.
        """
        if self._cache is not None:
            self._cache.invalidate_all()

    # ── scope indexing ────────────────────────────────────────────────────────

    def _scope_corpus(self, scope: str) -> tuple[list[MemoryEntry], BM25Okapi | None]:
        """Cached (context-filtered) corpus + index for ``scope``, rebuilt on change."""
        fingerprint = _scope_fingerprint(self._store, scope)
        cached = self._indexes.get(scope)
        if cached is not None and cached[0] == fingerprint:
            return cached[1], cached[2]
        entries = self._store.list(context=scope)
        corpus = [_entry_tokens(entry) for entry in entries]
        index: BM25Okapi | None = BM25Okapi(corpus) if corpus else None
        self._indexes[scope] = (fingerprint, entries, index)
        return entries, index

    def _rank_scope(self, query_tokens: list[str], scope: str, top_k: int) -> list[
        tuple[MemoryEntry, float]
    ]:
        """BM25-ranked ``top_k`` hits for ``scope`` (zero-score hits dropped)."""
        entries, index = self._scope_corpus(scope)
        if index is None or not entries:
            return []
        scores = index.get_scores(query_tokens)
        ranked = sorted(
            (
                (entry, float(score))
                for entry, score in zip(entries, scores, strict=True)
                if score != 0.0
            ),
            key=lambda pair: (-pair[1], pair[0].id),
        )
        return ranked[:top_k]

    # ── public API (plan Phase 4) ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        context: str = "global",
    ) -> list[RetrievedEntry]:
        """Top-K candidates for ``query`` in ``context`` + global (plan P4.T1).

        ``context="global"`` searches only global; any other scope unions
        that named context with global (deduplicated by id, best score kept)
        and never touches other contexts. An empty/whitespace query (no
        tokens), ``top_k <= 0`` or an unsafe context name yield an empty
        list / :class:`ValueError` respectively. With ``vector_backend``
        set, the merged BM25 candidate pool is re-ranked by cosine
        similarity and the cosine becomes :attr:`RetrievedEntry.score`.

        With a P22.T2 ``cache`` injected, the computed result is looked up
        BEFORE the BM25 pass (keyed on query + scope + ``top_k``) and stored
        afterwards — a cache hit skips indexing, ranking and any vector
        re-ranking entirely. Invalidation on memory updates is the store's
        ``on_update`` hook wired to :meth:`invalidate_cache` (see the module
        docstring); degenerate results (empty query / ``top_k <= 0``) are
        returned directly without touching the cache.
        """
        if context != "global" and not _CONTEXT_RE.fullmatch(context):
            raise ValueError(f"context must be a safe single path component, got {context!r}")
        query_tokens = tokenize(query)
        if not query_tokens or top_k <= 0:
            return []
        cache_key = _cache_key(query, top_k, context)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached.value)
        scopes = ("global",) if context == "global" else (context, "global")
        pool: dict[str, tuple[MemoryEntry, float]] = {}
        for scope in scopes:
            for entry, score in self._rank_scope(query_tokens, scope, top_k):
                current = pool.get(entry.id)
                if current is None or score > current[1]:
                    pool[entry.id] = (entry, score)
        candidates = sorted(pool.values(), key=lambda pair: (-pair[1], pair[0].id))[:top_k]
        if self._vector_backend is not None:
            query_vector = self._vector_backend(query)
            candidates = sorted(
                (
                    (entry, _cosine(query_vector, self._vector_backend(entry.text)))
                    for entry, _bm25_score in candidates
                ),
                key=lambda pair: (-pair[1], pair[0].id),
            )
        results = [RetrievedEntry(entry=entry, score=score) for entry, score in candidates]
        if self._cache is not None:
            self._cache.put(cache_key, results)
        return results
