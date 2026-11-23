"""MemoryCache: TTL + LRU memoization of memory results (plan P22.T2).

The plan Phase 22 caching layer — memoize retrieval results (same query ->
same result within a TTL) and batch multiple memory ops into a single
"API call" where possible (the extractor+reviewer combination case).

Two memo surfaces, one store:

- **Per-key memo** (:meth:`MemoryCache.get` / :meth:`MemoryCache.put`) —
  the retrieval-result cache :class:`hermes.memory.retrieval.RetrievalEngine`
  consults BEFORE the BM25 pass. Keys are opaque strings; values are the
  computed results (``list[RetrievedEntry]`` in the retrieval wiring).

- **Batch-key memo** (:meth:`MemoryCache.batch`) — the "combine
  extractor+reviewer into fewer calls" acceptance: a batch of memory-op
  inputs (JSON-serializable dicts) plus a compute callable. Identical batch
  inputs (same entries, same order) are computed ONCE; every repeat is
  served from the cache and counts as one model call avoided (tracked in
  :attr:`MemoryCache.batch_calls_avoided`).

Stats
-----
:func:`stats()` returns ``CacheStats(hits, misses, evictions, hit_rate)``
— the cumulative counters over every lookup (per-key AND batch-key).
``evictions`` counts capacity-driven LRU drops ONLY; a TTL expiry drops the
entry silently and surfaces as a ``miss`` on the next
lookup, not as an eviction. Invalidations
(:meth:`invalidate_key` / :meth:`invalidate_prefix` / :meth:`invalidate_all`)
are explicit operator actions and are not counted in any stat.

For the P22.T1 :class:`hermes.optimization.performance.PerformanceProfiler`
seam — ``cache_stats: Callable[[], tuple[int, int]]`` sampling CUMULATIVE
``(hits, misses)`` — use :func:`MemoryCache.cache_stats`, which returns
exactly that 2-tuple::

    profiler = PerformanceProfiler(..., cache_stats=cache.cache_stats)

(``CacheStats`` itself is a 4-field NamedTuple, so it cannot be the profiler
seam directly — the profiler unpacks exactly two values.)

Invalidation contract (plan P22.T2 acceptance: "cache invalidated when
underlying memory entries change"): the cache NEVER watches the store
itself. The invalidation signal comes from the store's ``on_update`` hook
(plan P7.T2) pointed at
:func:`hermes.memory.retrieval.RetrievalEngine.invalidate_cache`, which
calls :func:`MemoryCache.invalidate_all`. The clock is injectable so tests
can drive TTL expiry deterministically.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

__all__ = ["CacheStats", "CachedResult", "MemoryCache"]

T = TypeVar("T")

#: Key prefix for batch-memo entries, so :meth:`invalidate_prefix` can drop
#: every batch memo without touching per-key (retrieval) entries.
_BATCH_KEY_PREFIX = "batch::"

#: Length of the sha256 digest used for batch keys (collision-resistant
#: enough for a memo; keeps the OrderedDict keys small).
_BATCH_KEY_DIGEST_CHARS = 32


class CacheStats(NamedTuple):
    """Cumulative cache counters (plan P22.T2 stats interface).

    ``hit_rate`` is ``hits / (hits + misses)`` over ALL lookups (0.0 when
    nothing has been looked up yet). ``evictions`` counts capacity-driven
    LRU drops only (see the module docstring for the expiry convention).
    """

    hits: int
    misses: int
    evictions: int
    hit_rate: float


@dataclass(frozen=True)
class CachedResult(Generic[T]):
    """One cache slot: the stored value plus its TTL bookkeeping.

    ``expires_at`` is an absolute reading of the injected clock; a lookup at
    ``now >= expires_at`` is an expired miss.
    """

    value: T
    created_at: float
    expires_at: float


class MemoryCache(Generic[T]):
    """Bounded TTL cache with LRU eviction and a batch-key memo (P22.T2).

    ``ttl_seconds`` bounds how long an entry stays fresh, ``max_entries``
    bounds the slot count (least-recently-USED entries are evicted first —
    both ``get`` hits and ``put`` refreshes count as use), and ``clock``
    (default ``time.monotonic``) makes expiry testable. All counters are
    cumulative over the cache's lifetime; every public method is
    thread-safe.
    """

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._clock = clock
        self._entries: OrderedDict[str, CachedResult[T]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._batch_calls_avoided = 0
        self._lock = threading.RLock()

    # ── per-key memo (the retrieval-result surface) ────────────────────────
    def get(self, key: str) -> CachedResult[T] | None:
        """Return the fresh cached result for ``key``, else ``None``.

        A hit refreshes the entry's LRU recency. An expired entry is dropped
        on sight and the lookup counts as a miss (see the module docstring).
        """
        with self._lock:
            cached = self._entries.get(key)
            if cached is None:
                self._misses += 1
                return None
            if self._clock() >= cached.expires_at:
                del self._entries[key]
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return cached

    def put(self, key: str, value: T) -> None:
        """Store ``value`` under ``key`` with a fresh TTL; evict LRU over capacity.

        Overwriting an existing key refreshes its timestamp and recency (it
        does not count as an eviction).
        """
        with self._lock:
            now = self._clock()
            self._entries[key] = CachedResult(
                value=value, created_at=now, expires_at=now + self._ttl_seconds
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    # ── batch-key memo (the extractor+reviewer combination surface) ────────
    def batch(
        self, entries: list[dict[str, Any]], compute: Callable[[list[dict[str, Any]]], T]
    ) -> T:
        """Compute-or-memoize one combined result for a batch of op inputs.

        The canonical key is the sha256 of the JSON-serialized ``entries``
        (object keys sorted; LIST ORDER IS SIGNIFICANT — a reordered batch
        is a different batch, since the combined prompt order could change
        the result). On a miss, ``compute(entries)`` runs exactly once and
        its result is cached; every repeat of identical inputs is served
        from the cache and counted in :attr:`batch_calls_avoided` (one
        avoided model call each — the P22.T2 "combine extractor+reviewer
        into fewer calls" acceptance). Batch lookups go through
        :meth:`get`/:meth:`put`, so they appear in :func:`stats` like any
        other lookup and honor the same TTL/capacity rules.

        Entries must be JSON-serializable (turn text, roles, ids — the
        extractor/reviewer payloads are plain strings), else ``json.dumps``
        raises ``TypeError`` before any compute runs.
        """
        key = _BATCH_KEY_PREFIX + hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:_BATCH_KEY_DIGEST_CHARS]
        cached = self.get(key)
        if cached is not None:
            with self._lock:
                self._batch_calls_avoided += 1
            return cached.value
        result = compute(entries)
        self.put(key, result)
        return result

    @property
    def batch_calls_avoided(self) -> int:
        """Model calls avoided by the batch memo (cache-served batch repeats)."""
        with self._lock:
            return self._batch_calls_avoided

    # ── stats (the P22.T1 profiler seam + the full report) ──────────────────
    def stats(self) -> CacheStats:
        """Cumulative :class:`CacheStats` over every lookup so far."""
        with self._lock:
            lookups = self._hits + self._misses
            hit_rate = (self._hits / lookups) if lookups else 0.0
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )

    def cache_stats(self) -> tuple[int, int]:
        """Cumulative ``(hits, misses)`` — the P22.T1 PerformanceProfiler seam.

        Bound this method (NOT :meth:`stats`, which returns the 4-field
        :class:`CacheStats`) as the profiler's ``cache_stats`` callable::

            PerformanceProfiler(..., cache_stats=cache.cache_stats)
        """
        with self._lock:
            return (self._hits, self._misses)

    # ── invalidation (the memory-update signal) ─────────────────────────────
    def invalidate_key(self, key: str) -> int:
        """Drop ``key``; returns 1 when it existed, 0 otherwise."""
        with self._lock:
            return 1 if self._entries.pop(key, None) is not None else 0

    def invalidate_prefix(self, prefix: str) -> int:
        """Drop every key starting with ``prefix``; returns the number dropped."""
        with self._lock:
            stale = [key for key in self._entries if key.startswith(prefix)]
            for key in stale:
                del self._entries[key]
            return len(stale)

    def invalidate_all(self) -> int:
        """Drop everything; returns the number of entries dropped."""
        with self._lock:
            dropped = len(self._entries)
            self._entries.clear()
            return dropped
