"""P22.T2 acceptance: MemoryCache (plan Phase 22 caching layer).

Covered: TTL-bounded memoization with hits/misses/evictions/hit_rate stats
(the stats interface the P22.T1 PerformanceProfiler ``cache_stats`` seam
consumes via :func:`MemoryCache.cache_stats`), invalidation when the
underlying memory entries change (the store's P7.T2 ``on_update`` hook
wired to ``engine.invalidate_cache``), and the batch-key memo that combines
extractor+reviewer-style op inputs into fewer computed calls (call count
asserted through a counting fake).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hermes.memory.cache import MemoryCache
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore


@pytest.mark.unit
def test_cache_hit_within_ttl() -> None:
    """Same key within the TTL is a hit; past the TTL it expires to a miss.

    Also pins the hit-rate acceptance on a repeated-query workload (5 unique
    queries x 20 repeats, the shape P22.T2 prescribes): 5 compulsory misses
    + 95 hits = 95% hit rate, comfortably above the 30% target.
    """
    now = [100.0]
    cache: MemoryCache[str] = MemoryCache(
        ttl_seconds=60.0, clock=lambda: now[0]
    )
    cache.put("k", "v")
    hit = cache.get("k")
    assert hit is not None and hit.value == "v"
    assert hit.created_at == 100.0 and hit.expires_at == 160.0

    # 30s later, still within the TTL: fresh hit, LRU recency refreshed.
    now[0] += 30.0
    hit = cache.get("k")
    assert hit is not None and hit.value == "v"

    # At exactly ttl_seconds past creation the entry is expired: dropped and
    # reported as a miss (module contract: expiry is not an eviction).
    now[0] += 30.0
    assert cache.get("k") is None
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.evictions) == (2, 1, 0)
    assert stats.hit_rate == pytest.approx(2 / 3)
    # The profiler seam reports the same cumulative counters as a 2-tuple.
    assert cache.cache_stats() == (2, 1)

    # Capacity: the LRU cap evicts the least-recently-used entry.
    tiny: MemoryCache[str] = MemoryCache(ttl_seconds=60.0, max_entries=2, clock=lambda: now[0])
    tiny.put("a", "1")
    tiny.put("b", "2")
    tiny.put("c", "3")  # evicts "a" (oldest)
    assert tiny.get("a") is None
    assert tiny.get("b") is not None and tiny.get("c") is not None
    assert tiny.stats().evictions == 1

    # Hit-rate acceptance: 5 unique queries x 20 repeats within the TTL.
    workload: MemoryCache[list[str]] = MemoryCache(clock=lambda: now[0])
    queries = [f"query-{i}" for i in range(5)]
    for _ in range(20):
        for query in queries:
            key = f"retrieve::global::20::{query}"
            if workload.get(key) is None:
                workload.put(key, [query])  # the "computed" retrieval result
    workload_stats = workload.stats()
    assert (workload_stats.hits, workload_stats.misses) == (95, 5)
    assert workload_stats.hit_rate == pytest.approx(0.95)
    assert workload_stats.hit_rate >= 0.30  # the P22.T2 acceptance threshold


@pytest.mark.unit
def test_cache_invalidated_on_memory_update(tmp_path: Path) -> None:
    """A committed memory update invalidates the result cache via on_update.

    The documented P7.T2 wiring point: ``store.on_update =
    engine.invalidate_cache``. The store calls the hook with the updated
    entry id, so the engine's ``invalidate_cache`` must accept (and ignore)
    that argument. After the update, a previously-cached query recomputes
    and reflects the new entry text instead of the stale cached one.
    """
    store = MemoryStore(root=tmp_path / "memory-repo")
    entry = store.add("Kubernetes cluster autoscaling adds pods under load", type="Reference")
    # BM25 needs a term rare enough in the corpus to score positively (a term
    # in half the docs has idf 0 under BM25Okapi), hence 4 unrelated entries.
    store.add("Grandma's apple pie recipe uses cinnamon and nutmeg", type="User")
    store.add("Weekly team sync notes about the offsite", type="User")
    store.add("Gardening log: tomatoes and basil on the balcony", type="User")
    store.add("Book club meets on Thursdays to discuss novels", type="User")

    cache = MemoryCache()
    engine = RetrievalEngine(store, cache=cache)

    first = engine.retrieve("kubernetes", top_k=5)
    assert first, "sanity: the seeded entry must be retrievable"
    second = engine.retrieve("kubernetes", top_k=5)
    assert [r.entry.id for r in second] == [r.entry.id for r in first]
    primed = cache.stats()
    assert primed.hits == 1, "the repeated query must be served from the cache"

    # Wire the P7.T2 hook to the P22.T2 invalidation point, then mutate the
    # underlying memory: the committed update must clear the result cache.
    store.on_update = engine.invalidate_cache
    store.update(entry.id, text="Kubernetes replaced by Nomad for batch scheduling")
    after_update = cache.stats()
    assert after_update.hits == primed.hits, "invalidation must not record hits"
    assert after_update.misses == primed.misses, "invalidation is not a lookup"

    fresh = engine.retrieve("kubernetes", top_k=5)
    texts = " ".join(r.entry.text for r in fresh)
    assert "Nomad" in texts and "autoscaling" not in texts
    recomputed = cache.stats()
    assert recomputed.misses == primed.misses + 1
    assert recomputed.hits == primed.hits, "the post-update query was a recompute, not a hit"


@pytest.mark.unit
def test_batch_reduces_api_calls() -> None:
    """Identical batch inputs compute once; repeats are cache-served.

    The extractor+reviewer combination case (plan P22.T2): a counting fake
    stands in for the combined model call, and the memo must keep it to one
    invocation per distinct batch. Batch lookups flow through the same
    stats/hit-rate surface as per-key lookups.
    """
    calls: list[list[dict[str, Any]]] = []

    def compute(entries: list[dict[str, Any]]) -> str:
        calls.append(entries)
        return "combined:" + ",".join(str(entry["turn"]) for entry in entries)

    cache = MemoryCache[str]()
    batch = [
        {"turn": "turn-1", "op": "extraction"},
        {"turn": "turn-1", "op": "review"},
    ]

    first = cache.batch(batch, compute)
    # Identical inputs as NEW containers: memoization is value-keyed, not
    # object-identity keyed.
    repeat_one = cache.batch([dict(item) for item in batch], compute)
    repeat_two = cache.batch([dict(item) for item in batch], compute)
    assert first == repeat_one == repeat_two == "combined:turn-1,turn-1"
    assert len(calls) == 1, "identical batch inputs must compute exactly once"
    assert cache.batch_calls_avoided == 2

    # A different batch is a different key: one more compute call.
    other = cache.batch([{"turn": "turn-2", "op": "extraction"}], compute)
    assert other == "combined:turn-2"
    assert len(calls) == 2

    # List order is significant (documented): a reordered batch recomputes.
    cache.batch(list(reversed(batch)), compute)
    assert len(calls) == 3

    # Batch activity flows through the shared stats surface: 5 lookups
    # (3 misses: the two distinct batches + the reordered one; 2 hits).
    stats = cache.stats()
    assert (stats.hits, stats.misses) == (2, 3)
    assert stats.hit_rate == pytest.approx(0.4)
    assert cache.cache_stats() == (2, 3)
