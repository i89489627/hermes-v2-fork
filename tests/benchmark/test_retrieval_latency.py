"""P4.T1 benchmark: BM25 retrieval latency budget (plan Phase 4 acceptance).

*BM25 search < 100ms for 1000 entries.* The corpus is 1000 deterministic
filler entries written straight onto the store's on-disk layout with the
public :func:`hermes.memory.trust_card.serialize` (git commits are skipped
deliberately — the budget covers BM25 retrieval, not storage write
throughput; the retrieval path reads exactly the layout ``store.add``
produces). The engine warms its index with one query, then the measured
query must complete in under 100ms (best of three runs, to absorb scheduler
noise on shared CI hosts).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import TrustCard, serialize
from hermes.memory.types import VerificationEvent

ENTRY_COUNT = 1000
LATENCY_BUDGET_SECONDS = 0.100


def _write_corpus(root: Path, count: int) -> str:
    """Write ``count`` deterministic filler entries; returns a target id."""
    facts = root / "global" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    event = VerificationEvent(
        event_id="ev-bench-seed",
        timestamp=now,
        source="agent-inferred",
        verifier="bench",
        evidence="seed corpus",
    )
    topics = ("deploy", "billing", "gateway", "cache", "logs")
    target_id = ""
    for index in range(count):
        entry_id = f"bench-{index:04d}"
        text = (
            f"entry{index} fact about topic-{topics[index % len(topics)]} "
            f"item {index} shared filler alpha beta gamma delta epsilon"
        )
        card = TrustCard(
            id=entry_id,
            type="Reference",
            created_at=now,
            last_verified_at=now,
            trust_score=0.5,
            verification_events=[event],
            context="global",
        )
        (facts / f"{entry_id}.md").write_text(serialize(card, text), encoding="utf-8")
        if index == 777:
            target_id = entry_id
    return target_id


@pytest.mark.benchmark
def test_bm25_under_100ms_for_1000_entries(tmp_path: Path) -> None:
    """Warm BM25 retrieval over 1000 entries stays under the 100ms budget."""
    root = tmp_path / "memory-repo"
    target_id = _write_corpus(root, ENTRY_COUNT)

    store = MemoryStore(root=root)
    engine = RetrievalEngine(store)

    # Warmup: first query builds the per-scope index (not part of the budget).
    warm = engine.retrieve("entry777", top_k=5, context="global")
    assert warm, "warmup query must match the seeded corpus"
    assert warm[0].entry.id == target_id

    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        results = engine.retrieve("entry777", top_k=5, context="global")
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        assert results[0].entry.id == target_id

    assert best < LATENCY_BUDGET_SECONDS, (
        f"BM25 retrieval over {ENTRY_COUNT} entries took {best * 1000:.1f}ms "
        f"(budget {LATENCY_BUDGET_SECONDS * 1000:.0f}ms)"
    )
