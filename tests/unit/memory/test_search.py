"""P1.T4 acceptance: BM25 search over memory entries.

Covers ``MemoryStore.search`` (relevant entries ranked into the top-k,
cache invalidation on mutations, context filtering) and the plan's latency
budget — search over a 1000-entry corpus stays under 100ms (asserted here
on the warmed index; the corpus is built in memory to keep the unit test
fast, the store-backed relevance path is separate).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.search import BM25Search
from hermes.memory.store import MemoryStore

_TOPICS = [
    ("pnpm is the preferred package manager for this repo", "User", ["package-manager"]),
    ("docker images are rebuilt weekly with buildkit", "Project", ["containers"]),
    ("pytest markers are mandatory on every new test", "Feedback", ["testing"]),
    ("the kubernetes cluster runs in eu-central-1", "Project", ["infra"]),
    ("the office wifi password rotates monthly", "Reference", ["office"]),
    ("release notes are written in spanish first", "Project", ["docs"]),
]


def _make_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(root=tmp_path / "memory-repo")
    for text, entry_type, tags in _TOPICS:
        store.add(text, type=entry_type, tags=tags)
    return store


@pytest.mark.unit
def test_bm25_returns_relevant_top_k(tmp_path: Path) -> None:
    """The pnpm entry outranks distractors; new entries appear after mutation."""
    store = _make_store(tmp_path)
    hits = store.search("pnpm package manager", top_k=5)
    assert hits, "relevant query must return at least one hit"
    assert len(hits) <= 5
    assert hits[0].text.startswith("pnpm is the preferred package manager")
    assert hits[0].context == "global"
    assert all(hit.id != _TOPICS[4][0][:20] for hit in hits)  # no wifi distractor

    # cache invalidation: an entry added after a warmed search is findable
    late = store.add("pnpm workspaces are required for monorepos", type="Project")
    hits_after = store.search("pnpm workspaces monorepos", top_k=5)
    assert late.id in {hit.id for hit in hits_after}
    assert hits_after[0].id == late.id

    # zero-score entries are never returned (query matching nothing)
    assert store.search("xyzzyplugh quantum entanglement orb", top_k=5) == []

    # latency budget (plan P1.T4): <100ms on a 1000-entry corpus, warmed index
    now = datetime.now(UTC)
    corpus = [
        MemoryEntry(
            id=f"2026-01-01-doc-{i:04d}",
            type="Reference",
            text=f"routine note {i} covering topic {i % 50} with marker token u{i}",
            created_at=now,
        )
        for i in range(1000)
    ]
    needle = MemoryEntry(
        id="2026-01-01-quokka-0001",
        type="Reference",
        text="quokka breeding habits observed at the perth zoo enclosure",
        created_at=now,
    )
    engine = BM25Search(entries=[*corpus, needle])
    engine.search("warm the index")  # builds the cached BM25 index
    start = perf_counter()
    latency_hits = engine.search("quokka breeding perth", top_k=5)
    elapsed = perf_counter() - start
    assert elapsed < 0.1, f"1000-entry search took {elapsed * 1000:.1f}ms (>100ms)"
    assert latency_hits and latency_hits[0].id == needle.id


@pytest.mark.unit
def test_empty_query_returns_empty(tmp_path: Path) -> None:
    """Empty / punctuation-only queries and top_k<=0 yield empty results."""
    store = _make_store(tmp_path)
    assert store.search("") == []
    assert store.search("   \t  ") == []
    assert store.search("!!! ??? ...") == []
    assert store.search("pnpm", top_k=0) == []

    engine = BM25Search()
    assert engine.search("pnpm") == []  # no corpus configured


@pytest.mark.unit
def test_search_filters_by_context(tmp_path: Path) -> None:
    """Context-filtered search ranks within the context only — no bleeding."""
    store = _make_store(tmp_path)
    work = store.add("pnpm lockfile conflicts surface in ci", type="Project", context="work")
    home = store.add("pnpm is banned on the family laptop", type="User", context="home")

    everywhere = store.search("pnpm", top_k=10)
    assert len(everywhere) == 3  # one match per context, ranked across all
    assert {hit.context for hit in everywhere} == {"global", "work", "home"}
    assert {hit.id for hit in everywhere} >= {work.id, home.id}

    only_work = store.search("pnpm lockfile conflicts", top_k=10, context="work")
    assert [hit.id for hit in only_work] == [work.id]

    only_global = store.search("pnpm", top_k=10, context="global")
    assert {hit.id for hit in only_global}.isdisjoint({work.id, home.id})
    assert only_global[0].text.startswith("pnpm is the preferred package manager")

    assert store.search("pnpm", top_k=10, context="no-such-context") == []
    assert store.search("pnpm", top_k=10, context="home")[0].id == home.id
