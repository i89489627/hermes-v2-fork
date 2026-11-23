"""P4.T1 acceptance: RetrievalEngine BM25 candidates with context scoping.

Covers :class:`hermes.memory.retrieval.RetrievalEngine` over the real git-backed
store: BM25 relevance ranking (top-k, scores descending), the empty-query
contract, and the plan Phase 4 scope rule — retrieving for context X returns
X + global entries (union, deduped) and NEVER another context's entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.retrieval import RetrievalEngine, RetrievedEntry
from hermes.memory.store import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


@pytest.mark.unit
def test_bm25_returns_relevant_top_k(tmp_path: Path) -> None:
    """'kubernetes' ranks only the kubernetes entries, best first, at top_k."""
    store = _make_store(tmp_path)
    store.add("Kubernetes cluster autoscaling adds pods under load", type="Reference")
    store.add("Grandma's apple pie recipe uses cinnamon and nutmeg", type="User")
    store.add("Kubernetes ingress routes traffic to the public API", type="Project")
    store.add("Weekly team sync notes about the offsite", type="User")
    store.add("Gardening log: tomatoes and basil on the balcony", type="User")

    engine = RetrievalEngine(store)
    results = engine.retrieve("kubernetes", top_k=2)

    assert len(results) == 2
    assert all(isinstance(result, RetrievedEntry) for result in results)
    assert all("kubernetes" in result.entry.text.lower() for result in results)
    # Scores are BM25 relevance, descending; ties broken deterministically.
    assert results[0].score >= results[1].score > 0.0
    assert results[0].entry.id != results[1].entry.id

    # A wider net still never promotes non-matching entries.
    everything = engine.retrieve("kubernetes", top_k=10)
    assert {result.entry.context for result in everything} == {"global"}
    assert len(everything) == 2


@pytest.mark.unit
def test_empty_query_returns_empty(tmp_path: Path) -> None:
    """Empty/whitespace/punctuation-only queries yield no candidates."""
    store = _make_store(tmp_path)
    store.add("The billing service invoices EU clients monthly", type="Project")

    engine = RetrievalEngine(store)
    assert engine.retrieve("", top_k=5) == []
    assert engine.retrieve("   ", top_k=5) == []
    assert engine.retrieve("?!...---", top_k=5) == []
    assert engine.retrieve("billing", top_k=0) == []
    assert engine.retrieve("billing", top_k=-3) == []


@pytest.mark.unit
def test_filters_by_context(tmp_path: Path) -> None:
    """Context X = X ∪ global; other contexts never leak (plan Phase 4)."""
    store = _make_store(tmp_path)
    store.add("Deploy runbook for the billing service", type="Project", context="billing")
    store.add("Billing questions route to the shared support desk", type="User")
    store.add("Billing invoice formatting rules for EU clients", type="Project", context="hr")

    engine = RetrievalEngine(store)

    # Named context: matches from billing AND global, never from hr.
    results = engine.retrieve("billing", top_k=10, context="billing")
    contexts = {result.entry.context for result in results}
    assert contexts == {"billing", "global"}
    assert all(result.entry.context != "hr" for result in results)
    assert len(results) == 2

    # global stays global-only.
    global_only = engine.retrieve("billing", top_k=10, context="global")
    assert {result.entry.context for result in global_only} == {"global"}
    assert len(global_only) == 1

    # The hr entry is invisible from the billing scope even though its text
    # would match BM25 — filtering happens before ranking.
    hr_ids = {result.entry.id for result in engine.retrieve("invoice", top_k=10, context="billing")}
    hr_only = engine.retrieve("invoice formatting rules", top_k=10, context="hr")
    assert hr_ids.isdisjoint({result.entry.id for result in hr_only})
    assert len(hr_only) == 1
