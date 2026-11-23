"""P10.T4 acceptance: Pareto front store behavior (plan Phase 10).

The front is stored per component (single-model architecture — no model
dimension), returns non-dominated candidates, supports best-for-context
routing, and persists atomically under ``hermes_home()`` by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.optimization.pareto import Candidate, ParetoFrontStore


@pytest.mark.unit
def test_add_candidate(tmp_path: Path) -> None:
    """Candidates validate, store, persist and reload losslessly."""
    store = ParetoFrontStore()
    # Default path derives from hermes_home() so tests stay hermetic.
    assert store.path == (
        hermes_home() / "memory-repo" / "optimization" / "pareto-front.json"
    )

    candidate = store.add_candidate("extraction", "prompt v1", {"f1": 0.8, "recall": 0.7})
    assert isinstance(candidate, Candidate)
    assert candidate.component == "extraction"
    assert candidate.prompt == "prompt v1"
    assert candidate.scores == {"f1": 0.8, "recall": 0.7}
    assert candidate.context is None
    assert candidate.created_at > 0.0
    assert store.get_front("extraction") == [candidate]

    # Scores are copied — mutating the caller's dict must not leak into the store.
    raw_scores = {"recall": 0.95}
    stored = store.add_candidate("extraction", "prompt v2", raw_scores)
    raw_scores["recall"] = 0.99
    assert stored.scores == {"recall": 0.95}

    # Persisted to the default location; a fresh store sees the same data.
    assert store.path.exists()
    reloaded = ParetoFrontStore(store.path)
    assert len(reloaded.get_front("extraction")) == 2

    # Validation: empty score mapping, empty component, bad score values.
    with pytest.raises(ValueError):
        store.add_candidate("extraction", "p", {})
    with pytest.raises(ValueError):
        store.add_candidate("  ", "p", {"f1": 0.5})
    with pytest.raises(TypeError):
        store.add_candidate("extraction", "p", {"f1": "high"})
    with pytest.raises(ValueError):
        store.add_candidate("extraction", "p", {"f1": float("nan")})


@pytest.mark.unit
def test_get_front_returns_non_dominated(tmp_path: Path) -> None:
    """Only non-dominated candidates survive; dominated ones are pruned."""
    store = ParetoFrontStore(path=tmp_path / "pareto.json")
    strong = store.add_candidate("extraction", "strong", {"f1": 0.9, "recall": 0.9})
    dominated = store.add_candidate("extraction", "weak", {"f1": 0.5, "recall": 0.5})
    tradeoff = store.add_candidate("extraction", "tradeoff", {"f1": 0.2, "recall": 1.0})
    # Equal scores tie — neither dominates the other, so both stay.
    equal = store.add_candidate("extraction", "equal", {"f1": 0.9, "recall": 0.9})

    front = store.get_front("extraction")
    assert len(front) == 3
    assert strong in front
    assert tradeoff in front
    assert equal in front
    assert dominated not in front
    # Deterministic order: highest mean score first.
    assert front[0].prompt == "strong"

    # A dimension missing from a candidate's scores counts as 0.0, so a
    # candidate covering more dimensions can dominate a partial one.
    store.add_candidate("gating", "partial", {"precision": 1.0})
    fuller = store.add_candidate("gating", "fuller", {"precision": 1.0, "coverage": 0.8})
    assert store.get_front("gating") == [fuller]

    # Unknown component -> empty front.
    assert store.get_front("missing") == []


@pytest.mark.unit
def test_get_best_for_context(tmp_path: Path) -> None:
    """Context-specific fronts are preferred; global pool is the fallback."""
    store = ParetoFrontStore(path=tmp_path / "pareto.json")
    coding = store.add_candidate(
        "injection", "coding template", {"quality": 0.6}, context="coding"
    )
    chat = store.add_candidate(
        "injection", "chat template", {"quality": 0.9}, context="chat"
    )
    store.add_candidate("injection", "generic template", {"quality": 0.75})

    # No context -> global pool; the dominated candidates lose to "chat".
    assert store.get_best("injection") == chat
    # Context-specific front preferred even though globally weaker.
    assert store.get_best("injection", context="coding") == coding
    assert store.get_best("injection", context="chat") == chat
    # Context without recorded candidates falls back to the global pool.
    assert store.get_best("injection", context="voice") == chat
    # Unknown component -> KeyError.
    with pytest.raises(KeyError):
        store.get_best("extraction")


@pytest.mark.unit
def test_per_component_isolation(tmp_path: Path) -> None:
    """Fronts and best-pick never leak candidates across components."""
    store = ParetoFrontStore(path=tmp_path / "pareto.json")
    extraction_best = store.add_candidate("extraction", "ext winner", {"f1": 0.95})
    store.add_candidate("extraction", "ext loser", {"f1": 0.3})
    gating_best = store.add_candidate("gating", "gate winner", {"precision": 0.4})

    ext_front = store.get_front("extraction")
    assert [c.prompt for c in ext_front] == ["ext winner"]
    assert all(c.component == "extraction" for c in ext_front)
    # The gating front is independent despite much lower absolute scores.
    gate_front = store.get_front("gating")
    assert [c.prompt for c in gate_front] == ["gate winner"]
    assert all(c.component == "gating" for c in gate_front)

    # Best-pick routing is per component too.
    assert store.get_best("extraction") == extraction_best
    assert store.get_best("gating") == gating_best

    # Persistence keeps both components separate across reloads.
    reloaded = ParetoFrontStore(tmp_path / "pareto.json")
    assert [c.prompt for c in reloaded.get_front("extraction")] == ["ext winner"]
    assert [c.prompt for c in reloaded.get_front("gating")] == ["gate winner"]
    assert reloaded.get_front("missing") == []
