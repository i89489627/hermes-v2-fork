"""P4.T3 benchmark: full injection latency budget (plan Phase 4 acceptance).

*Full injection (retrieve+gate+inject): typical <2s, hard budget <5s (hook
timeout).* The corpus is 200 deterministic filler entries written straight
onto the store's on-disk layout (git commits are skipped deliberately — the
budget covers retrieval + gating + composition, not storage write
throughput; the retrieval path reads exactly the layout ``store.add``
produces). The gate model is an instant fake (single-model architecture —
never called live), so the measured cost is the real per-turn injection
plumbing: BM25 pre-filter, gate thread dispatch + budget pass, composition.

The stable context is loaded ONCE before the measured loop (the session-start
contract, plan P4.T3), then every measured call is a full per-turn injection.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.memory.context import ContextManager
from hermes.memory.gating import CBDCGate
from hermes.memory.injector import ContextInjector
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import TrustCard, serialize
from hermes.memory.types import VerificationEvent

ENTRY_COUNT = 200
RUNS = 20
#: Plan Phase 4 acceptance: "typical <2s".
TYPICAL_BUDGET_SECONDS = 2.0
#: Plan Phase 4 acceptance: "hard budget <5s (hook timeout)".
HARD_BUDGET_SECONDS = 5.0

_QUERY = "zebra quixotic deployment pipeline"
_TARGET_ID = "bench-0199"


def _write_corpus(root: Path, count: int) -> None:
    """Write ``count`` deterministic filler entries straight onto the layout."""
    facts = root / "global" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    event = VerificationEvent(
        event_id="ev-injector-bench-seed",
        timestamp=now,
        source="agent-inferred",
        verifier="bench",
        evidence="seed corpus",
    )
    topics = ("deploy", "billing", "gateway", "cache", "logs")
    for index in range(count):
        entry_id = f"bench-{index:04d}"
        if entry_id == _TARGET_ID:
            text = f"zebra quixotic deployment pipeline flag item {index}"
        else:
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
    (root / "MEMORY.md").write_text("global index: bench corpus\n", encoding="utf-8")


def _instant_gate_model(prompt: str) -> str:
    """Fake gate model: use for every candidate id found in the prompt."""
    found = re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
    return json.dumps([
        {"candidate_id": candidate_id, "decision": "use", "reason": "benchmark"}
        for candidate_id in found
    ])


@pytest.mark.benchmark
def test_full_injection_under_5s(tmp_path: Path) -> None:
    """Full retrieve+gate+inject stays inside the 5s hard budget (typical <2s)."""
    root = tmp_path / "memory-repo"
    _write_corpus(root, ENTRY_COUNT)

    injector = ContextInjector(
        retrieval=RetrievalEngine(MemoryStore(root=root)),
        gate=CBDCGate(gate_model=_instant_gate_model),
        context_manager=ContextManager(repo_root=root),
    )

    # session start (NOT part of the per-turn budget): stable loaded once
    stable = injector.load_stable_context({})
    assert stable.context_name == "global"

    # warm-up: first query builds the per-scope BM25 index (not the budget)
    warm = injector.build_dynamic_context(_QUERY, "global", stable)
    assert [
        claim["candidate_id"] for claim in warm["dynamic"]["claims"]
    ] == [_TARGET_ID]

    samples: list[float] = []
    start = time.perf_counter()
    for _ in range(RUNS):
        turn_start = time.perf_counter()
        injected = injector.build_dynamic_context(_QUERY, "global", stable)
        samples.append(time.perf_counter() - turn_start)
    wall_s = time.perf_counter() - start

    worst_s = max(samples)
    median_s = statistics.median(samples)
    assert worst_s < HARD_BUDGET_SECONDS, (
        f"full injection took {worst_s * 1000:.1f}ms "
        f"(hard budget {HARD_BUDGET_SECONDS * 1000:.0f}ms)"
    )
    assert median_s < TYPICAL_BUDGET_SECONDS, (
        f"typical full injection took {median_s * 1000:.1f}ms "
        f"(typical budget {TYPICAL_BUDGET_SECONDS * 1000:.0f}ms)"
    )
    assert wall_s < HARD_BUDGET_SECONDS  # the whole burst fits one hard budget
    # stable block unchanged across every measured turn (prefix cache contract)
    assert injected["stable"] == warm["stable"]
