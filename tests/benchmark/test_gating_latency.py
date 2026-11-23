"""P4.T2 benchmark: CBDC gate latency budget (plan Phase 4 acceptance).

*Gate timeout 3.5s -> trust-only fallback.* With an instant fake gate model
(single-model architecture — never called live) every gate call must
complete far inside the plan's 3.5s hard gate timeout, and a sustained burst
of gate calls (one per turn) must keep the whole batch within the budget:
thread-spawn + join overhead per call is the only real cost here, so the
benchmark guards against regressions that push per-call latency toward the
timeout.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.gating import CBDCGate
from hermes.memory.retrieval import RetrievedEntry

RUNS = 200
#: Plan Phase 4 hard gate timeout (plan Section 2.5 / Phase 4 acceptance).
LATENCY_BUDGET_MS = 3500.0

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_MODEL_RESPONSE = json.dumps([
    {"candidate_id": "bench-1", "decision": "use", "reason": "relevant"},
    {"candidate_id": "bench-2", "decision": "verify", "reason": "stale"},
    {"candidate_id": "bench-3", "decision": "ignore", "reason": "off-topic"},
])


def _gate() -> CBDCGate:
    def _instant_model(_prompt: str) -> str:
        return _MODEL_RESPONSE

    return CBDCGate(gate_model=_instant_model)


def _candidates() -> list[RetrievedEntry]:
    return [
        RetrievedEntry(
            entry=MemoryEntry(
                id="bench-1",
                type="Project",
                text="Deploy pipeline runs through GitHub Actions on main",
                created_at=_NOW,
                trust_score=0.95,
            ),
            score=9.0,
        ),
        RetrievedEntry(
            entry=MemoryEntry(
                id="bench-2",
                type="User",
                text="User prefers concise answers with diffs first",
                created_at=_NOW,
                trust_score=0.55,
            ),
            score=5.0,
        ),
        RetrievedEntry(
            entry=MemoryEntry(
                id="bench-3",
                type="Reference",
                text="API reference lives at docs/api.md",
                created_at=_NOW,
                trust_score=0.85,
            ),
            score=1.0,
        ),
    ]


@pytest.mark.benchmark
def test_cbdc_gate_under_3500ms() -> None:
    """Every gate call stays under the 3.5s budget; 200 calls fit it too."""
    gate = _gate()
    candidates = _candidates()
    gate.gate(candidates)  # warm-up (lazy wiring, interpreter caches)

    worst_ms = 0.0
    start = time.perf_counter()
    for _ in range(RUNS):
        result = gate.gate(candidates, turn_text="benchmark turn")
        worst_ms = max(worst_ms, result.elapsed_ms)
    wall_s = time.perf_counter() - start

    assert worst_ms < LATENCY_BUDGET_MS  # single-call budget (plan hard timeout)
    assert wall_s < LATENCY_BUDGET_MS / 1000.0  # the whole burst fits one budget
    assert not result.fallback_used
