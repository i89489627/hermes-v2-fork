"""P20.T4 acceptance: performance regression tests vs stored baselines (plan Phase 20).

*Baselines stored per release; regression >10% fails CI; trend tracked over
time.* :func:`measure_current` re-measures the four key memory operations on
THIS host with the SAME deterministic fakes and datasets every run:

- **extraction** — :class:`~hermes.memory.extractor.ExtractorAgent.extract`
  with an instant fake model returning a fixed 3-draft JSON array (single-
  model architecture — the LLM is never called live; the measured cost is
  the real per-turn extraction plumbing: prompt build + JSON parse +
  per-item validation).
- **gating** — :meth:`~hermes.memory.gating.CBDCGate.gate` over 3
  deterministic candidates with an instant fake gate model (thread-spawn +
  join + deterministic budget pass — the real per-turn gate cost).
- **retrieval** — warm
  :meth:`~hermes.memory.retrieval.RetrievalEngine.retrieve` over a 200-entry
  corpus written straight onto the store's on-disk layout (git commits are
  skipped deliberately — the budget covers BM25 + freshness fingerprint, not
  storage write throughput; mirrors the P4.T1/P4.T3 benchmark pattern).
- **injection** — full retrieve+gate+inject via
  :meth:`~hermes.memory.injector.ContextInjector.build_dynamic_context` over
  the same 200-entry corpus (stable context loaded once per session, the
  session-start contract).

Measurement method (documented so baselines stay comparable): per operation,
3 batches each of one warm-up + 30 timed calls; a batch's p50 is
``statistics.median`` of its per-call milliseconds and the op's measured
value is the BEST batch p50 (min of 3) — the same noise-absorption pattern
the P4.T1 retrieval benchmark uses ("best of three runs, to absorb scheduler
noise on shared CI hosts"). A full battery running alongside other suites on
this host inflated a single-batch retrieval p50 by ~4x; the best-batch p50
recovers the true cost while remaining an honest measured p50. Values are
rounded to 3 decimals (microsecond precision — no false significance).

Tolerance reasoning (Windows timing jitter honesty): a regression fails when
``current > baseline + allowance``. The allowance is ``baseline * 10%``
(plan: ">10% regression fails"), EXCEPT for baselines under 5ms where 10% is
smaller than a scheduler tick — there the allowance is
``max(baseline * 10%, 2.0ms)`` (an absolute floor). The floor stays generous
on purpose: tiny ops (extraction/gating with fakes) sit in the sub-millisecond
range where ordinary Windows timer and thread-scheduling jitter exceeds any
percentage, and a percentage-only comparison would flake CI without catching
real regressions (a real regression on those ops is a 2ms+ p50 shift).

Baseline lifecycle (plan acceptance "baselines stored per release tag"):
CI refreshes ``baseline_metrics.json`` per release tag (re-measure on the
release runner via ``measure_current()`` and commit alongside the tag), so a
release branch always compares against its predecessor's host-equivalent
numbers. PR CI fails when any op exceeds its allowance (regression >10%).

Weekly trend report (plan acceptance "trend report generated weekly" —
design, no code this task): a weekly CI job re-runs ``measure_current()`` on
the pinned runner and APPENDS
``{"week": <ISO week>, "release_tag": <tag>, "p50s": {...}}`` to
``tests/benchmark/trend.jsonl`` (published as a CI artifact). Each week's
snapshot is therefore comparable against both the previous week (immediate
drift) and the release-tag baselines (release drift); the JSONL append-only
log keeps the full history queryable without extra infrastructure.
"""

from __future__ import annotations

import json
import re
import statistics
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.memory.context import ContextManager
from hermes.memory.entry import MemoryEntry
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.gating import CBDCGate
from hermes.memory.injector import ContextInjector
from hermes.memory.retrieval import RetrievalEngine, RetrievedEntry
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import TrustCard, serialize
from hermes.memory.types import VerificationEvent

#: Timed calls per batch (p50 over 30 samples per batch).
RUNS = 30

#: Best-of batches per operation (min of the batch p50s — the P4.T1
#: benchmark's "best of three runs to absorb scheduler noise" pattern).
BATCHES = 3

#: Deterministic corpus size shared by the retrieval + injection workloads.
ENTRY_COUNT = 200

#: Plan P20.T4: fail when current exceeds baseline by more than 10%.
REGRESSION_RATE = 0.10

#: Baselines below this get the absolute jitter floor (10% of a sub-5ms op is
#: smaller than one Windows scheduler tick — see the module docstring).
FLOOR_APPLIES_BELOW_MS = 5.0

#: Absolute tolerance floor for tiny (sub-5ms) baselines.
ABS_FLOOR_MS = 2.0

#: The per-turn text every fake-driven op runs on (deterministic workload).
_TURN = "we deploy the acme service via GitHub Actions every tuesday"

#: Retrieval/injection query: only bench-0199 carries these distinctive tokens.
_QUERY = "zebra quixotic deployment pipeline"
_TARGET_ID = "bench-0199"

_BASELINE_PATH = Path(__file__).with_name("baseline_metrics.json")

#: Fixed extractor output: schema-valid 3-draft JSON array (no model call).
_EXTRACT_MODEL_RESPONSE = json.dumps([
    {
        "type": "Project",
        "text": "deploy pipeline runs through GitHub Actions on main",
        "proposed_source": "user-explicit",
        "confidence": 0.9,
    },
    {
        "type": "User",
        "text": "user prefers concise answers with diffs first",
        "proposed_source": "agent-inferred",
        "confidence": 0.8,
    },
    {
        "type": "Reference",
        "text": "API reference lives at docs/api.md",
        "proposed_source": "file-read",
        "confidence": 0.7,
    },
])


def _write_corpus(root: Path, count: int) -> None:
    """Write ``count`` deterministic filler entries straight onto the layout."""
    facts = root / "global" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    event = VerificationEvent(
        event_id="ev-regression-bench-seed",
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
    (root / "MEMORY.md").write_text("global index: regression bench corpus\n", encoding="utf-8")


def _instant_extract_model(_prompt: str) -> str:
    """Fake extractor model: the fixed 3-draft JSON array."""
    return _EXTRACT_MODEL_RESPONSE


def _instant_gate_model(prompt: str) -> str:
    """Fake gate model: use for every candidate id found in the prompt."""
    found = re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
    # skip the system prompt's literal "<id>" placeholder (never a real candidate)
    return json.dumps([
        {"candidate_id": candidate_id, "decision": "use", "reason": "benchmark"}
        for candidate_id in found
        if candidate_id != "<id>"
    ])


def _gate_candidates(now: datetime) -> list[RetrievedEntry]:
    """Three deterministic in-memory candidates (one per gate decision class)."""
    entries = [
        MemoryEntry(
            id="bench-1",
            type="Project",
            text="Deploy pipeline runs through GitHub Actions on main",
            created_at=now,
            trust_score=0.95,
        ),
        MemoryEntry(
            id="bench-2",
            type="User",
            text="User prefers concise answers with diffs first",
            created_at=now,
            trust_score=0.55,
        ),
        MemoryEntry(
            id="bench-3",
            type="Reference",
            text="API reference lives at docs/api.md",
            created_at=now,
            trust_score=0.85,
        ),
    ]
    return [RetrievedEntry(entry=entry, score=9.0 - position * 4.0) for position, entry in
            enumerate(entries)]


def _batch_p50_ms(fn: Callable[[], object], runs: int = RUNS) -> float:
    """p50 of ``fn`` over ``runs`` timed calls in ms (one warm-up call first)."""
    fn()  # warm-up: lazy wiring + interpreter caches are not the measured cost
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _p50_ms(fn: Callable[[], object]) -> float:
    """Best-batch p50 of ``fn``: min of ``BATCHES`` batch p50s, rounded to 3 decimals.

    The min-of-batches absorbs scheduler/contention noise (see the module
    docstring): under transient host load a whole batch inflates, but rarely
    do ALL batches inflate, so the best batch still reflects the true cost.
    """
    return round(min(_batch_p50_ms(fn) for _ in range(BATCHES)), 3)


def measure_current() -> dict[str, float]:
    """Measure all four key operations with the SAME fakes/datasets every run.

    Returns ``{op: p50_ms}`` keyed exactly like the ``baselines`` section of
    ``baseline_metrics.json``. The 200-entry corpus lives in a throwaway temp
    dir and serves BOTH the retrieval and the injection workload (same
    dataset, as the plan requires for comparability).
    """
    with tempfile.TemporaryDirectory(prefix="hermes-regression-bench-") as tmp:
        root = Path(tmp) / "memory-repo"
        _write_corpus(root, ENTRY_COUNT)
        store = MemoryStore(root=root)
        engine = RetrievalEngine(store)
        injector = ContextInjector(
            retrieval=engine,
            gate=CBDCGate(gate_model=_instant_gate_model),
            context_manager=ContextManager(repo_root=root),
        )
        # session-start contract: stable loaded ONCE, outside the measured ops
        stable = injector.load_stable_context({})
        warm = injector.build_dynamic_context(_QUERY, "global", stable)
        assert [claim["candidate_id"] for claim in warm["dynamic"]["claims"]] == [_TARGET_ID], (
            "deterministic corpus must rank the target first (workload sanity)"
        )

        extractor = ExtractorAgent(model=_instant_extract_model)
        gate = CBDCGate(gate_model=_instant_gate_model)
        candidates = _gate_candidates(datetime.now(UTC))

        return {
            "extraction_ms_p50": _p50_ms(lambda: extractor.extract(_TURN)),
            "gating_ms_p50": _p50_ms(lambda: gate.gate(candidates, turn_text=_TURN)),
            "retrieval_ms_p50": _p50_ms(
                lambda: engine.retrieve(_QUERY, top_k=5, context="global")
            ),
            "injection_ms_p50": _p50_ms(
                lambda: injector.build_dynamic_context(_QUERY, "global", stable)
            ),
        }


def _load_baseline() -> dict[str, float]:
    """Load the release-tag baselines from the JSON next to this test."""
    document = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {op: float(ms) for op, ms in document["baselines"].items()}


def _allowance_ms(baseline_ms: float) -> float:
    """Regression allowance: 10%, with the 2ms absolute floor under 5ms."""
    if baseline_ms < FLOOR_APPLIES_BELOW_MS:
        return max(baseline_ms * REGRESSION_RATE, ABS_FLOOR_MS)
    return baseline_ms * REGRESSION_RATE


def _violations(current: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """One human-readable line per op whose p50 regressed beyond its allowance."""
    problems: list[str] = []
    for op, base in baseline.items():
        now = current.get(op)
        if now is None:
            problems.append(f"{op}: no current measurement (baseline {base:.3f}ms)")
            continue
        limit = base + _allowance_ms(base)
        if now > limit:
            problems.append(
                f"{op}: current p50 {now:.3f}ms exceeds baseline {base:.3f}ms "
                f"by more than the allowance ({limit:.3f}ms limit) — >10% regression"
            )
    return problems


@pytest.mark.benchmark
def test_extraction_latency_vs_baseline() -> None:
    """Extraction p50 (fake model, 3-draft output) is within its baseline allowance."""
    baseline = _load_baseline()
    current = measure_current()
    problems = _violations(
        {"extraction_ms_p50": current["extraction_ms_p50"]},
        {"extraction_ms_p50": baseline["extraction_ms_p50"]},
    )
    assert not problems, "; ".join(problems)


@pytest.mark.benchmark
def test_gating_latency_vs_baseline() -> None:
    """Gating p50 (fake model, 3 candidates) is within its baseline allowance."""
    baseline = _load_baseline()
    current = measure_current()
    problems = _violations(
        {"gating_ms_p50": current["gating_ms_p50"]},
        {"gating_ms_p50": baseline["gating_ms_p50"]},
    )
    assert not problems, "; ".join(problems)


@pytest.mark.benchmark
def test_retrieval_latency_vs_baseline() -> None:
    """Warm BM25 retrieval p50 (200 entries) is within its baseline allowance."""
    baseline = _load_baseline()
    current = measure_current()
    problems = _violations(
        {"retrieval_ms_p50": current["retrieval_ms_p50"]},
        {"retrieval_ms_p50": baseline["retrieval_ms_p50"]},
    )
    assert not problems, "; ".join(problems)


@pytest.mark.benchmark
def test_injection_latency_vs_baseline() -> None:
    """Full retrieve+gate+inject p50 (200 entries) is within its baseline allowance."""
    baseline = _load_baseline()
    current = measure_current()
    problems = _violations(
        {"injection_ms_p50": current["injection_ms_p50"]},
        {"injection_ms_p50": baseline["injection_ms_p50"]},
    )
    assert not problems, "; ".join(problems)


@pytest.mark.benchmark
def test_no_regression_greater_than_10_percent() -> None:
    """Aggregate check: NO key operation regressed >10% (floor-guarded) vs baseline."""
    baseline = _load_baseline()
    current = measure_current()
    problems = _violations(current, baseline)
    assert not problems, (
        "performance regression(s) greater than 10% detected: " + "; ".join(problems)
    )
