"""PerformanceProfiler: latency/token/cache profiling of the memory ops (plan P22.T1).

``profile_memory_ops(n_turns=100) -> ProfileReport`` runs a deterministic
synthetic workload (n turns, seeded turn texts — no RNG) through the four
key memory operations and reports WHERE the time and tokens go:

- **retrieval** — the injected retrieval engine's ``retrieve(query)``.
- **gating** — the injected gate's batched ``gate(candidates, turn_text)``.
- **extraction** — the injected extractor's ``extract(turn_text)``.
- **review** — the injected reviewer's ``review(draft)`` per extracted draft.

Collaborators are injected (single-model architecture — tests and profiling
runs always pass deterministic fakes; the plan's hard inputs are the P4.T4
pipeline components). ``cache_stats`` is the P22.T2 ``MemoryCache`` stats
seam: a ``Callable[[], (hits, misses)]]`` over CUMULATIVE counters. The
profiler samples it around EVERY operation and attributes each delta to the
operation that was running, so whatever the cache memoizes lands on the
right op; ``None`` (before P22.T2 lands) simply yields zeros everywhere.

Estimated tokens use the documented chars/4 heuristic
(:func:`estimate_tokens`, ceil) over each op's model-facing payload — the
text the profiler feeds the op:

- retrieval: the query text;
- gating: the turn text plus every candidate's entry text (the batched
  gate-prompt payload);
- extraction: the turn text;
- review: each reviewed draft's text.

Rankings: :attr:`ProfileReport.slowest` is the top-3 operations by CUMULATIVE
``total_ms`` (what dominates the window — the optimization target), and
:attr:`ProfileReport.most_expensive` the top-3 by cumulative ``est_tokens``;
ties break alphabetically by op name (deterministic reports). Only
operations with at least one call rank (a zero-call op carries no signal).
The report covers ALL FOUR operations regardless (plan P22.T1 acceptance);
percentiles are nearest-rank over the per-call samples (p50 = the median
rank, p95 = the 95th-nearest rank — no interpolation, so reports never
contain values no call actually achieved).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from hermes.memory.extractor import ExtractorAgent
from hermes.memory.gating import CBDCGate
from hermes.memory.retrieval import RetrievalEngine, RetrievedEntry
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore

__all__ = [
    "OpProfile",
    "PerformanceProfiler",
    "ProfileReport",
    "estimate_tokens",
]

#: Op name: BM25 (+ vector) candidate retrieval (plan P4.T1).
RETRIEVAL_OP: Final[str] = "retrieval"

#: Op name: batched CBDC gating over the retrieved candidates (plan P4.T2).
GATING_OP: Final[str] = "gating"

#: Op name: turn -> draft-fact extraction (plan P3.T2).
EXTRACTION_OP: Final[str] = "extraction"

#: Op name: dedup/update/conflict review per draft (plan P3.T3).
REVIEW_OP: Final[str] = "review"

#: All covered operations, in pipeline order (plan P22.T1 acceptance list).
_ALL_OPS: Final[tuple[str, ...]] = (RETRIEVAL_OP, GATING_OP, EXTRACTION_OP, REVIEW_OP)

#: Size of the deterministic synthetic query pool the window cycles over.
#: Repeats are INTENTIONAL: real sessions revisit topics, and repeated
#: queries are what exercise the P22.T2 retrieval cache in the profile.
_TOPIC_POOL_SIZE: Final[int] = 5

#: Retrieval top_k used while profiling (plan Section 2.5 retrieval default).
_PROFILE_TOP_K: Final[int] = 20

#: How many chars of a store entry ground a synthetic query (when the store
#: has entries, every third turn queries real memory content instead).
_GROUNDED_QUERY_WORDS: Final[int] = 6


def estimate_tokens(text: str) -> int:
    """Documented chars/4 token heuristic, ceiling (0 for empty text).

    ~4 characters per token is the usual English-prose approximation; the
    ceiling keeps a non-empty payload from estimating zero tokens.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile of an ASCENDING-sorted sample list."""
    if not sorted_samples:
        return 0.0
    rank = max(1, math.ceil(pct * len(sorted_samples)))
    return sorted_samples[min(rank, len(sorted_samples)) - 1]


@dataclass
class OpProfile:
    """Per-operation profile over one profiling window (plan P22.T1).

    ``calls`` is the number of timed invocations, ``total_ms`` their summed
    wall time, ``p50_ms``/``p95_ms`` the nearest-rank percentiles over the
    per-call samples, ``est_tokens`` the cumulative chars/4 payload estimate
    and ``cache_hits``/``cache_misses`` the P22.T2 cache deltas attributed to
    this op (0/0 when no cache is wired).
    """

    name: str
    calls: int
    total_ms: float
    p50_ms: float
    p95_ms: float
    est_tokens: int
    cache_hits: int
    cache_misses: int


@dataclass
class ProfileReport:
    """Result of one :meth:`PerformanceProfiler.profile_memory_ops` run.

    ``ops`` covers every memory operation (retrieval, gating, extraction,
    review); ``slowest`` / ``most_expensive`` are the top-3 rankings; the
    cache miss rate spans the whole window; ``window_turns`` records the
    synthetic-turn count the profile ran on.
    """

    ops: dict[str, OpProfile]
    slowest: list[tuple[str, float]]
    most_expensive: list[tuple[str, int]]
    cache_miss_rate: float
    window_turns: int


class PerformanceProfiler:
    """Profiles the four memory operations over a synthetic turn window.

    All collaborators are injected; ``store`` grounds the workload (when it
    holds entries, some synthetic queries are derived from real entry texts,
    deterministically) and is otherwise untouched — profiling never writes.
    """

    def __init__(
        self,
        store: MemoryStore,
        retrieval: RetrievalEngine,
        gate: CBDCGate,
        extractor: ExtractorAgent,
        reviewer: ReviewerAgent,
        cache_stats: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self._store = store
        self._retrieval = retrieval
        self._gate = gate
        self._extractor = extractor
        self._reviewer = reviewer
        #: P22.T2 MemoryCache stats seam: cumulative (hits, misses) snapshot.
        self._cache_stats = cache_stats

    # ── workload generation (deterministic, no RNG) ────────────────────────
    def _turn_texts(self, n_turns: int) -> list[str]:
        """Seeded synthetic turn texts for the profiling window.

        Base pool: ``_TOPIC_POOL_SIZE`` templates cycled over the window
        (repeats intentional — see the constant). When the store holds
        entries, every third turn queries the first words of a store entry
        instead, grounding the workload in real memory content; the store
        listing is workload SETUP, not part of any measured op.
        """
        texts = [
            f"topic-{i % _TOPIC_POOL_SIZE} synthetic memory query" for i in range(n_turns)
        ]
        entries = self._store.list()
        if entries:
            for i in range(0, n_turns, 3):
                grounded = " ".join(
                    entries[i % len(entries)].text.split()[:_GROUNDED_QUERY_WORDS]
                )
                if grounded:
                    texts[i] = grounded
        return texts

    # ── cache stats attribution ─────────────────────────────────────────────
    def _sample_cache(self) -> tuple[int, int]:
        """Current cumulative (hits, misses); ``(0, 0)`` without a cache."""
        if self._cache_stats is None:
            return (0, 0)
        hits, misses = self._cache_stats()
        return (hits, misses)

    @staticmethod
    def _cache_delta(
        before: tuple[int, int], after: tuple[int, int]
    ) -> tuple[int, int]:
        """Non-negative (hits, misses) delta between two cumulative samples.

        Deltas are clamped at 0 so a cache resetting its counters mid-profile
        can never attribute negative activity to an operation.
        """
        return (max(0, after[0] - before[0]), max(0, after[1] - before[1]))

    # ── public API (plan P22.T1) ────────────────────────────────────────────
    def profile_memory_ops(self, n_turns: int = 100) -> ProfileReport:
        """Run ``n_turns`` synthetic turns through the four ops; return the report.

        Each turn runs retrieval -> gating -> extraction -> review (review
        once per extracted draft) with per-call latency sampling, payload
        token estimation and per-op cache-stat attribution. ``n_turns`` must
        be positive. Latencies are wall-clock ``perf_counter`` deltas in
        milliseconds; nothing is written to the store.
        """
        if n_turns <= 0:
            raise ValueError(f"n_turns must be >= 1, got {n_turns}")

        samples: dict[str, list[float]] = {op: [] for op in _ALL_OPS}
        tokens: dict[str, int] = {op: 0 for op in _ALL_OPS}
        cache_totals: dict[str, tuple[int, int]] = {op: (0, 0) for op in _ALL_OPS}

        for turn_text in self._turn_texts(n_turns):
            # cumulative cache snapshot carried across the turn's ops so each
            # delta is attributed to EXACTLY the op that ran since the last
            cache_snapshot = self._sample_cache()

            # retrieval (plan P4.T1) — the cache seam memoizes exactly this op
            start = time.perf_counter()
            candidates: list[RetrievedEntry] = self._retrieval.retrieve(
                turn_text, top_k=_PROFILE_TOP_K, context="global"
            )
            samples[RETRIEVAL_OP].append((time.perf_counter() - start) * 1000.0)
            tokens[RETRIEVAL_OP] += estimate_tokens(turn_text)
            after = self._sample_cache()
            delta = self._cache_delta(cache_snapshot, after)
            cache_totals[RETRIEVAL_OP] = (
                cache_totals[RETRIEVAL_OP][0] + delta[0],
                cache_totals[RETRIEVAL_OP][1] + delta[1],
            )
            cache_snapshot = after

            # gating (plan P4.T2): ONE batched call over the turn's candidates
            start = time.perf_counter()
            self._gate.gate(candidates, context="global", turn_text=turn_text)
            samples[GATING_OP].append((time.perf_counter() - start) * 1000.0)
            payload = turn_text + "".join(c.entry.text for c in candidates)
            tokens[GATING_OP] += estimate_tokens(payload)
            after = self._sample_cache()
            delta = self._cache_delta(cache_snapshot, after)
            cache_totals[GATING_OP] = (
                cache_totals[GATING_OP][0] + delta[0],
                cache_totals[GATING_OP][1] + delta[1],
            )
            cache_snapshot = after

            # extraction (plan P3.T2): turn -> draft facts
            start = time.perf_counter()
            drafts = self._extractor.extract(turn_text)
            samples[EXTRACTION_OP].append((time.perf_counter() - start) * 1000.0)
            tokens[EXTRACTION_OP] += estimate_tokens(turn_text)
            after = self._sample_cache()
            delta = self._cache_delta(cache_snapshot, after)
            cache_totals[EXTRACTION_OP] = (
                cache_totals[EXTRACTION_OP][0] + delta[0],
                cache_totals[EXTRACTION_OP][1] + delta[1],
            )
            cache_snapshot = after

            # review (plan P3.T3): ONE decision per draft
            for draft in drafts:
                start = time.perf_counter()
                self._reviewer.review(draft)
                samples[REVIEW_OP].append((time.perf_counter() - start) * 1000.0)
                tokens[REVIEW_OP] += estimate_tokens(draft.text)
            after = self._sample_cache()
            delta = self._cache_delta(cache_snapshot, after)
            cache_totals[REVIEW_OP] = (
                cache_totals[REVIEW_OP][0] + delta[0],
                cache_totals[REVIEW_OP][1] + delta[1],
            )

        ops: dict[str, OpProfile] = {}
        for op in _ALL_OPS:
            call_samples = samples[op]
            ordered = sorted(call_samples)
            hits, misses = cache_totals[op]
            ops[op] = OpProfile(
                name=op,
                calls=len(call_samples),
                total_ms=round(sum(call_samples), 3),
                p50_ms=_percentile(ordered, 0.50),
                p95_ms=_percentile(ordered, 0.95),
                est_tokens=tokens[op],
                cache_hits=hits,
                cache_misses=misses,
            )

        active = [profile for profile in ops.values() if profile.calls > 0]
        slowest = [
            (profile.name, profile.total_ms)
            for profile in sorted(active, key=lambda p: (-p.total_ms, p.name))[:3]
        ]
        most_expensive = [
            (profile.name, profile.est_tokens)
            for profile in sorted(active, key=lambda p: (-p.est_tokens, p.name))[:3]
        ]
        total_hits = sum(profile.cache_hits for profile in ops.values())
        total_misses = sum(profile.cache_misses for profile in ops.values())
        lookups = total_hits + total_misses
        cache_miss_rate = (total_misses / lookups) if lookups else 0.0

        return ProfileReport(
            ops=ops,
            slowest=slowest,
            most_expensive=most_expensive,
            cache_miss_rate=cache_miss_rate,
            window_turns=n_turns,
        )
