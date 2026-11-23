# Performance characteristics (hermes-v2)

Plan reference: **P22.T4** ("Re-profile and document performance
characteristics", plan Phase 22). This document records the REAL measured
numbers from the post-optimization profiling runs (P22.T1 profiler, P22.T2
`MemoryCache`, P22.T3 `PromptOptimizer`), what the cache changes — and,
just as important, what it does **not** change.

All numbers below were measured on the development host
(Windows 10.0.26300 x64, CPython 3.13.17 from the repo `.venv`,
repository-local run, 2026-11-21). They are host-specific: re-measure on
your host with the tooling in [Reproducing the numbers](#reproducing-the-numbers).

## Measurement method (same as P20.T4)

Identical to the pre-optimization baseline runs, so before/after numbers
are comparable:

- **Workload**: deterministic — no RNG. A 200-entry corpus written straight
  onto the store's on-disk layout (git commits skipped: the budget covers
  retrieval/gating/composition, not storage write throughput). Extractor
  and gate run instant fake models (single-model architecture — the LLM is
  never called live; what is measured is the real per-turn plumbing).
- **Latency**: per operation, 3 batches each of 1 warm-up + 30 timed calls;
  a batch's p50 is the median of its per-call milliseconds; the op's value
  is the **best batch p50** (min of 3) — the P4.T1 benchmark's
  "best of three runs" pattern that absorbs scheduler noise on shared
  hosts. Values in milliseconds, wall-clock `perf_counter`.
- **Tokens**: the documented chars/4 heuristic (`estimate_tokens`, ceiling)
  over each operation's model-facing payload — retrieval: the query text;
  gating: turn text + every candidate's entry text; extraction: turn text;
  review: each reviewed draft's text.
- **Repeated-query workload**: 5 distinct queries cycled (real sessions
  revisit topics — the profiler's own synthetic pool repeats
  intentionally). This is the workload where the P22.T2 result cache
  legitimately changes the numbers; unique-query numbers are reported
  alongside for honesty.

## Baselines vs after the optimizations

### Stored pre-optimization baseline (P20.T4, no cache)

`tests/benchmark/baseline_metrics.json` (v2.1-r7, rounded UP to 0.1 ms so
the stored reference never understates the measured cost):

| operation | p50 (ms) |
|---|---|
| extraction | 0.1 |
| gating | 0.4 |
| retrieval | 0.8 |
| injection (retrieve+gate+inject) | 1.3 |

Current no-cache re-measurement on this host (same method, sanity check —
all within the P20.T4 10% allowance): extraction 0.052, gating 0.378,
retrieval 0.723, injection 1.262 ms.

### After the P22.T2 result cache (repeated-query pool)

`RetrievalEngine(cache=MemoryCache(ttl=300s, max_entries=256))`, same
200-entry corpus, same best-of-3 batch-p50 method, 5-query repeated pool
(`top_k=5`):

| measurement | cache OFF | cache ON | reduction |
|---|---|---|---|
| retrieval p50 (repeated pool) | 0.857 ms | 0.004 ms | **99.5%** |
| retrieval p50 vs the stored 0.8 ms baseline | — | 0.004 ms | **99.5%** |
| retrieval p50, UNIQUE queries (cache on) | — | 0.667 ms | **0%** |

- The >=20% latency target (plan P22.T4) holds against **both** the stored
  baseline and the same-workload no-cache measurement — a cache hit skips
  the BM25 pass entirely, so the hit cost is a dict lookup + list copy of
  the cached ranking (microseconds).
- **A unique-query workload sees NO benefit** — every call is a miss and
  pays the full BM25 pass (0.667 ms, i.e. the uncached cost). The cache is
  a repeated-query optimization, nothing else.
- Cold start note: the FIRST retrieve on a fresh engine pays a one-time
  store listing + parse of the 200 fact files plus the BM25 index build
  (~3 s on this host, fresh interpreter). It sits outside the steady-state
  numbers above (the measurement method's warm-up call absorbs it) and is
  amortized by the engine's per-scope index cache afterwards; a cache hit
  skips even the per-call fingerprint re-validation of that index.

### Profiler window (n=100 synthetic turns, real engine + cache)

`PerformanceProfiler.profile_memory_ops(n_turns=100)` with the real
`RetrievalEngine` over the 200-entry corpus; the profiler's pool repeats
by design (5 topic queries + every-3rd-turn grounded queries = 39 distinct
queries over 100 calls):

| op | calls | p50 off / on (ms) | est. tokens off | est. tokens on |
|---|---|---|---|---|
| retrieval | 100 | 0.848 / 0.011 | 866 | 866 |
| gating | 100 | <0.1 / <0.1 (fake gate) | 42,396 | 42,396 |
| extraction | 100 | <0.1 / <0.1 (fake extractor) | 866 | 866 |
| review | 100 | <0.1 / <0.1 (fake reviewer) | 1,200 | 1,200 |

Profiler window cache activity: **61 hits / 39 misses (hit rate 0.61)**,
all attributed to the retrieval op (the only op the cache memoizes).

## Token usage: what the cache does and does not save

Honest decomposition — the cache does **not** shrink any prompt:

- Per-op payload estimates are **byte-identical** with the cache on vs off
  (866 = 866 retrieval tokens above). The profiler still records the query
  text for a cache-served call.
- But on a hit **no retrieval-model call happens** — the cached ranking is
  served locally, so the query payload never reaches BM25 (or a vector
  backend's embedding call, which is what the retrieval token estimate
  models; pure-BM25 retrieval makes no model call at all and its real
  token cost is zero either way). The window's model-facing retrieval
  token cost therefore scales with the **miss share**:

  | metric | cache OFF | cache ON | reduction |
  |---|---|---|---|
  | model-facing retrieval tokens (n=100 window) | 866 | 338 | **61%** |

- System-wide, the same window totals 45,328 payload tokens off vs
  ~44,800 model-facing on — a **~1.2%** total reduction. Why so small:
  the gate prompt (42,396 tokens of candidate texts) must be rebuilt and
  re-sent **every turn** regardless of the cache — the injected claims are
  needed for the response. The cache touches only the op it memoizes.
- Unique-query window: miss share 1.0 — **0% token reduction**, same as
  latency.
- The P22.T3 `PromptOptimizer` is the lever for the model-facing prompt
  ops (duplicate lines, boilerplate headers, verbose connectors, over-long
  examples — each suggestion quantified in est. tokens). It is
  suggestions-only by design: prompts are never auto-mutated, because the
  GEPA loop (plan Phase 9/10) owns prompt text. It writes a per-op usage
  ledger to `<hermes_home()>/diagnostics/prompt_usage.jsonl`.

## Cache hit-rate characteristics by workload

`MemoryCache.stats().hit_rate` (cumulative, per-key + batch-key lookups,
LRU + TTL; 300 s TTL / 256 slots in these runs):

| workload | lookups | hit rate |
|---|---|---|
| 5 distinct queries x 10 cycles (dedicated repeated pool) | 50 | **0.90** |
| Profiler's mixed pool (39 distinct queries, n=100) | 100 | **0.61** |
| Unique queries (every call new) | n | **0.00** |

Both repeated-query workloads clear the plan's >=30% acceptance with
margin; the unique-query row is the honest lower bound. Practical
implications:

- Hits dominate when the pool of distinct queries is small relative to the
  call volume (chat sessions that revisit the same topics) — exactly the
  P22.T2 acceptance scenario.
- TTL expiry and LRU eviction surface as misses; capacity-driven drops are
  counted separately as `evictions`.
- The cache never watches the store: every committed memory update must
  invalidate it — wire the store's `on_update` hook to
  `RetrievalEngine.invalidate_cache` (plan P7.T2 hook), or the next
  retrieve serves stale rankings until the TTL expires.

## End-to-end injection budget

Full per-turn chain (retrieve + gate + inject) with the cache wired,
instant fake gate model, repeated 5-query pool, 30 measured turns:
**worst 1.68 ms, median 0.45 ms** — orders of magnitude inside the plan's
budgets (hard <5 s hook timeout, typical <2 s; both are Phase 4 acceptance
values carried into P22.T4). The budget protects the real-model case: with
live gate models the model call dominates, and the plumbing measured here
is the part v2 code controls.

Remaining fixed costs per turn, cache or no cache:

- **Gating** must run every turn (the gate prompt re-sends candidate texts
  — the largest model-facing payload in the window: 42,396 of 45,328
  tokens).
- **Extraction / review** are LLM-bound in production; the sub-millisecond
  fake-model numbers here measure only the plumbing (prompt build, JSON
  parse, validation). Their real cost is the provider's latency and price,
  which is why P22.T3's prompt-size suggestions and the plan's cheaper
  memory-tier option (plan Phase 22 step 6) exist.

## Tooling

- **Profiler** (where time and tokens go, cache miss rates):
  `PerformanceProfiler(store, retrieval, gate, extractor, reviewer,
  cache_stats=cache.cache_stats).profile_memory_ops(n_turns=100)` —
  see `hermes/optimization/performance.py`.
- **Regression tests vs the stored baseline**: `pytest
  tests/benchmark/test_regression.py` — fails CI on any op regressing
  >10% (2 ms absolute floor below 5 ms baselines). CI refreshes
  `baseline_metrics.json` per release tag; a weekly job appends snapshots
  to `tests/benchmark/trend.jsonl` (P20.T4 design).
- **Optimization acceptance** (P22.T4): `pytest
  tests/benchmark/test_optimization.py` pins the >=20% latency, >=15%
  token and >=30% hit-rate targets on the repeated-query workload;
  `pytest tests/integration/test_performance.py` pins the 5 s end-to-end
  budget.
- **Prompt usage ledger**: `PromptOptimizer.record_usage` /
  `usage_report` (`hermes/core/prompt_optimizer.py`), persisted under
  `diagnostics/prompt_usage.jsonl`.

## Reproducing the numbers

```bash
# stored-baseline regression battery (same method as every number above)
.venv/Scripts/python -m pytest tests/benchmark/test_regression.py -q

# P22.T4 acceptance tests (cache wired, repeated-query workloads)
.venv/Scripts/python -m pytest tests/benchmark/test_optimization.py -q
.venv/Scripts/python -m pytest tests/integration/test_performance.py -q -m "not live"

# one-off profiler report
.venv/Scripts/python -c "from hermes.optimization.performance import PerformanceProfiler; ..."
```

The P22.T4 test docstrings (`tests/benchmark/test_optimization.py`,
`tests/integration/test_performance.py`) carry the exact workload shapes
and the honesty notes reproduced here.
