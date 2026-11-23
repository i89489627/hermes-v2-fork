# hermes-v2 Architecture

This document describes hermes-v2 **as implemented** on the `v2-build` branch
(plan: `docs/plan/hermes-v2-improvement-plan-r7.md`, r7). Every module path
cited below exists in the tree. Fork-base (v2026.9.24) code is preserved
untouched; v2 lives in the `hermes/` package tree and `hermes_home()` v2
subdirectories (plan P0).

## 1. Overview

hermes-v2 is a **memory-first re-architecture** of the Hermes agent. The
system of record is a **Git-backed atomic markdown graph**: every memory is a
markdown file with a YAML trust-card frontmatter, and every write is a Git
commit (plan Phase 1). A **single-model architecture** is used throughout
(plan r3): the LLM is injected as a `Callable[[str], str]`
(`prompt -> raw completion`) — the extractor, reviewer, CBDC gate, topic
detector and GEPA reflection rewriter all take the same seam, so production
reaches the fork-base API lazily at call time (`build_default_model`) and
tests always run against fakes. There is no model stack routing in the
memory pipeline itself; model versioning concerns (registry, drift, A/B,
deprecation) live in `hermes/models/` around that single seam.

The fork-base agent (`run_agent.py`, `hermes_cli/`, upstream state machinery)
is preserved; v2 extends it via the hook points of `hermes/core/agent.py`
(`V2Agent`: pre-step, post-turn and command-execution hooks — the
UserPromptSubmit / Stop / PostToolUse equivalents) and adds the v2 command
surface (`/memory`, `/context`, `/soul`, `/skills`, `/learning`,
`/diagnostics`).

## 2. Component map

| Module path | Responsibility (plan task) |
|---|---|
| `hermes/memory/store.py` | `MemoryStore` — atomic trust-card markdown writes, index (`MEMORY.md`), archive dir, per-context facts (P1.T3) |
| `hermes/memory/git_backend.py` | `GitBackend` — pygit2 (libgit2) commits, branch/merge/log; batched commit window (P1.T2, P6.T2) |
| `hermes/memory/trust_card.py` | Trust-card frontmatter serialize/parse; `compute_trust_score` weighted mean (P1.T1, P2.T1) |
| `hermes/memory/types.py` | `EntryType`, `VerificationSource`, the plan 2.4 `SOURCE_WEIGHTS` table (P1.T1) |
| `hermes/memory/entry.py` | `MemoryEntry` runtime record (P1.T1) |
| `hermes/memory/draft.py` | `DraftFact` candidates from extraction (P3.T2) |
| `hermes/memory/extractor.py` | `ExtractorAgent` — turn (+ last 3–5 turns) to draft facts, max 5/turn (P3.T2, P3.T6) |
| `hermes/memory/reviewer.py` | `ReviewerAgent` — add/update/skip/conflict vs semantic top-5; conflict resolution protocol (P3.T3, P8.T2) |
| `hermes/memory/pipeline.py` | `MemoryPipeline` — extract -> review -> apply, one commit per write, audit events (P3.T4) |
| `hermes/memory/search.py` | BM25 keyword search (`rank_bm25`), tags boost (P1.T4) |
| `hermes/memory/retrieval.py` | `RetrievalEngine` — top-K candidates from active context + global (P4.T1) |
| `hermes/memory/gating.py` | `CBDCGate` — one batched use/verify/ignore decision on top-3 (P4.T2) |
| `hermes/memory/injector.py` | `ContextInjector` — stable/dynamic context split, prefix-cache preservation (P4.T3) |
| `hermes/memory/hooks/user_prompt_submit_hook.py` | Pre-step hook: retrieve -> gate -> inject, 5s budget, never raises (P4.T4) |
| `hermes/memory/hooks/stop_hook.py` | Stop hook: fires the pipeline async after each turn (P3.T5) |
| `hermes/memory/verification.py` | Verification event append + score recompute (P2.T1) |
| `hermes/memory/anchors.py` | Rule-based ground-truth anchors from test/build/command completions (P2.T5) |
| `hermes/memory/decay.py`, `decay_scheduler.py` | Forgetting curve `score * exp(-0.01 * days)` + stale archiving; daily cron job (P2.T2, P2.T4, P8.T6) |
| `hermes/memory/conflict_detector.py` | Embedding/BM25 near-duplicate + contradiction detection (P8.T1) |
| `hermes/memory/multimodal.py` (+ `ocr.py`, `transcription.py`) | Image/audio/data attachments under `memory-repo/assets/` (P8.T3) |
| `hermes/memory/context.py` | `ContextManager` — detect/switch/create contexts, inheritance (P5.T1) |
| `hermes/memory/context_window.py` | Recent-turn window for the extractor (P3.T6) |
| `hermes/memory/hot_reload.py` | Segmented system prompt (`<mem-segment>`), selective invalidation (P7.T1) |
| `hermes/memory/soul.py` | `SoulManager` — immutable `soul_core.md` + style bounded to 2000 chars (P15.T1) |
| `hermes/memory/locking.py` | `FileLock` — cross-process exclusive byte-range lock (`.lock`) (P1.T5) |
| `hermes/memory/adversarial_detector.py` | `AdversarialInputDetector` + `store_guard` blocking risk >0.7 (P17.T1) |
| `hermes/memory/rate_limiter.py` | 20 writes/session/h, 100/user/day rolling + burst flags (P17.T2) |
| `hermes/memory/pii_detector.py` | Regex PII detection, redact/block/warn (P17.T4) |
| `hermes/memory/trust_anomaly.py` | Trust anomaly detection (rapid raise, impossible verify) (P17.T3) |
| `hermes/memory/migrations/fork_to_v2_0.py` | One-shot fork-base -> v2.0 migration (P1.T6, P15.T2) |
| `hermes/concurrency/manager.py` | `ConcurrencyManager` — exclusive mutation lock, shared read lock (P6.T1) |
| `hermes/concurrency/writer.py` | `WriterProcess` — single-writer daemon, socket RPC, batching (P6.T2) |
| `hermes/concurrency/protocol.py` | Length-prefixed JSON wire frames, `WriteRequest/Response` (P6.T2) |
| `hermes/concurrency/recovery.py` | Client-side journal + bounded replay to a fresh writer (P6.T3) |
| `hermes/traces/store.py`, `schema.py` | Trace Store: JSONL per day + SQLite/FTS5 index (P11.T1) |
| `hermes/optimization/metrics.py` | Standard evaluation metrics for the optimization loop (P10.T1) |
| `hermes/optimization/pareto.py` | `ParetoFrontStore` — `memory-repo/optimization/pareto-front.json` (P10.T4) |
| `hermes/optimization/adapters/` | GEPA adapter protocol over the component metrics (P10.T3) |
| `hermes/optimization/gepa_runner.py` | `GEPARunner.run_weekly` — seed/evaluate/Pareto/deploy loop; deploys prompt templates (P10.T5) |
| `hermes/optimization/gepa_scheduler.py` | Weekly GEPA cron wiring (P10.T6) |
| `hermes/diagnostics/health_checks.py` | The six plan 5.1 checks + `HealthCheckContext` (P14.T1) |
| `hermes/diagnostics/metrics.py` | `MetricsCollector` — 5-min JSONL snapshots, 90-day retention (P14.T2) |
| `hermes/diagnostics/alerts.py` | The 12 plan 5.3 alert rules + `AlertEngine` (log/gateway/email channels) (P14.T3) |
| `hermes/diagnostics/auto_repair.py` | The five plan 5.4 repairs + `repairs.log` (P14.T4) |
| `hermes/diagnostics/system.py` | `DiagnosticsSystem` — `run_health_check() -> HealthReport`, background collection, alert wiring (P14.T5) |
| `hermes/diagnostics/model_metrics.py` | tok/s, quality, error/fallback rates ledger (P14.T7) |
| `hermes/diagnostics/learning_metrics.py` | GEPA before/after ledger + trend report (P14.T8) |
| `hermes/diagnostics/memory_quality_metrics.py` | Weekly trust/conflict/archive/recall points + trends (P14.T9) |
| `hermes/diagnostics/dashboard_cli.py` | Plain-text dashboard + interactive page system (P14.T10) |
| `hermes/diagnostics/dashboard_http.py` | HTML dashboard (`/`), reuses `/health`, `/metrics` (P14.T10) |
| `hermes/gateway/health.py` | `/health` + `/metrics` JSON endpoints, `/diagnostics` command logic (P14.T6) |
| `hermes/gateway/topic_detector.py`, `session_manager.py` | Virtual session boundaries on topic shift (P7.T3, P7.T4) |
| `hermes/gateway/clarify.py`, `shutdown.py`, `delivery.py` | P21 fixes: clarify empty-submission, fail-safe shutdown notice, no-reply delivery gate (P21.T7/T8/T10) |
| `hermes/skills/lifecycle.py`, `registry.py` | Skill state machine + version store with rollback (P13.T1, P13.T2) |
| `hermes/skills/auto_deprecator.py`, `bundle_suggester.py` | 30-day unused deprecation proposals; bundle suggestions (P13.T3, P13.T6) |
| `hermes/skills/search.py`, `index.py` | Fast bundled/cached skills search + local fallback (P21.T4) |
| `hermes/models/version_registry.py` | Append-only model version registry (P19.T1) |
| `hermes/models/drift_detector.py` | Eval on version change; regression alert + auto-rollback (P19.T2) |
| `hermes/models/router.py`, `ab_testing.py` | Current-version routing, deterministic 80/20 split, A/B comparison (P19.T2, P19.T4) |
| `hermes/models/cloud_client.py`, `cloud_deprecation.py` | Cloud provider deprecation probing (P19.T5) |
| `hermes/cron/scheduler.py`, `watchdog.py` | Cron ticks never silently dropped; run inactivity watchdog (P21.T5) |
| `hermes/mcp/client.py`, `protocol.py`, `verify_tool.py`, `argument_validator.py` | MCP client stack + the `verify` trust tool (P2.T3) |
| `hermes/sessions/store.py` | Durable session state (JSON file swap or SQLite mode) (P21.T12) |
| `hermes/storage/database.py` | SQLite WAL + `BEGIN IMMEDIATE` + connection pool for v2 state (P21.T6) |
| `hermes/backup/manager.py`, `recovery.py`, `git_recovery.py` | Daily `tar.gz` backups, destructive restore with safety copy, fsck/corruption checks (P18.T1–T3) |
| `hermes/install/migrations.py`, `updater.py` | Zero-loss config migration; OOM-safe updater (P21.T11, P21.T9) |
| `hermes/core/agent.py` | `V2Agent` hook points + slash-command dispatch registries (P0.T3, P13.T4, P15.T3/T4) |
| `hermes/core/tool_registry.py`, `tool_executor.py`, `execution_context.py`, `streaming.py` | Tool surface + internal-marker stripping (P0.T3) |
| `hermes/acp/server.py` | Transport-agnostic ACP server surviving editor crashes (P21.T12) |

## 3. Data flow

### 3.1. Turn lifecycle — pre-step injection (UserPromptSubmit)

```
 user prompt
     │
     ▼
 V2Agent.run_pre_step_hooks  (hermes/core/agent.py)
     │
     ▼
 UserPromptSubmitHook (hermes/memory/hooks/user_prompt_submit_hook.py)
     │  context detection: explicit override > session metadata (P5.T1)
     ▼
 RetrievalEngine.search      (hermes/memory/retrieval.py)
     │  BM25 over active context ∪ global, top-K
     ▼
 CBDCGate.classify_batch     (hermes/memory/gating.py)
     │  ONE batched LLM call on the top-3: use | verify | ignore
     │  timeout 3.5s -> trust-only fallback (Section 6 below)
     ▼
 ContextInjector.compose     (hermes/memory/injector.py)
     │  stable block (global + context MEMORY.md, cached per session,
     │  byte-identical for prefix cache) + dynamic block (gated facts)
     ▼
 additionalContext merged into the model call
```

No injection when the gate returns nothing (`{}` — the prompt is never
polluted by an empty wrapper). The whole hook runs on a bounded worker
thread joined for at most 5s; any failure degrades to `{}` and the turn
proceeds (never raises).

### 3.2. Turn lifecycle — post-turn memory write (Stop)

```
 completed turn
     │
     ▼
 V2Agent.run_post_turn_hooks -> StopHook (hermes/memory/hooks/stop_hook.py)
     │  async: daemon thread, returns immediately
     ▼
 MemoryPipeline.run_after_turn (hermes/memory/pipeline.py)
     │
     ├─► ExtractorAgent.extract   draft facts (≤5/turn, ContextWindow last 3–5 turns)
     │        │  main-model failure -> skip, retry next turn (5.5)
     │        ▼
     ├─► ReviewerAgent.review     per draft vs semantic top-5:
     │        add | update | skip | conflict
     │        │  conflict -> exactly ONE resolution pass (or the P8.T2
     │        │  detector protocol: merge/replace/skip); still-conflicting
     │        │  drafts are DROPPED — the loop is finite by design
     │        ▼
     ├─► apply_actions            MemoryStore.add/update — atomic markdown
     │        │  write + one Git commit per write, under the store lock;
     │        │  merge/replace appends an audit event (source `llm`, w 0.3)
     │        ▼
     └─► side effects             trace append (TraceStore), metrics
              (MetricsCollector), GEPA-recorded scores — each failure-safe
```

Anchoring runs outside the turn: the command-execution hook
(`hermes/memory/anchors.py`, `V2Agent.register_command_execution_hook`)
appends `test-green` (weight 1.0) events to entries related to a completed
test/build command. The `verify` MCP tool (`hermes/mcp/verify_tool.py`)
does the same explicitly. Decay is a daily cron job
(`hermes/memory/decay_scheduler.py`) that applies the forgetting curve and
archives stale entries into `memory-repo/archive/`.

### 3.3. GEPA weekly loop

```
 CronScheduler (weekly, hermes/optimization/gepa_scheduler.py)
     │
     ▼
 GEPARunner.run_weekly (hermes/optimization/gepa_runner.py)
     │  per component (extraction / gating / reviewer / conflict):
     │  seed = current prompt template file
     │  -> OptimizationLoop (deterministic; optional reflection LM via
     │     model_factory) -> adapter.evaluate each variant
     │  -> all candidates join ParetoFrontStore
     │     (memory-repo/optimization/pareto-front.json)
     │  -> best candidate DEPLOYED by writing
     │     <hermes_home>/optimization/prompts/<component>.txt
     │  -> ModelVersionRegistry.register (optional on_deploy hook)
     ▼
 LearningMetricsLedger.record_run (before/after scores, hermes/diagnostics/
 learning_metrics.py) -> trend_report + learning_no_improvement alert
 (3 consecutive runs without improvement)
```

### 3.4. Diagnostics cadence

```
 every 5 min:  run_health_check()  -> six checks + metrics block + alerts
               MetricsCollector background thread -> diagnostics/metrics.jsonl
                 (snapshots retained 90 days)
 weekly:       MemoryQualityTracker.record_weekly -> memory_quality.jsonl
               (trust distribution, conflict rate, archive rate, recall@5)
 per GEPA run: LearningMetricsLedger.record_run -> learning_metrics.jsonl
 per turn:     ModelMetricsCollector events -> model_metrics.jsonl
 daily:        BackupManager.create_backup -> backups/YYYY-MM-DD.tar.gz
 every report: AlertEngine.evaluate -> diagnostics/alerts.log (+ optional
               gateway sink / email); failing repairs -> repairs.log
 surfaces:     /diagnostics (hermes/gateway/health.py), /health + /metrics
               JSON, dashboard (CLI + HTTP, hermes/diagnostics/dashboard_*)
```

## 4. Storage layout (actual paths)

Everything derives from `hermes.config.hermes_home()` (honours
`HERMES_HOME`; default `~/.hermes`):

```
<hermes_home>/
├── memory-repo/                        # the Git-backed memory graph
│   ├── .git/                           # pygit2 commits (one per write)
│   ├── MEMORY.md                       # global index (ids per context)
│   ├── global/facts/*.md               # cross-context trust-card entries
│   ├── contexts/<name>/facts/*.md      # context-scoped entries (+ MEMORY.md)
│   ├── archive/*.md                    # decayed entries (recoverable, stamped `archived_at`)
│   ├── soul/soul_core.md               # immutable (no write API)
│   ├── soul/soul_style.md              # adaptive, hard-bounded 2000 chars
│   ├── soul/.style_history.log         # style updates (when not Git-visible)
│   ├── skills/_skill_registry.json     # lifecycle state machine store
│   ├── skills/_skills_index.json       # cached skills search index
│   ├── skills/skills_usage.json        # per-skill usage ledger (P13.T3)
│   ├── skills/pending_deprecations.json# auto-deprecation proposals
│   ├── assets/{images,audio,data}/     # multimodal attachments (P8.T3)
│   └── optimization/pareto-front.json  # GEPA Pareto fronts (P10.T4)
├── optimization/prompts/<component>.txt # GEPA deployment surface (templates)
├── traces/                             # NOT in Git (large)
│   ├── v1.0/YYYY-MM-DD.jsonl           # JSONL source of truth
│   └── traces.db                       # SQLite + FTS5 mirror index
├── backups/YYYY-MM-DD.tar.gz           # daily archives incl. .git
├── diagnostics/
│   ├── metrics.jsonl                   # 5-min snapshots, 90-day retention
│   ├── model_metrics.jsonl             # per-generation model events
│   ├── learning_metrics.jsonl          # per-GEPA-run outcomes
│   ├── memory_quality.jsonl            # weekly quality points
│   ├── alerts.log                      # fired alerts (JSON lines)
│   └── repairs.log                     # auto-repair attempts
├── journal/writer_journal.jsonl        # client-side write journal (P6.T3)
├── sessions/<session-id>.json          # durable ACP session state (JSON mode)
├── state-v2.db                         # v2 SQLite session/state (SQLite mode)
├── config.yaml                         # fork-base config; v2 keys under `v2:`
└── hermes.db                           # fork-base FTS5/session DB (untouched)
```

## 5. Trust model

**Weights** (`hermes/memory/types.py:SOURCE_WEIGHTS`, plan 2.4 table —
import-time validated):

| Verification source | Weight |
|---|---|
| `user-explicit` | 1.0 |
| `test-green` (rule-based anchor) | 1.0 |
| `code-review-human` | 0.95 |
| `tool-output` | 0.85 |
| `file-read` | 0.7 |
| `agent-inferred` | 0.5 |
| `llm` (no anchor) | 0.3 |

**Score**: `trust_score = Σ(w_i · v_i) / Σ(w_i)` over the entry's
verification events (`hermes/memory/trust_card.py:compute_trust_score`),
clamped to [0, 1]; an unverified history scores 0.5 (the migration default).
A negative anchor (failing test) contributes `v_i = 0`.

**Decay**: the forgetting curve multiplies the score by
`exp(-0.01 · days_since_verification)` (`hermes/memory/decay.py`); entries
below the staleness threshold are archived (recoverable) with an
`archived_at` stamp by the daily scheduler.

**Guards** (plan Phase 17): `AdversarialInputDetector`
(`hermes/memory/adversarial_detector.py`) blocks memory writes scoring
>0.7 (prompt-injection / trust-manipulation patterns; regex-only when the
model is down); `RateLimiter` caps 20 writes/session/h and 100/user/day and
flags bursts; `TrustAnomalyDetector` holds suspicious trust jumps (≥0.3 in
one event, impossible verifications) out of the applied history for human
review; `PIIDetector` redacts (default), blocks or warns on sensitive data
before the write.

## 6. Concurrency model

- **`FileLock`** (`hermes/memory/locking.py`): cross-process exclusive
  byte-range lock on `<memory-repo>/.lock` (msvcrt on Windows / fcntl on
  POSIX), 10s default wait, reentrant per handle; rejections raise
  `LockTimeoutError`.
- **`ConcurrencyManager`** (`hermes/concurrency/manager.py`): every mutating
  operation serializes on the exclusive lock; readers take the shared lock;
  `queue_write` funnels callables through the single writer.
- **`WriterProcess`** (`hermes/concurrency/writer.py`): one single-writer
  daemon per store; clients send length-prefixed JSON frames
  (`hermes/concurrency/protocol.py`) over a local socket; commits are
  batched (0.2s window / batch-max ops) and flushed on demand.
- **Journal recovery** (`hermes/concurrency/recovery.py`, plan P6.T3): the
  CLIENT journals every op (`<hermes_home>/journal/writer_journal.jsonl`,
  journal-before-send, fsync) and removes entries on ack. After a crash,
  `recover()` replays journaled-unacked ops to a fresh writer under a 5s
  budget, then commits the dirty repo state as one `[memory] recovery`
  commit.
- **libgit2 single-handle constraint**: libgit2 keeps one in-memory index
  per repository handle and never reloads it from disk; two handles
  committing interleaved on one repository drop each other's staged files.
  ALL writes therefore flow through ONE `MemoryStore`/`GitBackend` handle
  (the writer's); recovery reloads the index from disk before its durability
  commit (documented in `recovery.py`, implemented in
  `MemoryStore._stage_removal`).

## 7. Failure modes (plan 5.5 graceful degradation)

Every degradation is logged and surfaces in diagnostics
(`fallback_rate_high` alert via the P14.T7 model-metrics ledger):

| Main-model failure | Implemented behaviour |
|---|---|
| Memory extraction | skipped, retried next turn (`MemoryPipeline`, `on_persistent_failure: skip` config key) |
| CBDC gating | trust-only fallback: inject top-3 by trust score iff all >0.7, else skip — no LLM (`hermes/memory/gating.py`) |
| Conflict detection | embeddings/BM25 similarity only, contradiction LLM pass skipped (`hermes/memory/conflict_detector.py`) |
| Adversarial detection | regex patterns only (`hermes/memory/adversarial_detector.py`) |
| Reviewer / topic detector | a reviewer failure degrades that draft to a conservative `skip` (never auto-adds unaudited content; rules always constrain the model); the topic detector falls back to a fully local lexical classifier — the turn is never blocked |

Additional guarantees: hooks are advisory (a failing hook is logged and
skipped, never breaks the turn); the injection hook has a hard 5s budget;
health checks, alert rules, repair attempts, notification channels and
dashboard sections are each exception-isolated (one failure degrades that
piece to a logged/`unavailable` result); HTTP endpoints answer `500` JSON
instead of raising; a cron tick that is late is still executed, never
silently dropped (`hermes/cron/scheduler.py`); the shutdown notifier never
raises and always exits clean (`hermes/gateway/shutdown.py`).
