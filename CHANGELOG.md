# Changelog

All notable changes to hermes-v2 are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project versions with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

hermes-v2 is a memory-first re-architecture of the Hermes agent, built on the
fork base v2026.9.24 (commit f97608f). The authoritative feature inventory is
`docs/architecture.md` (as-built); the plan is
`docs/plan/hermes-v2-improvement-plan-r7.md` (v2.1-r7).

## [2.0.0] - 2026-11-21

First v2 release: the memory system of record moves to a Git-backed atomic
markdown graph with per-fact trust cards, a two-stage extraction/review
pipeline replaces the single memory tool, proactive injection is gated by a
batched CBDC step, and every turn leaves a quality-labeled trace that feeds
weekly GEPA prompt optimization. Fork-base agent code, config and session
storage are preserved; v2 lives in the `hermes/` package and
`hermes_home()` v2 subdirectories.

### Added

- **Storage layer** (P1): `MemoryStore` over a Git-backed atomic markdown
  graph (pygit2, one commit per write), YAML trust-card frontmatter with a
  weighted trust score, BM25 keyword search, cross-process `FileLock`, and a
  one-shot fork-base-to-v2.0 migration with dry-run and rollback.
- **Trust cards and provenance** (P2): verification events on every write
  (plan 2.4 weights: user-explicit/test-green 1.0 ... llm 0.3), rule-based
  ground-truth anchors that append test-green/test-red events on test/build
  completions, a `verify` MCP tool, the forgetting curve
  `score * exp(-0.01 * days)` with daily decay cron and recoverable stale
  archiving.
- **Two-stage memory pipeline** (P3, P7.T5, P8.T2): `ExtractorAgent` drafts
  (max 5/turn, multi-turn context window), `ReviewerAgent` decides
  add/update/skip/conflict against semantic top-5, one resolution pass with
  a finite termination rule, conflict-resolution audit trail in the trust
  card, async Stop-hook dispatch that never blocks the turn, compaction
  summaries as primary extraction input.
- **CBDC proactive injection** (P4): `RetrievalEngine` top-K, single batched
  `CBDCGate` call (use/verify/ignore on top-3, 3.5s timeout with trust-only
  fallback), `ContextInjector` stable/dynamic split that preserves the
  prefix cache, pre-step hook with a hard 5s budget.
- **Project/context layer** (P5): `ContextManager` (cwd / gateway chat id /
  `/context switch` detection and inheritance), per-context memories and
  user profiles, `/context` command.
- **Concurrent writers** (P6): `ConcurrencyManager` read/exclusive locking,
  single-writer `WriterProcess` daemon with length-prefixed JSON protocol
  and batched commits, client-side journal with bounded crash recovery.
- **Long-session handling** (P7): segmented system prompt with selective
  invalidation (memory updates visible mid-session, prefix cache kept),
  topic-shift detection and virtual session boundaries.
- **Memory quality** (P8): deterministic ConflictDetector (trigram TF-IDF
  backend, optional dense embeddings) with a duplicate/contradiction split,
  multimodal attachments under `memory-repo/assets/` with optional OCR and
  audio transcription.
- **GEPA self-optimization** (P10): standard per-component metrics, labeled
  eval sets, GEPA adapters, persistent Pareto fronts, weekly runner +
  scheduler that deploy improved prompt templates.
- **Trace Store** (P11): per-turn traces (JSONL per day + SQLite FTS5
  index), one-trace-per-turn capture wired through the pre-step and Stop
  hooks, rule-based auto quality labeling (`good | acceptable | bad |
  needs-review` — plan P11's skip/add/conflict mapping, refined to treat
  conservative degradation skips as `bad` and deliberate dedup-skips as
  `acceptable`) with a manual override mechanism that validates
  `good | bad | needs-review` and survives reloads.
- **Skills lifecycle** (P13): propose -> approve -> active -> deprecate
  state machine, versioned registry with rollback, 30-day auto-deprecation
  proposals, bundle suggestions, `/skills` and `/learning` commands.
- **Self-diagnostics** (P14): six health checks, 5-minute metrics collector
  (90-day retention), twelve alert rules (log / gateway / email channels),
  five auto-repairs, model-performance / learning-loop / memory-quality
  trend ledgers, CLI and HTTP dashboards, `/diagnostics` command and
  `/health` + `/metrics` endpoints.
- **SOUL split and UX commands** (P15): immutable `soul_core.md` plus style
  bounded to 2000 chars, `/soul` command, `/memory` (list / search / stats /
  export / import).
- **Expanded testing** (P20): chaos, property-based, mutation and
  multi-agent test tiers; mutmut configured with Tier-1/Tier-2 thresholds.
- **Docs** (P16): `docs/architecture.md` (as-built), migration guide, user
  guide, commands reference.

### Changed

- **Single-model architecture** (plan r3): one `Callable[[str], str]` model
  seam serves the extractor, reviewer, CBDC gate, topic detector and GEPA
  rewriter; no tiered model routing in the memory pipeline.
- **Storage layout**: memories move from fork-base `~/.hermes/memories/*.md`
  to the Git-backed `memory-repo/` graph (global + contexts + archive +
  soul + skills + assets); traces, backups and diagnostics live in
  dedicated `hermes_home()` subdirectories, outside Git.
- **Graceful degradation contract** (plan 5.5): hooks are advisory; a
  failing extractor skips and retries next turn; a failing reviewer degrades
  the draft to a conservative skip; the gate falls back to trust-only
  injection; every degradation is logged and surfaces in diagnostics.
- **Documentation** rewritten for v2: architecture (as-built), migration
  guide from v2026.9.24, user guide, commands reference, updated README.

### Fixed

- ThreadPoolExecutor `_interrupted_threads` leak: recycled thread IDs no
  longer inherit a stale interrupted flag (#35309, P21.T1).
- Hot memory is honored and disabled tools are no longer used; the fix is
  the new injection wiring itself (#26568, P21.T2).
- MCP stdio `tools/call` arguments are validated and echoed instead of
  being intermittently dropped (#156143, P21.T3).
- Skills search no longer takes ~86s on a cold cache: bundled/cached index
  with a local fallback (#156140, P21.T4).
- Cron jobs stuck in `running` are force-failed by an inactivity watchdog
  and surfaced as the `cron_job_stuck` alert (#156160, P21.T5).
- state.db WAL errors / session-storage unavailability: SQLite WAL with
  `BEGIN IMMEDIATE` and a connection pool for v2 state (P21.T6).
- Clarify prompts no longer misreport an empty free-text submission as a
  timeout (#156171, P21.T7).
- Gateway shutdown stops retrying after network loss and always exits clean
  (#156127, P21.T8).
- Webapp renderer updates can no longer exit 0 while the build was
  OOM-killed; the updater reports honest failures (#156161, P21.T9).
- The internal `NO_REPLY` marker no longer leaks into gateway delivery
  (#157239, P21.T10).
- Custom model configurations survive updates via zero-loss config
  migration (#25272, P21.T11).
- An ACP turn no longer keeps running after the editor exits: durable
  session state behind a transport-agnostic ACP server (#156173, P21.T12).

### Security

- PII detection before every memory write with configurable redact (default)
  / block / warn (P17.T4); traces carry a PII flag for downstream export
  curation.
- Adversarial input detection blocks memory writes scoring > 0.7 (prompt
  injection, trust manipulation; regex-only when the model is down) (P17.T1).
- Trust anomaly detection holds suspicious trust jumps for human review
  (P17.T3); per-session/per-user write rate limits with burst flags
  (P17.T2).

## Release checklist (v2.0.0)

Status at tag time (2026-11-21):

- **Final gate** — executed by the orchestrator before tagging; all green at
  authoring time: `pytest tests/unit tests/integration tests/property -q -m
  "not live"`, `.venv/Scripts/ruff check hermes/` (line 100;
  E,F,W,I,N,UP,B,SIM), `.venv/Scripts/python -m mypy hermes/` (strict).
- **Mutation CI** (plan Section 4.7): mutmut configured in `pyproject.toml`;
  per-PR diff mutation on changed in-scope modules, nightly full run
  published as a CI artifact. Release exit criterion: Tier-1 >= 95%, Tier-2
  >= 85%, zero unexplained survived Tier-1 mutants.
- **A/B memory-quality test** (plan Section 4.5) — **REQUIRED-BEFORE-PUBLISH**.
  The 50-turn v2026.9.24-vs-v2.1 comparison (extraction precision/recall,
  retrieval relevance, injection usefulness, memory growth rate,
  false-positive rate; pass: >= baseline on all metrics or >= 10% better on
  >= 3) requires live main-model traffic on both versions and is NOT
  runnable in the hermetic CI suite — the planned
  `tests/e2e/test_ab_memory_quality.py` is not present in this build. It
  must be executed against a live model and pass before the release is
  published; per plan 4.5, a failure blocks the release.
- **Packaging**: runtime and optional-dependency groups are declared in
  `pyproject.toml` (P16.T4 clean-install tests live in
  `tests/integration/test_installation.py`); uv.lock regeneration against
  the final dependency set is pending and must land before the PyPI upload.
- **Tag**: the `v2.0.0` tag and GitHub release are created and pushed by the
  orchestrator after the final gate; this CHANGELOG ships inside the tag.
