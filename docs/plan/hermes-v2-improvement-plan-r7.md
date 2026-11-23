# Hermes Improvement Plan — v2.1 "Memory-First Hermes"

**Engineering plan / executable specification** — 2026-11-18
**Version:** v2.1-r7 (after 6 checks; r6 — full English translation; r7 — dry-run remediation: fork-base pinned, create-before-modify fixed, 9 dependency edges, pytest importlib mode)
**Supersedes:** v2.0 (which had internally inconsistent round 1/2/3 addendum sections)

> This document is a self-contained improvement plan for Hermes Agent (Nous Research).
> It can be handed to another coding agent in another harness without any context from
> the original session. It contains the architecture, per-phase tasks, tests as
> executable specs, and a self-diagnostics system.

**Fork base:** `github.com/NousResearch/hermes-agent` (v2026.9.24, commit f97608f, MIT license)
**Target version:** Hermes v2.0.0
**Executor:** coding agent / harness (not a human engineer). Labor-time estimates (hours/days/weeks) were removed in r5: for agent execution the work order is defined by the dependency graph (Section 7 + tasks.yaml); runtime budgets (gate 3.5s etc.) are kept as product requirements.

---

## Revision history

| Round | Date | Changes | Trigger |
|---|---|---|---|
| v2.0-r1 | 2026-11-17 | Initial plan: 17 phases, 62 tasks | — |
| v2.0-r2 | 2026-11-17 | +4 phases, 13 tasks; fixed dependencies | GLM-5.3 review (verdict: Rework needed) |
| v2.0-r3 | 2026-11-17 | Removed tiered models; added Phase 21 bug fixes | User feedback (simplify) |
| v2.1 | 2026-11-18 | Cleaned and integrated: fixed 13 errors from self-review; added cost estimation, rollback strategy, A/B regression test, prioritized bug fixes, Phase 22 performance optimization | Self-review after GLM-5.3 round 2 (verdict: Ship with revisions) |
| v2.1-r3.1 | 2026-11-18 | Sync pass: tasks.yaml aligned to plan (removed phases 9/12 refs, merged supplements, added Phase 22 tasks, fixed stale Ollama/distillation references); added P2.T5 automatic ground-truth anchors and P7.T5 session compaction summaries; fixed tool-output weight in example (0.55 → 0.85); removed stale plan PDF | Final audit vs original ZCode-plugin hypothesis |
| v2.1-r3.2 | 2026-11-18 | Check 3: mutation testing policy — new Section 4.7 (mutmut, Tier-1 ≥95% / Tier-2 ≥85% killed), tasks P20.T5/T6, Phase 21 regression tests must kill bug re-mutants, mutation CI gates in Phase 0 + Section 4.4. Fixed: day/hour mismatches (P3 6.5d, P10 7d, P14 7d, P19 3d, P21 10d — 80h ≠ 5 days); Phase 4 gating → single batched call + 3.5s timeout trust-only fallback; P21.T5 watchdog alerts surface via Phase 14 diagnostics (single alerting channel, `cron_job_stuck`); GEPA rewriter = compaction generator; phase numbering note | User check 3: "where applicable — all tests must be mutation-tested" |
| v2.1-r4 | 2026-11-18 | Check 4 (final): diagram 2.1 synced with check 3 (gate = 1 batched call; "query rewriter" → compactor/rewriter, double counting of components removed in 1.4/2.2/5.2); phase-level Dependencies lines reconciled with the task graph (P4.T3 ← P5.T1, P3.T6 ← P11.T1, P18.T3 ← P14.T1, P20 extended, P22: hard deps vs scheduling); P21.T11 decoupled from P1.T6/P18.T4 — critical bugs T1+T11 are now actually executable right after Phase 0, T2 lands with Phase 4; Phase 14 = 6 health checks (backup_freshness added, P14.T1 ← P18.T1) and 12 alert rules (+fallback_rate_high, synced with 5.3); Phase 4 latency wording (typical <2s / hard 5s / gate 3.5s, benchmark under_5s); cost table 2.6 recalculated ($15-90/1K turns, was $15-180; break-even Sonnet vs Haiku); phase_21 goal 10→12; Appendix A: + mcp/, sessions/, install/, cron/scheduler.py; ~264k stars; P0.T4 nightly mutation job. Files renamed with the check version (-r4) | User check 4 + "add the check version to file names" |
| v2.1-r5 | 2026-11-18 | Labor-time estimates removed: effort_hours (YAML: 112 tasks + 21 phases), estimated_total_hours and effort_model (meta), Effort lines and day counts in phase headers, Hours column in Phase 21 tables, days in the Section 7 diagram and Total estimate, engineer-weeks in the header. Execution order is now defined by the dependency graph only; runtime budgets (gate 3.5s, backup 25h, GEPA <8 days etc.) untouched. Files renamed: -r4 → -r5 | User: "the plan is meant to be executed by a coding agent or harness, not a human engineer" |
| v2.1-r6 | 2026-11-19 | Full English translation of the plan and tasks.yaml — the documents are handed to coding agents in any harness, single-language now; no content changes (the only content edit: the Section 3 intro no longer mentions the removed effort estimate). Files renamed: -r5 → -r6 | User: "translate plan+tasks fully to English, bump to r6, then run check 5" |
| v2.1-r7 | 2026-11-20 | Check 6 dry-run remediation (the 7 verdict points): fork-base pinned to v2026.9.24 (commit f97608f — v0.18.2 does not exist upstream; migration artifacts renamed to fork-neutral names); creator-first file plan — every files_to_modify target now has a creator: 20 paths moved from files_to_modify to files_to_create (Phase 21 tasks author the v2 modules they fix, per Appendix A; hermes/core/agent.py created in P0.T3 per Phase 0 Files; P14.T6 creates hermes/gateway/health.py instead of the bare directory); 9 dependency edges added (write races P18.T3/P8.T2, missing edges P11.T2/P20.T6, router.py ordering P19.T4, sessions/store.py P21.T6); tests/**/__init__.py package skeleton (P0.T3) keeps pytest default import mode working (pytest 9 removed the importmode ini option — the verdict's option A is unexecutable, option B implemented); 3 duplicate tests renamed with owning-phase suffix (_p20, _p8); P0.T4 ci.yml moved to files_to_modify; Section 7 flags now document P14.T1→P18.T1 and P3.T6→P11.T1 | User: "apply the 7 r7 points with a validator" |

---

## 0. Executive Summary

Hermes v2.1 is a re-architecting of Hermes memory that integrates proven patterns from
other harnesses (DSH, Claude Code, ZCode) and two hypotheses from the architectural
discussion:

1. **Atomic markdown graph in Git** with trust cards, a two-stage extractor+reviewer
   pipeline, CBDC-gated proactive injection (fixes: silent truncation, no versioning,
   no provenance, no conflict detection)
2. **GEPA self-optimization** — evolving the prompts of all memory agents against explicit
   metrics, with a Pareto front per component (fixes: opaque learning loop, no quality metric)

**Tiered model architecture was considered and rejected** (round 3): all memory LLM
operations use the main model, as in Hermes v2026.9.24. Rationale — see Section 1.5.

Additionally: 12 improvements from the Hermes architectural review (project/context layer,
concurrent writers, memory hot-reload, forgetting curve, multi-modal memory, skill
lifecycle UI, learning loop dashboard, topic detection on gateways, SOUL.md split,
memory introspection commands, memory export/import, model-aware memory strategy),
plus **Phase 21**: fixes for 12 known Hermes bugs from open GitHub issues, and
**Phase 22**: performance optimization after the feature work is implemented.

**Result:** Hermes remains an agent-first long-lived autonomous agent, but the memory
pipeline becomes Git-backed, trust-aware, self-optimizing, and observable through an
explicit diagnostics system. Single-model architecture (main model for all LLM ops).

---

## 1. Context & Motivation

### 1.1. What Hermes is (recap)

Hermes Agent is an open-source autonomous AI agent by Nous Research, v2026.9.24 (September 2026),
MIT license. It runs as a CLI, TUI, desktop app, OpenAI-compatible API server, and gateway
on 20+ messaging platforms (Telegram, Discord, Slack, WhatsApp, Teams, Signal, Matrix,
Mattermost, Email, SMS, LINE, Google Chat). ~264k GitHub stars (Nov 2026).

Architecture: a single `AIAgent` class in `run_agent.py` routes all surfaces (CLI, gateway,
API server, cron scheduler). Multi-provider (200+ models via Nous Portal, OpenRouter,
direct API keys, local llama.cpp/MLX/vLLM). Minimum 64K context. ThreadPoolExecutor with
8 workers for tool calls. Streaming tokens.

### 1.2. Strengths of the current Hermes memory

- **Three-file separation**: MEMORY.md (personal notes), USER.md (user profile), SOUL.md
  (immutable voice) — clean separation of concerns
- **Frozen snapshot pattern**: memory is injected into the system prompt once at session
  start and does not change within it — preserves prefix cache
- **Hard char limits without auto-compaction**: 2200 chars for MEMORY.md, 1375 for
  USER.md — the agent self-manages via the memory tool (add/patch/remove) and must
  consolidate when limits overflow
- **Skills system (procedural memory)**: autonomous skill creation (YAML in
  `~/.hermes/skills/`), Skill Bundles group skills under slash commands
- **Episodic memory with FTS5**: cross-session recall via SQLite FTS5 + LLM summarization
- **Learning loop**: observe → distill → reuse → refine cycle
- **Dynamic tool calling**: 60+ built-in tools in toolsets, MCP support, dynamic skill
  invocation (progressive disclosure), Dynamic Workflow for fan-out tasks

### 1.3. Weaknesses (what v2.0 fixes)

A full architectural review was performed while preparing this plan. In brief:

**Architectural:**
1. Frozen snapshot breaks on long messaging sessions (Telegram chats lasting weeks)
2. One writer per Hermes home — no concurrent agents
3. No project/context layer — all memories are mixed together
4. No memory versioning/rollback

**Memory quality:**
5. No provenance / trust score on entries
6. No forgetting curve — memory grows forever
7. No conflict detection pre-write
8. Memory is text-only, no multi-modal

**Learning loop:**
9. Skills creation is opaque — no UI, no lifecycle
10. Learning loop has no dashboard — no visibility
11. SOUL.md immutable vs adaptive identity tension
12. Skills bundles static

**UX:**
13. No memory introspection commands
14. No memory export/import
15. Topic detection on gateways — the user has to run `/new` manually
16. USER.md one-dimensional

**Performance:**
17. Hard dependency on prefix cache — does not adapt to different models
18. Char limits are arbitrary and rigid — power users outgrow them

### 1.4. Hypotheses implemented in v2.1

**Hypothesis 1: Atomic markdown graph + trust cards + extractor/reviewer + CBDC gating**

- Atomic markdown notes (one fact per file) with a MEMORY.md index
- Git as the primary store (versioning, audit trail, merge conflicts as a signal)
- Trust card with provenance: `verified_by`, `verified_at`, `evidence`, `trust_score`
- Two-stage pipeline: extractor → reviewer (dedup + in-place update), both on the main model
- CBDC gating: retrieval → use/verify/ignore decision → inject only "use" items within budget
- Proactive injection via pre-step hooks with prefix cache preservation (segment-level
  invalidation, stable/dynamic context split)

**Hypothesis 2: GEPA self-optimization**

- GEPA (Genetic-Pareto) optimizer evolving the prompts of all memory subagents
- Pareto front per component (extractor, reviewer, gate, compactor/rewriter — 4 components;
  the rewriter is the compaction-summary generator, Phase 7 SessionCompactor)
- Trace Store accumulates execution traces for GEPA reflection
- Offline weekly runs via the scheduler

### 1.5. Rejected hypothesis: Tiered model architecture

**Considered in round 2, rejected in round 3.** The full description of the approach is
preserved in [tiered-model-architecture.md](./tiered-model-architecture.md) for historical
reference.

The rejected approach in essence: lightweight local models (Qwen-2.5-3B for
extractor/reviewer, Llama-3.2-1B for gating, etc.) for memory ops, the main cloud model
for reasoning, and a distillation loop to close the quality gap.

**Reasons for rejection:**
- Resource-intensive at this stage: requires GPU/NPU, Ollama runtime, Unsloth for fine-tuning
- Hermes runs on a single model — the added architectural complexity is not justified
- The latency benefit (50-200ms vs 1-5s) is less critical than assumed: the memory pipeline
  runs async after the Stop hook and does not block the turn
- The distillation loop adds operational complexity (dataset quality filtering,
  A/B testing, model versioning) without a guaranteed early-stage benefit

**Possible return path:** if a future v3.0 needs cost/latency optimization, the tiered
model architecture can be reintroduced from git history (see
[tiered-model-architecture.md](./tiered-model-architecture.md) for the full design).

---

## 2. Target Architecture

### 2.1. High-level diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER SURFACES (unchanged)                                           │
│  CLI | TUI | Desktop | API server | 20+ gateways                     │
└───────────────────────────────────────────▲──────────────────────────┘
                                            │
┌───────────────────────────────────────────┴──────────────────────────┐
│  AGENT CORE (AIAgent class — extended)                               │
│  Main model (cloud or heavy local — one model for all)               │
│  • Actual task work (coding, reasoning, tool use)                    │
│  • Memory pipeline ops (extractor, reviewer, gate, compaction)       │
│    — all via AIAgent.chat() with the main model                      │
│  • Receives additionalContext via pre-step hooks                     │
└───────────────────────────────────────────▲──────────────────────────┘
                                            │ verified context injection
                                            │
┌───────────────────────────────────────────┴──────────────────────────┐
│  MEMORY PIPELINE (single-model, all LLM ops via the main model)      │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  extractor  │  │  reviewer   │  │    gate     │  │  compactor  │  │
│  │ main model  │  │ main model  │  │ main model  │  │ (= rewriter)│  │
│  │  (async)    │  │  (async)    │  │ (1 batched) │  │ main model  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │                │         │
│         └────────────────┴────────────────┴────────────────┘         │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  MEMORY STORAGE LAYER                                          │  │
│  │  • Atomic markdown files in Git (${HERMES_HOME}/memory-repo/)  │  │
│  │  • MEMORY.md index + per-fact *.md with trust card frontmatter │  │
│  │  • Multi-context: global/ + contexts/<ctx>/                    │  │
│  │  • Multi-modal assets: assets/ (images, audio)                 │  │
│  │  • Skills: skills/ (YAML, versioned)                           │  │
│  │  • SOUL split: soul_core.md (immutable) + soul_style.md        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  TRACE STORE (JSONL + SQLite FTS5)                             │  │
│  │  • Per-turn execution traces (tool calls, reasoning, outputs)  │  │
│  │  • Quality labels (good | acceptable | bad)                    │  │
│  │  • Optimization-ready subset                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  CONCURRENCY LAYER                                             │  │
│  │  • File locking (flock) on memory-repo                         │  │
│  │  • Writer queue (single writer process per home)               │  │
│  │  • Multi-agent support (subagents + main agent)                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  SELF-DIAGNOSTICS                                              │  │
│  │  • Health checks (storage, Git state, API reachability)        │  │
│  │  • Metrics (latency, token usage, trust distribution)          │  │
│  │  • Learning loop status                                        │  │
│  │  • /diagnostics slash command + HTTP endpoint                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  RETRY & FALLBACK LAYER                                        │  │
│  │  • Exponential backoff for API failures                        │  │
│  │  • Graceful degradation: memory ops skip on persistent failure │  │
│  │  • Rate limit handling                                         │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

                                ▲
                                │ reflective dataset
                                │
┌───────────────────────────────┴──────────────────────────────────────┐
│  OFFLINE OPTIMIZATION (weekly batch)                                 │
│  ┌──────────────────┐  ┌──────────────────┐                          │
│  │  GEPA optimizer  │  │  Pareto front    │                          │
│  │  Main model      │  │  per component   │                          │
│  │  (via API)       │  │  (extractor,     │                          │
│  │                  │  │   reviewer, ...) │                          │
│  └──────────────────┘  └──────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2. Component overview

| Component | Responsibility | Files |
|---|---|---|
| `AgentCore` | Main reasoning, tool execution, streaming, memory pipeline calls | `hermes/core/agent.py` (extended from `run_agent.py`) |
| `MemoryStore` | Atomic markdown + Git + trust cards | `hermes/memory/store.py`, `hermes/memory/git_backend.py` |
| `MemoryPipeline` | Extractor + reviewer + gate + compactor (rewriter) orchestration | `hermes/memory/pipeline.py` |
| `RetryHandler` | Exponential backoff for main model API failures | `hermes/core/retry.py` |
| `TraceStore` | JSONL + SQLite FTS5 traces | `hermes/traces/store.py` |
| `GEPAOptimizer` | Offline GEPA runs | `hermes/optimization/gepa_runner.py` |
| `ConcurrencyManager` | File locking + writer queue | `hermes/concurrency/manager.py` |
| `DiagnosticsSystem` | Health + metrics + dashboard | `hermes/diagnostics/system.py` |
| `SkillLifecycle` | Skill propose/approve/version/deprecate | `hermes/skills/lifecycle.py` |
| `TopicDetector` | Automatic session boundaries on gateways | `hermes/gateway/topic_detector.py` |
| `AdversarialDetector` | Prompt injection + trust manipulation detection | `hermes/memory/adversarial_detector.py` |
| `BackupManager` | Daily backups + recovery | `hermes/backup/manager.py` |
| `ModelVersionTracker` | Cloud model deprecation detection | `hermes/models/version_registry.py` |
| `TestFramework` | Unit + integration + property + benchmark + chaos + mutation (mutmut) | `tests/` |

### 2.3. Memory storage layout

```
${HERMES_HOME}/                    # default: ~/.hermes/
├── memory-repo/                   # Git-backed atomic markdown graph
│   ├── .git/
│   ├── MEMORY.md                  # global index (loaded at session start)
│   ├── global/                    # cross-context memory
│   │   ├── user-profile.md        # was USER.md
│   │   └── facts/
│   │       ├── 2026-11-18-pnpm-preference.md
│   │       └── 2026-11-19-deploy-branch.md
│   ├── contexts/                  # context-specific memory
│   │   ├── work-project-x/
│   │   │   ├── MEMORY.md          # context index (loaded when context active)
│   │   │   └── facts/
│   │   ├── personal-coding/
│   │   │   ├── MEMORY.md
│   │   │   └── facts/
│   │   └── telegram-chat-12345/
│   │       └── ...
│   ├── soul/
│   │   ├── soul_core.md           # immutable (safety/alignment)
│   │   └── soul_style.md          # adaptive (communication preferences)
│   ├── skills/                    # procedural memory
│   │   ├── code-review-v3.yaml
│   │   ├── testing-workflow-v2.yaml
│   │   └── _skill_registry.json   # versioning metadata
│   ├── assets/                    # multi-modal memory attachments
│   │   ├── images/
│   │   ├── audio/
│   │   └── data/
│   ├── archive/                   # decayed entries (recoverable)
│   └── optimization/              # GEPA state
│       ├── pareto-front.json
│       └── candidate-pool/
├── traces/                        # Trace Store (not in Git, large)
│   ├── v1.0/
│   │   └── 2026-11-18.jsonl
│   ├── optimization-ready/
│   │   └── train.jsonl
│   └── traces.db                  # SQLite with FTS5 index
├── backups/                       # daily backups (Phase 18)
│   └── 2026-11-18.tar.gz
├── diagnostics/
│   ├── metrics.jsonl              # time series
│   ├── health.json                # last health check
│   └── alerts.log
├── config.yaml                    # single-model configuration
└── hermes.db                      # SQLite for FTS5, sessions, etc.
```

### 2.4. Trust card schema (YAML frontmatter on every fact)

```yaml
---
id: 2026-11-18-pnpm-preference
type: User                  # User | Feedback | Project | Reference
text: "Prefers pnpm and conventional commits"
created_at: 2026-11-18T14:32:00Z
last_verified_at: 2026-11-18T14:32:00Z
trust_score: 0.85           # 0.0-1.0, weighted avg of verification events
verification_events:
  - event_id: v1
    timestamp: 2026-11-18T14:32:00Z
    source: user-explicit    # user-explicit | agent-inferred | file-read | tool-output | test-green | code-review-human | llm
    verifier: "user"
    weight: 1.0
    evidence: "User said: 'always use pnpm' in turn abc123"
  - event_id: v2
    timestamp: 2026-11-19T10:15:00Z
    source: tool-output
    verifier: "main-model"
    weight: 0.85
    evidence: "Detected pnpm-lock.yaml in project root"
context: personal-coding     # which context this fact belongs to (or "global")
scope:                      # privacy/visibility
  - personal
  - coding
ttl_days: null              # null = no expiry, N = expire after N days unverified
tags: [package-manager, preferences]
pii_checked: true           # PII scan completed (Phase 17.T4)
attachments: []             # multi-modal attachments (Phase 8)
---

Prefers pnpm and conventional commits.

## Notes
- Confirmed via pnpm-lock.yaml presence
- User mentioned conventional commits preference on 2026-11-18
```

**Trust score weights** (per verification event source):

| Source | Weight |
|---|---|
| `user-explicit` | 1.0 |
| `test-green` (rule-based anchor) | 1.0 |
| `code-review-human` | 0.95 |
| `tool-output` | 0.85 |
| `file-read` | 0.7 |
| `agent-inferred` | 0.5 |
| `llm` (no anchor) | 0.3 |

Trust score = weighted average of all verification events, decayed over time
(see Phase 2 forgetting curve).

### 2.5. Single-model configuration

```yaml
# config.yaml
model:
  # Hermes v2026.9.24 model configuration — unchanged
  # All memory pipeline ops use the same model
  default: "claude-sonnet-4"
  provider: "anthropic"

memory_pipeline:
  extraction:
    max_drafts_per_turn: 5
    async_after_stop: true
  gating:
    budget_claims: 3
    budget_chars: 1200
    bm25_prefilter_top_k: 20   # retrieve 20, gate only top 3
    mode: batched              # all top-3 candidates in ONE main model call
    timeout_seconds: 3.5       # hard timeout → trust-score-only fallback (5.5)
  retry:
    max_retries: 3
    backoff_base_seconds: 2
    backoff_max_seconds: 60
    on_persistent_failure: "skip"  # skip | fail
```

### 2.6. API cost estimation

Since all memory ops use the main model, every operation costs API tokens.
Estimated monthly cost for typical usage:

| Operation | Calls per turn | Tokens per call (in+out) | Cost per 1K turns* |
|---|---|---|---|
| Extraction | 1 | ~2,000 | ~$6-30 |
| Review | 1 | ~1,500 | ~$5-23 |
| Gating (batched top-3) | 1 | ~1,500 | ~$5-23 |
| Compaction | 0-1 (when session long) | ~1,000 | ~$0-15 |
| **Total** | **~3-4 calls/turn** | **~5,000-6,000 tokens/turn** | **~$15-90 per 1K turns** |

*Cost range depends on model: Claude Sonnet ($3/$15 per M tokens; the range spans "all
tokens at input price" to "all tokens at output price" for the same volumes) vs Claude
Haiku ($0.25/$1.25 per M tokens, ~12x cheaper) vs GLM-5.3-Flash. Use a cheaper model for
memory ops if
provider supports multiple models — still single-model architecture, but a cheaper
tier of the same provider is acceptable.

**Break-even analysis:** v2.1 makes sense if:
- >100 turns/day (heavy usage — memory value grows with volume; memory ops ≈$100/month
  at Sonnet rates, ≈$10/month at Haiku rates)
- OR privacy requirement (PII filtering + Git audit trail)
- OR team usage (multiple users benefit from shared infrastructure)

For <20 turns/day single user with no privacy requirement, v2026.9.24 may be sufficient.

---

## 3. Phased Implementation Plan

Each phase documents: goal, components, files (path + purpose + key functions),
implementation steps, tests, acceptance criteria, dependencies.

> **Note on numbering:** phase numbers reflect the plan's creation history (r1→r3.2),
> not the execution order. Phase 16 (Documentation) appears in the document after Phase 22
> but executes last. Execution order — see Section 7.

### Phase 0: Foundation & fork setup

**Goal:** Fork Hermes, set up development environment, define v2.0 module skeleton.

**Components:**
- Fork `github.com/NousResearch/hermes-agent` to `github.com/<org>/hermes-v2`
- Set up Python 3.11+ project with `pyproject.toml`, `ruff`, `mypy`, `pytest`
- Create module skeleton: `hermes/memory/`, `hermes/traces/`, `hermes/optimization/`,
  `hermes/concurrency/`, `hermes/diagnostics/`, `hermes/skills/`, `hermes/gateway/`,
  `hermes/backup/`
- Preserve v2026.9.24 behavior behind feature flags
- Set up CI (GitHub Actions): lint, type-check, test, build

**Files:**
- `pyproject.toml` — dependencies, tooling config (pytest markers; the `tests/**/__init__.py` skeleton from P0.T3 keeps default import mode working, check 6)
- `hermes/__init__.py` — package init
- `hermes/core/agent.py` — extended from `run_agent.py` (kept temporarily)
- `hermes/config.py` — config loading from YAML
- `.github/workflows/ci.yml` — CI pipeline (created in P0.T1, extended in P0.T4)
- `.github/workflows/nightly.yml` — nightly E2E + mutation run (Section 4.7; full scope
  and thresholds are configured in P20.T5)

**Implementation steps:**
1. Fork repo, clone, install dependencies
2. Run existing test suite, verify all pass
3. Add `ruff` config (line length 100, select E/F/W/I/N/UP/B/SIM)
4. Add `mypy` strict mode for new modules only (existing code: gradual typing)
5. Add `pytest` config with markers: `unit`, `integration`, `property`, `benchmark`, `e2e`; create the `tests/**/__init__.py` package skeleton (P0.T3) so duplicate test-file basenames across the new `tests/**` dirs stay importable in pytest's default mode — pytest 9 has no importmode ini option, so the verdict's alternative (option B) is implemented (check 6)
6. Create empty module skeleton with `__init__.py` and stub README per module; create `hermes/core/agent.py` extended from upstream `run_agent.py` (P0.T3 — the creator of the v2 agent module, check 6)
7. Set up pre-commit hooks (ruff, mypy, pytest-quick)
8. Configure GitHub Actions: lint + type-check + unit tests on every PR,
   integration tests on merge to main, nightly mutation testing job
   (mutmut, scope and thresholds per Section 4.7)

**Tests (sanity, not new functionality):**
- `tests/test_fork_baseline.py::test_existing_tests_pass` — v2026.9.24 test suite still passes
- `tests/test_fork_baseline.py::test_imports_clean` — all modules import without errors

**Acceptance criteria:**
- Fork pushed to GitHub
- CI green on main branch
- All v2026.9.24 tests pass
- `ruff check hermes/` clean
- `mypy hermes/memory/` clean (new modules only)
- Nightly mutation-testing CI job configured and green on baseline skeleton
  (Section 4.7)

**Dependencies:** none

---

### Phase 1: Storage layer — atomic markdown graph in Git

**Goal:** Migrate storage from `~/.hermes/memories/*.md` (unversioned) to
Git-backed atomic markdown graph with trust card frontmatter.

**Components:**
- `MemoryStore` class: CRUD operations on memory entries
- `GitBackend`: Git operations (commit, branch, merge, log)
- Trust card parser/serializer (YAML frontmatter)
- Migration script from v2026.9.24 format

**Files:**
- `hermes/memory/store.py` — `MemoryStore` class
  - `add(text, type, context, source, verifier, evidence) -> MemoryEntry`
  - `get(entry_id) -> MemoryEntry | None`
  - `update(entry_id, text=None, trust_score=None) -> MemoryEntry`
  - `remove(entry_id) -> bool`
  - `list(context=None, type=None, tag=None) -> list[MemoryEntry]`
  - `search(query, top_k=10) -> list[MemoryEntry]` (BM25)
- `hermes/memory/git_backend.py` — `GitBackend` class
  - `init(repo_path) -> GitBackend`
  - `commit(message, files) -> str` (returns commit sha)
  - `branch(name) -> None`
  - `merge(branch) -> MergeResult`
  - `log(path=None, limit=10) -> list[CommitEntry]`
  - `is_clean() -> bool`
- `hermes/memory/trust_card.py` — `TrustCard` dataclass + parser
  - `parse(markdown_content) -> (TrustCard, body)`
  - `serialize(card, body) -> str`
  - `compute_trust_score(events) -> float` (weighted average)
- `hermes/memory/entry.py` — `MemoryEntry` dataclass
- `hermes/memory/migrations/fork_to_v2_0.py` — one-time migration

**Implementation steps:**
1. Define `MemoryEntry` dataclass with all fields from trust card schema
2. Implement `TrustCard` parser using `python-frontmatter` library
3. Implement `GitBackend` using `pygit2` (preferred) or `gitpython`
4. Implement `MemoryStore` with file CRUD + Git commit on every write
5. Implement BM25 search using `rank_bm25` library over all entries
6. Implement context-aware storage: global/ vs contexts/<name>/
7. Write migration script: read old MEMORY.md, parse entries, create new atomic files
8. Add file locking via `flock` to prevent concurrent write corruption
9. Add automatic Git garbage collection (weekly)

**Tests:**
- `tests/unit/memory/test_trust_card.py::test_parse_roundtrip` — parse → serialize → parse equality
- `tests/unit/memory/test_trust_card.py::test_compute_trust_score_weighted_avg`
- `tests/unit/memory/test_trust_card.py::test_verification_event_weights`
- `tests/unit/memory/test_store.py::test_add_creates_file_and_commits`
- `tests/unit/memory/test_store.py::test_get_returns_entry`
- `tests/unit/memory/test_store.py::test_update_modifies_trust_score`
- `tests/unit/memory/test_store.py::test_remove_deletes_file`
- `tests/unit/memory/test_store.py::test_list_filters_by_context_type_tag`
- `tests/unit/memory/test_store.py::test_search_bm25_returns_relevant`
- `tests/unit/memory/test_store.py::test_context_isolation` — global fact not in context list
- `tests/unit/memory/test_git_backend.py::test_commit_returns_sha`
- `tests/unit/memory/test_git_backend.py::test_log_returns_history`
- `tests/unit/memory/test_git_backend.py::test_branch_and_merge`
- `tests/unit/memory/test_git_backend.py::test_is_clean_after_commit`
- `tests/unit/memory/test_concurrency.py::test_concurrent_writes_serialized` — flock works
- `tests/integration/test_migration.py::test_fork_to_v2_0_preserves_entries`
- `tests/integration/test_migration.py::test_fork_to_v2_0_sets_default_trust`

**Property-based tests:**
- `tests/property/test_trust_card.py::test_parse_never_crashes` (hypothesis)
- `tests/property/test_trust_card.py::test_trust_score_in_range_0_1`
- `tests/property/test_store.py::test_add_then_get_roundtrip`

**Acceptance criteria:**
- All v2026.9.24 memory entries migrate without data loss
- Git commit created on every memory write (add/update/remove)
- Trust score computed correctly (weighted avg, range 0-1)
- BM25 search returns relevant entries in top-5 for typical queries
- Concurrent writes don't corrupt files (verified by stress test)

**Dependencies:** Phase 0

---

### Phase 2: Trust cards & provenance

**Goal:** Trust card schema fully implemented, verification events tracked, trust score
computed and used for injection prioritization.

**Components:**
- `VerificationEvent` tracking: every memory operation creates an event
- Trust score computation with source-aware weights
- Trust score decay over time (forgetting curve)
- `verify` MCP tool for explicit verification (test green, code review)

**Files:**
- `hermes/memory/trust_card.py` (extended)
  - `add_verification_event(entry_id, event) -> MemoryEntry`
  - `decay_trust_scores() -> int` (returns count of decayed entries)
- `hermes/memory/decay.py` — forgetting curve logic
  - `compute_decay(current_score, last_verified_at, now) -> float`
  - Formula: `score * exp(-decay_rate * days_since_verification)`, decay_rate=0.01
- `hermes/mcp/verify_tool.py` — MCP tool
  - `verify(entry_id, source, evidence) -> MemoryEntry`
- `hermes/memory/anchors.py` — automatic ground-truth anchors
  - Hook on command execution (PostToolUse equivalent): when a test suite or build
    completes, find memory entries cited in the session / touching changed files,
    and append a `test-green` (weight 1.0) or `test-red` event automatically
  - This implements the hypothesis element "who verifies the verifier":
    anchors are rule-based, not LLM-based

**Implementation steps:**
1. Define verification event types and weights (from config)
2. Implement `add_verification_event` with trust score recompute
3. Implement forgetting curve with daily cron job
4. Add `verify` MCP tool callable by main agent
5. Trust score weights table in `config.yaml`
6. Implement automatic test/build anchors (`hermes/memory/anchors.py`): test-green
   / test-red events appended without manual `verify` calls

**Tests:**
- `tests/unit/memory/test_trust_card.py::test_add_verification_event_updates_score`
- `tests/unit/memory/test_trust_card.py::test_multiple_events_weighted_avg`
- `tests/unit/memory/test_trust_card.py::test_user_explicit_higher_than_llm`
- `tests/unit/memory/test_decay.py::test_decay_reduces_score_over_time`
- `tests/unit/memory/test_decay.py::test_decay_zero_after_1_year`
- `tests/unit/memory/test_decay.py::test_recently_verified_unchanged`
- `tests/integration/test_verify_tool.py::test_verify_via_mcp_adds_event`
- `tests/unit/memory/test_anchors.py::test_test_green_appends_weight_1_event`
- `tests/unit/memory/test_anchors.py::test_test_red_downweights_related_entries`
- `tests/unit/memory/test_anchors.py::test_unrelated_entries_untouched`
- `tests/integration/test_anchors.py::test_full_run_updates_trust_scores`

**Acceptance criteria:**
- Every memory entry has ≥1 verification event
- Trust score in [0, 1], weighted avg of all events
- Forgetting curve runs daily, decay verified
- `verify` MCP tool callable, updates trust score
- Test suite run automatically appends test-green/test-red events to related entries
  (no manual verify call needed)

**Dependencies:** Phase 1

---

### Phase 3: Two-stage extractor + reviewer pipeline

**Goal:** Replace single memory tool with two-stage pipeline: extractor extracts
draft facts from turn, reviewer dedups and updates existing notes. **Both use main
model via `AIAgent.chat()`**. Runs async after Stop hook — doesn't block turn.

**Components:**
- `ExtractorAgent`: turn + context window → list[DraftFact]
- `ReviewerAgent`: (DraftFact, existing entries) → Action (add | update | skip | conflict)
- Pipeline orchestrator: hooks into `Stop` event, runs async after turn
- Multi-turn context window: last 3-5 turns for reference resolution

**Files:**
- `hermes/memory/pipeline.py` — `MemoryPipeline` orchestrator
  - `run_after_turn(turn_id) -> PipelineResult`
  - `extract_drafts(turn) -> list[DraftFact]`
  - `review_drafts(drafts) -> list[ReviewAction]`
  - `apply_actions(actions) -> ApplyResult`
- `hermes/memory/extractor.py` — `ExtractorAgent`
  - System prompt: extraction rules (4 types: User, Feedback, Project, Reference)
  - Output: structured JSON (validated, retry on malformed)
- `hermes/memory/reviewer.py` — `ReviewerAgent`
  - System prompt: dedup rules, conflict detection, merge rules
  - Input: draft + semantic search results (top 5 existing entries)
  - Output: `{"action": "add|update|skip|conflict", "target_id": "...", "rationale": "..."}`
- `hermes/memory/types.py` — `DraftFact`, `ReviewAction`, `PipelineResult` dataclasses
- `hermes/memory/context_window.py` — `ContextWindow` (multi-turn context)
  - `get_recent_turns(session_id, n=3) -> list[Turn]`
  - Filters by session, handles missing turns

**Implementation steps:**
1. Define DraftFact and ReviewAction dataclasses
2. Write extraction prompt template (structured output JSON schema)
3. Write review prompt template (with existing context injection)
4. Implement ExtractorAgent via `AIAgent.chat()` (main model)
5. Implement ReviewerAgent with semantic search for existing entries
6. Implement multi-turn context window (P3.T6)
7. Pipeline runs async after `Stop` hook — doesn't block turn
8. Apply actions to MemoryStore (with Git commit per action)
9. Track pipeline metrics (extraction count, review actions, conflicts)

**Tests:**
- `tests/unit/memory/test_extractor.py::test_extracts_user_preference`
- `tests/unit/memory/test_extractor.py::test_extracts_project_fact`
- `tests/unit/memory/test_extractor.py::test_extracts_feedback_correction`
- `tests/unit/memory/test_extractor.py::test_empty_turn_no_drafts`
- `tests/unit/memory/test_extractor.py::test_structured_output_schema_valid`
- `tests/unit/memory/test_extractor.py::test_extracts_with_multi_turn_context`
- `tests/unit/memory/test_extractor.py::test_does_not_reextract_confirmed_facts`
- `tests/unit/memory/test_extractor.py::test_resolves_references_across_turns`
- `tests/unit/memory/test_reviewer.py::test_add_action_for_new_fact`
- `tests/unit/memory/test_reviewer.py::test_update_action_for_related_existing`
- `tests/unit/memory/test_reviewer.py::test_skip_action_for_exact_duplicate`
- `tests/unit/memory/test_reviewer.py::test_conflict_detection_returns_special_action`
- `tests/integration/test_pipeline.py::test_full_pipeline_turn_to_committed_entry`
- `tests/integration/test_pipeline.py::test_pipeline_async_does_not_block_turn`
- `tests/integration/test_pipeline.py::test_metrics_recorded`
- `tests/integration/test_pipeline.py::test_multi_turn_extraction_workflow`
- `tests/property/test_extractor.py::test_output_always_valid_json`
- `tests/property/test_reviewer.py::test_action_always_in_allowed_set`

**Acceptance criteria:**
- Extractor produces valid JSON for 95%+ of turns
- Reviewer correctly identifies duplicates in 90%+ cases
- Pipeline completes within 10s for typical turn (async, doesn't block)
- Multi-turn references resolved correctly
- All actions logged with rationale

**Dependencies:** Phase 1, Phase 2; T6 (multi-turn context) additionally needs Phase 11
Trace Store (P11.T1) — or the session-history fallback (see the note in the task)

---

### Phase 4: CBDC gating for proactive injection

**Goal:** Before each turn, retrieve relevant entries and gate them through
Claim → Belief → Decision → Consumption (use/verify/ignore). Inject only "use"
with budget cap. Preserve prefix cache via stable/dynamic context split.
**Gating decisions use main model.** BM25 pre-filter minimizes calls (only top-3
candidates gated, not all retrieved).

**Components:**
- `RetrievalEngine`: BM25 + optional vector search → top-K candidates
- `CBDCGate`: gating via main model decides use/verify/ignore per candidate
- `ContextInjector`: format injected context, split stable/dynamic
- Pre-step hook integration (replaces frozen snapshot pattern)

**Files:**
- `hermes/memory/retrieval.py` — `RetrievalEngine`
  - `retrieve(query, top_k=20) -> list[RetrievedEntry]`
  - BM25 by default, optional vector via `sentence-transformers`
- `hermes/memory/gating.py` — `CBDCGate`
  - `gate(candidates, context, budget_claims=3, budget_chars=1200) -> list[GatedEntry]`
  - Returns only "use" decisions within budget
  - Uses main model via `AIAgent.chat()` — **single batched call**: all top-3
    candidates classified in one prompt (one JSON decision list), not 3 sequential
    calls. 3 sequential main-model calls (1-3s each) would blow the 5s hook budget
    on slow providers — batching keeps worst case ≈1 model round-trip
  - Hard timeout 3.5s → fallback to trust-score-only gating (Section 5.5),
    so the pre-step hook never blocks a turn on a slow model
  - Top-3 candidates only (BM25 pre-filter)
- `hermes/memory/injector.py` — `ContextInjector`
  - `build_context(turn) -> InjectedContext`
  - Splits stable (loaded at session start) vs dynamic (per-turn)
- `hermes/memory/hooks.py` — pre-step hook
  - On `UserPromptSubmit`: call retrieval + gating + inject to additionalContext
  - On `SessionStart`: load stable context (global MEMORY.md + active context MEMORY.md)

**Implementation steps:**
1. Implement RetrievalEngine with BM25 (rank_bm25)
2. Implement CBDCGate using main model — one batched call for all top-3 candidates
   (decision list JSON), 3.5s timeout with trust-only fallback
3. Implement budget enforcement (≤3 claims, ≤1200 chars)
4. Implement ContextInjector with stable/dynamic split
5. Stable context: global user profile + active context index → loaded once per session
6. Dynamic context: per-turn retrieved facts → injected via additionalContext
7. Pre-step hook fires before every model call
8. Prefix cache preservation: stable part never changes mid-session

**Tests:**
- `tests/unit/memory/test_retrieval.py::test_bm25_returns_relevant_top_k`
- `tests/unit/memory/test_retrieval.py::test_empty_query_returns_empty`
- `tests/unit/memory/test_gating.py::test_use_decision_for_high_trust_relevant`
- `tests/unit/memory/test_gating.py::test_verify_decision_for_medium_trust`
- `tests/unit/memory/test_gating.py::test_ignore_decision_for_irrelevant`
- `tests/unit/memory/test_gating.py::test_budget_claims_enforced`
- `tests/unit/memory/test_gating.py::test_budget_chars_enforced`
- `tests/unit/memory/test_gating.py::test_single_batched_call_per_turn` — one model call for top-3, not 3 calls
- `tests/unit/memory/test_gating.py::test_timeout_falls_back_to_trust_only`
- `tests/unit/memory/test_injector.py::test_stable_context_loaded_at_session_start`
- `tests/unit/memory/test_injector.py::test_dynamic_context_changes_per_turn`
- `tests/unit/memory/test_injector.py::test_stable_context_preserves_prefix_cache`
- `tests/integration/test_hooks.py::test_pre_step_hook_injects_context`
- `tests/integration/test_hooks.py::test_no_injection_when_no_relevant_facts`

**Property-based tests:**
- `tests/property/test_gating.py::test_injected_count_never_exceeds_budget`
- `tests/property/test_gating.py::test_all_injected_have_use_decision`

**Benchmark tests:**
- `tests/benchmark/test_retrieval_latency.py::test_bm25_under_100ms_for_1000_entries`
- `tests/benchmark/test_gating_latency.py::test_cbdc_gate_under_3500ms`
- `tests/benchmark/test_injector_latency.py::test_full_injection_under_5s`

**Acceptance criteria:**
- Pre-step hook: typical <2s, hard timeout 5s (gate itself: hard timeout 3.5s → trust-only fallback)
- Budget enforced (≤3 claims, ≤1200 chars)
- Stable context changes ≤1 time per session (prefix cache preserved)
- Injected facts improve downstream task success (A/B test: with vs without)
- Gating uses ≤1 batched main model call per turn (top-3 candidates in one prompt;
  check 3 replaced the former "≤3 sequential calls" — 3×1-3s calls would exceed the 5s hook budget)

**Dependencies:** Phase 1, Phase 2, Phase 3; Phase 5 (P4.T3 injector needs the P5.T1
ContextManager — so Phase 5 T1 is done before Phase 4 T3)

---

### Phase 5: Project/context layer

**Goal:** Support multiple contexts (work, personal, project-X, telegram-chat-Y).
Active context determines which MEMORY.md + facts are loaded.

**Components:**
- `ContextManager`: detect/switch/create contexts
- Context detection rules: by cwd, by Telegram chat id, by explicit `/context switch`
- Per-context MEMORY.md index + facts/

**Files:**
- `hermes/memory/context.py` — `ContextManager`
  - `detect_context(session_metadata) -> str` (returns context name)
  - `switch_context(name) -> Context`
  - `create_context(name, parent=None) -> Context`
  - `list_contexts() -> list[Context]`
- `hermes/memory/types.py` (extended) — `Context` dataclass
- Slash command `commands/context.md` — `/context list|switch|create|inspect`

**Implementation steps:**
1. Define Context dataclass (name, parent, path, active_sessions)
2. Implement context detection rules (cwd mapping, gateway chat id mapping)
3. Implement context switch with stable context reload
4. Add `/context` slash command
5. Migration: existing v2026.9.24 memory becomes `global/` context
6. Multi-faceted USER.md: `global/user-profile.md` + `contexts/<name>/user-profile.md`

**Tests:**
- `tests/unit/memory/test_context.py::test_detect_by_cwd`
- `tests/unit/memory/test_context.py::test_detect_by_gateway_chat_id`
- `tests/unit/memory/test_context.py::test_switch_reloads_stable_context`
- `tests/unit/memory/test_context.py::test_create_with_parent`
- `tests/unit/memory/test_context.py::test_list_returns_all`
- `tests/integration/test_context_workflow.py::test_user_creates_switches_contexts`
- `tests/integration/test_context_workflow.py::test_facts_isolated_per_context`

**Acceptance criteria:**
- Context auto-detected for CLI (cwd) and gateway (chat id)
- Switch reloads stable context (preserves prefix cache session boundary)
- Facts isolated per context (work fact not in personal context)
- Migration preserves all existing memory as global context

**Dependencies:** Phase 1

---

### Phase 6: Concurrent writers & multi-agent support

**Goal:** Multiple agent processes (main + subagents) can share one Hermes home
without corrupting memory.

**Components:**
- `ConcurrencyManager`: file locking + writer queue
- Single writer process per Hermes home
- Subagents queue writes to main writer

**Files:**
- `hermes/concurrency/manager.py` — `ConcurrencyManager`
  - `acquire_lock(timeout=10) -> LockHandle`
  - `release_lock(handle) -> None`
  - `queue_write(operation) -> WriteResult` (async)
- `hermes/concurrency/writer.py` — `WriterProcess`
  - Background process reading from queue
  - Applies writes to MemoryStore + Git
- `hermes/concurrency/protocol.py` — IPC protocol (Unix socket)

**Implementation steps:**
1. Implement file locking via `fcntl.flock` on `${HERMES_HOME}/.lock`
2. Implement writer process as daemon (started on first agent launch)
3. Implement IPC protocol (Unix socket, JSON messages)
4. Subagents connect to writer socket, send write ops
5. Writer serializes, applies, returns result
6. Main agent can also bypass queue for direct writes (with lock)
7. Crash recovery: if writer dies, agents detect and restart

**Tests:**
- `tests/unit/concurrency/test_lock.py::test_acquire_release`
- `tests/unit/concurrency/test_lock.py::test_concurrent_acquire_blocks`
- `tests/unit/concurrency/test_lock.py::test_lock_timeout`
- `tests/integration/test_writer.py::test_writer_processes_queue`
- `tests/integration/test_writer.py::test_subagent_writes_via_queue`
- `tests/integration/test_writer.py::test_writer_crash_recovery`
- `tests/integration/test_multi_agent.py::test_two_agents_share_home`

**Acceptance criteria:**
- Two agents can run simultaneously without corruption
- Writer process handles 100 writes/sec
- Crash recovery within 5s

**Dependencies:** Phase 1

---

### Phase 7: Long-session handling — hot-reload & topic detection

**Goal:** Fix frozen snapshot breakage on long messaging sessions. Memory updates
become visible mid-session via selective prefix cache invalidation. Topic shifts
auto-detect and create virtual session boundaries.

**Deliberate trade-off:** stable context reloads only at session boundaries (not
mid-session). In-session updates use segment-level invalidation. This preserves
≥80% prefix cache hit rate. Full memory evolution mid-session would require
accepting prefix cache invalidation cost — deliberately not done.

**Components:**
- `MemoryHotReloader`: selective invalidation of system prompt segments
- `TopicDetector`: detect topic shifts in conversation (main model classification)
- Virtual session boundaries for memory extraction (without closing gateway chat)
- `SessionCompactor`: compaction summaries as first-class memory citizens
  (hypothesis element from DSH pattern: compacted summaries are the primary
  extraction/retrieval input for long sessions, dramatically less noisy than raw turns)

**Files:**
- `hermes/memory/hot_reload.py` — `MemoryHotReloader`
  - `invalidate_segment(segment_id) -> None`
  - `reload_segment(segment_id) -> str`
  - System prompt segmented by memory entry ID
- `hermes/gateway/topic_detector.py` — `TopicDetector`
  - `detect_shift(recent_messages) -> ShiftResult`
  - Uses main model for classification
- `hermes/gateway/session_manager.py` — virtual sessions
  - `start_virtual_session(topic) -> VirtualSession`
  - Memory extraction scoped to virtual session
- `hermes/memory/compaction.py` — `SessionCompactor`
  - `compact_session(session) -> MemoryEntry` (type=Reference, source points to
    trace range, not a copy — "memory points, doesn't mirror")
  - Generated when a virtual session closes or session exceeds N turns
  - Extraction uses compaction summaries as primary input (raw turns secondary)

**Implementation steps:**
1. Segment system prompt: each memory entry wrapped with markers
2. On memory tool update: find affected segment, invalidate
3. Re-inject only that segment on next model call
4. Prefix cache: most of prompt unchanged, only segment invalidated
5. Topic detector: every N messages, classify topic vs previous
6. On topic shift: create virtual session boundary
7. Memory extraction runs per virtual session, not per gateway chat
8. User can override: `/session new` for explicit boundary
9. On virtual session close (or N turns): generate compaction summary entry
10. Extraction reads compaction summaries first, raw turns only on demand

**Tests:**
- `tests/unit/memory/test_hot_reload.py::test_invalidate_specific_segment`
- `tests/unit/memory/test_hot_reload.py::test_other_segments_unchanged`
- `tests/unit/memory/test_hot_reload.py::test_reload_returns_updated_content`
- `tests/unit/gateway/test_topic_detector.py::test_detects_clear_topic_shift`
- `tests/unit/gateway/test_topic_detector.py::test_no_shift_for_continuation`
- `tests/integration/test_long_session.py::test_memory_update_visible_mid_session`
- `tests/integration/test_long_session.py::test_virtual_session_extraction_isolated`
- `tests/integration/test_long_session.py::test_prefix_cache_hit_rate_above_80_percent`
- `tests/unit/memory/test_compaction.py::test_compact_session_creates_reference_entry`
- `tests/unit/memory/test_compaction.py::test_summary_points_to_trace_range`
- `tests/integration/test_compaction.py::test_extraction_uses_summary_first`
- `tests/property/test_compaction.py::test_summary_never_exceeds_char_limit`

**Acceptance criteria:**
- Memory updates visible within 1 turn of being written
- Prefix cache hit rate ≥80% (only segments invalidated)
- Topic detector 85%+ accurate on labeled test set
- Virtual sessions create clean memory boundaries
- Compaction summary created on session close, stored as Reference entry with
  trace-range provenance
- Extraction prefers compaction summaries over raw turns

**Dependencies:** Phase 1, Phase 4

---

### Phase 8: Memory quality — forgetting curve, conflict detection, multi-modal

**Goal:** Memory stays high-quality over time. Stale facts decay. Conflicts caught
before write. Multi-modal attachments supported. **Conflict detection uses main model.**

**Components:**
- Forgetting curve (extended from Phase 2): auto-archive low-trust entries
- Conflict detection pre-write: semantic similarity check (embeddings, local — no LLM)
- Multi-modal memory: image, audio, structured data attachments

**Files:**
- `hermes/memory/decay.py` (extended)
  - `archive_stale(threshold=0.2) -> int`
  - Archived entries moved to `memory-repo/archive/`
- `hermes/memory/conflict_detector.py` — `ConflictDetector`
  - `check_conflict(new_entry, existing) -> ConflictResult`
  - Uses embeddings similarity (sentence-transformers) — local, no LLM
  - Threshold: similarity >0.85 → potential conflict
  - For contradiction detection (vs duplicate): uses main model
- `hermes/memory/multimodal.py` — `MultimodalAttachment`
  - `attach_image(entry_id, image_path) -> Attachment`
  - `attach_audio(entry_id, audio_path) -> Attachment`
  - Stored in `memory-repo/assets/`

**Implementation steps:**
1. Extend decay: entries below threshold archived (not deleted, recoverable)
2. Implement conflict detector using sentence-transformers embeddings
3. Pre-write hook: check conflict before applying ReviewAction
4. If conflict: return to ReviewerAgent with conflict context for resolution
5. Multi-modal: attachments stored as files, referenced in trust card `attachments: []`
6. Image OCR (optional): extract text from images for searchability
7. Audio transcription (optional): transcribe voice notes

**Tests:**
- `tests/unit/memory/test_decay.py::test_archive_stale_moves_files_p8`
- `tests/unit/memory/test_decay.py::test_archived_recoverable_p8`
- `tests/unit/memory/test_conflict_detector.py::test_detects_semantic_duplicate`
- `tests/unit/memory/test_conflict_detector.py::test_no_conflict_for_unrelated`
- `tests/unit/memory/test_conflict_detector.py::test_contradiction_returns_special_result`
- `tests/integration/test_multimodal.py::test_attach_image_to_entry`
- `tests/integration/test_multimodal.py::test_image_searchable_via_ocr`
- `tests/integration/test_conflict_workflow.py::test_conflict_triggers_reviewer_resolution`

**Acceptance criteria:**
- Stale entries (>1 year unverified, trust <0.2) archived
- Conflicts detected in 90%+ of test cases
- Multi-modal attachments accessible and searchable
- Conflict resolution creates audit trail in trust card

**Dependencies:** Phase 1, Phase 2, Phase 3

---

### Phase 10: GEPA self-optimization

**Goal:** Weekly GEPA runs optimize prompts for all memory subagents against
explicit metrics. Pareto front per component (no model variant dimension —
single-model architecture).

**Components:**
- `GEPAAdapter` implementations per memory operation
- `GEPARunner`: orchestrates weekly offline runs
- `ParetoFrontStore`: persistent Pareto front storage
- Metrics: retrieval recall@k, extraction F1, gating precision

**Files:**
- `hermes/optimization/gepa_runner.py` — `GEPARunner`
  - `run_weekly() -> OptimizationResult`
  - `evaluate_candidate(prompt, eval_set) -> (score, feedback)`
- `hermes/optimization/adapters/extractor_adapter.py`
- `hermes/optimization/adapters/reviewer_adapter.py`
- `hermes/optimization/adapters/gate_adapter.py`
- `hermes/optimization/adapters/rewriter_adapter.py`   # rewriter = compaction-summary generator (Phase 7 SessionCompactor)
- `hermes/optimization/pareto.py` — `ParetoFrontStore`
  - `add_candidate(component, prompt, scores) -> None`
  - `get_front(component) -> list[Candidate]`
  - `get_best(component, context=None) -> Candidate`
- `hermes/optimization/metrics.py`
  - `extraction_f1(predictions, ground_truth) -> float`
  - `retrieval_recall_at_k(results, relevant, k) -> float`
  - `gating_precision(decisions, ground_truth) -> float`

**Implementation steps:**
1. Define metrics per component (extraction F1, recall@k, gating precision)
2. Build eval sets: 50-100 labeled examples per component
3. Implement GEPAAdapter for each component (evaluate + make_reflective_dataset)
4. Implement GEPARunner using `gepa-ai/gepa` library
5. Weekly cron: load traces from past week, run GEPA on each component
6. Pareto front: candidates stored per component
7. Deploy: best candidate per component → update prompt template
8. Audit: every deployment creates Git commit with rationale

**Tests:**
- `tests/unit/optimization/test_metrics.py::test_extraction_f1_correct`
- `tests/unit/optimization/test_metrics.py::test_recall_at_k`
- `tests/unit/optimization/test_metrics.py::test_gating_precision`
- `tests/unit/optimization/test_pareto.py::test_add_candidate`
- `tests/unit/optimization/test_pareto.py::test_get_front_returns_non_dominated`
- `tests/unit/optimization/test_pareto.py::test_get_best_for_context`
- `tests/integration/test_gepa_runner.py::test_weekly_run_produces_candidates`
- `tests/integration/test_gepa_runner.py::test_deploy_updates_prompt_template`
- `tests/integration/test_gepa_runner.py::test_audit_trail_in_git`

**Acceptance criteria:**
- Weekly GEPA run completes within 1 hour
- Pareto front per component populated
- Deployed prompts improve metric by ≥5% over baseline
- Full audit trail in Git commits

**Dependencies:** Phase 3, Phase 4, Phase 11 (Trace Store)

---

### Phase 11: Trace Store

**Goal:** Capture per-turn execution traces for GEPA reflection and optimization.

**Components:**
- `TraceStore`: JSONL append + SQLite FTS5 index
- Trace schema: turn, tool calls, model used, extraction results, review actions
- Quality labels: `good | acceptable | bad` (auto + manual)

**Files:**
- `hermes/traces/store.py` — `TraceStore`
  - `append(trace) -> str` (returns trace_id)
  - `search(query, limit=10) -> list[Trace]`
  - `label_quality(trace_id, label) -> None`
  - `get_optimization_ready() -> list[Trace]`
- `hermes/traces/schema.py` — `Trace` dataclass
- `hermes/traces/migrations.sql` — SQLite schema

**Implementation steps:**
1. Define Trace dataclass with all fields
2. JSONL append (one file per day) for fast writes
3. SQLite FTS5 index for search (synced from JSONL)
4. Auto-quality-label: extract from review outcomes (skip = bad, add = good)
5. Manual labeling via `/trace label <id> <good|bad>` command
6. Optimization-ready filter: quality_label=good

**Tests:**
- `tests/unit/traces/test_store.py::test_append_creates_trace`
- `tests/unit/traces/test_store.py::test_search_returns_relevant`
- `tests/unit/traces/test_store.py::test_label_quality_updates`
- `tests/unit/traces/test_store.py::test_get_optimization_ready_filters`
- `tests/integration/test_trace_capture.py::test_turn_creates_trace`
- `tests/integration/test_trace_capture.py::test_quality_auto_labeled`

**Acceptance criteria:**
- Every turn produces a trace
- Search returns relevant traces in <100ms
- Quality labels auto-applied for 80%+ of traces
- Optimization-ready subset filterable

**Dependencies:** Phase 0 (T1 TraceStore); T2 (wiring) additionally needs Phases 3-4
(P3.T4, P3.T5, P4.T4)

---

### Phase 13: Skills lifecycle & learning loop visibility

**Goal:** Explicit skill lifecycle: propose → approve → version → deprecate.
Learning loop dashboard for visibility.

**Components:**
- `SkillLifecycle`: state machine for skills
- Slash commands: `/skills list|inspect|approve|deprecate`
- Dashboard: `/learning stats|traces|pending`

**Files:**
- `hermes/skills/lifecycle.py` — `SkillLifecycle`
  - States: `proposed | approved | active | deprecated | archived`
  - `propose(name, yaml, rationale) -> Skill`
  - `approve(skill_id) -> Skill`
  - `version(skill_id, new_yaml) -> Skill`
  - `deprecate(skill_id, reason) -> Skill`
- `hermes/skills/registry.py` — `_skill_registry.json` manager
- `commands/skills.md` — slash command
- `commands/learning.md` — slash command

**Implementation steps:**
1. Define skill state machine
2. Implement registry with versioning
3. Agent proposes skills (was autonomous) → user approves
4. Skills versioned: each version stored, rollback possible
5. Auto-deprecate: skills not used in 30 days → proposed for deprecation
6. Dashboard: stats (total skills, active, deprecated), traces count, pending approvals
7. Dynamic bundle suggestion: co-occurrence analysis → propose bundles

**Tests:**
- `tests/unit/skills/test_lifecycle.py::test_propose_creates_in_proposed_state`
- `tests/unit/skills/test_lifecycle.py::test_approve_moves_to_active`
- `tests/unit/skills/test_lifecycle.py::test_version_creates_new_version`
- `tests/unit/skills/test_lifecycle.py::test_deprecate_moves_to_deprecated`
- `tests/unit/skills/test_lifecycle.py::test_auto_deprecate_after_30_days_unused`
- `tests/integration/test_skills_workflow.py::test_propose_approve_use_deprecate`
- `tests/integration/test_skills_workflow.py::test_rollback_to_previous_version`

**Acceptance criteria:**
- All skills go through propose → approve flow
- Versioning with rollback
- Auto-deprecation after 30 days unused
- Dashboard shows real-time stats

**Dependencies:** Phase 0; T5 (learning dashboard) additionally needs Phase 11 Trace
Store (P11.T1)

---

### Phase 14: Self-diagnostics system

**Goal:** System can check its own health, report metrics, alert on issues.

**Components:**
- `DiagnosticsSystem`: health checks, metrics collection, alerts
- `/diagnostics` slash command
- HTTP endpoint (if running as gateway): `/health`, `/metrics`
- Auto-repair for common issues

**Files:**
- `hermes/diagnostics/system.py` — `DiagnosticsSystem`
  - `run_health_check() -> HealthReport`
  - `collect_metrics() -> MetricsSnapshot`
  - `check_alerts() -> list[Alert]`
  - `auto_repair(issue) -> RepairResult`
- `hermes/diagnostics/health_checks.py`
  - `check_memory_store_integrity() -> Check`
  - `check_git_state() -> Check`
  - `check_main_model_reachable() -> Check`
  - `check_trace_store_size() -> Check`
  - `check_pareto_front_freshness() -> Check`
  - `check_backup_freshness() -> Check` (requires BackupManager — P18.T1)
- `hermes/diagnostics/metrics.py`
  - Latency percentiles (p50, p95, p99) per operation
  - Token usage by operation
  - Trust score distribution
  - Conflict rates
  - Retrieval recall metrics
  - Learning loop progress
- `hermes/diagnostics/alerts.py`
  - Rules: main model unreachable, Git dirty >1h, trace store >10GB, cron_job_stuck
    (surfaces P21.T5 watchdog incidents — watchdog detects, diagnostics reports;
    single alerting channel, no duplicate paths), etc.
  - Notification channels: log file, gateway message, email (optional)
- `hermes/diagnostics/auto_repair.py`
  - Repair memory store: re-run migration if entries missing trust cards
  - Repair Git: auto-commit dirty state
- `commands/diagnostics.md` — slash command

**Health report schema:**
```json
{
  "timestamp": "2026-11-18T14:32:00Z",
  "overall_status": "healthy | degraded | unhealthy",
  "checks": [
    {
      "name": "memory_store_integrity",
      "status": "pass | fail | warn",
      "details": "247 entries, all have trust cards",
      "auto_repair_attempted": false
    },
    {
      "name": "git_state",
      "status": "warn",
      "details": "dirty for 2h 15min",
      "auto_repair_attempted": true,
      "auto_repair_result": "committed pending changes"
    }
  ],
  "metrics": {
    "memory_ops_latency_p50_ms": 145,
    "memory_ops_latency_p95_ms": 380,
    "token_usage_today": 12450,
    "trust_score_avg": 0.72,
    "trust_score_min": 0.15,
    "conflict_rate_7d": 0.03,
    "retrieval_recall_at_5": 0.81,
    "skills_active": 18,
    "skills_pending_approval": 2,
    "gepa_last_run": "2026-11-17T03:00:00Z"
  },
  "alerts": []
}
```

**Implementation steps:**
1. Implement 6 health checks (memory, Git, main model, traces, Pareto, backup — the full
   list is in 5.1)
2. Implement metrics collection (time series JSONL)
3. Implement alert rules engine
4. Implement auto-repair for top 5 common issues
5. Add `/diagnostics` slash command (full report)
6. Add `/diagnostics quick` (just overall status)
7. HTTP endpoints if running as gateway
8. Periodic background collection (every 5 min)

**Tests:**
- `tests/unit/diagnostics/test_health_checks.py::test_memory_store_integrity_pass`
- `tests/unit/diagnostics/test_health_checks.py::test_memory_store_integrity_fail_on_missing_trust_card`
- `tests/unit/diagnostics/test_health_checks.py::test_git_state_warn_when_dirty`
- `tests/unit/diagnostics/test_health_checks.py::test_main_model_reachable_pass`
- `tests/unit/diagnostics/test_health_checks.py::test_main_model_fail_when_api_down`
- `tests/unit/diagnostics/test_health_checks.py::test_trace_store_size_pass`
- `tests/unit/diagnostics/test_health_checks.py::test_pareto_front_freshness_pass`
- `tests/unit/diagnostics/test_health_checks.py::test_backup_freshness_pass`
- `tests/unit/diagnostics/test_metrics.py::test_latency_percentiles_computed`
- `tests/unit/diagnostics/test_metrics.py::test_token_usage_accumulated`
- `tests/unit/diagnostics/test_alerts.py::test_alert_on_model_down`
- `tests/unit/diagnostics/test_alerts.py::test_alert_on_dirty_git_over_1h`
- `tests/unit/diagnostics/test_alerts.py::test_alert_on_cron_stuck`
- `tests/unit/diagnostics/test_auto_repair.py::test_repair_git_commits_dirty_state`
- `tests/integration/test_diagnostics.py::test_full_health_report_generated`
- `tests/integration/test_diagnostics.py::test_background_collection_every_5_min`

**Acceptance criteria:**
- `/diagnostics` returns full report in <2s
- Health checks catch all common failure modes
- Auto-repair resolves 80%+ of common issues without user intervention
- Metrics time series retained for 90 days

**Dependencies:** All previous phases (uses all components; the health checks require
Phases 1, 10, 11, 18 — task-level deps in tasks.yaml)

---

### Phase 15: SOUL.md split & UX commands

**Goal:** Split SOUL.md into immutable core + adaptive style. Add memory introspection
and export/import commands.

**Components:**
- `soul_core.md`: immutable (safety, alignment, identity)
- `soul_style.md`: adaptive (communication preferences, learned terminology)
- Commands: `/memory list|search|stats|export|import`, `/soul inspect|edit-style`

**Files:**
- `hermes/memory/soul.py` — `SoulManager`
  - `get_core() -> str` (immutable, errors on write)
  - `get_style() -> str` (adaptive, agent-managed with char limit)
  - `update_style(new_content) -> None` (with bounded size)
- `commands/memory.md` — slash command with subcommands
- `commands/soul.md` — slash command

**Implementation steps:**
1. Split existing SOUL.md: safety/alignment → soul_core.md, rest → soul_style.md
2. SoulManager enforces immutability of core (Git-protected, requires manual override)
3. Soul_style.md bounded (2000 chars), agent can update via memory tool
4. `/memory list [context]` — list entries
5. `/memory search <query>` — semantic search
6. `/memory stats` — counts, trust distribution, size
7. `/memory export > file.json` — export all entries
8. `/memory import < file.json` — import entries (with conflict resolution)
9. `/soul inspect` — show current soul content
10. `/soul edit-style` — manual edit of style layer

**Tests:**
- `tests/unit/memory/test_soul.py::test_core_immutable`
- `tests/unit/memory/test_soul.py::test_style_updatable_within_limit`
- `tests/unit/memory/test_soul.py::test_style_over_limit_errors`
- `tests/integration/test_memory_commands.py::test_list_returns_entries`
- `tests/integration/test_memory_commands.py::test_search_returns_relevant`
- `tests/integration/test_memory_commands.py::test_export_import_roundtrip`
- `tests/integration/test_memory_commands.py::test_import_handles_conflicts`

**Acceptance criteria:**
- SOUL core never changes without explicit user override
- SOUL style evolves within 2000 char limit
- All slash commands work as specified
- Export/import preserves trust cards

**Dependencies:** Phase 1

---

### Phase 17: Adversarial input detection & safety

**Goal:** Detect and mitigate attempts to inject false information or manipulate
trust system. **Classification uses main model.**

**Components:**
- `AdversarialInputDetector`: prompt injection, trust manipulation, flooding detection
- Rate limiting on memory writes
- Trust score anomaly detection
- PII detection with configurable action

**Files:**
- `hermes/memory/adversarial_detector.py` — `AdversarialInputDetector`
  - `detect(input_text, context) -> RiskResult`
  - `risk_score: float` in [0, 1]
  - Uses main model for classification
  - Patterns: prompt injection, trust manipulation, contradiction flooding
- `hermes/memory/rate_limiter.py` — `RateLimiter`
  - Max 20 writes per session per hour
  - Max 100 writes per day per user
  - Burst detection (10 writes in 1 minute = suspicious)
- `hermes/memory/trust_anomaly.py` — `TrustAnomalyDetector`
  - Rapid trust score increase (>0.3 in one event)
  - Verifier flooding (many events from same verifier in short time)
- `hermes/memory/pii_detector.py` — `PIIDetector`
  - Email, phone, SSN, credit card, API keys
  - Configurable action: redact | block | warn

**Implementation steps:**
1. Implement AdversarialInputDetector with main model classification
2. Implement rate limiting with configurable thresholds
3. Implement trust anomaly detection
4. Implement PII detection with regex patterns + main model fallback
5. Wire into MemoryStore pre-write hooks
6. High-risk inputs (>0.7) blocked from memory write
7. Alerts on repeated adversarial attempts

**Tests:**
- `tests/unit/memory/test_adversarial_detector.py::test_detects_prompt_injection`
- `tests/unit/memory/test_adversarial_detector.py::test_detects_trust_manipulation`
- `tests/unit/memory/test_adversarial_detector.py::test_detects_contradiction_flooding`
- `tests/unit/memory/test_adversarial_detector.py::test_normal_input_low_risk`
- `tests/unit/memory/test_rate_limiter.py::test_allows_under_limit`
- `tests/unit/memory/test_rate_limiter.py::test_blocks_over_limit`
- `tests/unit/memory/test_trust_anomaly.py::test_detects_rapid_increase`
- `tests/unit/memory/test_pii_detector.py::test_detects_email`
- `tests/unit/memory/test_pii_detector.py::test_detects_api_key`
- `tests/integration/test_adversarial_workflow.py::test_high_risk_input_blocked`
- `tests/integration/test_pii_workflow.py::test_pii_redacted_before_write`

**Acceptance criteria:**
- Risk score in [0, 1]
- False positive rate <5% on normal inputs
- Detects 90%+ of known adversarial patterns
- High-risk inputs (>0.7) blocked from memory write
- PII detection 95%+ recall, <2% false positive
- Rate limits configurable in config.yaml

**Dependencies:** Phase 1, Phase 2

---

### Phase 18: Backup & recovery strategy

**Goal:** Recover from Git repo corruption, memory store loss, or partial migrations.

**Components:**
- `BackupManager`: daily compressed backups
- `Recovery`: restore from backup
- Git corruption recovery
- Migration safety (dry-run + rollback)

**Files:**
- `hermes/backup/manager.py` — `BackupManager`
  - `create_backup() -> str` (returns backup path)
  - Daily at 02:00, retention 30 days, Git fsck integrity check
- `hermes/backup/recovery.py` — `Recovery`
  - `restore(backup_path) -> RecoveryResult`
- `hermes/backup/git_recovery.py` — Git corruption detection + repair
- `hermes/memory/migrations/base.py` — migration base with dry-run + rollback

**Implementation steps:**
1. Implement daily backup with retention
2. Implement Git fsck integrity verification
3. Implement restore workflow
4. Add Git corruption detection to health checks
5. Add auto-repair (gc, fsck --fix) for repairable corruption
6. Add restore-from-backup for unrepairable corruption
7. Add dry-run + rollback to migration base class

**Tests:**
- `tests/unit/backup/test_manager.py::test_creates_daily_backup`
- `tests/unit/backup/test_manager.py::test_retention_enforced`
- `tests/unit/backup/test_manager.py::test_integrity_verified`
- `tests/integration/test_backup.py::test_full_backup_restore_cycle`
- `tests/integration/test_recovery.py::test_restore_from_backup`
- `tests/integration/test_git_recovery.py::test_detects_corruption`
- `tests/integration/test_migration.py::test_dry_run_reports_only`
- `tests/integration/test_migration.py::test_rollback_after_failed_migration`

**Acceptance criteria:**
- Daily backup created at 02:00 local
- Backups older than 30 days auto-deleted
- Each backup Git-fsck verified
- Restore completes within 5 minutes for 1000-entry repo
- Migration dry-run never modifies state
- Rollback restores pre-migration state

**Dependencies:** Phase 1; T3 (Git-corruption wiring) additionally needs Phase 14 health
checks (P14.T1, P14.T4)

---

### Phase 19: Model version drift handling

**Goal:** Detect and mitigate performance changes when the cloud model provider
deprecates or changes models. Hermes uses one cloud model — deprecation is a real risk.

**Components:**
- `ModelVersionRegistry`: track current model version
- `DeprecationDetector`: detect cloud deprecation warnings
- Drift detection: run eval set on model change

**Files:**
- `hermes/models/version_registry.py` — `ModelVersionRegistry`
  - `register(version, eval_score) -> None`
  - `get_current() -> ModelVersion`
  - `get_history() -> list[ModelVersion]`
- `hermes/models/deprecation.py` — `DeprecationDetector`
  - `check_model_availability() -> AvailabilityResult`
  - Auto-fallback to configured alternative on removal
- `hermes/models/drift_detector.py` — `DriftDetector`
  - `on_version_change(old, new) -> DriftResult`
  - Run eval set, compare scores
  - Alert on >2% regression, auto-rollback on >5%

**Implementation steps:**
1. Implement model version registry
2. Implement deprecation detection (periodic API check)
3. Implement drift detection (eval on version change)
4. Auto-rollback on significant regression
5. Alerts on deprecation warnings

**Tests:**
- `tests/unit/models/test_version_registry.py::test_register_version`
- `tests/unit/models/test_deprecation.py::test_detects_deprecation_warning`
- `tests/unit/models/test_deprecation.py::test_auto_fallback_on_removal`
- `tests/unit/models/test_drift_detector.py::test_eval_runs_on_version_change`
- `tests/unit/models/test_drift_detector.py::test_auto_rollback_on_5_percent_regression`
- `tests/integration/test_drift_workflow.py::test_version_change_triggers_eval`

**Acceptance criteria:**
- Deprecation warnings detected within 24h
- Eval runs automatically on version change
- >2% regression alerts
- >5% regression auto-rollback
- Rollback within 5 minutes

**Dependencies:** Phase 0, Phase 10 (eval sets); T4 (A/B testing) additionally needs
P19.T2 (router.py author)

---

### Phase 20: Expanded testing — chaos, property, mutation, multi-agent

**Goal:** Address testing gaps: chaos testing, property tests for critical paths,
mutation testing for all deterministic logic (Section 4.7 — where applicable,
all tests are mutation-tested), multi-agent integration tests, performance
regression tests.

**Components:**
- Chaos testing framework: system resilience under failure conditions
- Property tests for pipeline, gating, reviewer, trust card, decay, conflict detector
- Mutation testing infrastructure: mutmut + per-PR diff mode + nightly full run + Tier-1/Tier-2 score gates (Section 4.7)
- Multi-agent integration tests
- Performance regression tests

**Files:**
- `tests/chaos/` — chaos tests (main-model API failure, Git corruption, network failure, disk full, writer crash)
- `tests/property/` — property tests for all critical paths
- `tests/integration/test_multi_agent.py` — multi-agent scenarios
- `tests/benchmark/test_regression.py` — performance regression detection
- `pyproject.toml` (mutmut config section: paths-to-mutate, thresholds)
- `.github/workflows/mutation.yml` — per-PR mutation diff + nightly full mutation run

**Implementation steps:**
1. Implement chaos tests: kill main model mid-request, corrupt Git, network failure, disk full, writer crash
2. Implement property tests: any turn → valid output, gating budget, trust score range, decay monotonicity, conflict symmetry
3. Implement mutation infrastructure (P20.T5): mutmut config with in-scope module list and Tier-1/Tier-2 thresholds, per-PR diff mode, nightly full run, CI gates
4. Raise mutation scores to thresholds on Tier-1/Tier-2 modules (P20.T6): fix weak tests, triage survivors, validate Phase 21 bug-fix regression mutants
5. Implement multi-agent tests: concurrent writes, subagent coordination, race conditions
6. Implement performance regression tests: baseline metrics, >10% regression fails CI

**Tests:**
- `tests/chaos/test_model_failure.py::test_model_killed_mid_request`
- `tests/chaos/test_git_corruption.py::test_corrupted_index_recoverable`
- `tests/chaos/test_network_failure.py::test_cloud_unreachable_local_only`
- `tests/chaos/test_disk_full.py::test_write_fails_gracefully`
- `tests/chaos/test_writer_crash.py::test_subagent_detects_writer_death`
- `tests/property/test_pipeline.py::test_any_turn_produces_valid_output`
- `tests/property/test_gating.py::test_injected_never_exceeds_budget`
- `tests/property/test_trust_card.py::test_score_always_in_range`
- `tests/property/test_decay.py::test_decay_never_increases`
- `tests/property/test_conflict_detector.py::test_symmetric`
- `tests/integration/test_multi_agent.py::test_two_agents_concurrent_writes`
- `tests/integration/test_multi_agent.py::test_main_plus_three_subagents`
- `tests/benchmark/test_regression.py::test_no_regression_greater_than_10_percent`

**Mutation testing checks (Section 4.7):**
- Mutation CI job green: Tier-1 modules ≥95% killed, Tier-2 ≥85% killed
- Nightly mutation report artifact published
- Re-mutants of fixed Phase 21 bugs (reintroduce original bug) are killed by regression tests

**Acceptance criteria:**
- All chaos tests pass (system recovers gracefully)
- No data loss in any failure scenario
- All property tests pass with 1000 examples each
- Mutation score: Tier-1 ≥95%, Tier-2 ≥85%, zero unexplained survived mutants in Tier-1
- Multi-agent tests pass without corruption
- Performance regression >10% fails CI

**Dependencies:** Phase 6, Phase 14, Phase 18; T2 (property tests) additionally
Phases 3-4, 8; T3 (multi-agent) additionally Phase 13 (P20.T5: Phase 0 CI;
P20.T6: P21.T1, P21.T2, P21.T5, P21.T11, P17.T2, P17.T4, P19.T2)

---

### Phase 21: Fix known Hermes bugs (distributed)

**Goal:** Fix 12 most impactful open bugs from Hermes GitHub issues (Nov 2026).
Prioritized: critical bugs first, then major, then UX.

**Priority order:**

**Critical (P0) — fix first:**

| Task | Bug | Issue |
|---|---|---|
| P21.T1 | ThreadPoolExecutor _interrupted_threads leak | [#35309](https://github.com/NousResearch/hermes-agent/issues/35309) |
| P21.T2 | Hot memory ignored + disabled tools used | [#26568](https://github.com/NousResearch/hermes-agent/issues/26568) |
| P21.T11 | Custom model configs vanish after update | [#25272](https://github.com/NousResearch/hermes-agent/issues/25272) |

**Major (P1) — fix second:**

| Task | Bug | Issue |
|---|---|---|
| P21.T3 | MCP tools/call arguments dropped (stdio) | [#156143](https://github.com/NousResearch/hermes-agent/issues/156143) |
| P21.T5 | Cron stuck running — no watchdog | [#156160](https://github.com/NousResearch/hermes-agent/issues/156160) |
| P21.T6 | state.db WAL errors / session storage | (troubleshooting docs) |
| P21.T9 | Webapp OOM-killed renderer | [#156161](https://github.com/NousResearch/hermes-agent/issues/156161) |
| P21.T12 | ACP turn runs after editor exits | [#156173](https://github.com/NousResearch/hermes-agent/issues/156173) |

**UX (P2) — fix last:**

| Task | Bug | Issue |
|---|---|---|
| P21.T4 | Skills search 86s cold cache | [#156140](https://github.com/NousResearch/hermes-agent/issues/156140) |
| P21.T7 | Clarify empty free-text misreported as timeout | [#156171](https://github.com/NousResearch/hermes-agent/issues/156171) |
| P21.T8 | Gateway shutdown retries after network loss | [#156127](https://github.com/NousResearch/hermes-agent/issues/156127) |
| P21.T10 | NO_REPLY marker leaks into gateway delivery | [#157239](https://github.com/NousResearch/hermes-agent/issues/157239) |

**Total: 12 bug fixes**

Each task in [hermes-v2-tasks-r7.yaml](./hermes-v2-tasks-r7.yaml) (phase_21 section) includes:
- Direct link to GitHub issue
- Bug description with reproduction scenario
- Specific fix approach
- Files to modify/create
- Tests (unit + integration + chaos where relevant)
- Acceptance criteria tied to issue reproduction scenarios

**Implementation order:** critical → major → UX. Each fix includes regression tests
tied to the original issue reproduction scenario.

**Scheduling (check 4):** T1 and T11 are done immediately after Phase 0 (baseline
stability). T2 keeps its critical priority but lands together with Phase 4: its fix IS
the new injection wiring (deps P4.T3, P1.T3) — fixing the old frozen-snapshot wiring
before replacing it is wasted work. T4 — after Phase 13, T6 — after Phase 6.

**Acceptance criteria (per task):**
- Original issue reproduction scenario passes
- No regression in other functionality
- Fix includes regression tests verified by mutation testing: a mutant
  reintroducing the original bug is killed by the new tests (Section 4.7, task P20.T6)

**Dependencies:** Phase 0 for most tasks; exceptions (task-level): T2 →
Phases 1+4 (the fix lives in the new injector), T4 → Phase 13, T6 → Phase 6 and P21.T12 (author of sessions/store.py — check 6)

---

### Phase 22: Performance optimization

**Goal:** After all features implemented, do a performance pass: latency optimization,
token reduction, caching, cost tuning.

**Components:**
- Latency profiling: identify slow memory ops
- Token usage optimization: reduce prompt sizes, cache aggressively
- Caching layer: memoize repeated retrievals, batch operations
- Cost tuning: use cheaper model tier for memory ops if provider supports

**Files:**
- `hermes/optimization/performance.py` — performance profiler
  - `profile_memory_ops(n_turns=100) -> ProfileReport`
  - Identifies: slowest ops, most expensive ops, cache miss rates
- `hermes/memory/cache.py` — `MemoryCache`
  - Memoize retrieval results (same query → same result within TTL)
  - Batch multiple memory ops into single API call where possible
- `hermes/core/prompt_optimizer.py`
  - Analyze prompt sizes, suggest reductions
  - Track token usage per operation

**Implementation steps:**
1. Profile all memory operations (100-turn benchmark)
2. Identify top-3 slowest and top-3 most expensive operations
3. Implement caching layer for repeated retrievals
4. Batch operations where possible (combine extractor+reviewer into fewer calls)
5. Optimize prompt sizes (remove redundant instructions)
6. Consider cheaper model tier for memory ops (if provider supports multi-model)
7. Re-profile, verify improvement
8. Document performance characteristics

**Tests:**
- `tests/benchmark/test_optimization.py::test_latency_improved_after_optimization`
- `tests/benchmark/test_optimization.py::test_token_usage_reduced`
- `tests/benchmark/test_optimization.py::test_cache_hit_rate_above_30_percent`
- `tests/integration/test_performance.py::test_end_to_end_latency_within_budget`

**Acceptance criteria:**
- Memory ops latency reduced by ≥20% after optimization
- Token usage reduced by ≥15% after optimization
- Cache hit rate ≥30% for repeated queries
- End-to-end latency within budget (5s for injection)
- Performance characteristics documented

**Dependencies:** Hard: Phase 4 (P4.T4), Phase 11 (P11.T1) — profiler inputs.
Scheduling: after all functional phases (Section 7), so the final system is optimized

---

### Phase 16: Documentation & packaging

**Goal:** User-facing docs, migration guide, packaging for distribution.

**Components:**
- README.md updated for v2.0
- Migration guide from v2026.9.24
- Architecture documentation
- User guide (commands, contexts, diagnostics)
- PyPI package

**Files:**
- `docs/architecture.md`
- `docs/migration-from-fork-base.md`
- `docs/user-guide.md`
- `docs/commands-reference.md`
- `README.md` (updated)
- `pyproject.toml` (finalized for PyPI)

**Implementation steps:**
1. Write architecture doc (target state, component overview, data flow)
2. Write migration guide (step-by-step, troubleshooting)
3. Write user guide (contexts, memory commands, diagnostics, skills lifecycle)
4. Write commands reference (all slash commands)
5. Update README with v2.0 features
6. Finalize pyproject.toml for PyPI
7. Build wheel, test installation in clean env
8. Tag release v2.0.0

**Tests:**
- `tests/integration/test_installation.py::test_clean_install_works`
- `tests/integration/test_installation.py::test_migration_script_runs`

**Acceptance criteria:**
- All docs accurate and complete
- Clean install in fresh venv works
- Migration from v2026.9.24 preserves all data
- PyPI package builds and uploads

**Dependencies:** All previous phases

---

## 4. Testing Strategy

### 4.1. Test pyramid

```
        ┌───────────┐
        │    E2E    │  ~10 tests, slow, full system
        ├───────────┤
        │ Integration│  ~100 tests, medium, multi-component
        ├───────────┤
        │   Unit    │  ~350 tests, fast, single component
        ├───────────┤
        │ Property  │  ~40 tests, hypothesis-based
        ├───────────┤
        │  Chaos    │  ~15 tests, failure injection
        └───────────┘
        │ Benchmark │  ~20 tests, performance regression
```

Mutation testing (Section 4.7) is not a separate layer — it is a **quality gate over
the Unit + Property layers**: it measures whether those tests actually detect
behavior changes (mutants) in deterministic logic.

### 4.2. Test markers

```bash
pytest tests/ -m unit          # fast unit tests only
pytest tests/ -m integration   # integration tests
pytest tests/ -m property      # property-based tests
pytest tests/ -m benchmark     # performance benchmarks
pytest tests/ -m e2e           # end-to-end tests
pytest tests/ -m chaos         # chaos/failure injection tests
pytest tests/                  # all
```

### 4.3. Coverage requirements

- Unit test coverage: ≥85% for new modules
- Integration test coverage: all critical paths
- Property-based tests: all parsers, serializers, state machines
- Mutation score (Section 4.7): Tier-1 logic ≥95% killed, Tier-2 logic ≥85% killed
- Benchmark tests: all operations with latency requirements
- Chaos tests: all external dependency failure modes

### 4.4. Continuous integration

GitHub Actions workflow:
- On PR: lint + type-check + unit tests + property tests + mutation diff on changed modules
- On merge to main: + integration tests + benchmark tests
- Nightly: full E2E tests + chaos tests + full mutation run (Section 4.7) + diagnostics health check
- Weekly: GEPA optimization run (if traces available)

### 4.5. A/B memory quality regression test

**What if v2.1 memory is worse than v2026.9.24?** This is the key risk of the re-architecture.
Explicit A/B comparison required before v2.0.0 release:

1. **Setup:** Same 50-turn conversation run against v2026.9.24 and v2.1
2. **Metrics compared:**
   - Fact extraction precision/recall (labeled ground truth)
   - Retrieval relevance (human judgment on top-5 results)
   - Injection usefulness (downstream task success with vs without)
   - Memory growth rate (entries per week)
   - False positive rate (irrelevant facts stored)
3. **Pass criteria:** v2.1 must be ≥v2026.9.24 on all metrics, or ≥10% better on ≥3 metrics
4. **If fail:** investigate, fix, re-run before release
5. **Test location:** `tests/e2e/test_ab_memory_quality.py`

### 4.6. Test fixtures

- `tests/fixtures/traces/` — sample traces from real sessions (anonymized)
- `tests/fixtures/memory_states/` — sample memory repos for migration tests
- `tests/fixtures/eval_sets/` — labeled eval sets for GEPA
- `tests/fixtures/models/` — mock model responses for unit tests (no real API calls)

### 4.7. Mutation testing (check-3 requirement: where applicable, all tests must be mutation-tested)

**Principle:** a test suite proves its value by killing mutants. Every unit/property
test that verifies deterministic logic must be mutation-tested: each survived mutant
is either a missing assertion (fix the test) or an equivalent mutant (mark
`# pragma: no mutate` with a one-line justification, reviewed in PR). Coverage %
alone does not prove assertions exist — mutation score does.

**Tool:** `mutmut` (Python), configured in `pyproject.toml` + `.github/workflows/mutation.yml`.

**In scope — deterministic logic modules (mutation-tested):**

| Tier | Modules | Mutated logic | Threshold (killed) |
|---|---|---|---|
| 1 — critical | `memory/trust_card.py` | weighted trust score, parse/serialize roundtrip | ≥95% |
| 1 — critical | `memory/gating.py` | budget enforcement (≤3 claims, ≤1200 chars), decision post-processing, trust-only fallback | ≥95% |
| 1 — critical | `memory/pii_detector.py` | PII regex patterns, redaction | ≥95% |
| 1 — critical | `memory/rate_limiter.py` | write limits, burst detection | ≥95% |
| 1 — critical | `models/drift_detector.py` | 2% alert / 5% rollback thresholds | ≥95% |
| 1 — critical | `cron/watchdog.py` (P21.T5) | inactivity timeout, force-fail | ≥95% |
| 2 — logic | `memory/decay.py`, `injector.py` | forgetting curve, stable/dynamic split, char budget | ≥85% |
| 2 — logic | `memory/store.py`, `context.py`, `hot_reload.py`, `compaction.py`, `soul.py` | CRUD, boundaries, segment invalidation | ≥85% |
| 2 — logic | `memory/conflict_detector.py`, `retrieval.py`, `trust_anomaly.py` | similarity thresholds, BM25 top-k, anomaly rules | ≥85% |
| 2 — logic | `optimization/metrics.py`, `pareto.py` | F1/recall/precision, dominance | ≥85% |
| 2 — logic | `concurrency/manager.py`, `protocol.py` | lock semantics, message framing | ≥85% |
| 2 — logic | `diagnostics/alerts.py`, `health_checks.py`, `metrics.py` | alert rules, percentiles | ≥85% |
| 2 — logic | `skills/lifecycle.py`, `registry.py` | state machine transitions, versioning | ≥85% |

**Out of scope (mutation runs give no signal):**
- LLM-boundary glue (`extractor.py`, `reviewer.py`, `adversarial_detector.py`,
  `topic_detector.py`, the gate model call itself): behavior is non-deterministic
  behind a mocked model; mutations of glue add cost without signal. Covered by
  structured-output validation, property tests and eval sets instead.
- Integration / E2E / chaos / benchmark suites: too slow to mutate.
- One-time migration scripts: covered by dry-run + rollback integration tests.
- I/O-bound wrappers (`git_backend.py`, `backup/manager.py`): covered by integration tests.

**Workflow:**
1. Per-PR: mutation diff on changed in-scope modules only (target <10 min)
2. Nightly: full mutation run over all in-scope modules, report published as CI artifact
3. Survivor triage: fix the test, or mark equivalent mutant with `# pragma: no mutate` + justification
4. Gate: PR fails if a Tier-1 module's mutation score drops below its threshold
5. Phase 21 cross-check: each bug-fix regression test must kill a mutant reintroducing
   the original bug — "tests that would have caught the bug" becomes a verifiable
   criterion (validated in P20.T6)

**Exit criterion for v2.0.0 release:** Tier-1 ≥95%, Tier-2 ≥85%, zero unexplained
survived mutants in Tier-1.

---

## 5. Self-Diagnostics System

(See Phase 14 for full spec)

### 5.1. Health checks (run every 5 min)

| Check | Status if | Auto-repair |
|---|---|---|
| memory_store_integrity | All entries have valid trust cards | Re-run migration for missing cards |
| git_state | Clean working tree | Auto-commit dirty state >1h |
| main_model_reachable | Main model API responds to ping | None (alert user, skip ops) |
| trace_store_size | <10GB | Archive old traces |
| pareto_front_freshness | Last GEPA run <8 days | Trigger GEPA run |
| backup_freshness | Last backup <25 hours | Trigger backup |

### 5.2. Metrics (collected every 5 min)

- Latency: p50, p95, p99 per operation (extractor, reviewer, gate, compactor/rewriter)
- Token usage: per operation, per day, estimated cost
- Trust score distribution: avg, min, max, count below 0.3
- Conflict rate: 7-day rolling
- Retrieval recall@5, recall@10
- Skills: active count, pending approvals, deprecated 30d
- GEPA: last run timestamp, candidates in Pareto front
- Model version: current, last deprecation check

### 5.3. Alerts

| Alert | Trigger | Severity |
|---|---|---|
| main_model_down | Model API health check fails 3x in a row | critical |
| git_dirty_long | Git dirty >1h despite auto-repair | warning |
| trace_store_full | Trace store >10GB | warning |
| gepa_stale | No GEPA run in 8 days | warning |
| low_trust_entries | >10 entries with trust <0.3 | info |
| skill_pending_long | Skill pending approval >7 days | info |
| pii_detected_high_rate | PII detection rate >10% of writes | warning |
| adversarial_attempts | >5 blocked adversarial attempts in 1h | warning |
| cost_anomaly | Daily token cost >2x rolling average | warning |
| backup_stale | Last backup >25 hours old | warning |
| cron_job_stuck | Cron watchdog (P21.T5) reports run with no progress >30min | warning |
| fallback_rate_high | Trust-only/regex fallback >20% of ops in 1h (main model slow/unavailable) | warning |

### 5.4. Auto-repair

Top 5 common issues auto-repaired:
1. Dirty Git state → auto-commit with message "auto-repair: commit pending changes"
2. Missing trust cards → re-run migration for affected entries
3. Trace store full → archive traces older than 90 days
4. Pareto front stale → trigger GEPA run
5. Backup stale → trigger backup

### 5.5. Graceful degradation on main model failure

When main model API is unreachable:
- Memory extraction (Phase 3) → **skip**, retry next turn
- CBDC gating (Phase 4) → **fallback to trust-score-only gating** (no LLM: inject top-3
  by trust score if all >0.7, skip otherwise)
- Conflict detection (Phase 8) → **fallback to embeddings-only** (BM25/embedding
  similarity, skip contradiction detection)
- Adversarial detection (Phase 17) → **fallback to regex patterns only**
- All degradations logged and reported in diagnostics

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Memory ops latency exceeds 5s timeout (main model slow) | Medium | High | Batched single-call gating (≤1 round-trip); BM25 pre-filter (top-3 only); 3.5s gate timeout + trust-only fallback; async pipeline; skip on timeout |
| API cost higher than estimated | Medium | Medium | Cost tracking in diagnostics (Section 5.2); cheaper model tier option; budget alerts |
| GEPA optimization overfits to eval set | Medium | Medium | Diverse eval sets; periodic re-evaluation; Pareto front prevents single-point optimization |
| Git repo grows unbounded | Low | Medium | Weekly GC; archive stale entries |
| Migration loses data from v2026.9.24 | Low | Critical | Backup before migration; dry-run mode; rollback script (Phase 18) |
| Concurrent writers deadlock | Low | High | Timeout on lock; writer crash recovery (Phase 6) |
| Prefix cache invalidation too aggressive | Medium | Medium | Segment-level invalidation; stable/dynamic split (Phase 7) |
| Multi-modal attachments bloat storage | Low | Low | Size limits per attachment; optional OCR/transcription |
| v2.1 memory quality worse than v2026.9.24 | Medium | Critical | A/B regression test (Section 4.5) gates release |
| Adversarial detection false positives annoy users | Medium | Medium | Configurable sensitivity; allowlist for trusted sources |
| Main model API rate limits hit during heavy usage | Medium | High | Exponential backoff; rate limiter (Phase 17); graceful degradation (Section 5.5) |
| Cloud model deprecation breaks system | Low | High | Deprecation detection + auto-fallback (Phase 19) |

---

## 7. Implementation Order

```
Phase 0 (Foundation)
   │
   ├──> Phase 1 (Storage)
   │      │
   │      ├──> Phase 2 (Trust cards)
   │      │      │
   │      │      └──> Phase 3 (Extractor + reviewer)
   │      │             │
   │      │             └──> Phase 4 (CBDC gating)
   │      │                    │
   │      │                    └──> Phase 7 (Hot-reload)
   │      │
   │      ├──> Phase 5 (Context layer) [parallel; T1 before P4.T3]
   │      ├──> Phase 6 (Concurrent writers) [parallel]
   │      └──> Phase 15 (SOUL split & UX) [parallel]
   │
   ├──> Phase 11 (Trace Store) [before GEPA; T1 before P3.T6 (multi-turn context)]
   │      │
   │      └──> Phase 10 (GEPA)
   │
   ├──> Phase 8 (Memory quality) [after Phase 3]
   ├──> Phase 13 (Skills lifecycle) [parallel]
   ├──> Phase 17 (Adversarial detection) [parallel, after Phase 2]
   ├──> Phase 18 (Backup & recovery) [parallel]
   ├──> Phase 19 (Model version drift) [parallel]
   ├──> Phase 21 (Fix known bugs) [parallel; T1/T11 right after Phase 0; T2 with Phase 4; T4 after P13; T6 after P6]
   │
   ├──> Phase 14 (Self-diagnostics) [after most phases; T1 after P18.T1 (backup_freshness)]
   ├──> Phase 20 (Expanded testing + mutation) [after Phase 6, 14, 18]
   ├──> Phase 22 (Performance optimization) [after all functional]
   └──> Phase 16 (Documentation) [last]
```

**Backbone (primary dependency chain):** Phase 0 → 1 → 2 → 3 → 4 → 7
**Parallel tracks:** Phases 5, 6, 11, 13, 15, 17, 18, 19, 21 can run alongside the backbone

**Bug fix priority (Phase 21):** Critical bugs T1 and T11 are fixed first, immediately
after Phase 0 — they affect baseline stability. T2 (hot memory) is critical by priority,
but its fix IS the new Phase 4 injection wiring (deps P4.T3, P1.T3); fixing the old wiring
before replacing it is double work. Major and UX bugs are spread across the timeline
(T4 after Phase 13, T6 after Phase 6).

---

## 8. References

### Hermes base
- Hermes Agent repo: https://github.com/NousResearch/hermes-agent
- Hermes docs: https://hermes-agent.nousresearch.com/docs
- Hermes persistent memory docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- Hermes MCP integration: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- Hermes skills system: https://hermes-agent.nousresearch.com/docs/user-guide/features/skills

### Referenced systems (patterns borrowed from)
- ZCode memory: https://zcode.z.ai/en/docs/memory
- Claude Code memory: https://code.claude.com/docs/en/memory
- DeepSeek Harness: https://www.deepseek.com/en/harness
- DSH memory plugin discussion: https://github.com/deepseek-ai/deepseek-harness/discussions/1345

### Historical design docs (in this repo)
- Tiered model architecture (rejected in round 3): [tiered-model-architecture.md](./tiered-model-architecture.md)

### Tools & libraries
- GEPA: https://github.com/gepa-ai/gepa
- GEPA paper: https://arxiv.org/abs/2507.19457
- rank_bm25: https://github.com/dorianbrown/rank_bm25
- pygit2: https://www.pygit2.org/
- python-frontmatter: https://github.com/eyeseast/python-frontmatter
- sentence-transformers: https://www.sbert.net/
- pytest: https://docs.pytest.org/
- hypothesis (property-based): https://hypothesis.readthedocs.io/

### Models
- Qwen-2.5: https://qwenlm.github.io/blog/qwen2.5
- Claude models: https://docs.anthropic.com/en/docs/about-claude/models
- GLM models: https://open.bigmodel.cn/

---

## Appendix A: File layout (target)

```
hermes-v2/
├── hermes/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py                  # extended from run_agent.py
│   │   ├── retry.py                  # RetryHandler (exponential backoff)
│   │   ├── tool_executor.py          # fixed ThreadPoolExecutor
│   │   ├── tool_registry.py          # fixed tool enable/disable
│   │   └── execution_context.py      # per-execution interrupt state
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                  # MemoryStore
│   │   ├── git_backend.py            # GitBackend
│   │   ├── trust_card.py             # TrustCard parser
│   │   ├── entry.py                  # MemoryEntry dataclass
│   │   ├── context.py                # ContextManager
│   │   ├── context_window.py         # multi-turn context
│   │   ├── retrieval.py              # RetrievalEngine (BM25)
│   │   ├── gating.py                 # CBDCGate
│   │   ├── injector.py               # ContextInjector
│   │   ├── hooks.py                  # pre-step hooks
│   │   ├── hot_reload.py             # MemoryHotReloader
│   │   ├── decay.py                  # forgetting curve
│   │   ├── conflict_detector.py      # ConflictDetector
│   │   ├── multimodal.py             # MultimodalAttachment
│   │   ├── soul.py                   # SoulManager
│   │   ├── pipeline.py               # MemoryPipeline orchestrator
│   │   ├── extractor.py              # ExtractorAgent
│   │   ├── reviewer.py               # ReviewerAgent
│   │   ├── adversarial_detector.py   # AdversarialInputDetector
│   │   ├── rate_limiter.py           # RateLimiter
│   │   ├── trust_anomaly.py          # TrustAnomalyDetector
│   │   ├── pii_detector.py           # PIIDetector
│   │   ├── types.py                  # dataclasses
│   │   └── migrations/
│   │       ├── base.py               # migration base (dry-run + rollback)
│   │       └── fork_to_v2_0.py
│   ├── traces/
│   │   ├── __init__.py
│   │   ├── store.py                  # TraceStore
│   │   └── schema.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── gepa_runner.py
│   │   ├── pareto.py
│   │   ├── metrics.py
│   │   ├── performance.py            # performance profiler (Phase 22)
│   │   └── adapters/
│   │       ├── extractor_adapter.py
│   │       ├── reviewer_adapter.py
│   │       ├── gate_adapter.py
│   │       └── rewriter_adapter.py
│   ├── concurrency/
│   │   ├── __init__.py
│   │   ├── manager.py                # ConcurrencyManager
│   │   ├── writer.py                 # WriterProcess
│   │   └── protocol.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── system.py                 # DiagnosticsSystem
│   │   ├── health_checks.py
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   └── auto_repair.py
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── lifecycle.py              # SkillLifecycle
│   │   ├── registry.py
│   │   ├── search.py                 # fixed skills search (P21.T4)
│   │   └── index.py                  # bundled index
│   ├── backup/
│   │   ├── __init__.py
│   │   ├── manager.py                # BackupManager
│   │   ├── recovery.py
│   │   └── git_recovery.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── version_registry.py       # ModelVersionRegistry
│   │   ├── deprecation.py            # DeprecationDetector
│   │   └── drift_detector.py         # DriftDetector
│   ├── cron/
│   │   ├── __init__.py
│   │   ├── scheduler.py              # cron scheduler (P21.T5 fix)
│   │   └── watchdog.py               # cron inactivity watchdog (P21.T5)
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── verify_tool.py            # verify MCP tool (Phase 2)
│   │   ├── client.py                 # MCP client (P21.T3 fix)
│   │   ├── protocol.py               # stdio JSON-RPC framing (P21.T3 fix)
│   │   └── argument_validator.py     # MCP arg schema validation (P21.T3)
│   ├── sessions/
│   │   └── store.py                  # session storage (P21.T6/T12 fixes)
│   ├── install/
│   │   └── migrations.py             # update-time migrations (P21.T11 fix)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── database.py               # fixed SQLite WAL (P21.T6)
│   ├── acp/
│   │   ├── __init__.py
│   │   └── server.py                 # fixed ACP (P21.T12)
│   └── gateway/
│       ├── __init__.py
│       ├── health.py                 # /health, /metrics endpoints (P14.T6)
│       ├── topic_detector.py
│       ├── session_manager.py
│       ├── clarify.py                # fixed clarify (P21.T7)
│       ├── shutdown.py               # fixed shutdown (P21.T8)
│       └── delivery.py               # fixed NO_REPLY leak (P21.T10)
├── commands/
│   ├── context.md
│   ├── skills.md
│   ├── learning.md
│   ├── memory.md
│   ├── soul.md
│   └── diagnostics.md
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   ├── benchmark/
│   ├── chaos/
│   ├── e2e/
│   │   └── test_ab_memory_quality.py # A/B regression test
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── migration-from-fork-base.md
│   ├── user-guide.md
│   └── commands-reference.md
├── scripts/
│   ├── migrate_fork_to_v2_0.py
│   └── hermes-restore.py
├── pyproject.toml
├── README.md
└── .github/workflows/
    ├── ci.yml
    └── mutation.yml          # nightly full mutation run + per-PR diff (Section 4.7)
```

---

## Appendix B: Task list (machine-readable)

See [hermes-v2-tasks-r7.yaml](./hermes-v2-tasks-r7.yaml) for structured task list parseable
by coding agents. Each task has: id, phase, title, description, files_to_create,
files_to_modify, tests_to_write, acceptance_criteria, dependencies,
github_issue (for Phase 21 bug fixes).

---

## Appendix C: How to use this plan

### For a coding agent picking this up cold

1. **Read sections 0-2** for context and target architecture
2. **Read [hermes-v2-tasks-r7.yaml](./hermes-v2-tasks-r7.yaml)** for structured task list
3. **Start with Phase 0** — fork repo, set up skeleton
4. **Fix Phase 21 critical bugs** (P21.T1, T11 — right after Phase 0; T2 lands together
   with Phase 4, see Section 3 / Phase 21 Scheduling) — baseline stability
5. **Follow dependency graph** in section 7 — don't skip phases
6. **For each task**: implement files, write tests (test names are specs), meet
   acceptance criteria; deterministic-logic tests must be mutation-strong
   (Section 4.7 — survivors are fixed or justified)
7. **Run diagnostics** (Phase 14) to verify system health after each phase
8. **Commit to Git** after each task with task ID in commit message
9. **Run A/B memory quality test** (Section 4.5) before release

### For a human reviewer

1. **Sections 0-2**: high-level vision
2. **Section 2.6**: API cost estimation — is this acceptable for your use case?
3. **Section 3**: detailed phases (read per-phase if reviewing specific part)
4. **Section 4.5**: A/B memory quality regression test — key release gate
5. **Section 5**: self-diagnostics
6. **Section 6**: risks
7. **Section 7**: implementation order (Phase 21 critical bugs first)
