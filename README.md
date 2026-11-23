# Hermes v2 Fork — Memory-First Build

A fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT)
that re-architects agent memory around a **Git-backed atomic markdown graph**, adds a
quality-gated extraction pipeline, and fixes 12 known upstream bugs.

This repository is a **snapshot build** (single commit, upstream history omitted — see the
upstream repo for prior history; CI workflows stripped for publishing). Synced with
upstream `main` as of **2026-11-23**. The original upstream README is preserved at
[README-upstream.md](README-upstream.md).

## What changed in agent memory

The stock Hermes memory is a frozen three-file snapshot (MEMORY.md / USER.md / SOUL.md)
injected once per session, with hard char limits and FTS5 episodic recall. This fork
replaces it with a pipeline that is versioned, provenance-aware, cache-safe, and
observable (modules under `hermes/memory/`):

**Storage & versioning**
- **Atomic markdown graph** — one fact per file, `MEMORY.md` as a generated index, all in
  a dedicated git repo via a `pygit2` backend (`memory/git_backend.py`): full version
  history, audit trail, merge conflicts as an explicit signal.
- **Trust cards** — every entry carries provenance (`verified_by`, `verified_at`,
  `evidence`, `trust_score`); `trust_anomaly.py` flags entries whose evidence no longer
  holds.
- **Forgetting curve** — time-decay scheduling (`decay.py`) instead of unbounded growth.
- **Concurrent writers** — file locking (`locking.py`) + a write journal with 5s-bounded
  replay and dedupe (crash recovery proven under ENOSPC and mid-batch kill).

**Quality pipeline (extractor → reviewer → gate)**
- Two-stage extraction/review on the main model (`extractor.py`, `reviewer.py`): dedup,
  in-place update, conflict detection pre-write (`conflict_detector.py`), adversarial
  input screening (`adversarial_detector.py`).
- **PII scanning** with redact/skip modes (`pii_detector.py`): email/phone/API-key/
  credit-card (+Luhn), SSN.
- **Gated proactive injection** (`gating.py`, `injector.py`): retrieval → use/verify/
  ignore decision → only "use" items injected, within budget. Injection is
  timeout-guarded (2 s typical, 5 s hard) and never breaks the turn (1.68 ms worst
  observed vs a 5 s budget).

**Session & prompt-cache behavior**
- **Hot reload with segment-level invalidation** (`hot_reload.py`) — mid-session memory
  updates swap in per-entry prompt segments with stable markers; non-invalidated segments
  stay byte-identical, preserving the prefix cache (measured hit-rate ≈ 0.90).
- **Topic detection** on gateways — model-injected classification with a lexical
  fallback (trigram cosine, 1–2 ms); closes virtual session boundaries automatically
  instead of requiring manual `/new`.
- **Session compaction** (`compaction.py`) — summaries stored as reference entries with
  trace-range provenance (they point at raw turns, they don't mirror them).
- **Memory cache** (`cache.py`) — TTL + LRU with per-op stats; measured **−61%
  model-facing tokens** for the memory pipeline.

**Observability & ops**
- Trace store (`hermes/traces/`) capturing per-turn traces (extraction / review / gating /
  injection), GEPA-ready — offline prompt self-optimization scaffolding (pilot frozen).
- `memory_ops` MCP server (`hermes/mcp/`) — stats / search / get.
- Diagnostics + dashboard (`hermes/diagnostics/`) — CLI pages and an offline HTTP
  `/dashboard` with trend charts.
- SQLite-backed optional session store (`hermes/sessions/`) — pooled connections,
  `BEGIN IMMEDIATE`; zero `BUSY` errors under thread + multi-process load.

## Bug fixes applied (upstream issues)

| Fix | Upstream issue |
|---|---|
| ThreadPoolExecutor interrupt-state leak | #35309 |
| Hot memory ignored + disabled tools still callable | #26568 |
| MCP stdio arguments silently dropped | #156143 |
| Skills search: 86 s cold cache → <1 s via bundled index | #156140 |
| Cron job stuck in `running` — inactivity watchdog + alerting | #156160 |
| ACP turn runs after editor exit (peer-EOF detection) | #156173 |
| Webapp renderer OOM — heap cap + explicit exit codes | #156161 |
| Gateway shutdown retry loop after network loss | #156127 |
| Clarify: empty free-text misreported as timeout | #156171 |
| Custom model configs vanished after update (backup + safe merge) | #25272 |
| `NO_REPLY` internal marker leaked into chat | #157239 |
| state.db WAL `BUSY` storms under concurrent writers | — |

Fork-specific fixes beyond upstream: `hermes_home()` resolution on Windows
(`%LOCALAPPDATA%\hermes` vs a phantom `~/.hermes`), Telegram polling-conflict
self-healing, and a test-home I/O guard so the test suite can never read or write the
real Hermes home.

## Install from scratch

Windows (PowerShell), Linux/macOS (bash):

```bash
git clone https://github.com/i89489627/hermes-v2-fork.git
cd hermes-v2-fork
. ./activate            # PowerShell: . .\activate.ps1
hermes --version
```

`activate` bootstraps everything via Hermes' `pm` runtime: a pinned Python, tools
(uv, node, pnpm, ffmpeg, ripgrep, chromium, cua-driver) and all dependencies from
`uv.lock`, hash-verified, into an isolated Hermes home (default
`%LOCALAPPDATA%\hermes` on Windows, `~/.hermes` elsewhere; override with
`HERMES_HOME`). First run downloads a few hundred MB.

Notes:
- `pip install` is intentionally refused by upstream's `setup.py` — Hermes is not
  distributed as a wheel. Use the activate flow above (dev-install), matching how
  this fork is actually run.
- The fork's `hermes/` package occupies the repo path of upstream's root `hermes`
  launcher script, so `activate` invokes `hermes_cli/main.py` directly — the same
  entry point the installed gateway uses.

## Test status

- Fork v2 suites (unit + property + chaos): **263 passed** on the synced tree.
- Upstream memory slice: 246 passed; 7 known failures reproduce on pure upstream `main`
  (path-case sensitivity on Windows — upstream-side, not introduced here).

## License

MIT, inherited from upstream. All credit for the original Hermes Agent belongs to
[Nous Research](https://github.com/NousResearch/hermes-agent) and its contributors.
