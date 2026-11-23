# hermes-v2 Commands Reference

Every hermes-v2 slash command, its subcommands and flags. Handlers live in
`hermes/core/agent.py` (dispatch registries `CONTEXT_COMMANDS`,
`MEMORY_COMMANDS`, `SOUL_COMMANDS`, `SKILL_COMMANDS`, `LEARNING_COMMANDS`)
and `hermes/gateway/health.py` (`/diagnostics`); full behaviour docs are in
`commands/*.md` and [docs/user-guide.md](user-guide.md). Expected failures
(unknown names, bad slugs, illegal transitions) come back as `error: ...`
reply strings — a command never crashes the session.

## /context

Context management (`hermes.memory.context.ContextManager`, plan P5). Facts
are isolated per context; every context inherits the global facts up its
parent chain.

| Command | Description |
|---|---|
| `/context list` | Show every context with parent, active sessions and fact count. |
| `/context switch <name>` | Switch the session to `<name>`; the stable memory block reloads at this boundary. Unknown names are an error. |
| `/context create <name> [--parent <p>]` | Create a context; `<name>` must be a lowercase slug (`[a-z0-9][a-z0-9-]*`); `--parent` selects the inherited context (default: global). |
| `/context inspect [name]` | Details for one context — parent, backing path, active sessions, fact count. Defaults to `global`. |
| `/context delete <name> [--force]` | Delete the context's facts dir + metadata. Refuses `global` and contexts with active sessions unless `--force`; children re-parent to global. |

## /memory

Memory introspection and transfer (`hermes.memory.store.MemoryStore`, plan
P15.T3). Entries are trust-card markdown files; export preserves ids, trust
cards and the full verification history.

| Command | Description |
|---|---|
| `/memory list [context]` | Table of entries — id, type, context, trust score, text snippet. Without `context`: all contexts; with: only that one. |
| `/memory search <query>` | BM25 keyword search, best first; only entries sharing a query token are returned (archived entries excluded). |
| `/memory stats` | Counts by entry type and context plus the trust distribution (avg, min, count below 0.3). |
| `/memory export [out_path]` | Dump every entry (complete markdown file per entry) to one JSON file; default path `memory-export-<timestamp>.json` in the Hermes home. |
| `/memory import <path> [--skip]` | Import an export file; unknown ids are written verbatim (batch Git commit). Default conflict policy `--skip`. |
| `/memory import <path> --overwrite` | Same, but existing ids are replaced (also across contexts); near-duplicates are flagged (`conflicts_flagged=N`), never blocked. |

## /soul

SOUL.md split surface (`hermes.memory.soul.SoulManager`, plan P15.T4). The
core is immutable (no write API exists); only the style is editable, hard
bounded to 2000 characters.

| Command | Description |
|---|---|
| `/soul inspect` | Show both layers with char counts — the immutable core and the adaptive style (`<n>/2000 chars`). |
| `/soul edit-style <text>` | Replace the adaptive style; atomic, appended to `.style_history.log` and committed (`[P15.T1] update soul style`). Over-limit updates are refused, nothing written. |
| `/soul edit-style -` | Read the new style text from stdin. |
| `/soul history` | Style change history: `git log` of `soul/soul_style.md` in a Git-backed repo (sha, timestamp, message), otherwise `.style_history.log`. |

## /skills

Governed skill management (`hermes.skills.lifecycle.SkillLifecycle` +
`hermes.skills.registry.SkillRegistry`, plan P13.T4). Lifecycle:
`proposed -> approved -> active -> deprecated -> archived`.

| Command | Description |
|---|---|
| `/skills list` | Every registered skill with state, version and `last_used_at`. |
| `/skills inspect <name>` | One skill's state, version history, rationale, `last_used_at`, deprecation reason. Accepts skill id or display name. |
| `/skills approve <id>` | User approval: `proposed -> active` (an intermediate `approved` state is accepted too). |
| `/skills deprecate <name> [reason]` | Retire an active skill, recording the reason (default: "deprecated via /skills"). This is also how an AutoDeprecator proposal is approved. |
| `/skills rollback <name> <version>` | Restore a prior version's content as a NEW latest version (non-destructive, reversible). Requires `approved` or `active` state. |

## /learning

Learning-loop dashboard (plan P13.T5) — read-only over the skill lifecycle,
skill registry, auto-deprecator, trace store and Pareto front store.
Unwired components render placeholder lines, never errors.

| Command | Description |
|---|---|
| `/learning stats` | Skill totals per lifecycle state, pending approvals (proposals + auto-deprecations) and trace volume. |
| `/learning traces` | The recent 5 traces, newest first, with quality labels (`good`/`acceptable`/`bad`/`needs-review`/`unlabeled`). |
| `/learning pending` | Skills awaiting your decision: proposed skills (approve via `/skills approve`) + pending auto-deprecation proposals (retire via `/skills deprecate`). |
| `/learning history` | GEPA run history — per-component Pareto front sizes with best mean score. |

## /diagnostics

Self-diagnostics (`hermes.diagnostics.system.DiagnosticsSystem` via
`hermes.gateway.health.handle_diagnostics_command`, plan P14.T6). Bare
`/diagnostics` is the full report; generation completes in <2s.

| Command | Description |
|---|---|
| `/diagnostics` (or `full`) | The full plan 5.1 report: overall status, every check with status/details, the metrics block, fired alerts. |
| `/diagnostics quick` | Just the overall status one-liner (`healthy` / `degraded` / `unhealthy`). |
| `/diagnostics repair` | Run health check with auto-repair: every non-passing check with a plan 5.4 repair gets repaired and reported. |
| `/diagnostics repair <issue>` | Run one repair: `dirty_git`, `missing_trust_cards`, `trace_store_full`, `stale_pareto`, `stale_backup` (unknown keys reply with the known list). |

### HTTP endpoints (gateway / dashboard)

| Route | Description |
|---|---|
| `GET /health` | Full health report JSON (`HealthReport.to_json()`), refreshed per request (`hermes.gateway.health`). |
| `GET /metrics` | Latest `MetricsSnapshot` from the JSONL time series (fresh collect when none persisted) plus the model-metrics summary (`null` when unwired). |
| `GET /` or `GET /dashboard` | The HTML dashboard — tables + inline-SVG trend sparklines, offline-capable (`hermes.diagnostics.dashboard_http`). |
| CLI dashboard | `hermes.diagnostics.dashboard_cli.render_dashboard_text(...)` / interactive pager (arrow-key page navigation: HEALTH / TRENDS / ALERTS / ...; `q` quits). |

Repair attempts are logged to `~/.hermes/diagnostics/repairs.log`; fired
alerts to `~/.hermes/diagnostics/alerts.log`.
