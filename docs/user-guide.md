# hermes-v2 User Guide

hermes-v2 is the memory-first re-architecture of the Hermes agent: every
fact the agent learns is a markdown file with a **trust card** in a Git
repository, verified, decayed and injected back into your conversations
according to how trustworthy it is. This guide covers the features a user
interacts with. Command syntax is summarized in
[docs/commands-reference.md](commands-reference.md); the internals are in
[docs/architecture.md](architecture.md); upgrading from the fork base is in
[docs/migration-from-fork-base.md](migration-from-fork-base.md).

Where files live: everything below is under the Hermes home
(`HERMES_HOME` environment variable overrides it; default `~/.hermes`). The
memory graph is `~/.hermes/memory-repo/` — an ordinary Git repository, so
you can inspect it with any Git tooling.

## Contexts (`/context`)

Memory is scoped by contexts: the implicit `global` context plus any number
of named contexts (`memory-repo/contexts/<name>/facts/`). Facts never bleed
between contexts, and every context **inherits the global facts** up its
parent chain.

```
/context list                  # every context: parent, sessions, fact count
/context create work           # new context (lowercase-slug name)
/context create project-x --parent work
/context switch work           # switch this session to a context
/context inspect work          # details for one context (default: global)
/context delete work --force   # delete context + its facts (global is protected)
```

- `switch` reloads the stable memory block at the session boundary — nothing
  is cached across a switch.
- `delete` refuses contexts with attached active sessions unless `--force`.
- The merged fact view (context's own facts + inherited global facts,
  nearest copy wins) is what `/context inspect` reports; `/memory list
  <name>` shows only that one context's own entries.

## Memory (`/memory`)

Every entry is a markdown file whose YAML frontmatter is its **trust card**:
id, type (`User` / `Feedback` / `Project` / `Reference`), the full
verification history, and a `trust_score` in `[0, 1]`. The score is the
weighted average of verification events — a fact you stated explicitly
(`user-explicit`, weight 1.0) or that a passing test anchored
(`test-green`, 1.0) outranks a guess the agent inferred on its own
(`agent-inferred`, 0.5) or that an LLM asserted without evidence (`llm`,
0.3). Scores decay slowly over time when unverified (the forgetting curve)
and stale, low-trust entries eventually get archived (see Archives below).

```
/memory list                   # table: id, type, context, trust, snippet
/memory list work              # only one context
/memory search pnpm            # BM25 keyword search, best first
/memory stats                  # counts by type/context + trust distribution
/memory export                 # dump EVERY entry (+ full trust cards) to JSON
/memory export backups/mem.json
/memory import backups/mem.json            # import (conflicts skip by default)
/memory import backups/mem.json --overwrite
```

- **Export** writes one JSON file carrying each entry's complete markdown
  file (frontmatter + body), so ids, trust cards and the whole verification
  history survive a move. Without a path the file lands in the Hermes home
  as `memory-export-<timestamp>.json`.
- **Import** writes unknown ids verbatim (one batch Git commit) and handles
  existing ids as conflicts: `--skip` (default) keeps the original,
  `--overwrite` replaces it. Near-duplicates/contradictions are flagged in
  the summary (`conflicts_flagged=N`) but never blocked — the flag is advice,
  the policy is yours.
- The agent maintains this memory autonomously (extraction/review after each
  turn); the commands above are your window into it. To explicitly vouch for
  a fact, the `verify` MCP tool appends a verification event to its card.

## Soul (`/soul`)

The agent's identity is split into two files under
`memory-repo/soul/`:

- `soul_core.md` — safety, alignment, identity. **Immutable**: there is no
  API that writes it, by design. Changing it requires an explicit manual Git
  operation by you.
- `soul_style.md` — the adaptive communication style, hard-bounded to 2000
  characters.

```
/soul inspect                  # both layers with char counts
/soul edit-style Prefer concise replies. Use bullet lists for multi-step plans.
/soul edit-style -             # read the new style from stdin
/soul history                  # git log (or .style_history.log) of style changes
```

An update beyond the 2000-char limit is refused and nothing is written.

## Learning loop (`/learning`) and skills lifecycle (`/skills`)

The agent proposes skills autonomously; you govern them. Skills move
through `proposed -> approved -> active -> deprecated -> archived`, and the
state machine is the single source of truth (`memory-repo/skills/`).

```
/learning stats                # skill counts, pending approvals, trace volume
/learning traces               # recent 5 turns with quality labels
/learning pending              # what waits for your approval
/learning history              # GEPA optimization history
/skills list                   # every skill: state, version, last_used_at
/skills inspect greeter-skill  # details incl. version history
/skills approve greeter-skill  # proposed -> active
/skills deprecate old-skill superseded by the new greeter
/skills rollback greeter-skill 1   # restore v1 as a NEW latest version
```

- **Auto-deprecation**: skills unused for 30+ days are *proposed* for
  retirement (`pending_deprecations.json`); your `/skills deprecate` of such
  a skill IS the approval. `/learning pending` lists them.
- **Rollback** is non-destructive: old content is restored as a new latest
  version, so history is never rewritten and the rollback itself is
  reversible.
- **GEPA** (weekly self-optimization) rewrites the pipeline's own prompts:
  before/after scores per run are tracked (`/learning history`), and the
  system alerts when a component stops improving.

## Diagnostics (`/diagnostics`, dashboards, HTTP)

The system checks its own health every 5 minutes (six checks: memory store
integrity, git state, main-model reachability, trace store size, Pareto
front freshness, backup freshness), collects metrics, fires alerts and can
auto-repair common issues.

```
/diagnostics quick             # just "overall status: healthy|degraded|unhealthy"
/diagnostics                   # full report: checks + metrics + alerts
/diagnostics repair            # auto-repair everything repairable
/diagnostics repair dirty_git  # one specific repair (dirty_git,
                               #   missing_trust_cards, trace_store_full,
                               #   stale_pareto, stale_backup)
```

Dashboards (plan P14.T10) show the same data, friendlier:

- **CLI dashboard** — `render_dashboard_text` / the interactive pager from
  `hermes.diagnostics.dashboard_cli` (`python -m
  hermes.diagnostics.dashboard_cli`): sections HEALTH / METRICS / TRENDS /
  ALERTS / RECENT ACTIVITY, arrow keys (left/right) switch pages, `q` quits.
- **HTTP dashboard** — `hermes.diagnostics.dashboard_http.serve_dashboard(...)`
  serves the HTML page (tables + inline-SVG trend sparklines, works fully
  offline) on `GET /`, plus the JSON routes:
  - `GET /health` — the full health report JSON, refreshed per request;
  - `GET /metrics` — latest metrics snapshot + model-metrics summary.

Alerts (12 rules) are appended to `~/.hermes/diagnostics/alerts.log` and can
be fanned out to a gateway message or email by the wiring. Auto-repair
attempts land in `diagnostics/repairs.log`.

## Hot memory & injection (what appears in context)

Each turn, before the model call, the pipeline retrieves facts relevant to
your prompt (BM25 over the active context + global) and injects the useful
ones. Budgets (all configurable, see Configuration):

- only the **top-3** retrieved candidates are gated by the CBDC gate
  (`budget_claims`);
- at most **1200 characters** are injected per turn (`budget_chars`);
- the gate classifies each candidate `use` / `verify` / `ignore` in ONE
  batched model call with a **3.5s timeout** — on timeout it falls back to
  trust-only injection (only facts scoring above 0.7).

The **stable block** (global + context `MEMORY.md`) is loaded once per
session and then replayed byte-identically, so prompt prefix caches stay
warm. If the gate injects nothing, nothing is added — your prompt is never
polluted. On long gateway chats, virtual session boundaries (topic shift
detection, checked every N messages) scope extraction without ending your
chat, and mid-session memory updates become visible via segmented prompt
invalidation (only changed segments are refreshed).

## Archives (decay & recover)

Facts age. The daily decay job applies the forgetting curve; an entry is
archived (moved to `memory-repo/archive/` in one Git commit, stamped
`archived_at`) when it is BOTH low-trust (score below 0.2) AND unverified
for more than a year. Archived facts:

- are excluded from `/memory search` and injection by default;
- are searchable again via
  `MemoryStore.search(query, include_archived=True)` (programmatic);
- are recoverable: `hermes.memory.decay.recover(entry_id, store)` moves the
  file back to its original facts directory.

## Backup & restore

`hermes.backup.manager.BackupManager` archives the whole memory repository
(including `.git`) daily to `~/.hermes/backups/YYYY-MM-DD.tar.gz` — same-day
re-runs refresh in place; retention prunes old archives. Restore is
`hermes.backup.recovery.Recovery.restore(archive)`: it keeps a safety copy
of the current (possibly broken) repo as `<repo>-pre-restore` first, then
replaces the live tree. Git corruption is detected by the opt-in
`git_corruption` health check (`hermes/backup/git_recovery.py`).

## Multi-agent (writer + crash recovery)

Multiple agent processes can share one memory repository safely: every
mutation serializes on the store's `FileLock`, and the single-writer daemon
(`hermes/concurrency/writer.py`) applies queued writes in batches. If the
writer crashes, the client-side journal
(`~/.hermes/journal/writer_journal.jsonl`) has every journaled operation and
`hermes.concurrency.recovery.recover(...)` replays the un-acked ones to a
fresh writer (bounded to 5s) — no write is lost. One practical constraint:
all writes flow through ONE store handle (the writer's) because libgit2
keeps a single in-memory index per repository handle.

## Configuration (`config.yaml`)

v2 reads the `v2:` section of `~/.hermes/config.yaml` (the fork-base
sections stay valid). Missing file or key -> default. Invalid values raise
a `ConfigError` at load; unknown keys are ignored (forward compatible).

```yaml
v2:
  memory_pipeline:
    max_drafts_per_turn: 5        # extraction cap per turn
    async_after_stop: true        # pipeline runs on a daemon thread
    gating:
      budget_claims: 3            # max facts injected per turn
      budget_chars: 1200          # max injected characters per turn
      bm25_prefilter_top_k: 20    # candidates retrieved before gating
      mode: batched               # gate call shape (only "batched" today)
      timeout_seconds: 3.5        # gate budget -> trust-only fallback
    retry:
      max_retries: 3              # main-model call retries
      backoff_base_seconds: 2.0   # exponential backoff base
      backoff_max_seconds: 60.0   # backoff ceiling
      on_persistent_failure: skip # skip | fail — plan 5.5 degradation
```

Config updates are zero-loss: the migration layer
(`hermes/install/migrations.py`) backs up `config.yaml`
(`config.yaml.bak-<timestamp>`) before any change and merges so your values
win (`config_schema_version` drift prompts you).
