---
name: memory
description: Introspect and transfer the memory graph — list, search, stats, export, import.
---

# /memory

Memory introspection and export/import (plan P15.T3). Implemented by
`hermes.core.agent.handle_memory_command` (dispatch registry
`MEMORY_COMMANDS`) as thin orchestration over
`hermes.memory.store.MemoryStore` (plan P1.T3) — entries are trust-card
markdown files under `memory-repo/global/facts/` and
`memory-repo/contexts/<name>/facts/` (plan Phase 1). Spec doc for the plan
task `P15.T3` (`docs/plan/hermes-v2-improvement-plan-r7.md`, Phase 15).

Usage: `/memory <subcommand> [args]`

## Subcommands

### list [context]

Formatted table of entries — id, type, context, trust score and a text
snippet. Without `context`, all contexts are shown; with `context`, only
that one (contexts never bleed into each other).

```
/memory list
/memory list work
```

### search <query>

BM25 keyword search (plan P1.T4), best first. Only entries sharing at least
one query token are returned.

```
/memory search pnpm
```

### stats

Counts by entry type and by context plus the trust distribution
(average, minimum, and how many entries score below 0.3 — the
attention-hungry tail).

```
/memory stats
```

Example output:

```
total: 7
by type: Project=3, User=4
by context: global=5, work=2
trust: avg=0.81 min=0.22 below_0.3=1
```

### export [out_path]

Dump EVERY entry to one JSON file — each entry carries its complete
markdown file content (trust-card frontmatter + text body), so the export
preserves ids, trust cards and the full verification history. Without
`out_path` the file lands under the Hermes home
(`memory-export-<timestamp>.json`); the reply names the written path.

```
/memory export backups/memory-2026.json
```

### import <path> [--skip|--overwrite]

Import an export file. For each entry:

- id unknown -> the entry file is written exactly as exported (full trust
  card, all verification events) and the store index is regenerated with
  one batch Git commit;
- id already exists -> conflict: `--skip` (default) keeps the original,
  `--overwrite` replaces it (also across contexts). The optional
  `ConflictDetector` (plan P8.T1) flags near-duplicate/contradicting adds in
  the summary (`conflicts_flagged=N`) without blocking them — the policy
  governs, the detector informs.

The summary line reports `added / skipped / overwritten /
conflicts_flagged / errors`.

```
/memory import backups/memory-2026.json
/memory import backups/memory-2026.json --overwrite
```

## Notes

- Expected failures (unknown file, malformed export) come back as
  `error: ...` reply strings — the command never crashes the session.
- The merged view (context facts + inherited global facts) is available
  via `/context inspect` and
  `hermes.core.agent.list_effective_entries(store, context)`.
