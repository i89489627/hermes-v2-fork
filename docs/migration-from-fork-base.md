# Migrating from the fork-base memory (v2026.9.24) to hermes-v2

Step-by-step migration guide from the fork-base memory layout (v2026.9.24:
plain markdown under `~/.hermes/memories/` plus a single `SOUL.md`) to the
hermes-v2 git-backed trust-card memory repository. Every command below is the
real, implemented interface — there are no placeholders. (plan P16.T2)

## 1. Overview — what changes

The migration converts two things and leaves the originals untouched:

**Memory directory → memory repository.** The fork-base layout is plain
markdown without metadata:

| Fork-base file | Becomes in v2 |
|---|---|
| `~/.hermes/memories/MEMORY.md` | One entry per non-empty line/bullet, type `Feedback` (list markers `-`, `*`, `+`, `1.` are stripped as structure, the text is kept verbatim) |
| `~/.hermes/memories/USER.md` | One entry with the whole file as body, type `User` |
| any other `*.md` in that directory | One entry per file (body byte-exact), type `Reference` |

The result is the v2 memory repository (default `~/.hermes/memory-repo/`):
one markdown file per entry at `<repo>/global/facts/<id>.md`, each with a
trust-card YAML frontmatter (id, type, context, trust score, verification
history), plus a regenerated `MEMORY.md` index — all in one Git commit, so
the whole migration is a single, inspectable commit in the repository
history. Per plan P5.T4 every migrated entry lands in the explicit `global`
context (`context: global` in the frontmatter).

**SOUL.md → soul split.** The fork-base `SOUL.md` is split section-wise into
the v2 soul layout under `<repo>/soul/`:

- `soul_core.md` — safety, alignment, identity and NEVER/ALWAYS guardrail
  blocks. IMMUTABLE from then on (the agent has no API to write it).
- `soul_style.md` — voice/tone/formatting blocks. Agent-writable, bounded to
  2000 characters.

The classification is purely rule-based (keyword lists, first match wins, no
LLM), so it is deterministic and reproducible.

## 2. Pre-migration checklist

1. **Stop Hermes processes.** Close running gateways/agents that could write
   to `~/.hermes/memories/` while the migration runs. (The v2 memory-repo
   writers coordinate via file locks; the fork-base directory has none.)
2. **Full-home backup.** The migration creates its own pre-migration backup
   of the source directory automatically (see below) — but take a general
   home backup as well so configurations and sessions are covered:

   ```
   hermes backup          # zip of the Hermes home (config, skills, sessions, data)
   ```

3. **Memory-repo backup (if one already exists).** If `~/.hermes/memory-repo/`
   exists from earlier v2 use, snapshot it first — `BackupManager().create_backup()`
   writes `<hermes-home>/backups/YYYY-MM-DD.tar.gz` (one archive per day,
   `git fsck`-verified before archiving, 30-day retention).

4. **Disk space.** The migration copies the source directory (backup) and
   writes the target repository plus a Git object per migrated entry. Rule
   of thumb: free space ≈ 3× the size of `~/.hermes/memories/`.
5. **Know your paths.** Default source is `~/.hermes/memories` (on Windows:
   `%LOCALAPPDATA%\hermes\memories` or your configured `HERMES_HOME`),
   default target is `~/.hermes/memory-repo`. Both are explicit CLI
   arguments below — nothing is guessed.

## 3. Step-by-step migration

### Step 1 — dry run (always do this first)

```
python scripts/migrate_fork_to_v2_0.py ~/.hermes/memories ~/.hermes/memory-repo --dry-run
```

A dry run produces the full report and writes NOTHING to the target. The
pre-migration backup of the source is created even on a dry run and kept.

Sample output:

```
Fork-base memory migration to v2.0 (plan P1.T6)
  source:           ~/.hermes/memories
  target:           ~/.hermes/memory-repo
  dry run:          yes
  entries found:    87
  entries migrated: 0
  skipped:          2
  backup:           ~/.hermes/memories-premigration-backup
errors:
  - corrupted-notes.md: unreadable, skipped ([Errno 13] Permission denied)
```

`entries found` is the number of candidate entries discovered; `skipped`
counts unreadable files and empty lines (each with a message under
`errors:`). **Read the errors list before proceeding** — unreadable files
are skipped, not fatal, but you may want to fix them and re-run so their
content migrates.

### Step 2 — real run

```
python scripts/migrate_fork_to_v2_0.py ~/.hermes/memories ~/.hermes/memory-repo
```

Same report with `dry run: no` and `entries migrated: 87`. The entries are
written into `<repo>/global/facts/`, the `MEMORY.md` index is regenerated,
and everything lands in one Git commit. The source directory is left
untouched on success (it stays as historical data; Hermes v2 no longer reads
it).

Exit codes: `0` on a clean run, `1` when the report carries errors
(e.g. unreadable source files) or the migration failed outright.

### Step 3 — verify the global-context contract

The post-migration verifier is read-only and safe to run any time:

```python
from pathlib import Path
from hermes.memory.migrations.fork_to_v2_0 import verify_global_context

problems = verify_global_context(Path.home() / ".hermes" / "memory-repo")
print(problems)   # [] means the repository satisfies the contract
```

It reports entries stored outside the global context, trust cards whose
frontmatter does not say `context: global`, and unreadable cards.

### Step 4 — split SOUL.md

Dry-run the split first (`apply=False` reports the classification and
character counts without writing anything), then apply:

```python
from pathlib import Path
from hermes.memory.migrations.fork_to_v2_0 import migrate_soul

source = Path("~/.hermes/SOUL.md").expanduser()
target = Path("~/.hermes/memory-repo")

plan = migrate_soul(source, target, apply=False)
print(plan.core_chars, plan.style_chars)
for rule in plan.rules_applied:          # audit trail: "<rule> <- <section>"
    print(rule)

result = migrate_soul(source, target, apply=True)
print(result.core_path, result.style_path)
```

This writes `soul/soul_core.md` (immutable from now on) and
`soul/soul_style.md`, and commits both when the target is Git-backed.

### Step 5 — post-migration checks

- `git -C ~/.hermes/memory-repo log --oneline` shows the migration commit(s);
  `git -C ~/.hermes/memory-repo status` is clean (everything committed).
- Entries are readable through the store:

  ```python
  from hermes.memory.store import MemoryStore
  print(len(MemoryStore().list()))        # migrated entry count
  ```

- In a running Hermes session, `/memory stats` and `/diagnostics` (see the
  user guide) summarize the memory store and its health; run them after
  restart so the new repository is picked up.
- Restart Hermes now (it was stopped in step 2 of the checklist).

## 4. Backup & rollback

### Where the pre-migration backup lives

The migration ALWAYS creates `<source>-premigration-backup` (a byte-exact
copy of the source directory) BEFORE writing anything, and keeps it. If that
name already exists from an earlier attempt, a timestamped sibling
(`<source>-premigration-backup-YYYYMMDDTHHMMSS`) is created instead — earlier
backups are never clobbered. The report's `backup:` line shows the exact
path used.

### Rolling back a memory migration

The rollback removes the target repository and restores the source directory
byte-exact from the backup (a failed or partial run is fully repaired):

```
python scripts/migrate_fork_to_v2_0.py ~/.hermes/memory-repo --rollback ~/.hermes/memories-premigration-backup
```

Note the argument order: TARGET first (the v2 repo to remove), then
`--rollback BACKUP` (the backup directory to restore from). Exit code `0` on
success, `1` on failure (missing backup, malformed backup name).

### Restoring the memory repository from an archive (v2 backups)

For corruption or accidents AFTER a successful migration, restore the
memory repository from a `BackupManager` archive:

```
python scripts/hermes-restore.py ~/.hermes/backups/2026-11-21.tar.gz
```

Options: `--hermes-root ROOT` (when the Hermes home is not the default) and
`--no-safety-backup` (skip the pre-restore safety copy — not recommended).
The restore keeps a safety copy of the current repository as a `-pre-restore`
sibling, wipes the live repository, extracts the archive (entry files AND the
Git history), and verifies the result (entry count > 0, non-empty commit
log). Stop Hermes before running it and restart afterwards — the tool cannot
signal live processes.

## 5. Troubleshooting

**"unreadable, skipped" in the errors list.** A source file could not be
read (permissions, encoding). The migration skips it and continues; the exit
code is `1` so scripts notice. Fix the file and re-run the migration —
already-migrated entries get suffixed ids (see below) rather than
duplicates.

**Duplicate ids get `-2`/`-3` suffixes.** Entry ids come from the content
slug; duplicate bullets (or slugs already present in the target repo) get a
numeric suffix (`my-note`, `my-note-2`, ...). Content is never dropped on
collision.

**Trust scores are all 0.5 — is that a bug?** No. Migrated entries carry no
verification evidence, so each gets the documented migration default: a
single `agent-inferred` verification event with value `0.5`, whose recorded
history re-derives exactly 0.5 under the weighted-average rule (instead of
drifting to 1.0 on the next recomputation). Scores rise as entries get
verified in normal v2 use.

**Partial / failed migration.** The applied run writes files first and
commits once at the end; if the commit fails, the written files are removed
before the error propagates. If anything else went wrong mid-way, roll back
(Section 4) — the target repository is removed and the source is restored
byte-exact from the pre-migration backup, then re-run the migration after
fixing the cause.

**`refusing to overwrite the immutable soul_core.md` (FileExistsError).**
`soul_core.md` already exists in the target — the core never changes, so a
re-run is refused (and the existing style file is not touched either). Roll
back or remove the repository first if you really want a fresh split.

**Windows: "PermissionError" when deleting a memory repository.** libgit2
marks packed objects under `.git/objects` read-only, so a plain `rmtree` of a
v2 repository fails. The rollback and the restore workflow both use a
force-rmtree that clears the read-only bit first — no manual `attrib`
hunting needed. For manual cleanups outside those tools, clear the bit on
`.git` contents before deleting.

**Unreadable trust card reported by `verify_global_context`.** An entry file
under `global/facts/` lost its frontmatter (manual edit). Restore it from a
memory-repo backup (`hermes-restore.py`, Section 4) or repair the
frontmatter by hand; the verifier never mutates anything.

**Rollback failed: "not a pre-migration backup".** The `--rollback` path
must be the backup directory the migration created (name ending in
`-premigration-backup`) — the source path is derived from that name.
Point it at the path printed in the migration report's `backup:` line.
