---
name: soul
description: Inspect the immutable soul core and manage the bounded adaptive style.
---

# /soul

SOUL.md split surface (plan P15.T4). The soul lives in two files under
`memory-repo/soul/` (plan P15.T1): `soul_core.md` — safety, alignment and
identity, IMMUTABLE (the agent has no API that writes it; changes require an
explicit manual Git operation by the user) — and `soul_style.md`, the
adaptive communication style, hard-bounded to 2000 characters. Implemented
by `hermes.core.agent.handle_soul_command` (dispatch registry
`SOUL_COMMANDS`) as thin orchestration over
`hermes.memory.soul.SoulManager`. Spec doc for the plan task `P15.T4`
(`docs/plan/hermes-v2-improvement-plan-r7.md`, Phase 15).

Usage: `/soul <subcommand> [args]`

## Subcommands

### inspect

Show both layers with char counts: the immutable core and the adaptive
style (`<n>/2000 chars`).

```
/soul inspect
```

Example output:

```
soul core (immutable, 842 chars):
# Soul — Core (immutable)
...
soul style (18/2000 chars):
Prefer concise replies.
```

### edit-style <text|->

Replace the adaptive style (manual override). `-` reads the new text from
stdin. An update beyond the 2000-char limit is refused with an error and
the style is left untouched — nothing is written or logged in that case.
Every accepted update is atomic, appended to `.style_history.log` and, in a
Git-backed memory repo, committed (`[P15.T1] update soul style (n chars)`).

```
/soul edit-style Prefer concise replies. Use bullet lists for multi-step plans.
/soul edit-style -
```

### history

The style change history. In a Git-backed memory repo: the git log of
`soul/soul_style.md` (sha, timestamp, commit message — one line per style
update, newest first). Without Git: the `.style_history.log` content
(`<timestamp>  <char count> chars` per line).

```
/soul history
```

## Notes

- The core is never editable through this command — by design there is no
  `update_core` API (plan P15.T1 acceptance: "SOUL core never changes
  without explicit user override"). Hooks can detect out-of-band core
  edits via `SoulManager.verify_core_integrity(expected_hash)`.
- Expected failures come back as `error: ...` reply strings — the command
  never crashes the session.
