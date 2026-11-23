---
name: context
description: Manage memory contexts — list, switch, create, inspect, delete.
---

# /context

Project/context layer management (plan P5.T2). Contexts scope the memory
graph: the implicit `global` root plus named contexts under
`memory-repo/contexts/<name>/facts/` (plan Phase 5). Facts are isolated per
context and every context inherits the global facts up its parent chain
unless overridden. Implemented by `hermes.core.agent.handle_context_command`
(dispatch registry `CONTEXT_COMMANDS`) as thin orchestration over
`hermes.memory.context.ContextManager` (plan P5.T1). Spec doc for the plan
task `P5.T2` (`docs/plan/hermes-v2-improvement-plan-r7.md`, Phase 5).

Usage: `/context <subcommand> [args]`

## Subcommands

### list

Show every context with its parent, active session count and fact count.

```
/context list
```

Example output:

```
contexts:
  global   parent=-         sessions=0 (none)  facts=12
  work     parent=global    sessions=1 (sess-9)  facts=3
```

### switch <name>

Switch the session to `<name>` (explicit detection, the highest-precedence
rule in plan P5.T1). The stable context (`global/MEMORY.md` + the context's
`MEMORY.md`) is reloaded at this session boundary — nothing is cached across
the switch. Unknown names return an error (typos must not be silently
ignored).

```
/context switch work
```

### create <name> [--parent <p>]

Create a named context. `<name>` must be a lowercase slug
(`[a-z0-9][a-z0-9-]*`); `--parent` names the context whose facts are
inherited (default: inherit global directly).

```
/context create work
/context create project-x --parent work
```

### inspect [name]

Details for one context — parent, backing path, active sessions, fact
count. Defaults to `global`.

```
/context inspect work
```

### delete <name> [--force]

Remove a context: its facts directory and its `contexts.json` metadata entry
are deleted (the facts are gone; global memory is untouched). Refuses the
implicit `global` root and any context with attached active sessions unless
`--force` is given. Children whose `parent` pointed at the deleted context
fall back to direct global inheritance.

```
/context delete work
/context delete work --force
```

## Notes

- Expected failures (unknown context, bad slug, guarded delete) come back
  as `error: ...` reply strings — the command never crashes the session.
- The merged fact view (context's own facts + inherited global facts,
  nearest copy wins) is available programmatically via
  `hermes.core.agent.list_effective_entries(store, context, context_manager)`.
