---
name: skills
description: Inspect and govern the skill lifecycle — list, inspect, approve, deprecate, rollback.
---

# /skills

Governed skill management (plan P13.T4). Every skill moves through the
lifecycle `proposed -> approved -> active -> deprecated -> archived` under
user control: the agent *proposes* skills autonomously (plan P13.T1), and the
user approves, retires, or reverts them here. Implemented by
`hermes.core.agent.handle_skills_command` (dispatch registry `SKILL_COMMANDS`)
as thin orchestration over `hermes.skills.lifecycle.SkillLifecycle` +
`hermes.skills.registry.SkillRegistry` — the lifecycle state machine is the
single source of truth for transitions.

Usage: `/skills <subcommand> [args]`

`list` and `inspect` surface `last_used_at` from the session's usage ledger
(`memory-repo/skills/skills_usage.json`, plan P13.T3; auto-deprecation
proposals from the same module are confirmed indirectly — a `deprecate` of a
skill proposed by `AutoDeprecator` is the user's approval of that proposal).

## Subcommands

### list

Show every registered skill with its state, version and `last_used_at`.

```
/skills list
```

Example output:

```
skills:
  draft-helper     proposed   v1  last_used_at=never
  greeter-skill    active     v2  last_used_at=2026-11-20T09:15:00+00:00
  old-skill        deprecated v1  last_used_at=never
```

### inspect <name>

Detailed view of one skill: state, version, version history, rationale,
`last_used_at`, and the deprecation reason when retired. Accepts the skill id
or the display name.

```
/skills inspect greeter-skill
/skills inspect "Greeter Skill"
```

### approve <id>

User approval of a proposed skill: `proposed -> active` (an intermediate
`approved` skill is accepted too). The skill becomes usable by the agent.

```
/skills approve greeter-skill
```

### deprecate <name> [reason]

Retire an active skill: `active -> deprecated`, recording the reason
(default: "deprecated via /skills"). This is also how an
`AutoDeprecator` proposal (plan P13.T3) is approved by the user.

```
/skills deprecate greeter-skill superseded by the new greeter
```

### rollback <name> <version>

Restore a prior version's content as a NEW latest version (non-destructive:
history is never rewritten, so the rollback itself is reversible). The
lifecycle record is synced to the rolled-back version. Requires the skill to
be in `approved` or `active` state.

```
/skills rollback greeter-skill 1
```

Example output:

```
rolled back greeter-skill to version 1: content restored as version 3
```

## Notes

- Cache-aware by design: lifecycle changes take effect on the next session;
  no mid-conversation system-prompt invalidation (upstream invariant).
- Expected failures (unknown skill, bad version, illegal transition) come
  back as `error: ...` replies, never as harness crashes.
- Related: auto-deprecation proposals (plan P13.T3) land in
  `memory-repo/skills/pending_deprecations.json`; the learning dashboard
  (`/learning pending`, plan P13.T5) lists them for review.
