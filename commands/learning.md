---
name: learning
description: Learning-loop dashboard — skill stats, recent traces, pending approvals, GEPA history.
---

# /learning

Learning-loop visibility dashboard (plan P13.T5). Read-only orchestration
over the v2 learning components: `hermes.skills.lifecycle.SkillLifecycle`
(state machine, P13.T1), `hermes.skills.registry.SkillRegistry` (version
store, P13.T2), `hermes.skills.auto_deprecator.AutoDeprecator` (P13.T3),
`hermes.traces.store.TraceStore` (P11.T1) and
`hermes.optimization.pareto.ParetoFrontStore` (P10.T4). Implemented by
`hermes.core.agent.handle_learning_command` (dispatch registry
`LEARNING_COMMANDS`). Spec doc for the plan task `P13.T5`
(`docs/plan/hermes-v2-improvement-plan-r7.md`, Phase 13).

Usage: `/learning <subcommand>`

## Subcommands

### stats

Skill totals per lifecycle state, pending approvals and trace volume.

```
/learning stats
```

Example output:

```
learning dashboard:
  skills: total=5 active=2 deprecated=1 (proposed=2 approved=0 archived=0)
  registry: 5 skill record(s)
  pending approvals: 3 (2 proposed skill(s), 1 auto-deprecation proposal(s))
  traces: 42
```

### traces

The recent 5 traces, newest first, with their quality labels
(`good` / `acceptable` / `bad` / `needs-review` / `unlabeled` — labels are
curated for the GEPA training set, plan P11/P12).

```
/learning traces
```

### pending

Skills awaiting user approval: proposed-state skills (approve via
`/skills approve <id>`) plus the auto-deprecator's pending proposals
(retire via `/skills deprecate <name>`), read from
`pending_deprecations.json` when present.

```
/learning pending
```

### history

GEPA run history. Until the P10.T5 optimization runner exists this is the
documented proxy: per-component Pareto front sizes with the best mean
score (plan P10.T4 store).

```
/learning history
```

Example output:

```
GEPA run history (pareto front sizes as proxy until the P10.T5 runner exists):
  extraction: front=2 best_mean=0.82
  gating: front=1 best_mean=0.91
```

## Notes

- Every component is optional; an unwired component renders a placeholder
  line (`no trace store wired ...`) instead of failing — the dashboard is
  advisory and never crashes the session.
- Trace counts and the recent list are read from the TraceStore JSONL
  source of truth (`<root>/v1.0/<day>.jsonl`), de-duplicated by trace id.
