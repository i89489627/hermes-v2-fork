---
name: diagnostics
description: Self-diagnostics — health report, quick status, auto-repair; plus HTTP /health and /metrics endpoints when running as gateway.
---

# /diagnostics

Self-diagnostics command (plan P14.T6,
`docs/plan/hermes-v2-improvement-plan-r7.md`, Phase 14). Thin orchestration
over `hermes.diagnostics.system.DiagnosticsSystem` (plan P14.T5): runs the
six health checks (P14.T1), aggregates the metrics snapshot (P14.T2),
evaluates the alert rules (P14.T3) and can trigger auto-repair (P14.T4).

Usage: `/diagnostics [quick|full|repair]`

Bare `/diagnostics` is the full report (plan implementation step 5: "Add
`/diagnostics` slash command (full report)").

## Subcommands

### (bare) / full

Render the full plan 5.1 health report as text: overall status, every check
with its status/details, the metrics block, and any fired alerts. Report
generation completes in <2s (plan acceptance).

```
/diagnostics
/diagnostics full
```

Example output:

```
diagnostics report @ 2026-11-18T14:32:00+00:00
overall status: degraded
checks:
  [pass] memory_store_integrity: 247 entries, all have trust cards
  [warn] git_state: working tree dirty for 2h 15min (auto-repair: auto-commit pending changes)
  [pass] main_model_reachable: probe ok
  [pass] trace_store_size: 1.2 GB / 10.0 GB
  [pass] pareto_front_freshness: pareto.json age 3h
  [pass] backup_freshness: last backup 3h ago
metrics:
  memory_ops_latency_p50_ms=145.0
  memory_ops_latency_p95_ms=380.0
  token_usage_today=12450
  trust_score_avg=0.72
  trust_score_min=0.15
  conflict_rate_7d=0.03
  retrieval_recall_at_5=0.81
  skills_active=18
  skills_pending_approval=2
  gepa_last_run=2026-11-17T03:00:00+00:00
alerts:
  [warning] git_dirty_long: working tree dirty since 2026-11-18T12:17:00+00:00 (>1h)
```

### quick

Just the overall status one-liner (plan implementation step 6: "Add
`/diagnostics quick` (just overall status)").

```
/diagnostics quick
```

Example output:

```
overall status: degraded
```

### repair

Trigger auto-repair (plan Phase 14: "Auto-repair for common issues"). Two
forms:

- bare `/diagnostics repair` — runs the health check with
  `attempt_auto_repair=True`: every non-passing check that maps to a plan 5.4
  issue (memory store, git state, trace store, Pareto freshness, backup
  freshness) is repaired via the P14.T4 backend and its result is reported.
  `main_model_reachable` has no auto-repair (the plan says alert the user,
  skip ops).
- `/diagnostics repair <issue>` — runs one plan 5.4 repair directly, where
  `<issue>` is one of `dirty_git`, `missing_trust_cards`, `trace_store_full`,
  `stale_pareto`, `stale_backup` (plus the P18.T3 `git_corruption` escalation
  when wired). Unknown issue keys reply with the known list; expected
  failures come back as reply text — the command never crashes the session.

```
/diagnostics repair
/diagnostics repair dirty_git
```

Example output:

```
diagnostics repair run @ 2026-11-18T14:33:00+00:00
overall status: degraded
repairs:
  git_state: committed 3 pending path(s) in a1b2c3d4e5f6
  backup_freshness: failed: backup trigger not configured
```

## HTTP endpoints (gateway)

When running as gateway, the same diagnostics system is served over HTTP
(plan Phase 14: "HTTP endpoint (if running as gateway): `/health`,
`/metrics`"), implemented by `hermes.gateway.health`:

- `GET /health` — the full plan 5.1 health report JSON
  (`HealthReport.to_json()`), refreshed per request;
- `GET /metrics` — the latest `MetricsSnapshot` from the JSONL time series
  (fresh collection when none persisted) plus the P14.T7 model-metrics
  summary (`model_metrics` is `null` when no collector is wired).

`hermes.gateway.health.serve_health(diagnostics, host="127.0.0.1", port=0)`
runs both routes on a threaded `http.server` in a daemon thread (ephemeral
port by default) and `stop_health_server(address)` shuts it down; unknown
paths answer `404`, crashing reports degrade to `500` JSON errors.

## Notes

- Deviation from the P13.T4 pattern, documented per plan P14.T6: the P13.T4
  slash-command handlers live in `hermes.core.agent` (dispatch registry) —
  but `hermes/core/agent.py` is frozen this wave, so
  `handle_diagnostics_command(args, diagnostics)` lives in
  `hermes.gateway.health` until the wiring wave registers it on the agent.
  The signature and behaviour above are the registration contract.
- Every subcommand is read-only except `repair`; repair attempts are logged
  to `<hermes_home>/diagnostics/repairs.log` by the P14.T4 backend.
- A missing `AutoRepair` backend reports `failed: no AutoRepair backend
  configured` for the affected checks instead of silently doing nothing.
