"""hermes-v2 DiagnosticsSystem orchestrator (plan P14.T5, plan Phase 14 + Section 5.1).

One object that wires the Phase 14 pieces together and produces the plan 5.1
health report:

- :meth:`DiagnosticsSystem.run_health_check` runs the six P14.T1 health
  checks (:func:`~hermes.diagnostics.health_checks.run_all_checks`),
  aggregates the metrics snapshot (:meth:`collect_metrics` -> plan 5.1
  ``metrics`` block), evaluates the alert engine (plan 5.3 rules) and
  derives ``overall_status`` — any ``fail`` check -> ``unhealthy``, else any
  ``warn`` -> ``degraded``, else ``healthy``. The returned
  :class:`HealthReport` carries exactly the plan 5.1 JSON schema via
  :meth:`HealthReport.to_dict` / :meth:`HealthReport.to_json`:
  ``timestamp`` / ``overall_status`` / ``checks[]`` (``name`` ``status``
  ``details`` ``auto_repair_attempted`` ``auto_repair_result``) /
  ``metrics`` / ``alerts``. With the fast injected fakes the plan's
  acceptance budget (report generation < 2 s) holds; generation over the
  budget is logged as a warning.
- :meth:`DiagnosticsSystem.auto_repair` maps a plan 5.4 issue key to the
  P14.T4 :class:`~hermes.diagnostics.auto_repair.AutoRepair` method, and
  ``run_health_check(attempt_auto_repair=True)`` additionally repairs every
  non-passing check that has a plan 5.1 auto-repair column entry, filling
  the check's ``auto_repair_attempted`` / ``auto_repair_result`` fields
  (``main_model_reachable`` has none — the plan says "alert user, skip ops").
- :meth:`DiagnosticsSystem.start_background` /
  :meth:`DiagnosticsSystem.stop` run metrics collection on a background
  thread at :data:`DEFAULT_COLLECTION_INTERVAL_MINUTES` (plan acceptance:
  every 5 minutes; ``interval_minutes`` is injectable so tests scale the
  cadence). Each tick appends a :class:`~hermes.diagnostics.metrics.
  MetricsSnapshot` to the collector's JSONL time series; a tick that raises
  is logged and skipped — the collector thread never dies.
- when no :class:`~hermes.diagnostics.alerts.AlertEngine` is injected, the
  system builds the default wiring the alerts module anticipates ("the
  zero-argument callables the engine evaluates are built by currying those
  functions at the wiring site, e.g. the P14.T5 DiagnosticsSystem"):
  ``trace_store_full`` and ``backup_stale`` from the context paths,
  ``fallback_rate_high`` from the optional model-metrics collector, and the
  two stateful rules the orchestrator tracks across reports —
  ``main_model_down`` (consecutive failed reachability checks) and
  ``git_dirty_long`` (first dirty observation).

The ``metrics`` block keys follow plan Section 5.1 exactly:
``memory_ops_latency_p50_ms`` / ``memory_ops_latency_p95_ms`` (percentiles
of the ``"memory_ops"`` latency op — the op name the pipeline records memory
operations under, see :data:`MEMORY_OPS_OP`; 0.0 when nothing recorded),
``token_usage_today`` (token totals minus the last pre-today history
baseline — the collector's totals are cumulative, so the daily figure needs
the day boundary), ``trust_score_avg`` / ``trust_score_min``,
``conflict_rate_7d``, ``retrieval_recall_at_5`` (``retrieval_hits_at_5`` /
``retrieval_queries`` snapshot counters; 0.0 when no queries),
``skills_active`` / ``skills_pending_approval`` (same counter contract) and
``gepa_last_run`` (Pareto front mtime — it is rewritten on every GEPA run;
``None`` when no run has completed).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from hermes.diagnostics.alerts import (
    Alert,
    AlertEngine,
    rule_backup_stale,
    rule_fallback_rate_high,
    rule_git_dirty_long,
    rule_main_model_down,
    rule_trace_store_full,
)
from hermes.diagnostics.auto_repair import AutoRepair, RepairResult
from hermes.diagnostics.health_checks import Check, HealthCheckContext, run_all_checks
from hermes.diagnostics.metrics import MetricsCollector, MetricsSnapshot
from hermes.diagnostics.model_metrics import ModelMetricsCollector

__all__ = [
    "DEFAULT_COLLECTION_INTERVAL_MINUTES",
    "MEMORY_OPS_OP",
    "REPORT_BUDGET_SECONDS",
    "OverallStatus",
    "DiagnosticsSystem",
    "HealthReport",
]

_LOGGER = logging.getLogger("hermes.diagnostics.system")

#: The latency op name the pipeline records memory operations under; the
#: plan 5.1 ``memory_ops_latency_*`` metrics read this op's percentiles.
MEMORY_OPS_OP: Final[str] = "memory_ops"

#: Default background collection cadence (plan P14.T5 acceptance: 5 minutes).
DEFAULT_COLLECTION_INTERVAL_MINUTES: Final[float] = 5.0

#: Plan P14.T5 acceptance: report generation completes within 2 seconds.
REPORT_BUDGET_SECONDS: Final[float] = 2.0

#: plan 5.1 overall status vocabulary.
OverallStatus = Literal["healthy", "degraded", "unhealthy"]

#: Plan 5.1 auto-repair column: check name -> plan 5.4 issue key. Checks
#: absent from this mapping (``main_model_reachable``) have no auto-repair.
_REPAIR_BY_CHECK: Final[dict[str, str]] = {
    "memory_store_integrity": "missing_trust_cards",
    "git_state": "dirty_git",
    "trace_store_size": "trace_store_full",
    "pareto_front_freshness": "stale_pareto",
    "backup_freshness": "stale_backup",
}


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass
class HealthReport:
    """The plan 5.1 health report — schema-exact via :meth:`to_dict`."""

    #: Report generation moment (aware UTC).
    timestamp: datetime
    #: ``healthy`` / ``degraded`` / ``unhealthy`` (any fail -> unhealthy,
    #: else any warn -> degraded, else healthy).
    overall_status: OverallStatus
    #: The six (plus any registered extra) P14.T1 check results.
    checks: list[Check] = field(default_factory=list)
    #: The plan 5.1 metrics block (see module docstring for derivations).
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Alerts fired by the engine during this report (plan 5.3).
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The exact plan 5.1 JSON schema (JSON-safe values throughout)."""
        return {
            "timestamp": _as_utc(self.timestamp).isoformat(),
            "overall_status": self.overall_status,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "details": check.details,
                    "auto_repair_attempted": check.auto_repair_attempted,
                    "auto_repair_result": check.auto_repair_result,
                }
                for check in self.checks
            ],
            "metrics": dict(self.metrics),
            "alerts": [alert.to_dict() for alert in self.alerts],
        }

    def to_json(self) -> str:
        """``to_dict`` serialized (one JSON document, UTF-8 text)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def derive_overall_status(checks: list[Check]) -> OverallStatus:
    """Any ``fail`` -> ``unhealthy``; else any ``warn`` -> ``degraded``; else healthy."""
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "unhealthy"
    if "warn" in statuses:
        return "degraded"
    return "healthy"


def report_metrics(
    snapshot: MetricsSnapshot,
    history: list[MetricsSnapshot],
    *,
    pareto_path: Path | None = None,
) -> dict[str, Any]:
    """Build the plan 5.1 ``metrics`` block from a snapshot (+ its history).

    ``history`` (oldest first, the collector's JSONL time series) supplies
    the pre-today token baseline so ``token_usage_today`` is a daily figure
    rather than the collector's cumulative total. ``gepa_last_run`` comes
    from ``pareto_path``'s mtime (``None`` when the file is absent).
    """
    percentiles = snapshot.latency_percentiles.get(MEMORY_OPS_OP, {})
    trust_stats = snapshot.trust_stats
    counters = snapshot.counters
    retrieval_queries = counters.get("retrieval_queries", 0)
    retrieval_hits = counters.get("retrieval_hits_at_5", 0)
    latest_total = sum(snapshot.token_usage.values())
    day = _as_utc(snapshot.timestamp).date()
    baseline: MetricsSnapshot | None = None
    for past in history:
        if _as_utc(past.timestamp).date() < day:
            baseline = past  # keep the last one strictly before today
    if baseline is not None:
        token_usage_today = max(0, latest_total - sum(baseline.token_usage.values()))
    else:
        token_usage_today = latest_total
    gepa_last_run: str | None = None
    if pareto_path is not None and pareto_path.is_file():
        try:
            gepa_last_run = datetime.fromtimestamp(
                pareto_path.stat().st_mtime, tz=UTC
            ).isoformat()
        except OSError:
            gepa_last_run = None
    return {
        "memory_ops_latency_p50_ms": percentiles.get("p50", 0.0) * 1000.0,
        "memory_ops_latency_p95_ms": percentiles.get("p95", 0.0) * 1000.0,
        "token_usage_today": token_usage_today,
        "trust_score_avg": trust_stats.get("avg", 0.0),
        "trust_score_min": trust_stats.get("min", 0.0),
        "conflict_rate_7d": snapshot.conflict_rate_7d,
        "retrieval_recall_at_5": (
            retrieval_hits / retrieval_queries if retrieval_queries > 0 else 0.0
        ),
        "skills_active": counters.get("skills_active", 0),
        "skills_pending_approval": counters.get("skills_pending_approval", 0),
        "gepa_last_run": gepa_last_run,
    }


class DiagnosticsSystem:
    """Orchestrates health checks, metrics, alerts and auto-repair (P14.T5).

    Args:
        context: paths + probe for the health checks (default:
            :class:`~hermes.diagnostics.health_checks.HealthCheckContext`,
            i.e. ``hermes_home()`` defaults).
        collector: metrics collector backing :meth:`collect_metrics` and the
            background cadence (default: a fresh
            :class:`~hermes.diagnostics.metrics.MetricsCollector` on the
            default JSONL path).
        alert_engine: engine evaluated on every
            :meth:`run_health_check` / :meth:`check_alerts`. Default: the
            wiring built by :meth:`_default_alert_engine` (see module
            docstring).
        repair: optional P14.T4 auto-repair backend for
            :meth:`auto_repair` and ``run_health_check(attempt_auto_repair=
            True)``.
        model_metrics: optional P14.T7 collector; held for the default
            ``fallback_rate_high`` rule wiring and the dashboard (P14.T10).
    """

    def __init__(
        self,
        context: HealthCheckContext | None = None,
        collector: MetricsCollector | None = None,
        alert_engine: AlertEngine | None = None,
        repair: AutoRepair | None = None,
        model_metrics: ModelMetricsCollector | None = None,
    ) -> None:
        self.context = context if context is not None else HealthCheckContext()
        self.collector = collector if collector is not None else MetricsCollector()
        self.model_metrics = model_metrics  # read by _default_alert_engine() — must precede it
        self.alert_engine = (
            alert_engine if alert_engine is not None else self._default_alert_engine()
        )
        self.repair = repair
        # stateful alert wiring (see _default_alert_engine)
        self._consecutive_model_failures: int = 0
        self._git_dirty_since: datetime | None = None
        # background collection state
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._interval_seconds = DEFAULT_COLLECTION_INTERVAL_MINUTES * 60.0
        self._thread: threading.Thread | None = None
        self._collected: list[MetricsSnapshot] = []

    # ── plan Phase 14 component API ───────────────────────────────────────────

    def collect_metrics(self) -> MetricsSnapshot:
        """Aggregate the collector's current windows into one snapshot."""
        return self.collector.collect_snapshot()

    def check_alerts(self) -> list[Alert]:
        """Evaluate the alert engine and return the fired alerts (plan 5.3)."""
        fired: list[Alert] = self.alert_engine.evaluate()
        return fired

    def run_health_check(self, *, attempt_auto_repair: bool = False) -> HealthReport:
        """Produce the full plan 5.1 report; generation stays under 2 s.

        With ``attempt_auto_repair=True`` (and a ``repair`` backend
        configured) every non-passing check that maps to a plan 5.4 repair
        is repaired and its ``auto_repair_*`` fields filled in before the
        status is derived. Either way the report side effects are exactly:
        alert engine evaluation (which logs fired alerts to the alert log)
        and nothing else.
        """
        started = time.monotonic()
        checks = run_all_checks(self.context)
        if attempt_auto_repair and self.repair is not None:
            checks = [self._maybe_repair(check) for check in checks]
        self._update_alert_trackers(checks)
        metrics = report_metrics(
            self.collect_metrics(),
            self._safe_history(),
            pareto_path=self.context.pareto_path,
        )
        alerts = self.check_alerts()
        elapsed = time.monotonic() - started
        if elapsed >= REPORT_BUDGET_SECONDS:
            _LOGGER.warning(
                "health report generation took %.2fs (plan budget: %.1fs)",
                elapsed,
                REPORT_BUDGET_SECONDS,
            )
        return HealthReport(
            timestamp=datetime.now(UTC),
            overall_status=derive_overall_status(checks),
            checks=checks,
            metrics=metrics,
            alerts=alerts,
        )

    def auto_repair(self, issue: str) -> RepairResult:
        """Run the plan 5.4 repair for ``issue`` via the injected backend.

        ``issue`` is one of the plan 5.4 keys (``dirty_git``,
        ``missing_trust_cards``, ``trace_store_full``, ``stale_pareto``,
        ``stale_backup``). No backend configured -> a ``failed`` result
        (the same "not configured" convention the P14.T4 repairs use);
        an unknown issue raises ``ValueError``.
        """
        if self.repair is None:
            return RepairResult(
                issue=issue, action="none", status="failed",
                details="no AutoRepair backend configured",
            )
        methods = {
            "dirty_git": self.repair.repair_dirty_git,
            "missing_trust_cards": self.repair.repair_missing_trust_cards,
            "trace_store_full": self.repair.repair_trace_store_full,
            "stale_pareto": self.repair.repair_stale_pareto,
            "stale_backup": self.repair.repair_stale_backup,
        }
        method = methods.get(issue)
        if method is None:
            raise ValueError(f"unknown repair issue: {issue!r}; known: {sorted(methods)}")
        return method()

    # ── background collection (plan acceptance: every 5 minutes) ─────────────

    @property
    def collected(self) -> list[MetricsSnapshot]:
        """Snapshots taken by the background collector (oldest first)."""
        with self._lock:
            return list(self._collected)

    @property
    def is_collecting(self) -> bool:
        """True while the background collector thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start_background(
        self, interval_minutes: float = DEFAULT_COLLECTION_INTERVAL_MINUTES
    ) -> None:
        """Collect metrics every ``interval_minutes`` on a daemon thread.

        The first collection happens immediately, then one snapshot is
        appended to the collector's JSONL time series per interval. Raises
        ``RuntimeError`` when a collection thread is already running and
        ``ValueError`` for a non-positive interval.
        """
        if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int | float):
            raise TypeError(f"interval_minutes must be a number, got {interval_minutes!r}")
        if interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be > 0, got {interval_minutes!r}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("background collection is already running")
            self._stop_event.clear()
            self._interval_seconds = float(interval_minutes) * 60.0
            thread = threading.Thread(
                target=self._background_loop,
                name="hermes-diagnostics-collector",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background collector and join its thread (idempotent)."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    # ── internals ─────────────────────────────────────────────────────────────

    def _background_loop(self) -> None:
        """Collect immediately, then on every interval tick until stopped."""
        self._collect_once()
        while not self._stop_event.wait(self._interval_seconds):
            self._collect_once()

    def _collect_once(self) -> None:
        """One background tick: snapshot + JSONL append; never raises."""
        try:
            snapshot = self.collector.collect_snapshot()
            self.collector.append_snapshot(snapshot)
            with self._lock:
                self._collected.append(snapshot)
        except Exception:  # noqa: BLE001 - the collector thread must not die
            _LOGGER.exception("background metrics collection failed")

    def _safe_history(self) -> list[MetricsSnapshot]:
        """The collector's persisted snapshots; unreadable history is empty."""
        try:
            history: list[MetricsSnapshot] = self.collector.load_history()
            return history
        except Exception:  # noqa: BLE001 - metrics must not break the report
            _LOGGER.exception("metrics history could not be read")
            return []

    def _maybe_repair(self, check: Check) -> Check:
        """Repair a non-passing check per the plan 5.1 auto-repair column."""
        if check.status == "pass":
            return check
        issue = _REPAIR_BY_CHECK.get(check.name)
        if issue is None:
            return check  # main_model_reachable: plan says alert user, no repair
        try:
            result = self.auto_repair(issue)
        except Exception as exc:  # noqa: BLE001 - a crashing repair is a failed one
            return replace(
                check,
                auto_repair_attempted=True,
                auto_repair_result=f"failed: {exc}",
            )
        if result.status == "repaired":
            outcome = result.action
        else:
            outcome = f"{result.status}: {result.details or result.action}"
        return replace(check, auto_repair_attempted=True, auto_repair_result=outcome)

    def _update_alert_trackers(self, checks: list[Check]) -> None:
        """Refresh the stateful rule inputs from the fresh check results."""
        for check in checks:
            if check.name == "main_model_reachable":
                if check.status == "fail":
                    self._consecutive_model_failures += 1
                elif check.status == "pass":
                    self._consecutive_model_failures = 0
                # warn ("no provider configured") leaves the streak unchanged
            elif check.name == "git_state":
                if check.status == "warn" and check.details.startswith("working tree dirty"):
                    if self._git_dirty_since is None:
                        self._git_dirty_since = datetime.now(UTC)
                else:
                    self._git_dirty_since = None

    def _default_alert_engine(self) -> AlertEngine:
        """The P14.T5 rule wiring over the observables this system holds."""
        checks: list[Any] = [
            lambda: rule_trace_store_full(traces_root=self.context.traces_root),
            lambda: rule_backup_stale(self._latest_backup_time()),
            lambda: rule_main_model_down(self._consecutive_model_failures),
            lambda: rule_git_dirty_long(self._git_dirty_since),
        ]
        if self.model_metrics is not None:
            collector = self.model_metrics

            def _fallback_rule() -> Alert | None:
                report = collector.report(window_hours=1.0)
                total = report.generations + report.errors + report.fallbacks
                return rule_fallback_rate_high(report.fallbacks, total)

            checks.append(_fallback_rule)
        return AlertEngine(checks=checks)

    def _latest_backup_time(self) -> datetime | None:
        """mtime of the newest dated backup archive (``None`` = never ran)."""
        backups_dir = self.context.backups_dir
        if not backups_dir.is_dir():
            return None
        newest: tuple[float, datetime] | None = None
        for path in backups_dir.glob("*.tar.gz"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, datetime.fromtimestamp(mtime, tz=UTC))
        return newest[1] if newest is not None else None
