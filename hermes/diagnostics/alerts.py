"""hermes-v2 alert rules engine (plan P14.T3, plan Section 5.3).

Twelve alert rules — the exact plan 5.3 table — evaluated by one engine and
reported through three notification channels:

- **log** (always): fired alerts are appended as JSON lines to
  ``<hermes_home>/diagnostics/alerts.log``;
- **gateway message** (optional): a ``sink`` callable taking the
  :class:`Alert`, injected by the gateway wiring (P14.T6);
- **email** (optional): an ``email`` callable taking the :class:`Alert`.

Design (plan P14.T3): :class:`AlertRule` is the rule protocol — a callable
that inspects its inputs and returns an :class:`Alert` or ``None``. Each rule
is a plain function of its observable inputs plus an injectable ``now``
(deterministic tests); the zero-argument callables the engine evaluates are
built by currying those functions (``functools.partial``/lambdas at the
wiring site, e.g. the P14.T5 DiagnosticsSystem). :data:`ALL_RULES` registers
all twelve under their exact plan names; ``test_all_twelve_rules_registered``
guards that set.

The 12 rules (plan Section 5.3, names exact, severity per the plan table):

======================  ====================================================  ========
rule                    trigger                                               severity
======================  ====================================================  ========
main_model_down         probe fails 3x in a row (``consecutive_failures``)    critical
git_dirty_long          dirty >1h despite auto-repair (``dirty_since``)       warning
trace_store_full        trace store >10GB (``total_bytes`` or ``traces_root``) warning
gepa_stale              no GEPA run in 8 days (``last_run``)                  warning
low_trust_entries       >10 entries with trust <0.3 (``count``)               info
skill_pending_long      skill pending approval >7 days (``pending_since``)    info
pii_detected_high_rate  PII in >10% of writes (``detected_count``/``total``)  warning
adversarial_attempts    >5 blocked attempts in 1h (``blocked_attempts``)      warning
cost_anomaly            daily cost >2x rolling avg (``today_cost``/``rolling_avg``) warning
backup_stale            last backup >25h (``last_backup``)                    warning
cron_job_stuck          cron watchdog (P21.T5) incident input                 warning
fallback_rate_high      trust-only/regex fallback >20% of ops in 1h           warning
======================  ====================================================  ========

Documented edge semantics:

- ``git_dirty_long`` / ``gepa_stale`` / ``skill_pending_long`` with a ``None``
  timestamp return ``None``: "when it started" is unknown, and the plan's
  health checks (P14.T1) already report the fresh-install states (no repo, no
  Pareto front) as warnings — an alert without an onset would be noise.
- ``backup_stale`` with ``last_backup=None`` **does** fire: a never-backed-up
  system is maximally stale, matching :func:`hermes.backup.manager.due_for_backup`
  ("never backed up is due immediately").
- ``cron_job_stuck`` fires whenever an incident is supplied: the watchdog
  (P21.T5) already applied its 30-minute inactivity timeout when recording the
  incident — "watchdog DETECTS, diagnostics REPORTS" (single alerting path, no
  duplicate thresholds here).
- ``cost_anomaly`` with ``rolling_avg <= 0`` returns ``None``: no baseline
  exists to be 2x of (a brand-new install has no history).

Every rule and every engine step is exception-safe by construction: rules that
raise are reported in the engine's log (and skipped), a failing notification
channel never prevents the others — mirroring the failure isolation the
P14.T1 health checks and the P21.T5 watchdog already guarantee.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from hermes.config import hermes_home

__all__ = [
    "ADVERSARIAL_WINDOW_HOURS",
    "ADVERSARIAL_MAX_ATTEMPTS",
    "ALL_RULES",
    "Alert",
    "AlertEngine",
    "AlertRule",
    "BACKUP_STALE_MAX_HOURS",
    "COST_ANOMALY_FACTOR",
    "FALLBACK_RATE_THRESHOLD",
    "GEPA_STALE_MAX_DAYS",
    "GIT_DIRTY_MAX_HOURS",
    "LOW_TRUST_MAX_ENTRIES",
    "LOW_TRUST_THRESHOLD",
    "MAIN_MODEL_DOWN_THRESHOLD",
    "PII_RATE_THRESHOLD",
    "SEVERITIES",
    "SKILL_PENDING_MAX_DAYS",
    "TRACE_STORE_MAX_BYTES",
    "default_alert_log_path",
    "rule_adversarial_attempts",
    "rule_backup_stale",
    "rule_cost_anomaly",
    "rule_cron_job_stuck",
    "rule_fallback_rate_high",
    "rule_gepa_stale",
    "rule_git_dirty_long",
    "rule_low_trust_entries",
    "rule_main_model_down",
    "rule_pii_detected_high_rate",
    "rule_skill_pending_long",
    "rule_trace_store_full",
]

_LOGGER = logging.getLogger("hermes.diagnostics.alerts")

#: Valid alert severities (plan Section 5.3).
Severity = Literal["critical", "warning", "info"]
SEVERITIES: Final[frozenset[str]] = frozenset({"critical", "warning", "info"})

#: plan 5.3 thresholds, kept as named constants so callers/wiring share them.
MAIN_MODEL_DOWN_THRESHOLD: Final[int] = 3
GIT_DIRTY_MAX_HOURS: Final[float] = 1.0
TRACE_STORE_MAX_BYTES: Final[int] = 10 * 1024**3  # 10 binary GB (plan 5.3)
GEPA_STALE_MAX_DAYS: Final[float] = 8.0
LOW_TRUST_MAX_ENTRIES: Final[int] = 10  # fires when count > 10
LOW_TRUST_THRESHOLD: Final[float] = 0.3
SKILL_PENDING_MAX_DAYS: Final[float] = 7.0
PII_RATE_THRESHOLD: Final[float] = 0.1  # >10% of writes
ADVERSARIAL_MAX_ATTEMPTS: Final[int] = 5  # fires when count > 5 ...
ADVERSARIAL_WINDOW_HOURS: Final[float] = 1.0  # ... within 1 hour
COST_ANOMALY_FACTOR: Final[float] = 2.0
BACKUP_STALE_MAX_HOURS: Final[float] = 25.0
FALLBACK_RATE_THRESHOLD: Final[float] = 0.2  # >20% of ops in 1h

_SECONDS_PER_HOUR: Final[float] = 3600.0
_SECONDS_PER_DAY: Final[float] = 86400.0
_GB: Final[float] = float(1024**3)


def default_alert_log_path() -> Path:
    """Default alert log: ``<hermes_home>/diagnostics/alerts.log``."""
    path: Path = hermes_home() / "diagnostics" / "alerts.log"
    return path


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _utcnow() -> datetime:
    """Default clock: current aware UTC time."""
    return datetime.now(UTC)


class AlertRule(Protocol):
    """Protocol for alert rule callables (plan P14.T3 design).

    Rules are parameterized functions of their observable inputs plus a
    ``now`` keyword (aware-UTC wall clock, injectable for tests) returning the
    fired :class:`Alert` or ``None`` when the trigger condition is not met.
    The engine evaluates zero-argument curries of these.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Alert | None: ...


@dataclass
class Alert:
    """One fired alert (plan P14.T3 design).

    ``rule_name`` is one of the exact plan 5.3 names (see :data:`ALL_RULES`),
    ``severity`` one of ``critical``/``warning``/``info``, ``fired_at`` is
    aware UTC and ``context`` carries the observables the rule evaluated on
    (counts, ages, thresholds — machine-readable for the dashboard, P14.T10).
    """

    rule_name: str
    severity: Severity
    message: str
    fired_at: datetime
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_name:
            raise ValueError("rule_name must be a non-empty string")
        if self.severity not in SEVERITIES:
            allowed = sorted(SEVERITIES)
            raise ValueError(f"severity must be one of {allowed}, got {self.severity!r}")
        self.fired_at = _as_utc(self.fired_at)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection (alerts.log line / dashboard payload)."""
        return {
            "fired_at": self.fired_at.isoformat(),
            "rule_name": self.rule_name,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
        }


def _make_alert(
    rule_name: str,
    severity: Severity,
    message: str,
    now: datetime,
    context: dict[str, Any],
) -> Alert:
    """Build an :class:`Alert` with a UTC-stamped ``fired_at``."""
    return Alert(
        rule_name=rule_name,
        severity=severity,
        message=message,
        fired_at=now,
        context=context,
    )


# ── the 12 rules (plan Section 5.3, exact names) ─────────────────────────────


def rule_main_model_down(
    consecutive_failures: int,
    *,
    threshold: int = MAIN_MODEL_DOWN_THRESHOLD,
    now: datetime | None = None,
) -> Alert | None:
    """Main model API health probe failed ``threshold`` times in a row (critical)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if consecutive_failures < threshold:
        return None
    return _make_alert(
        "main_model_down",
        "critical",
        (
            f"main model down: {consecutive_failures} consecutive failed "
            f"health probes (threshold {threshold})"
        ),
        moment,
        {"consecutive_failures": consecutive_failures, "threshold": threshold},
    )


def rule_git_dirty_long(
    dirty_since: datetime | None,
    *,
    max_age_hours: float = GIT_DIRTY_MAX_HOURS,
    now: datetime | None = None,
) -> Alert | None:
    """Git working tree dirty longer than ``max_age_hours`` despite auto-repair."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if dirty_since is None:
        return None  # onset unknown (clean tree or no repo) — nothing to age
    age_hours = (moment - _as_utc(dirty_since)).total_seconds() / _SECONDS_PER_HOUR
    if age_hours <= max_age_hours:
        return None
    return _make_alert(
        "git_dirty_long",
        "warning",
        (
            f"git working tree dirty for {age_hours:.1f} h "
            f"(limit {max_age_hours:g} h) despite auto-repair"
        ),
        moment,
        {
            "dirty_since": _as_utc(dirty_since).isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
        },
    )


def rule_trace_store_full(
    total_bytes: int | None = None,
    traces_root: Path | None = None,
    *,
    max_bytes: int = TRACE_STORE_MAX_BYTES,
    now: datetime | None = None,
) -> Alert | None:
    """Trace store over ``max_bytes`` (10 GB binary, plan 5.3).

    Pass precomputed ``total_bytes`` or a ``traces_root`` to scan (every file
    under the root is summed — the same accounting the P14.T1 size check
    uses). Providing neither raises ``ValueError``.
    """
    moment = _as_utc(now) if now is not None else _utcnow()
    if total_bytes is None:
        if traces_root is None:
            raise ValueError("rule_trace_store_full requires total_bytes or traces_root")
        total_bytes = 0
        if traces_root.is_dir():
            for path in traces_root.rglob("*"):
                if path.is_file():
                    try:
                        total_bytes += path.stat().st_size
                    except OSError:
                        continue
    if total_bytes <= max_bytes:
        return None
    used_gb = total_bytes / _GB
    limit_gb = max_bytes / _GB
    return _make_alert(
        "trace_store_full",
        "warning",
        f"trace store full: {used_gb:.2f} GB used, limit {limit_gb:.1f} GB",
        moment,
        {"total_bytes": total_bytes, "max_bytes": max_bytes},
    )


def rule_gepa_stale(
    last_run: datetime | None,
    *,
    max_age_days: float = GEPA_STALE_MAX_DAYS,
    now: datetime | None = None,
) -> Alert | None:
    """No GEPA run within ``max_age_days`` (8 days, plan 5.3).

    ``last_run=None`` (never ran) returns ``None`` — documented fresh-install
    state; the P14.T1 pareto freshness check already warns on it.
    """
    moment = _as_utc(now) if now is not None else _utcnow()
    if last_run is None:
        return None
    age_days = (moment - _as_utc(last_run)).total_seconds() / _SECONDS_PER_DAY
    if age_days <= max_age_days:
        return None
    return _make_alert(
        "gepa_stale",
        "warning",
        f"no GEPA run in {age_days:.1f} days (limit {max_age_days:g} days)",
        moment,
        {
            "last_run": _as_utc(last_run).isoformat(),
            "age_days": age_days,
            "max_age_days": max_age_days,
        },
    )


def rule_low_trust_entries(
    count: int,
    *,
    max_count: int = LOW_TRUST_MAX_ENTRIES,
    trust_below: float = LOW_TRUST_THRESHOLD,
    now: datetime | None = None,
) -> Alert | None:
    """More than ``max_count`` entries with trust score below ``trust_below`` (info)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if count <= max_count:
        return None
    return _make_alert(
        "low_trust_entries",
        "info",
        (
            f"{count} entries with trust score below {trust_below} "
            f"(threshold: more than {max_count})"
        ),
        moment,
        {"count": count, "max_count": max_count, "trust_below": trust_below},
    )


def rule_skill_pending_long(
    pending_since: datetime | None,
    *,
    max_age_days: float = SKILL_PENDING_MAX_DAYS,
    now: datetime | None = None,
) -> Alert | None:
    """Oldest skill approval pending longer than ``max_age_days`` (info)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if pending_since is None:
        return None  # nothing pending (or onset unknown)
    age_days = (moment - _as_utc(pending_since)).total_seconds() / _SECONDS_PER_DAY
    if age_days <= max_age_days:
        return None
    return _make_alert(
        "skill_pending_long",
        "info",
        f"skill approval pending for {age_days:.1f} days (limit {max_age_days:g} days)",
        moment,
        {
            "pending_since": _as_utc(pending_since).isoformat(),
            "age_days": age_days,
            "max_age_days": max_age_days,
        },
    )


def rule_pii_detected_high_rate(
    detected_count: int,
    total: int,
    *,
    threshold: float = PII_RATE_THRESHOLD,
    now: datetime | None = None,
) -> Alert | None:
    """PII detected in more than ``threshold`` of memory writes (warning)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if total <= 0:
        return None  # no writes in the window — rate undefined, nothing to alert
    rate = detected_count / total
    if rate <= threshold:
        return None
    return _make_alert(
        "pii_detected_high_rate",
        "warning",
        (
            f"PII detected in {rate:.1%} of memory writes "
            f"(threshold {threshold:.0%}, {detected_count}/{total})"
        ),
        moment,
        {"detected_count": detected_count, "total": total, "rate": rate, "threshold": threshold},
    )


def rule_adversarial_attempts(
    blocked_attempts: int,
    *,
    max_attempts: int = ADVERSARIAL_MAX_ATTEMPTS,
    window_hours: float = ADVERSARIAL_WINDOW_HOURS,
    now: datetime | None = None,
) -> Alert | None:
    """More than ``max_attempts`` blocked adversarial attempts in the window (warning)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if blocked_attempts <= max_attempts:
        return None
    return _make_alert(
        "adversarial_attempts",
        "warning",
        (
            f"{blocked_attempts} blocked adversarial attempts within "
            f"{window_hours:g} h (threshold: more than {max_attempts})"
        ),
        moment,
        {
            "blocked_attempts": blocked_attempts,
            "max_attempts": max_attempts,
            "window_hours": window_hours,
        },
    )


def rule_cost_anomaly(
    today_cost: float,
    rolling_avg: float,
    *,
    factor: float = COST_ANOMALY_FACTOR,
    now: datetime | None = None,
) -> Alert | None:
    """Daily token cost more than ``factor`` times the rolling average (warning)."""
    moment = _as_utc(now) if now is not None else _utcnow()
    if rolling_avg <= 0:
        return None  # no baseline yet — a brand-new install has no history
    if today_cost <= factor * rolling_avg:
        return None
    return _make_alert(
        "cost_anomaly",
        "warning",
        (
            f"daily token cost {today_cost:.4f} exceeds {factor:g}x the "
            f"rolling average ({rolling_avg:.4f})"
        ),
        moment,
        {"today_cost": today_cost, "rolling_avg": rolling_avg, "factor": factor},
    )


def rule_backup_stale(
    last_backup: datetime | None,
    *,
    max_age_hours: float = BACKUP_STALE_MAX_HOURS,
    now: datetime | None = None,
) -> Alert | None:
    """Latest backup older than ``max_age_hours`` (warning).

    ``last_backup=None`` fires ("never backed up") — matching
    :func:`hermes.backup.manager.due_for_backup`, where a never-backed-up
    system is due immediately.
    """
    moment = _as_utc(now) if now is not None else _utcnow()
    if last_backup is None:
        return _make_alert(
            "backup_stale",
            "warning",
            f"no backup ever recorded (limit {max_age_hours:g} h)",
            moment,
            {"last_backup": None, "age_hours": None, "max_age_hours": max_age_hours},
        )
    age_hours = (moment - _as_utc(last_backup)).total_seconds() / _SECONDS_PER_HOUR
    if age_hours <= max_age_hours:
        return None
    return _make_alert(
        "backup_stale",
        "warning",
        f"latest backup {age_hours:.1f} h old (limit {max_age_hours:g} h)",
        moment,
        {
            "last_backup": _as_utc(last_backup).isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
        },
    )


def _incident_field(incident: Mapping[str, Any] | Any, name: str) -> Any:
    """Read ``name`` from an incident that is a mapping or an attribute object."""
    if isinstance(incident, Mapping):
        return incident.get(name)
    return getattr(incident, name, None)


def rule_cron_job_stuck(
    incident: Mapping[str, Any] | Any | None,
    *,
    now: datetime | None = None,
) -> Alert | None:
    """Surface a cron watchdog incident (P21.T5) as a warning (plan 5.3).

    The watchdog DETECTS (it already applied the 30-minute inactivity timeout
    when recording the :class:`~hermes.cron.watchdog.CronIncident`) and
    diagnostics REPORTS: the alert fires whenever an incident is supplied, so
    there is exactly one alerting path and no duplicated threshold. Accepts a
    real ``CronIncident``, any attribute-equivalent object, or a plain mapping.
    """
    moment = _as_utc(now) if now is not None else _utcnow()
    if incident is None:
        return None
    job_id = _incident_field(incident, "job_id")
    run_id = _incident_field(incident, "run_id")
    reason = _incident_field(incident, "reason") or "no progress"
    inactivity = _incident_field(incident, "inactivity_seconds")
    timeout = _incident_field(incident, "timeout_seconds")
    incident_id = _incident_field(incident, "incident_id")
    return _make_alert(
        "cron_job_stuck",
        "warning",
        f"cron job '{job_id}' run {run_id} stuck: {reason}",
        moment,
        {
            "job_id": job_id,
            "run_id": run_id,
            "reason": str(reason),
            "inactivity_seconds": inactivity,
            "timeout_seconds": timeout,
            "incident_id": incident_id,
        },
    )


def rule_fallback_rate_high(
    fallback_ops: int,
    total_ops: int,
    *,
    threshold: float = FALLBACK_RATE_THRESHOLD,
    now: datetime | None = None,
) -> Alert | None:
    """Trust-only/regex fallback in more than ``threshold`` of ops (warning).

    Callers compute the counts over the 1-hour window (e.g. from a
    :class:`~hermes.diagnostics.model_metrics.ModelMetricsReport`:
    ``fallback_ops=report.fallbacks``,
    ``total_ops=report.generations + report.errors + report.fallbacks``).
    """
    moment = _as_utc(now) if now is not None else _utcnow()
    if total_ops <= 0:
        return None  # no operations in the window — rate undefined
    rate = fallback_ops / total_ops
    if rate <= threshold:
        return None
    return _make_alert(
        "fallback_rate_high",
        "warning",
        (
            f"fallback rate {rate:.1%} of ops in the last hour "
            f"(threshold {threshold:.0%}, {fallback_ops}/{total_ops})"
        ),
        moment,
        {"fallback_ops": fallback_ops, "total_ops": total_ops, "rate": rate,
         "threshold": threshold},
    )


#: All 12 plan 5.3 rules under their exact plan names (registration guard:
#: tests/unit/diagnostics/test_alerts.py::test_all_twelve_rules_registered).
ALL_RULES: Final[dict[str, AlertRule]] = {
    "main_model_down": rule_main_model_down,
    "git_dirty_long": rule_git_dirty_long,
    "trace_store_full": rule_trace_store_full,
    "gepa_stale": rule_gepa_stale,
    "low_trust_entries": rule_low_trust_entries,
    "skill_pending_long": rule_skill_pending_long,
    "pii_detected_high_rate": rule_pii_detected_high_rate,
    "adversarial_attempts": rule_adversarial_attempts,
    "cost_anomaly": rule_cost_anomaly,
    "backup_stale": rule_backup_stale,
    "cron_job_stuck": rule_cron_job_stuck,
    "fallback_rate_high": rule_fallback_rate_high,
}


# ── the engine ────────────────────────────────────────────────────────────────


class AlertEngine:
    """Evaluates the configured rules and fans fired alerts out to channels.

    Args:
        checks: zero-argument rule callables, each returning an
            :class:`Alert` or ``None`` (curries of the :data:`ALL_RULES`
            functions, or fully custom rules).
        sink: optional gateway-message channel, ``Callable[[Alert], None]``.
        log_path: JSON-lines log (default
            ``<hermes_home>/diagnostics/alerts.log``).
        email: optional email channel, ``Callable[[Alert], None]``.

    :meth:`evaluate` runs every check (a raising rule is logged and skipped —
    one broken rule never disables the others), appends each fired alert to
    the log channel, then invokes ``sink`` and ``email``. A raising channel
    is logged and skipped, so the remaining channels still fire. Returns the
    fired alerts in check order.
    """

    def __init__(
        self,
        checks: Sequence[Callable[[], Alert | None]],
        sink: Callable[[Alert], None] | None = None,
        log_path: Path | None = None,
        email: Callable[[Alert], None] | None = None,
    ) -> None:
        self.checks: list[Callable[[], Alert | None]] = list(checks)
        #: Gateway-message notification channel (P14.T6 wiring injects it).
        self.sink = sink
        #: Optional email notification channel.
        self.email = email
        #: JSON-lines alert log (always-on channel, plan acceptance).
        self.log_path: Path = log_path if log_path is not None else default_alert_log_path()
        self._lock = threading.Lock()

    def evaluate(self) -> list[Alert]:
        """Run all rules; log, sink and email every fired alert; return them."""
        fired: list[Alert] = []
        for index, check in enumerate(self.checks):
            try:
                alert = check()
            except Exception:
                _LOGGER.exception("alert rule at position %d (%r) crashed", index, check)
                continue
            if alert is None:
                continue
            fired.append(alert)
            self._log_alert(alert)
            self._notify(alert)
        return fired

    # ── channels ──────────────────────────────────────────────────────────
    def _log_alert(self, alert: Alert) -> None:
        """Append one JSON line to the alert log (atomic single write + fsync)."""
        line = json.dumps(alert.to_dict(), ensure_ascii=False)
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _notify(self, alert: Alert) -> None:
        """Fan out to the optional gateway-message and email channels."""
        for channel_name, channel in (("gateway sink", self.sink), ("email", self.email)):
            if channel is None:
                continue
            try:
                channel(alert)
            except Exception:
                # A broken channel must never eat the other channels' delivery.
                _LOGGER.exception("%s notification channel failed for %s",
                                  channel_name, alert.rule_name)
