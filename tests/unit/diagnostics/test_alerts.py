"""P14.T3 acceptance: the alert rules engine (plan Section 5.3).

Each YAML-named test drives one plan-table rule with real inputs and checks
both the firing condition (correct rule name, severity per the plan table,
machine-readable context) and the non-firing boundary. The remaining six plan
rules get a smoke pass (``test_remaining_rules_smoke``), the twelve-rule
registration is guarded exactly, and the engine's three notification channels
(log file, gateway sink, optional email) plus its failure isolation are
exercised end to end. The cron rule is wired to a REAL watchdog incident:
watchdog DETECTS, diagnostics REPORTS (plan P21.T5 / 5.3 single-path rule).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from hermes.cron.watchdog import Watchdog, create_run
from hermes.diagnostics.alerts import (
    ALL_RULES,
    Alert,
    AlertEngine,
    default_alert_log_path,
    rule_adversarial_attempts,
    rule_backup_stale,
    rule_cost_anomaly,
    rule_cron_job_stuck,
    rule_fallback_rate_high,
    rule_gepa_stale,
    rule_git_dirty_long,
    rule_low_trust_entries,
    rule_main_model_down,
    rule_pii_detected_high_rate,
    rule_skill_pending_long,
    rule_trace_store_full,
)
from hermes.diagnostics.model_metrics import ModelMetricsCollector
from hermes.config import hermes_home

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
_GB = 1024**3


@pytest.mark.unit
def test_alert_on_main_model_down(tmp_path: Path) -> None:
    """3 consecutive failed probes fire a critical; 2 do not; engine logs+sink."""
    alert = rule_main_model_down(3, now=_NOW)
    assert alert is not None
    assert alert.rule_name == "main_model_down"
    assert alert.severity == "critical"
    assert alert.fired_at == _NOW
    assert alert.context["consecutive_failures"] == 3
    assert alert.context["threshold"] == 3

    assert rule_main_model_down(2, now=_NOW) is None

    # engine wiring: the curried rule is evaluated, logged and sunk
    sinked: list[Alert] = []
    log_path = tmp_path / "diagnostics" / "alerts.log"
    engine = AlertEngine(
        checks=[lambda: rule_main_model_down(3, now=_NOW)],
        sink=sinked.append,
        log_path=log_path,
    )
    fired = engine.evaluate()
    assert [fired_alert.rule_name for fired_alert in fired] == ["main_model_down"]
    assert [captured.rule_name for captured in sinked] == ["main_model_down"]
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["rule_name"] == "main_model_down"
    assert record["severity"] == "critical"


@pytest.mark.unit
def test_alert_on_dirty_git_over_1h() -> None:
    """Dirty >1h despite auto-repair fires a warning; under 1h and None do not."""
    alert = rule_git_dirty_long(_NOW - timedelta(hours=1, minutes=15), now=_NOW)
    assert alert is not None
    assert alert.rule_name == "git_dirty_long"
    assert alert.severity == "warning"
    assert alert.context["age_hours"] == pytest.approx(1.25)
    assert alert.context["dirty_since"] == (_NOW - timedelta(hours=1, minutes=15)).isoformat()

    assert rule_git_dirty_long(_NOW - timedelta(minutes=59), now=_NOW) is None
    assert rule_git_dirty_long(None, now=_NOW) is None  # onset unknown: silent


@pytest.mark.unit
def test_alert_on_trace_store_full(tmp_path: Path) -> None:
    """Trace store over the 10GB limit fires a warning, by bytes or by scan."""
    alert = rule_trace_store_full(total_bytes=10 * _GB + 1, now=_NOW)
    assert alert is not None
    assert alert.rule_name == "trace_store_full"
    assert alert.severity == "warning"
    assert alert.context["total_bytes"] == 10 * _GB + 1

    assert rule_trace_store_full(total_bytes=10 * _GB, now=_NOW) is None  # at-limit

    # real-path variant: a sparse file pushes the scanned store over the limit
    traces_root = tmp_path / "traces"
    traces_root.mkdir()
    big = traces_root / "v1.0" / "2026-11-21.jsonl"
    big.parent.mkdir()
    with open(big, "wb") as fh:
        fh.seek(10 * _GB + 512)
        fh.write(b"\0")
    scanned = rule_trace_store_full(traces_root=traces_root, now=_NOW)
    assert scanned is not None
    assert scanned.context["total_bytes"] == 10 * _GB + 513  # seek-to + 1 byte

    traces_root_small = tmp_path / "small"
    traces_root_small.mkdir()
    assert rule_trace_store_full(traces_root=traces_root_small, now=_NOW) is None


@pytest.mark.unit
def test_alert_on_gepa_stale() -> None:
    """No GEPA run in 8 days fires a warning; fresher runs and never-ran do not."""
    alert = rule_gepa_stale(_NOW - timedelta(days=8, minutes=30), now=_NOW)
    assert alert is not None
    assert alert.rule_name == "gepa_stale"
    assert alert.severity == "warning"
    assert alert.context["age_days"] == pytest.approx(8 + 30 / 1440)

    assert rule_gepa_stale(_NOW - timedelta(days=7), now=_NOW) is None
    # documented: never-ran is a legitimate fresh-install state (P14.T1 warns,
    # no onset to alert on)
    assert rule_gepa_stale(None, now=_NOW) is None


@pytest.mark.unit
def test_alert_on_fallback_rate_high(tmp_path: Path) -> None:
    """Fallback >20% of ops fires a warning; counts come from the real ledger."""
    alert = rule_fallback_rate_high(3, 10, now=_NOW)
    assert alert is not None
    assert alert.rule_name == "fallback_rate_high"
    assert alert.severity == "warning"
    assert alert.context["rate"] == pytest.approx(0.3)
    assert alert.context["threshold"] == pytest.approx(0.2)

    assert rule_fallback_rate_high(2, 10, now=_NOW) is None  # exactly 20%
    assert rule_fallback_rate_high(0, 0, now=_NOW) is None  # no ops in window

    # integration with the P14.T7 model-metrics ledger: derive the counts
    # from a real collector report (2 generations + 1 error + 2 fallbacks)
    collector = ModelMetricsCollector(
        ledger_path=tmp_path / "model_metrics.jsonl", now=lambda: _NOW
    )
    collector.record_generation("main", tokens=100, seconds=1.0)
    collector.record_generation("main", tokens=50, seconds=1.0)
    collector.record_error("main")
    collector.record_fallback("main")
    collector.record_fallback("main")
    report = collector.report(window_hours=1.0)
    derived = rule_fallback_rate_high(
        report.fallbacks, report.generations + report.errors + report.fallbacks, now=_NOW
    )
    assert derived is not None
    assert derived.context["fallback_ops"] == 2
    assert derived.context["total_ops"] == 5


@pytest.mark.unit
def test_alert_on_cron_stuck(tmp_path: Path) -> None:
    """A real watchdog incident surfaces as a cron_job_stuck warning."""
    # watchdog DETECTS: force-fail a run stuck past the 30-minute timeout
    clock_value = {"t": 1000.0}
    watchdog = Watchdog(clock=lambda: clock_value["t"])
    run = create_run("nightly-report", clock=lambda: clock_value["t"])
    watchdog.register_run(run)
    watchdog.mark_running(run.run_id)
    clock_value["t"] += 1900.0  # 31m40s without progress
    forced = watchdog.check_once()
    assert len(forced) == 1
    incident = watchdog.incidents[0]

    # diagnostics REPORTS: the rule fires whenever the incident is supplied
    alert = rule_cron_job_stuck(incident, now=_NOW)
    assert alert is not None
    assert alert.rule_name == "cron_job_stuck"
    assert alert.severity == "warning"
    assert alert.context["job_id"] == "nightly-report"
    assert alert.context["run_id"] == run.run_id
    assert "nightly-report" in alert.message
    assert alert.context["inactivity_seconds"] == pytest.approx(1900.0)

    # mapping form (duck-typed incidents, e.g. over the wire) also works
    mapping_alert = rule_cron_job_stuck(
        {"job_id": "j", "run_id": "r", "reason": "no progress for 1801s"}, now=_NOW
    )
    assert mapping_alert is not None
    assert mapping_alert.context["job_id"] == "j"

    assert rule_cron_job_stuck(None, now=_NOW) is None


@pytest.mark.unit
def test_all_twelve_rules_registered() -> None:
    """Guard: ALL_RULES carries the 12 plan 5.3 rules under exact names."""
    expected = {
        "main_model_down",
        "git_dirty_long",
        "trace_store_full",
        "gepa_stale",
        "low_trust_entries",
        "skill_pending_long",
        "pii_detected_high_rate",
        "adversarial_attempts",
        "cost_anomaly",
        "backup_stale",
        "cron_job_stuck",
        "fallback_rate_high",
    }
    assert set(ALL_RULES) == expected
    assert len(ALL_RULES) == 12
    # severity per the plan 5.3 table
    assert rule_main_model_down(3, now=_NOW).severity == "critical"
    assert rule_git_dirty_long(_NOW - timedelta(hours=2), now=_NOW).severity == "warning"
    assert rule_low_trust_entries(11, now=_NOW).severity == "info"
    assert rule_skill_pending_long(_NOW - timedelta(days=8), now=_NOW).severity == "info"
    for name in (
        "trace_store_full",
        "gepa_stale",
        "pii_detected_high_rate",
        "adversarial_attempts",
        "cost_anomaly",
        "backup_stale",
        "cron_job_stuck",
        "fallback_rate_high",
    ):
        assert ALL_RULES[name] is not None


@pytest.mark.unit
def test_remaining_rules_smoke() -> None:
    """Smoke pass over the other six plan 5.3 rules: fire + silent boundary."""
    # low_trust_entries (>10 entries with trust <0.3, info)
    fired = rule_low_trust_entries(11, now=_NOW)
    assert fired is not None and fired.rule_name == "low_trust_entries"
    assert fired.context["count"] == 11
    assert rule_low_trust_entries(10, now=_NOW) is None

    # skill_pending_long (pending >7 days, info)
    fired = rule_skill_pending_long(_NOW - timedelta(days=8), now=_NOW)
    assert fired is not None and fired.rule_name == "skill_pending_long"
    assert rule_skill_pending_long(_NOW - timedelta(days=7), now=_NOW) is None

    # pii_detected_high_rate (>10% of writes, warning)
    fired = rule_pii_detected_high_rate(2, 10, now=_NOW)
    assert fired is not None and fired.rule_name == "pii_detected_high_rate"
    assert rule_pii_detected_high_rate(1, 10, now=_NOW) is None
    assert rule_pii_detected_high_rate(1, 0, now=_NOW) is None

    # adversarial_attempts (>5 blocked attempts in 1h, warning)
    fired = rule_adversarial_attempts(6, now=_NOW)
    assert fired is not None and fired.rule_name == "adversarial_attempts"
    assert fired.context["window_hours"] == 1.0
    assert rule_adversarial_attempts(5, now=_NOW) is None

    # cost_anomaly (daily cost >2x rolling average, warning)
    fired = rule_cost_anomaly(10.0, 4.0, now=_NOW)
    assert fired is not None and fired.rule_name == "cost_anomaly"
    assert rule_cost_anomaly(8.0, 4.0, now=_NOW) is None
    assert rule_cost_anomaly(10.0, 0.0, now=_NOW) is None  # no baseline yet

    # backup_stale (last backup >25h, warning; never backed up fires)
    fired = rule_backup_stale(_NOW - timedelta(hours=26), now=_NOW)
    assert fired is not None and fired.rule_name == "backup_stale"
    assert rule_backup_stale(_NOW - timedelta(hours=25), now=_NOW) is None
    never = rule_backup_stale(None, now=_NOW)
    assert never is not None and "no backup ever recorded" in never.message


@pytest.mark.unit
def test_engine_channels_and_failure_isolation(tmp_path: Path) -> None:
    """Log+gateway+email all receive fired alerts; broken rule/channel isolated."""
    gateway_messages: list[str] = []
    emails: list[dict[str, Any]] = []

    def gateway_sink(alert: Alert) -> None:
        gateway_messages.append(f"gateway: {alert.rule_name}: {alert.message}")

    def email_channel(alert: Alert) -> None:
        emails.append(alert.to_dict())

    def broken_rule() -> Alert | None:
        raise RuntimeError("probe exploded")

    log_path = tmp_path / "diagnostics" / "alerts.log"
    engine = AlertEngine(
        checks=[
            broken_rule,  # raising rule must be skipped, not fatal
            lambda: rule_main_model_down(3, now=_NOW),  # fires
            lambda: rule_backup_stale(None, now=_NOW),  # fires
        ],
        sink=gateway_sink,
        log_path=log_path,
        email=email_channel,
    )
    fired = engine.evaluate()

    assert [alert.rule_name for alert in fired] == ["main_model_down", "backup_stale"]
    assert len(gateway_messages) == 2
    assert len(emails) == 2
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["rule_name"] for line in lines] == [
        "main_model_down",
        "backup_stale",
    ]

    # a broken channel must not prevent the other channels or the return value
    def broken_sink(alert: Alert) -> None:
        raise RuntimeError("gateway down")

    resilient = AlertEngine(
        checks=[lambda: rule_main_model_down(3, now=_NOW)],
        sink=broken_sink,
        log_path=tmp_path / "diagnostics" / "alerts2.log",
        email=email_channel,
    )
    assert [alert.rule_name for alert in resilient.evaluate()] == ["main_model_down"]
    assert len(emails) == 3

    # default log path derives from hermes_home() (the per-test temp dir)
    default_engine = AlertEngine(checks=[])
    assert default_engine.log_path == hermes_home() / "diagnostics" / "alerts.log"
    assert default_alert_log_path() == hermes_home() / "diagnostics" / "alerts.log"
