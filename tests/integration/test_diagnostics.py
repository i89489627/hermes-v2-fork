"""P14.T5 + P14.T6 acceptance: the DiagnosticsSystem and its surfaces (plan Phase 14).

Integration tests over the real wiring (MemoryStore-backed checks, a real
MetricsCollector JSONL time series, the AlertEngine): the health report
matches the plan 5.1 JSON schema exactly and generates in under 2 s with the
injected fakes, background collection runs on the 5-minute cadence
(scaled to 0.05 s here) and stops cleanly, and the report carries the plan's
metrics block plus fired alerts. P14.T6 adds the ``/diagnostics`` slash
command (quick|full|repair via ``hermes.gateway.health``) and the real-HTTP
``/health`` + ``/metrics`` gateway endpoints (http.client against an
ephemeral 127.0.0.1 port).
"""

from __future__ import annotations

import http.client
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.backup.manager import BackupManager
from hermes.diagnostics.alerts import Alert, AlertEngine
from hermes.diagnostics.auto_repair import AutoRepair, RepairContext
from hermes.diagnostics.health_checks import HealthCheckContext
from hermes.diagnostics.metrics import MetricsCollector
from hermes.diagnostics.model_metrics import ModelMetricsCollector
from hermes.diagnostics.system import (
    DEFAULT_COLLECTION_INTERVAL_MINUTES,
    DiagnosticsSystem,
)
from hermes.gateway.health import (
    handle_diagnostics_command,
    serve_health,
    stop_health_server,
)
from hermes.memory.store import MemoryStore

_PLAN_METRIC_KEYS = frozenset(
    {
        "memory_ops_latency_p50_ms",
        "memory_ops_latency_p95_ms",
        "token_usage_today",
        "trust_score_avg",
        "trust_score_min",
        "conflict_rate_7d",
        "retrieval_recall_at_5",
        "skills_active",
        "skills_pending_approval",
        "gepa_last_run",
    }
)

_PLAN_CHECK_NAMES = [
    "memory_store_integrity",
    "git_state",
    "main_model_reachable",
    "trace_store_size",
    "pareto_front_freshness",
    "backup_freshness",
]


def _build_store(tmp_path: Path) -> MemoryStore:
    """A real git-backed memory store with two valid entries."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("Entry one for diagnostics", type="Feedback", source="user-explicit")
    store.add("Entry two for diagnostics", type="Feedback", source="user-explicit")
    return store


def _fresh_context(tmp_path: Path, store: MemoryStore) -> HealthCheckContext:
    """Hermetic plan-5.1 context: injected probe + paths under ``tmp_path``."""
    pareto_path = tmp_path / "optimization" / "pareto.json"
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    pareto_path.write_text("{}", encoding="utf-8")
    return HealthCheckContext(
        store_root=store.root,
        repo_path=store.root,
        probe=lambda: True,
        traces_root=tmp_path / "traces",
        pareto_path=pareto_path,
        backups_dir=tmp_path / "backups",
    )


@pytest.mark.integration
def test_full_health_report_generated(tmp_path: Path) -> None:
    """The report matches plan 5.1 exactly and generates within the 2 s budget."""
    store = _build_store(tmp_path)
    context = _fresh_context(tmp_path, store)
    # fresh backup + fresh Pareto front so every one of the six checks passes
    BackupManager(hermes_root=tmp_path).create_backup(now=datetime.now(UTC))
    system = DiagnosticsSystem(
        context=context,
        collector=MetricsCollector(store_path=tmp_path / "metrics.jsonl"),
        alert_engine=AlertEngine(checks=[]),
    )

    started = time.perf_counter()
    report = system.run_health_check()
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"report generation took {elapsed:.2f}s (budget 2s)"
    assert report.overall_status == "healthy"  # every check passes
    assert [check.name for check in report.checks] == _PLAN_CHECK_NAMES

    document = report.to_dict()
    assert set(document) == {"timestamp", "overall_status", "checks", "metrics", "alerts"}
    assert datetime.fromisoformat(document["timestamp"]) is not None
    assert [check["name"] for check in document["checks"]] == _PLAN_CHECK_NAMES
    for check in document["checks"]:
        assert set(check) == {
            "name",
            "status",
            "details",
            "auto_repair_attempted",
            "auto_repair_result",
        }
        assert check["status"] == "pass"
        assert check["auto_repair_attempted"] is False
        assert check["auto_repair_result"] is None
    assert set(document["metrics"]) == _PLAN_METRIC_KEYS
    assert document["alerts"] == []

    # to_json is the same document, byte-for-byte parseable
    assert json.loads(report.to_json()) == document


@pytest.mark.integration
def test_background_collection_every_5_min(tmp_path: Path) -> None:
    """Snapshots land on the cadence (scaled 5 min -> 0.05 s) and stop cleanly."""
    assert DEFAULT_COLLECTION_INTERVAL_MINUTES == 5.0  # the plan's cadence
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    system = DiagnosticsSystem(
        context=_fresh_context(tmp_path, _build_store(tmp_path)),
        collector=collector,
        alert_engine=AlertEngine(checks=[]),
    )

    system.start_background(interval_minutes=0.05 / 60.0)  # 5 minutes scaled to 0.05 s
    time.sleep(0.2)
    assert len(system.collected) >= 2

    system.stop()
    assert not system.is_collecting
    count_at_stop = len(system.collected)
    time.sleep(0.1)
    assert len(system.collected) == count_at_stop  # cadence stopped cleanly

    # every collected snapshot was appended to the JSONL time series
    lines = [
        line
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == count_at_stop
    # restart works after a clean stop (and double-start is rejected)
    system.start_background(interval_minutes=0.05 / 60.0)
    with pytest.raises(RuntimeError):
        system.start_background(interval_minutes=0.05 / 60.0)
    system.stop()
    assert not system.is_collecting


@pytest.mark.integration
def test_report_includes_metrics_and_alerts(tmp_path: Path) -> None:
    """Latency/token/trust/conflict metrics and fired alerts ride the report."""
    store = _build_store(tmp_path)
    context = _fresh_context(tmp_path, store)
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    # yesterday's snapshot is the pre-today baseline for token_usage_today
    collector.record_tokens("memory_ops", 100)
    yesterday = datetime.now(UTC) - timedelta(days=1)
    collector.append_snapshot(collector.collect_snapshot(now=yesterday))
    # today: one memory-op latency sample (145 ms), 50 more tokens, trust, conflict
    collector.record_latency("memory_ops", 0.145)
    collector.record_latency("gate", 0.500)  # other ops stay out of the memory metric
    collector.record_tokens("memory_ops", 50)
    collector.record_trust_score(0.72)
    collector.record_trust_score(0.15)
    collector.record_conflict("entry-1")

    alert = Alert(
        rule_name="trace_store_full",
        severity="warning",
        message="trace store full: 10.5 GB used, limit 10.0 GB",
        fired_at=datetime.now(UTC),
        context={"total_bytes": 11 * 1024**3, "max_bytes": 10 * 1024**3},
    )
    system = DiagnosticsSystem(
        context=context,
        collector=collector,
        alert_engine=AlertEngine(checks=[lambda: alert]),
    )

    report = system.run_health_check()
    metrics = report.metrics

    assert metrics["memory_ops_latency_p50_ms"] == pytest.approx(145.0)
    assert metrics["memory_ops_latency_p95_ms"] == pytest.approx(145.0)
    assert metrics["token_usage_today"] == 50  # 150 cumulative - 100 pre-today baseline
    assert metrics["trust_score_avg"] == pytest.approx((0.72 + 0.15) / 2)
    assert metrics["trust_score_min"] == pytest.approx(0.15)
    assert metrics["conflict_rate_7d"] == pytest.approx(1 / (7 * 24))
    assert metrics["retrieval_recall_at_5"] == 0.0  # no retrieval counters recorded
    assert metrics["skills_active"] == 0
    assert metrics["skills_pending_approval"] == 0
    assert metrics["gepa_last_run"] is not None  # Pareto front file is present
    assert datetime.fromisoformat(metrics["gepa_last_run"]) is not None

    assert len(report.alerts) == 1
    assert report.alerts[0].rule_name == "trace_store_full"
    document = report.to_dict()
    assert document["alerts"] == [alert.to_dict()]
    # no backup archive exists -> the backup_freshness check warns -> degraded
    assert report.overall_status == "degraded"
    assert any(
        check.name == "backup_freshness" and check.status == "warn"
        for check in report.checks
    )


# ── P14.T6: /diagnostics slash command (hermes.gateway.health) ────────────────


def _healthy_system(tmp_path: Path) -> DiagnosticsSystem:
    """A diagnostics system whose six plan-5.1 checks all pass."""
    store = _build_store(tmp_path)
    context = _fresh_context(tmp_path, store)
    BackupManager(hermes_root=tmp_path).create_backup(now=datetime.now(UTC))
    return DiagnosticsSystem(
        context=context,
        collector=MetricsCollector(store_path=tmp_path / "metrics.jsonl"),
        alert_engine=AlertEngine(checks=[]),
    )


@pytest.mark.integration
def test_slash_command_quick(tmp_path: Path) -> None:
    """`/diagnostics quick` returns just the overall status one-liner."""
    system = _healthy_system(tmp_path)

    output = handle_diagnostics_command("quick", system)

    assert output == "overall status: healthy"  # the status and nothing else
    assert "\n" not in output

    # argument routing: bare /diagnostics is the FULL report (plan step 5),
    # unknown subcommands answer with the usage line instead of crashing
    bare = handle_diagnostics_command("", system)
    assert "overall status: healthy" in bare
    assert len(bare.splitlines()) > 1
    assert handle_diagnostics_command("nonsense", system).startswith(
        "usage: /diagnostics ["
    )


@pytest.mark.integration
def test_slash_command_full(tmp_path: Path) -> None:
    """`/diagnostics` returns the full report: status, checks, metrics, alerts."""
    system = _healthy_system(tmp_path)

    output = handle_diagnostics_command("full", system)
    lines = output.splitlines()

    assert lines[0].startswith("diagnostics report @ ")  # timestamped
    assert "overall status: healthy" in output
    for name in _PLAN_CHECK_NAMES:  # every check rendered with its status
        assert f"] {name}: " in output
    for metric in _PLAN_METRIC_KEYS:  # the plan 5.1 metrics block
        assert f"  {metric}=" in output
    assert "alerts: none" in output  # quiet engine -> no alert lines

    # an unhealthy check renders as a failed line, degrading the status
    broken = tmp_path / "memory-repo" / "global" / "facts" / "broken.md"
    broken.write_text("no frontmatter at all", encoding="utf-8")
    degraded = handle_diagnostics_command("full", system)
    assert "overall status: unhealthy" in degraded
    assert "[fail] memory_store_integrity" in degraded


@pytest.mark.integration
def test_slash_command_repair(tmp_path: Path) -> None:
    """`/diagnostics repair` triggers auto-repair and reports the outcomes."""
    store = _build_store(tmp_path)
    context = _fresh_context(tmp_path, store)
    repair_log = tmp_path / "repairs.log"
    system = DiagnosticsSystem(
        context=context,
        collector=MetricsCollector(store_path=tmp_path / "metrics.jsonl"),
        alert_engine=AlertEngine(checks=[]),
        repair=AutoRepair(
            RepairContext(
                store_root=store.root,
                repo_path=store.root,
                traces_root=tmp_path / "traces",
                pareto_path=context.pareto_path,
                backups_dir=tmp_path / "backups",
            ),
            log_path=repair_log,
        ),
    )
    # a fact file with no frontmatter fails memory_store_integrity...
    broken = store.root / "global" / "facts" / "broken.md"
    broken.write_text("no frontmatter at all", encoding="utf-8")
    before = system.run_health_check()
    integrity = next(check for check in before.checks if check.name == "memory_store_integrity")
    assert integrity.status == "fail"

    output = handle_diagnostics_command("repair", system)

    # ...and the repair pass writes the migration-default trust card: the
    # summary names the plan 5.4 issue key and the repair action, and the
    # P14.T4 backend logged the attempt with its "repaired" outcome
    assert "memory_store_integrity (missing_trust_cards)" in output
    assert "re-run trust-card migration" in output
    log_entries = [
        json.loads(line)
        for line in repair_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        entry["issue"] == "missing_trust_cards" and entry["status"] == "repaired"
        for entry in log_entries
    )

    # the next health check passes: every fact file parses again
    after = system.run_health_check()
    integrity_after = next(
        check for check in after.checks if check.name == "memory_store_integrity"
    )
    assert integrity_after.status == "pass"

    # single-issue form: the issue is gone now, so the repair is skipped
    again = handle_diagnostics_command("repair missing_trust_cards", system)
    assert "missing_trust_cards" in again and "skipped" in again
    unknown = handle_diagnostics_command("repair bogus_issue", system)
    assert unknown.startswith("unknown repair issue: bogus_issue")


# ── P14.T6: HTTP endpoints (gateway mode) ─────────────────────────────────────


def _get(address: tuple[str, int], path: str) -> http.client.HTTPResponse:
    """One real HTTP GET against the diagnostics server."""
    connection = http.client.HTTPConnection(address[0], address[1], timeout=5.0)
    connection.request("GET", path)
    response = connection.getresponse()
    connection.close()
    return response


@pytest.mark.integration
def test_http_health_endpoint(tmp_path: Path) -> None:
    """GET /health serves the fresh plan 5.1 report JSON over real HTTP."""
    system = _healthy_system(tmp_path)

    address = serve_health(system)
    try:
        response = _get(address, "/health")
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        document = json.loads(response.read().decode("utf-8"))
        assert document["overall_status"] == "healthy"
        assert [check["name"] for check in document["checks"]] == _PLAN_CHECK_NAMES
        assert set(document["metrics"]) == _PLAN_METRIC_KEYS
        assert datetime.fromisoformat(document["timestamp"]) is not None

        # unknown routes answer 404 with the route list, not a crash
        not_found = _get(address, "/definitely-not-a-route")
        assert not_found.status == 404
        body = json.loads(not_found.read().decode("utf-8"))
        assert body["error"] == "not found"
        assert body["routes"] == ["/health", "/metrics"]
    finally:
        stopped = stop_health_server(address)
    assert stopped is True  # the served server was shut down
    assert stop_health_server(address) is False  # idempotent second stop

    # the listening socket is actually closed after the stop
    with pytest.raises(OSError):
        _get(address, "/health")


@pytest.mark.integration
def test_http_metrics_endpoint(tmp_path: Path) -> None:
    """GET /metrics serves the latest snapshot + model metrics summary JSON."""
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    collector.record_latency("memory_ops", 0.145)
    collector.record_trust_score(0.72)
    collector.record_conflict("entry-1")
    snapshot = collector.collect_snapshot()
    collector.append_snapshot(snapshot)  # the persisted latest snapshot
    model_metrics = ModelMetricsCollector(ledger_path=tmp_path / "model_metrics.jsonl")
    model_metrics.record_generation("main-model", 300, 10.0)
    model_metrics.record_error("main-model")
    system = DiagnosticsSystem(
        context=_fresh_context(tmp_path, _build_store(tmp_path)),
        collector=collector,
        alert_engine=AlertEngine(checks=[]),
        model_metrics=model_metrics,
    )

    address = serve_health(system)
    try:
        response = _get(address, "/metrics")
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        document = json.loads(response.read().decode("utf-8"))

        assert set(document) == {"snapshot", "model_metrics"}
        assert document["snapshot"]["timestamp"] == snapshot.to_dict()["timestamp"]
        assert document["snapshot"]["trust_scores"] == [0.72]
        assert document["snapshot"]["conflict_rate_7d"] > 0.0
        assert document["model_metrics"]["generations"] == 1
        assert document["model_metrics"]["errors"] == 1
        assert document["model_metrics"]["tok_per_s_p50"] == pytest.approx(30.0)
    finally:
        assert stop_health_server(address) is True
