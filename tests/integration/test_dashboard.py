"""P14.T10 acceptance: the user-friendly dashboard, CLI + HTTP (plan Phase 14).

Integration tests over the real Phase 14 wiring (a git-backed MemoryStore,
a real MetricsCollector JSONL time series, the default alert engine,
MemoryQualityTracker, LearningMetricsLedger, ModelMetricsCollector — all on
per-test temp stores):

- the CLI render carries every section (HEALTH / METRICS / TRENDS / ALERTS /
  RECENT ACTIVITY) with the real health checks, the plan 5.1 metrics block,
  the weekly quality trends, the learning-loop trends and the fired alert;
- the interactive loop is driven through the INJECTED input stream (never
  raw keys) and navigates the page system;
- the HTTP server serves the HTML dashboard on ``GET /`` (200 + section
  headers + metric values) plus the reused P14.T6 JSON routes
  (``/health``, ``/metrics``) against an ephemeral 127.0.0.1 port;
- every key metric string is visible in BOTH renders (text + HTML).
"""

from __future__ import annotations

import http.client
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.diagnostics.alerts import AlertEngine, rule_backup_stale, rule_trace_store_full
from hermes.diagnostics.dashboard_cli import (
    pages,
    render_dashboard_text,
    render_page,
    run_cli_dashboard,
)
from hermes.diagnostics.dashboard_http import (
    render_dashboard_html,
    serve_dashboard,
    stop_dashboard_server,
)
from hermes.diagnostics.health_checks import HealthCheckContext
from hermes.diagnostics.learning_metrics import LearningMetricsLedger
from hermes.diagnostics.memory_quality_metrics import MemoryQualityTracker
from hermes.diagnostics.metrics import MetricsCollector
from hermes.diagnostics.model_metrics import ModelMetricsCollector
from hermes.diagnostics.system import DiagnosticsSystem
from hermes.memory.store import MemoryStore

_NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)

#: The section headers the plan P14.T10 dashboard must show.
SECTION_HEADERS = ("HEALTH", "METRICS", "TRENDS", "ALERTS", "RECENT ACTIVITY")

#: Key metric strings that must be visible in BOTH renders (plan P14.T10
#: acceptance: "all metrics, trends, alerts visible"): the plan 5.1 metrics
#: block keys, the P14.T7 model summary fields and the P14.T9 trend names.
KEY_METRIC_STRINGS = (
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
    "tok_per_s_p50",
    "tok_per_s_p95",
    "error_rate",
    "fallback_rate",
    "trust_avg",
    "trust_p10",
    "trust_p50",
    "trust_p90",
    "archive_count_this_week",
    "recall_at_5",
    "extraction",
)


def _wired(
    tmp_path: Path,
) -> tuple[DiagnosticsSystem, MemoryStore, MemoryQualityTracker, LearningMetricsLedger, ModelMetricsCollector]:
    """Real Phase 14 components on per-test temp stores, with recorded data."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("Dashboard entry one: pnpm is the package manager", type="Project", source="user-explicit")
    store.add("Dashboard entry two: deploy scripts live in scripts/", type="Reference", source="agent-inferred")

    collector = MetricsCollector(store_path=tmp_path / "diagnostics" / "metrics.jsonl", now=lambda: _NOW)
    collector.record_latency("memory_ops", 0.145)
    collector.record_latency("memory_ops", 0.380)
    collector.record_tokens("memory_ops", 12450)
    collector.record_trust_score(0.9)
    collector.record_trust_score(0.2)
    collector.record_conflict("entry-x", now=_NOW)
    collector.append_snapshot(collector.collect_snapshot())

    pareto_path = tmp_path / "optimization" / "pareto.json"
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    pareto_path.write_text("{}", encoding="utf-8")
    context = HealthCheckContext(
        store_root=store.root,
        repo_path=store.root,
        probe=lambda: True,
        traces_root=tmp_path / "traces",
        pareto_path=pareto_path,
        backups_dir=tmp_path / "backups",
    )

    quality_tracker = MemoryQualityTracker(
        store=store,
        ledger_path=tmp_path / "diagnostics" / "memory_quality.jsonl",
        recall_probe=lambda: 0.81,
    )
    quality_tracker.record_weekly(now=_NOW)

    learning_ledger = LearningMetricsLedger(
        ledger_path=tmp_path / "diagnostics" / "learning_metrics.jsonl", now=lambda: _NOW
    )
    learning_ledger.record_run("extraction", "run-1", 0.70, 0.80, deployed=True, finished_at=_NOW)

    model_metrics = ModelMetricsCollector(
        ledger_path=tmp_path / "diagnostics" / "model_metrics.jsonl", now=lambda: _NOW
    )
    model_metrics.record_generation("main-model", 120, 2.0)
    model_metrics.record_quality("main-model", 0.84)

    # Alert engine with the documented curried rules (the system.py default
    # wiring pattern): with an empty backups dir the backup_stale rule fires
    # ("no backup ever recorded") so the ALERTS section has content.
    alert_engine = AlertEngine(
        checks=[
            lambda: rule_trace_store_full(traces_root=context.traces_root),
            lambda: rule_backup_stale(None),
        ]
    )
    diagnostics = DiagnosticsSystem(
        context=context,
        collector=collector,
        alert_engine=alert_engine,
    )
    return diagnostics, store, quality_tracker, learning_ledger, model_metrics


@pytest.mark.integration
def test_cli_dashboard_renders(tmp_path: Path) -> None:
    """The text dashboard renders all sections; the loop navigates via the
    injected stream (never raw keys) through the page system."""
    diagnostics, _store, quality_tracker, learning_ledger, model_metrics = _wired(tmp_path)

    text = render_dashboard_text(diagnostics, quality_tracker, learning_ledger, model_metrics)
    for header in SECTION_HEADERS:
        assert header in text, f"missing section header: {header}"
    assert "overall status:" in text
    # real components visible: a health check, a metric, a trend, the alert
    assert "memory_store_integrity" in text
    assert "memory_ops_latency_p50_ms=" in text
    assert "trust_avg" in text
    assert "backup_stale" in text  # fired by the default engine (no backups yet)

    # the page system: one page per section + the overview, navigable
    assert pages() == ("overview", "health", "metrics", "trends", "alerts", "activity")
    trends_page = render_page(
        "trends", diagnostics, quality_tracker, learning_ledger, model_metrics
    )
    assert "\nTRENDS\n" in trends_page
    assert "\nHEALTH\n" not in trends_page  # a single-section page

    # interactive loop with an INJECTED input stream: overview -> right ->
    # health -> left -> overview -> quit (no raw keyboard access in tests)
    output = io.StringIO()
    final_render = run_cli_dashboard(
        diagnostics,
        quality_tracker,
        learning_ledger,
        model_metrics,
        input_stream=iter(["right", "left", "quit"]),
        output=output,
    )
    rendered = output.getvalue()
    assert rendered.count("hermes-v2 dashboard") == 3  # overview, health, overview
    assert final_render in rendered
    assert "overall status:" in final_render


@pytest.mark.integration
def test_http_dashboard_renders(tmp_path: Path) -> None:
    """GET / returns 200 text/html with all sections + metric values; the
    reused /health and /metrics JSON routes answer; unknown paths 404."""
    diagnostics, _store, quality_tracker, learning_ledger, model_metrics = _wired(tmp_path)
    address = serve_dashboard(diagnostics, quality_tracker, learning_ledger, model_metrics)
    try:
        host, port = address
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert response.getheader("Content-Type", "").startswith("text/html")
        for header in SECTION_HEADERS:
            assert f"<h2>{header}</h2>" in body, f"missing HTML section: {header}"
        assert "memory_ops_latency_p50_ms" in body
        assert "overall status" in body
        # inline SVG sparkline for the recorded weekly trend, no external CDN
        assert "<svg" in body
        assert "http://" not in body and "https://" not in body

        # the reused P14.T6 JSON endpoints (gateway/health handlers)
        connection.request("GET", "/health")
        health_response = connection.getresponse()
        health_body = json.loads(health_response.read().decode("utf-8"))
        assert health_response.status == 200
        assert health_body["overall_status"] in {"healthy", "degraded", "unhealthy"}

        connection.request("GET", "/metrics")
        metrics_response = connection.getresponse()
        metrics_body = json.loads(metrics_response.read().decode("utf-8"))
        assert metrics_response.status == 200
        assert "snapshot" in metrics_body

        connection.request("GET", "/dashboard")
        alias_response = connection.getresponse()
        alias_body = alias_response.read().decode("utf-8")
        assert alias_response.status == 200
        assert "<h2>HEALTH</h2>" in alias_body

        connection.request("GET", "/unknown")
        unknown_response = connection.getresponse()
        assert unknown_response.status == 404
        unknown_response.read()
        connection.close()
    finally:
        assert stop_dashboard_server(address) is True
        assert stop_dashboard_server(address) is False  # idempotent

    # the pure render function agrees with the served page
    html_text = render_dashboard_html(diagnostics, quality_tracker, learning_ledger, model_metrics)
    assert "<h2>HEALTH</h2>" in html_text


@pytest.mark.integration
def test_all_metrics_visible(tmp_path: Path) -> None:
    """Every key metric string (plan 5.1 block + model + trends) appears in
    BOTH the CLI text render and the HTTP HTML render."""
    diagnostics, _store, quality_tracker, learning_ledger, model_metrics = _wired(tmp_path)
    text = render_dashboard_text(diagnostics, quality_tracker, learning_ledger, model_metrics)
    html_text = render_dashboard_html(diagnostics, quality_tracker, learning_ledger, model_metrics)
    for metric in KEY_METRIC_STRINGS:
        assert metric in text, f"metric missing from CLI dashboard: {metric}"
        assert metric in html_text, f"metric missing from HTTP dashboard: {metric}"
