"""hermes-v2 user-friendly HTTP dashboard (plan P14.T10, plan Phase 14).

The HTML twin of :mod:`hermes.diagnostics.dashboard_cli` (plan P14.T10:
"HTTP: /dashboard endpoint (HTML page with charts)"):

- :func:`serve_dashboard` starts an :class:`http.server.ThreadingHTTPServer`
  on a daemon thread (the :func:`hermes.gateway.health.serve_health`
  pattern) and returns the resolved ``(host, port)``. Routes:

  - ``GET /`` — the dashboard page: HTML rendered from
    ``hermes/diagnostics/templates/dashboard.html`` (inline CSS, table rows
    for the plan 5.1 metrics, INLINE SVG sparklines for the memory-quality
    trends and a ``<script>`` chart-data placeholder for future JS charts —
    no external CDNs, the page works fully offline);
  - ``GET /health`` / ``GET /metrics`` — the plan P14.T6 JSON endpoints,
    REUSING the gateway handlers via
    :func:`hermes.gateway.health.build_health_handler` /
    :func:`hermes.gateway.health.build_metrics_handler`;
  - any other path answers ``404``.

- :func:`render_dashboard_html` is the pure render function the route calls
  (also usable directly, e.g. by tests or static exports). Like the CLI
  renderer it reads ONLY real components — the P14.T5
  :class:`~hermes.diagnostics.system.DiagnosticsSystem` health report (plan
  5.1 checks + metrics + fired alerts), the P14.T9
  :class:`~hermes.diagnostics.memory_quality_metrics.MemoryQualityTracker`
  weekly trends (rendered as sparklines), the P14.T8
  :class:`~hermes.diagnostics.learning_metrics.LearningMetricsLedger` trend
  report and the P14.T7
  :class:`~hermes.diagnostics.model_metrics.ModelMetricsCollector` report —
  and it never raises: a crashing component degrades to an ``unavailable``
  note.

- :func:`stop_dashboard_server` shuts a server started by
  :func:`serve_dashboard` back down (mirror of
  :func:`hermes.gateway.health.stop_health_server`).

Template contract
-----------------
``dashboard.html`` carries ``{placeholder}`` tokens — ``{generated_at}``,
``{health_status}``, ``{health_checks}``, ``{metrics_rows}``,
``{trend_rows}``, ``{alerts_items}``, ``{activity_items}``,
``{chart_data}`` — which :func:`render_dashboard_html` fills with exact
string replacements (never ``str.format``: the template's inline CSS braces
must survive untouched). All dynamic text is HTML-escaped.
"""

from __future__ import annotations

import html
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from hermes.diagnostics.learning_metrics import LearningMetricsLedger
from hermes.diagnostics.memory_quality_metrics import TRACKED_METRICS, MemoryQualityTracker
from hermes.diagnostics.model_metrics import ModelMetricsCollector
from hermes.diagnostics.system import DiagnosticsSystem, HealthReport
from hermes.gateway.health import build_health_handler, build_metrics_handler

__all__ = [
    "DASHBOARD_ROUTE",
    "TEMPLATE_PATH",
    "render_dashboard_html",
    "serve_dashboard",
    "sparkline_svg",
    "stop_dashboard_server",
]

_LOGGER = logging.getLogger("hermes.diagnostics.dashboard_http")

#: The dashboard page route (plan P14.T10: "/dashboard endpoint"). It IS the
#: site root, so ``/`` and ``/dashboard`` both serve it.
DASHBOARD_ROUTE: Final[str] = "/dashboard"

#: The vendored dashboard template (no external assets; works offline).
TEMPLATE_PATH: Final[Path] = Path(__file__).parent / "templates" / "dashboard.html"

#: Sparkline canvas geometry (fits inside a table row).
_SPARKLINE_WIDTH: Final[int] = 120
_SPARKLINE_HEIGHT: Final[int] = 28


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _esc(value: Any) -> str:
    """HTML-escape any rendered value (all dynamic text goes through this)."""
    return html.escape(str(value), quote=True)


def sparkline_svg(
    values: list[float], width: int = _SPARKLINE_WIDTH, height: int = _SPARKLINE_HEIGHT
) -> str:
    """One inline-SVG sparkline of ``values`` (chronological, offline-safe).

    The series is normalized to the canvas with a 2px margin; a constant or
    empty series renders a flat midline (an empty series renders a dimmed
    empty canvas so the row still aligns). Pure SVG — no scripts, no CDN.
    """
    margin = 2.0
    if values:
        low = min(values)
        high = max(values)
        span = high - low
        count = len(values)
        points: list[str] = []
        for index, value in enumerate(values):
            x = margin + (width - 2 * margin) * (index / (count - 1)) if count > 1 else width / 2.0
            y = (
                height - margin
                if span == 0.0
                else margin + (height - 2 * margin) * (1.0 - (value - low) / span)
            )
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        dot_x = points[-1].split(",")[0]
        dot_y = points[-1].split(",")[1]
        return (
            f'<svg class="spark" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="trend sparkline">'
            f'<polyline fill="none" stroke="#2563eb" stroke-width="1.5" '
            f'points="{polyline}"/>'
            f'<circle cx="{dot_x}" cy="{dot_y}" r="2" fill="#2563eb"/></svg>'
        )
    return (
        f'<svg class="spark spark-empty" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="no trend data">'
        f'<line x1="{margin}" y1="{height / 2:.1f}" x2="{width - margin}" '
        f'y2="{height / 2:.1f}" stroke="#cbd5e1" stroke-width="1.5"/></svg>'
    )


def _fmt(value: Any) -> str:
    """Compact rendering of one metric value (shared look with the CLI)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return str(value)


# ── section builders (mirror dashboard_cli's sections, HTML form) ────────────


def _health_rows(report: HealthReport) -> str:
    """The HEALTH section: one row per plan 5.1 check."""
    rows = [
        f"<tr><td>{_esc(check.name)}</td>"
        f'<td><span class="badge badge-{_esc(check.status)}">{_esc(check.status)}</span></td>'
        f"<td>{_esc(check.details)}</td></tr>"
        for check in report.checks
    ]
    return "".join(rows)


def _metrics_rows(report: HealthReport, model_metrics: ModelMetricsCollector | None) -> str:
    """The METRICS section: plan 5.1 metric rows + the P14.T7 model summary."""
    rows = [
        f"<tr><td>{_esc(key)}</td><td>{_esc(_fmt(value))}</td></tr>"
        for key, value in report.metrics.items()
    ]
    if model_metrics is None:
        rows.append('<tr><td>model metrics</td><td>not wired</td></tr>')
        return "".join(rows)
    try:
        summary = model_metrics.report().to_dict()
    except Exception as exc:  # noqa: BLE001 - advisory dashboard
        rows.append(
            f"<tr><td>model metrics</td><td>unavailable ({_esc(exc)})</td></tr>"
        )
    else:
        for key in (
            "tok_per_s_p50",
            "tok_per_s_p95",
            "quality_trend",
            "error_rate",
            "fallback_rate",
            "generations",
            "errors",
            "fallbacks",
        ):
            rows.append(
                f'<tr><td>model_{key}</td><td>{_esc(_fmt(summary[key]))}</td></tr>'
            )
    return "".join(rows)


def _trend_rows(
    quality_tracker: MemoryQualityTracker | None,
    learning_ledger: LearningMetricsLedger | None,
) -> str:
    """The TRENDS section: sparkline per memory-quality trend + learning trends."""
    rows: list[str] = []
    if quality_tracker is None:
        rows.append("<tr><td colspan=3>memory quality trends: not wired</td></tr>")
    else:
        trends = quality_tracker.trends()
        for metric in TRACKED_METRICS:
            trend = trends[metric]
            if trend.values:
                joined = ", ".join(_fmt(v) for v in trend.values)
                rows.append(
                    f"<tr><td>{_esc(metric)}</td>"
                    f"<td>{sparkline_svg(trend.values)}</td>"
                    f"<td>{_esc(trend.direction)} ({_esc(joined)})</td></tr>"
                )
            else:
                rows.append(
                    f"<tr><td>{_esc(metric)}</td><td>{sparkline_svg([])}</td>"
                    f"<td>no data yet</td></tr>"
                )
    if learning_ledger is None:
        rows.append("<tr><td colspan=3>learning loop trends: not wired</td></tr>")
    else:
        report = learning_ledger.trend_report()
        if not report.components:
            rows.append(
                "<tr><td colspan=3>learning loop: no optimization runs recorded</td></tr>"
            )
        for name, component_trend in report.components.items():
            rows.append(
                f"<tr><td>learning {_esc(name)}</td>"
                f"<td>{_esc(component_trend.trend)}</td>"
                f"<td>avg_improvement={_esc(_fmt(component_trend.avg_improvement))} "
                f"consecutive_no_improvement={component_trend.consecutive_no_improvement} "
                f"runs={component_trend.runs}</td></tr>"
            )
    return "".join(rows)


def _alerts_items(report: HealthReport) -> str:
    """The ALERTS section: a badge item per fired alert (or a quiet note)."""
    if not report.alerts:
        return '<li class="quiet">alerts: none</li>'
    items = [
        f'<li><span class="badge badge-{_esc(alert.severity)}">{_esc(alert.severity)}</span> '
        f"<strong>{_esc(alert.rule_name)}</strong>: {_esc(alert.message)}</li>"
        for alert in report.alerts
    ]
    return "".join(items)


def _activity_items(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None,
    learning_ledger: LearningMetricsLedger | None,
    model_metrics: ModelMetricsCollector | None,
) -> str:
    """The RECENT ACTIVITY section: the latest observable component events."""
    items: list[str] = []
    try:
        history = diagnostics.collector.load_history()
    except Exception:  # noqa: BLE001 - unreadable series hides, never breaks
        history = []
    if history:
        for snapshot in history[-3:]:
            tokens = sum(snapshot.token_usage.values())
            items.append(
                f"<li>metrics snapshot @ {_esc(_as_utc(snapshot.timestamp).isoformat())} "
                f"(tokens={tokens}, conflicts_total="
                f"{snapshot.counters.get('conflicts_total', 0)})</li>"
            )
    else:
        items.append("<li>metrics: no snapshots collected yet</li>")
    if quality_tracker is not None:
        try:
            points = quality_tracker.load_history()
        except Exception:  # noqa: BLE001 - advisory dashboard
            points = []
        for point in points[-3:]:
            items.append(
                f"<li>quality week @ {_esc(_as_utc(point.week_start).isoformat())} "
                f"(entries={point.entry_count}, trust_avg={_esc(_fmt(point.trust_avg))})</li>"
            )
        if not points:
            items.append("<li>quality: no weekly points recorded yet</li>")
    if learning_ledger is not None:
        report = learning_ledger.trend_report()
        for name, trend in report.components.items():
            items.append(
                f"<li>learning {_esc(name)}: {trend.runs} run(s) in window, "
                f"trend={_esc(trend.trend)}</li>"
            )
        if not report.components:
            items.append("<li>learning: no optimization runs recorded yet</li>")
    if model_metrics is not None:
        summary = model_metrics.report()
        items.append(
            f"<li>model window {summary.window_hours:g}h: generations="
            f"{summary.generations} errors={summary.errors} fallbacks={summary.fallbacks}</li>"
        )
    return "".join(items)


# ── template rendering ───────────────────────────────────────────────────────


def _fill_template(template: str, mapping: dict[str, str]) -> str:
    """Exact ``{token}`` replacements (NOT str.format — CSS braces survive)."""
    rendered = template
    for token, value in mapping.items():
        rendered = rendered.replace("{" + token + "}", value)
    return rendered


def render_dashboard_html(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None = None,
    learning_ledger: LearningMetricsLedger | None = None,
    model_metrics: ModelMetricsCollector | None = None,
) -> str:
    """Render the dashboard page from the vendored template (pure function).

    Fills the template's ``{placeholders}`` with server-rendered HTML: the
    plan 5.1 health report (status + checks), the metrics table, the trend
    table with inline-SVG sparklines, the alert list and the recent-activity
    list, plus a JSON ``<script>`` chart-data placeholder (escaped so it can
    never close its own tag). All dynamic text is HTML-escaped.
    """
    report = diagnostics.run_health_check()
    trend_values: dict[str, list[float]] = {}
    if quality_tracker is not None:
        try:
            trend_values = {
                metric: trend.values
                for metric, trend in quality_tracker.trends().items()
                if trend.values
            }
        except Exception:  # noqa: BLE001 - chart data is decorative
            trend_values = {}
    chart_data = json.dumps(
        {
            "overall_status": report.overall_status,
            "metrics": {key: value for key, value in report.metrics.items()},
            "trends": trend_values,
        },
        default=str,
    ).replace("<", "\\u003c")
    try:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"dashboard template unreadable: {TEMPLATE_PATH}") from exc
    return _fill_template(
        template,
        {
            "generated_at": _esc(_as_utc(report.timestamp).isoformat()),
            "health_status": _esc(report.overall_status),
            "health_checks": _health_rows(report),
            "metrics_rows": _metrics_rows(report, model_metrics),
            "trend_rows": _trend_rows(quality_tracker, learning_ledger),
            "alerts_items": _alerts_items(report),
            "activity_items": _safe_html(
                lambda: _activity_items(
                    diagnostics, quality_tracker, learning_ledger, model_metrics
                ),
                "recent activity",
            ),
            "chart_data": chart_data,
        },
    )


def _safe_html(render: Callable[[], str], label: str) -> str:
    """Failure isolation for one section (advisory dashboard, mirrors the CLI)."""
    try:
        return render()
    except Exception as exc:  # noqa: BLE001 - the dashboard never raises
        return _esc(f"{label}: unavailable ({exc})")


# ── the HTTP server (serve_health pattern) ───────────────────────────────────


class DashboardHttpServer(ThreadingHTTPServer):
    """Threaded HTTP server routing ``/`` (+ ``/dashboard``), /health, /metrics.

    Carries the HTML dashboard renderer inputs and the two reused P14.T6 JSON
    handlers; threads are daemonic so an abandoned server never keeps the
    process alive.
    """

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        diagnostics: DiagnosticsSystem,
        quality_tracker: MemoryQualityTracker | None,
        learning_ledger: LearningMetricsLedger | None,
        model_metrics: ModelMetricsCollector | None,
    ) -> None:
        self.render_page: Callable[[], str] = lambda: render_dashboard_html(
            diagnostics, quality_tracker, learning_ledger, model_metrics
        )
        self.health_handler = build_health_handler(diagnostics)
        self.metrics_handler = build_metrics_handler(diagnostics)
        super().__init__(address, _DashboardRequestHandler)


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    """GET router: the dashboard page + the reused diagnostics JSON routes."""

    server_version = "HermesDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib protocol name
        server = self.server
        assert isinstance(server, DashboardHttpServer)  # noqa: S101 - wiring invariant
        route = urlsplit(self.path).path
        if route != "/" and route.endswith("/"):
            route = route.rstrip("/")
        html_routes: dict[str, Callable[[], str]] = {
            "/": server.render_page,
            DASHBOARD_ROUTE: server.render_page,
        }
        json_routes: dict[str, Callable[[], tuple[int, str]]] = {
            "/health": server.health_handler,
            "/metrics": server.metrics_handler,
        }
        if route in html_routes:
            try:
                page = html_routes[route]()
                status = 200
                payload = page.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            except Exception as exc:  # noqa: BLE001 - endpoint answers, never crashes
                _LOGGER.exception("dashboard page render failed")
                status, content_type = 500, "application/json; charset=utf-8"
                payload = json.dumps(
                    {"error": "dashboard render failed", "detail": str(exc)}
                ).encode("utf-8")
        elif route in json_routes:
            status, body = json_routes[route]()
            payload, content_type = body.encode("utf-8"), "application/json; charset=utf-8"
        else:
            status = 404
            content_type = "application/json; charset=utf-8"
            payload = json.dumps(
                {"error": "not found", "routes": ["/", DASHBOARD_ROUTE, "/health", "/metrics"]}
            ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Route request logs to the module logger (debug, not stderr noise)."""
        _LOGGER.debug("dashboard http: %s", format % args)


#: Servers started by :func:`serve_dashboard`, keyed by bound address.
_ACTIVE_SERVERS: Final[dict[tuple[str, int], DashboardHttpServer]] = {}
_SERVERS_LOCK = threading.Lock()


def serve_dashboard(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None = None,
    learning_ledger: LearningMetricsLedger | None = None,
    model_metrics: ModelMetricsCollector | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[str, int]:
    """Start the dashboard HTTP server on a daemon thread; return its address.

    Binds ``(host, port)`` immediately (``port=0`` lets the OS pick an
    ephemeral port — the returned address carries the resolved port) and
    serves ``GET /`` (the HTML dashboard), ``GET /health`` and
    ``GET /metrics`` forever. Binding failures propagate to the caller;
    :func:`stop_dashboard_server` shuts the server down again.
    """
    server = DashboardHttpServer(
        (host, port), diagnostics, quality_tracker, learning_ledger, model_metrics
    )
    bound = server.server_address
    resolved: tuple[str, int] = (str(bound[0]), int(bound[1]))
    thread = threading.Thread(
        target=server.serve_forever, name="hermes-dashboard-http", daemon=True
    )
    with _SERVERS_LOCK:
        _ACTIVE_SERVERS[resolved] = server
    thread.start()
    return resolved


def stop_dashboard_server(address: tuple[str, int]) -> bool:
    """Shut down the server :func:`serve_dashboard` started at ``address``.

    Returns ``True`` when a server was found and stopped (``False`` for an
    already-stopped/unknown address — idempotent callers).
    """
    with _SERVERS_LOCK:
        server = _ACTIVE_SERVERS.pop(address, None)
    if server is None:
        return False
    server.shutdown()
    server.server_close()
    return True
