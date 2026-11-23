"""hermes-v2 diagnostics HTTP endpoints + /diagnostics command (plan P14.T6).

Two surfaces over the P14.T5 :class:`~hermes.diagnostics.system.
DiagnosticsSystem`:

- **HTTP endpoints** (plan Phase 14: "HTTP endpoint if running as gateway:
  ``/health``, ``/metrics``"). :func:`build_health_handler` /
  :func:`build_metrics_handler` are the WSGI-free minimal handler pair — each
  returns a zero-argument callable producing ``(status_code, json_body)`` so
  any serving stack (here: :mod:`http.server`; the gateway wiring can wrap the
  same callables) routes them without framework coupling. :func:`serve_health`
  runs both routes on an :class:`http.server.ThreadingHTTPServer` in a daemon
  thread (``port=0`` picks an ephemeral port; unknown paths answer 404);
  :func:`stop_health_server` shuts a server started by :func:`serve_health`
  back down. Handlers never raise: a crashing report/collection degrades to a
  ``500`` JSON error, mirroring the failure isolation the Phase 14 checks,
  alert engine and auto-repair already guarantee.

- **the ``/diagnostics`` slash command** (plan implementation steps 5-6):
  :func:`handle_diagnostics_command` — bare/``full`` renders the plan 5.1
  health report as text, ``quick`` returns just the ``overall_status``
  one-liner, and ``repair`` triggers auto-repair
  (``run_health_check(attempt_auto_repair=True)``; an explicit plan 5.4 issue
  key runs that single :meth:`~hermes.diagnostics.system.DiagnosticsSystem.
  auto_repair`).

Deviation (documented, plan P14.T6): the P13.T4 slash-command pattern put
``handle_*_command`` in ``hermes/core/agent.py`` with the dispatch registry —
but ``hermes/core/agent.py`` is frozen this wave, so the command logic lives
here in ``hermes/gateway/health.py`` until the wiring wave registers it on
the agent (see ``commands/diagnostics.md``). Tests call
:func:`handle_diagnostics_command` directly.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlsplit

from hermes.diagnostics.auto_repair import REPAIR_ISSUES
from hermes.diagnostics.metrics import MetricsSnapshot
from hermes.diagnostics.system import _REPAIR_BY_CHECK, DiagnosticsSystem, HealthReport

__all__ = [
    "HEALTH_ROUTE",
    "METRICS_ROUTE",
    "SUBCOMMANDS",
    "DiagnosticsHttpServer",
    "build_health_handler",
    "build_metrics_handler",
    "handle_diagnostics_command",
    "serve_health",
    "stop_health_server",
]

_LOGGER = logging.getLogger("hermes.gateway.health")

#: Plan Phase 14 route names (exact).
HEALTH_ROUTE: Final[str] = "/health"
METRICS_ROUTE: Final[str] = "/metrics"

#: The ``/diagnostics`` subcommand vocabulary (plan implementation steps 5-6
#: plus the auto-repair trigger; bare ``/diagnostics`` renders the full report).
SUBCOMMANDS: Final[tuple[str, ...]] = ("quick", "full", "repair")


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ── WSGI-free handler pair (plan P14.T6) ─────────────────────────────────────


def build_health_handler(diagnostics: DiagnosticsSystem) -> Callable[[], tuple[int, str]]:
    """``/health`` handler: ``(200, HealthReport.to_json())`` per call.

    Every call runs a fresh :meth:`DiagnosticsSystem.run_health_check`, so the
    endpoint always reflects the system's current state. A crashing report
    degrades to ``(500, <error json>)`` — the endpoint answers, never raises.
    """
    def handler() -> tuple[int, str]:
        try:
            report = diagnostics.run_health_check()
        except Exception as exc:  # noqa: BLE001 - endpoint must answer, not crash
            _LOGGER.exception("health endpoint report failed")
            return 500, json.dumps({"error": "health report failed", "detail": str(exc)})
        return 200, report.to_json()

    return handler


def build_metrics_handler(diagnostics: DiagnosticsSystem) -> Callable[[], tuple[int, str]]:
    """``/metrics`` handler: latest snapshot + model metrics summary JSON.

    The "latest" :class:`~hermes.diagnostics.metrics.MetricsSnapshot` is the
    newest persisted snapshot from the collector's JSONL time series; a fresh
    collection is substituted when nothing is persisted yet (or the series is
    unreadable). ``model_metrics`` is the P14.T7 trailing-window summary or
    ``null`` when no model-metrics collector is wired. A crashing collection
    degrades to ``(500, <error json>)``.
    """
    def handler() -> tuple[int, str]:
        try:
            document = _metrics_document(diagnostics)
        except Exception as exc:  # noqa: BLE001 - endpoint must answer, not crash
            _LOGGER.exception("metrics endpoint collection failed")
            return 500, json.dumps({"error": "metrics collection failed", "detail": str(exc)})
        return 200, json.dumps(document, ensure_ascii=False)

    return handler


def _metrics_document(diagnostics: DiagnosticsSystem) -> dict[str, Any]:
    """The ``/metrics`` response body (latest snapshot + model summary)."""
    history: list[MetricsSnapshot] = []
    try:
        history = diagnostics.collector.load_history()
    except Exception:  # noqa: BLE001 - unreadable series falls back to a fresh collect
        _LOGGER.exception("metrics history unreadable; collecting fresh snapshot")
    snapshot = history[-1] if history else diagnostics.collect_metrics()
    model_summary: dict[str, Any] | None = None
    if diagnostics.model_metrics is not None:
        model_summary = diagnostics.model_metrics.report().to_dict()
    return {"snapshot": snapshot.to_dict(), "model_metrics": model_summary}


# ── runnable server helper (http.server in a daemon thread) ─────────────────


class DiagnosticsHttpServer(ThreadingHTTPServer):
    """Threaded HTTP server routing :data:`HEALTH_ROUTE` / :data:`METRICS_ROUTE`.

    Carries the two handler callables built from the diagnostics system; any
    other path answers ``404``. Threads are daemonic so an abandoned server
    never keeps a process alive.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], diagnostics: DiagnosticsSystem) -> None:
        self.health_handler = build_health_handler(diagnostics)
        self.metrics_handler = build_metrics_handler(diagnostics)
        super().__init__(address, _DiagnosticsRequestHandler)


class _DiagnosticsRequestHandler(BaseHTTPRequestHandler):
    """GET router: ``/health`` + ``/metrics`` from the owning server, else 404."""

    server_version = "HermesDiagnostics/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib protocol name
        server = self.server
        assert isinstance(server, DiagnosticsHttpServer)  # noqa: S101 - wiring invariant
        route = urlsplit(self.path).path
        if route != "/" and route.endswith("/"):
            route = route.rstrip("/")
        routes: dict[str, Callable[[], tuple[int, str]]] = {
            HEALTH_ROUTE: server.health_handler,
            METRICS_ROUTE: server.metrics_handler,
        }
        handler = routes.get(route)
        if handler is None:
            status, body = 404, json.dumps(
                {"error": "not found", "routes": [HEALTH_ROUTE, METRICS_ROUTE]}
            )
        else:
            status, body = handler()
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Route request logs to the module logger (debug, not stderr noise)."""
        _LOGGER.debug("diagnostics http: %s", format % args)


#: Servers started by :func:`serve_health`, keyed by their bound address, so
#: :func:`stop_health_server` can shut them down again.
_ACTIVE_SERVERS: Final[dict[tuple[str, int], DiagnosticsHttpServer]] = {}
_SERVERS_LOCK = threading.Lock()


def serve_health(
    diagnostics: DiagnosticsSystem, host: str = "127.0.0.1", port: int = 0
) -> tuple[str, int]:
    """Start the diagnostics HTTP server on a daemon thread; return its address.

    Binds ``(host, port)`` immediately (``port=0`` lets the OS pick an
    ephemeral port — the returned address carries the resolved port), serves
    forever on a daemon thread and registers itself for
    :func:`stop_health_server`. Binding failures (port taken, permissions)
    propagate to the caller.
    """
    server = DiagnosticsHttpServer((host, port), diagnostics)
    bound = server.server_address
    resolved: tuple[str, int] = (str(bound[0]), int(bound[1]))
    thread = threading.Thread(
        target=server.serve_forever, name="hermes-diagnostics-http", daemon=True
    )
    with _SERVERS_LOCK:
        _ACTIVE_SERVERS[resolved] = server
    thread.start()
    return resolved


def stop_health_server(address: tuple[str, int]) -> bool:
    """Shut down the server :func:`serve_health` started at ``address``.

    Returns ``True`` when a server was found and stopped (idempotent callers
    get ``False`` for an already-stopped/unknown address).
    """
    with _SERVERS_LOCK:
        server = _ACTIVE_SERVERS.pop(address, None)
    if server is None:
        return False
    server.shutdown()
    server.server_close()
    return True


# ── /diagnostics slash command (plan implementation steps 5-6) ───────────────


def handle_diagnostics_command(args: str, diagnostics: DiagnosticsSystem) -> str:
    """Render the ``/diagnostics`` command output (plan P14.T6).

    ``args`` is everything after the command word: ``""``/``full`` renders the
    full plan 5.1 report as text, ``quick`` returns just the overall-status
    one-liner, and ``repair`` triggers auto-repair — bare ``repair`` runs
    ``run_health_check(attempt_auto_repair=True)`` and summarizes every repair
    it attempted, ``repair <issue>`` runs the single plan 5.4 repair for
    ``issue`` (unknown keys answer with the known-issue list, they never
    raise). Unknown subcommands answer with the usage line.
    """
    tokens = args.split()
    subcommand = tokens[0].lower() if tokens else "full"
    if subcommand == "quick":
        return f"overall status: {diagnostics.run_health_check().overall_status}"
    if subcommand == "full":
        return _format_report(diagnostics.run_health_check())
    if subcommand == "repair":
        issue = tokens[1].lower() if len(tokens) > 1 else None
        return _run_repair(diagnostics, issue)
    return f"usage: /diagnostics [{'|'.join(SUBCOMMANDS)}]"


def _format_report(report: HealthReport) -> str:
    """Human-readable rendering of the plan 5.1 report (full/repair output)."""
    lines = [
        f"diagnostics report @ {_as_utc(report.timestamp).isoformat()}",
        f"overall status: {report.overall_status}",
        "checks:",
    ]
    for check in report.checks:
        repair_note = ""
        if check.auto_repair_attempted:
            repair_note = f" (auto-repair: {check.auto_repair_result})"
        lines.append(f"  [{check.status}] {check.name}: {check.details}{repair_note}")
    lines.append("metrics:")
    for key, value in report.metrics.items():
        lines.append(f"  {key}={value}")
    if report.alerts:
        lines.append("alerts:")
        for alert in report.alerts:
            lines.append(f"  [{alert.severity}] {alert.rule_name}: {alert.message}")
    else:
        lines.append("alerts: none")
    return "\n".join(lines)


def _run_repair(diagnostics: DiagnosticsSystem, issue: str | None) -> str:
    """``repair`` output: one issue's repair, or a whole pass over the checks."""
    if issue is not None:
        try:
            result = diagnostics.auto_repair(issue)
        except ValueError:
            known = ", ".join(REPAIR_ISSUES)
            return f"unknown repair issue: {issue} (known issues: {known})"
        details = f" ({result.details})" if result.details else ""
        return f"repair {result.issue}: {result.status} — {result.action}{details}"
    report = diagnostics.run_health_check(attempt_auto_repair=True)
    lines = [
        f"diagnostics repair run @ {_as_utc(report.timestamp).isoformat()}",
        f"overall status: {report.overall_status}",
    ]
    attempted = [check for check in report.checks if check.auto_repair_attempted]
    if not attempted:
        lines.append("repairs: none needed (all checks pass)")
        return "\n".join(lines)
    lines.append("repairs:")
    for check in attempted:
        # annotate the line with the plan 5.4 issue key (the same column the
        # orchestrator's _REPAIR_BY_CHECK owns — single source of truth) so
        # `repair <issue>` can be copy-pasted from the summary
        issue = _REPAIR_BY_CHECK.get(check.name)
        label = f"{check.name} ({issue})" if issue is not None else check.name
        lines.append(f"  {label}: {check.auto_repair_result}")
    return "\n".join(lines)
