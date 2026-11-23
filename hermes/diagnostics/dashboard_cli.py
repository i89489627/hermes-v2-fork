"""hermes-v2 user-friendly CLI dashboard (plan P14.T10, plan Phase 14).

A rich-free, plain-text dashboard over the real Phase 14 components (plan
P14.T10: "CLI: hermes-dashboard (interactive TUI)" showing "health, metrics,
trends, alerts, recent activity"):

- :func:`render_dashboard_text` — the whole dashboard as one string with the
  five plan sections (``HEALTH`` / ``METRICS`` / ``TRENDS`` / ``ALERTS`` /
  ``RECENT ACTIVITY``), reading ONLY real components:
  the :class:`~hermes.diagnostics.system.DiagnosticsSystem` health report
  (plan 5.1 checks + metrics block + fired alerts), the
  :class:`~hermes.diagnostics.memory_quality_metrics.MemoryQualityTracker`
  weekly trends (:meth:`~MemoryQualityTracker.trends`), the
  :class:`~hermes.diagnostics.learning_metrics.LearningMetricsLedger` trend
  report (:meth:`~LearningMetricsLedger.trend_report`) and the
  :class:`~hermes.diagnostics.model_metrics.ModelMetricsCollector` report.
  Every component is exception-isolated: a crashing component renders an
  ``unavailable`` note — the dashboard is advisory and never raises.
- the page system — :func:`pages` returns the ordered page names the
  interactive loop navigates (``overview`` / ``health`` / ``metrics`` /
  ``trends`` / ``alerts`` / ``activity``) and :func:`render_page` renders one
  of them; :func:`navigate` is the pure navigation helper
  (``left``/``right`` move a page, ``quit`` exits, unknown keys stay).
- :func:`run_cli_dashboard` — the interactive loop. Key input comes from
  ``msvcrt.kbhit``/``getwch`` on Windows (a :func:`_read_key_windows`
  poll loop) and from a ``termios``/``tty`` cbreak one-shot read on POSIX
  (:func:`_read_key_posix`, documented fallback — imported lazily so the
  module loads on any platform). For TESTS the loop never touches raw keys:
  an injected ``input_stream`` (an iterable of key tokens, one per line)
  drives the same navigation — the loop renders the current page to
  ``output`` after every key and returns on ``quit`` or end of stream.

The loop is deliberately minimal and library-free (the fork base has no TUI
dependency for v2): render current page -> read one key -> navigate -> repeat.
Arrow keys map to page navigation (``left``/``right``/``up``/``down``),
``q``/``quit``/``esc`` exit.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from typing import Any, Final

from hermes.diagnostics.learning_metrics import LearningMetricsLedger
from hermes.diagnostics.memory_quality_metrics import TRACKED_METRICS, MemoryQualityTracker
from hermes.diagnostics.model_metrics import ModelMetricsCollector
from hermes.diagnostics.system import DiagnosticsSystem, HealthReport

__all__ = [
    "DASHBOARD_PAGES",
    "KEYS_NEXT",
    "KEYS_PREV",
    "KEYS_QUIT",
    "run_cli_dashboard",
    "navigate",
    "pages",
    "render_dashboard_text",
    "render_page",
]

#: The ordered pages of the interactive dashboard (the sections of
#: :func:`render_dashboard_text` plus the full ``overview`` first page).
DASHBOARD_PAGES: Final[tuple[str, ...]] = (
    "overview",
    "health",
    "metrics",
    "trends",
    "alerts",
    "activity",
)

#: Key tokens that move to the next page (arrow right/down and aliases).
KEYS_NEXT: Final[frozenset[str]] = frozenset({"right", "down", "l", "j", "next"})

#: Key tokens that move to the previous page (arrow left/up and aliases).
KEYS_PREV: Final[frozenset[str]] = frozenset({"left", "up", "h", "k", "prev"})

#: Key tokens that exit the dashboard.
KEYS_QUIT: Final[frozenset[str]] = frozenset({"q", "quit", "esc"})

#: Section headers of the full dashboard, in render order.
SECTION_HEADERS: Final[tuple[str, ...]] = (
    "HEALTH",
    "METRICS",
    "TRENDS",
    "ALERTS",
    "RECENT ACTIVITY",
)

_HORIZONTAL_RULE: Final[str] = "=" * 62


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _fmt(value: Any) -> str:
    """Compact JSON-ish rendering of one metric value for the text dashboard."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _safe(render: Callable[[], str], label: str) -> str:
    """Run one section renderer with failure isolation (advisory dashboard)."""
    try:
        return render()
    except Exception as exc:  # noqa: BLE001 - the dashboard never raises
        return f"{label}: unavailable ({exc})"


# ── section renderers (real components, one section each) ────────────────────


def _render_health(report: HealthReport) -> str:
    """The HEALTH section: overall status + every plan 5.1 check."""
    lines = [
        f"overall status: {report.overall_status}",
    ]
    for check in report.checks:
        repair_note = ""
        if check.auto_repair_attempted:
            repair_note = f" (auto-repair: {check.auto_repair_result})"
        lines.append(f"  [{check.status}] {check.name}: {check.details}{repair_note}")
    return "\n".join(lines)


def _render_metrics(report: HealthReport, model_metrics: ModelMetricsCollector | None) -> str:
    """The METRICS section: the plan 5.1 metrics block + model metrics."""
    lines = [f"{key}={_fmt(value)}" for key, value in report.metrics.items()]
    if model_metrics is None:
        lines.append("model metrics: not wired")
    else:
        try:
            summary = model_metrics.report().to_dict()
        except Exception as exc:  # noqa: BLE001 - advisory dashboard
            lines.append(f"model metrics: unavailable ({exc})")
        else:
            lines.append(
                f"model: tok_per_s_p50={_fmt(summary['tok_per_s_p50'])} "
                f"tok_per_s_p95={_fmt(summary['tok_per_s_p95'])} "
                f"quality_trend={summary['quality_trend']} "
                f"error_rate={_fmt(summary['error_rate'])} "
                f"fallback_rate={_fmt(summary['fallback_rate'])} "
                f"generations={summary['generations']} errors={summary['errors']} "
                f"fallbacks={summary['fallbacks']}"
            )
    return "\n".join(lines)


def _render_trends(
    quality_tracker: MemoryQualityTracker | None,
    learning_ledger: LearningMetricsLedger | None,
) -> str:
    """The TRENDS section: memory-quality weekly trends + learning-loop trends."""
    lines: list[str] = []
    if quality_tracker is None:
        lines.append("memory quality trends: not wired")
    else:
        trends = quality_tracker.trends()
        for metric in TRACKED_METRICS:
            trend = trends[metric]
            if not trend.values:
                lines.append(f"  {metric}: no data yet")
            else:
                values = ", ".join(_fmt(value) for value in trend.values)
                lines.append(f"  {metric}: {trend.direction} [{values}]")
    if learning_ledger is None:
        lines.append("learning loop trends: not wired")
    else:
        report = learning_ledger.trend_report()
        if not report.components:
            lines.append("  learning loop: no optimization runs recorded")
        for name, component_trend in report.components.items():
            lines.append(
                f"  learning {name}: {component_trend.trend} "
                f"avg_improvement={_fmt(component_trend.avg_improvement)} "
                f"consecutive_no_improvement={component_trend.consecutive_no_improvement} "
                f"runs={component_trend.runs}"
            )
    return "\n".join(lines)


def _render_alerts(report: HealthReport) -> str:
    """The ALERTS section: every fired alert (severity, rule, message)."""
    if not report.alerts:
        return "alerts: none"
    lines = [f"  [{alert.severity}] {alert.rule_name}: {alert.message}" for alert in report.alerts]
    return "\n".join(lines)


def _render_activity(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None,
    learning_ledger: LearningMetricsLedger | None,
    model_metrics: ModelMetricsCollector | None,
) -> str:
    """The RECENT ACTIVITY section: the latest observable component events."""
    lines: list[str] = []
    try:
        history = diagnostics.collector.load_history()
    except Exception:  # noqa: BLE001 - unreadable series hides, never breaks
        history = []
    if history:
        for snapshot in history[-3:]:
            tokens = sum(snapshot.token_usage.values())
            lines.append(
                f"  metrics snapshot @ {_as_utc(snapshot.timestamp).isoformat()} "
                f"(tokens={tokens}, conflicts_total="
                f"{snapshot.counters.get('conflicts_total', 0)})"
            )
    else:
        lines.append("  metrics: no snapshots collected yet")
    if quality_tracker is not None:
        try:
            points = quality_tracker.load_history()
        except Exception:  # noqa: BLE001 - advisory dashboard
            points = []
        if points:
            for point in points[-3:]:
                lines.append(
                    f"  quality week @ {_as_utc(point.week_start).isoformat()} "
                    f"(entries={point.entry_count}, trust_avg={_fmt(point.trust_avg)})"
                )
        else:
            lines.append("  quality: no weekly points recorded yet")
    if learning_ledger is not None:
        report = learning_ledger.trend_report()
        if report.components:
            for name, trend in report.components.items():
                lines.append(
                    f"  learning {name}: {trend.runs} run(s) in window, trend={trend.trend}"
                )
        else:
            lines.append("  learning: no optimization runs recorded yet")
    if model_metrics is not None:
        summary = model_metrics.report()
        lines.append(
            f"  model window {summary.window_hours:g}h: generations={summary.generations} "
            f"errors={summary.errors} fallbacks={summary.fallbacks}"
        )
    return "\n".join(lines)


# ── public render API ────────────────────────────────────────────────────────


def pages() -> tuple[str, ...]:
    """The ordered page names the interactive CLI navigates (page system)."""
    return DASHBOARD_PAGES


def navigate(page_list: Iterable[str], current: str, key: str) -> str | None:
    """Pure navigation helper: the page after ``key``, or ``None`` to quit.

    ``right``/``down`` move forward, ``left``/``up`` move backward (wrapping
    at the ends), ``quit`` (and its aliases) returns ``None``, and any
    unknown key keeps ``current`` (which also answers for an empty page
    list). The interactive loop is exactly "render :func:`navigate`'s page,
    repeat".
    """
    ordered = list(page_list)
    if current not in ordered:
        current = ordered[0] if ordered else current
    token = key.strip().lower()
    if token in KEYS_QUIT:
        return None
    if not ordered:
        return current
    index = ordered.index(current)
    if token in KEYS_NEXT:
        return ordered[(index + 1) % len(ordered)]
    if token in KEYS_PREV:
        return ordered[(index - 1) % len(ordered)]
    return current


def render_dashboard_text(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None = None,
    learning_ledger: LearningMetricsLedger | None = None,
    model_metrics: ModelMetricsCollector | None = None,
) -> str:
    """The full plain-text dashboard: HEALTH/METRICS/TRENDS/ALERTS/ACTIVITY.

    Reads real components only (health report, quality trends, learning
    trends, model metrics); each section is failure-isolated so one broken
    component renders an ``unavailable`` note instead of crashing the
    dashboard.
    """
    report = diagnostics.run_health_check()
    sections = [
        ("HEALTH", _render_health(report)),
        ("METRICS", _render_metrics(report, model_metrics)),
        (
            "TRENDS",
            _safe(
                lambda: _render_trends(quality_tracker, learning_ledger),
                "trends",
            ),
        ),
        ("ALERTS", _render_alerts(report)),
        (
            "RECENT ACTIVITY",
            _safe(
                lambda: _render_activity(
                    diagnostics, quality_tracker, learning_ledger, model_metrics
                ),
                "activity",
            ),
        ),
    ]
    header = (
        f"hermes-v2 dashboard @ {_as_utc(report.timestamp).isoformat()} "
        f"(overall status: {report.overall_status})"
    )
    lines = [header]
    for title, body in sections:
        lines.append(_HORIZONTAL_RULE)
        lines.append(title)
        lines.append(body)
    lines.append(_HORIZONTAL_RULE)
    lines.append("left/right: switch page - q: quit")
    return "\n".join(lines)


def render_page(
    page: str,
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None = None,
    learning_ledger: LearningMetricsLedger | None = None,
    model_metrics: ModelMetricsCollector | None = None,
) -> str:
    """Render ONE dashboard page (the interactive loop's unit of display).

    ``overview`` renders the full :func:`render_dashboard_text` dashboard;
    any other page of :func:`pages` renders exactly that section; an unknown
    page name falls back to ``overview``.
    """
    if page not in DASHBOARD_PAGES or page == "overview":
        return render_dashboard_text(diagnostics, quality_tracker, learning_ledger, model_metrics)
    report = diagnostics.run_health_check()
    header = (
        f"hermes-v2 dashboard [{page}] @ {_as_utc(report.timestamp).isoformat()} "
        f"(overall status: {report.overall_status})"
    )
    bodies: dict[str, str] = {
        "health": _render_health(report),
        "metrics": _render_metrics(report, model_metrics),
        "trends": _safe(
            lambda: _render_trends(quality_tracker, learning_ledger),
            "trends",
        ),
        "alerts": _render_alerts(report),
        "activity": _safe(
            lambda: _render_activity(diagnostics, quality_tracker, learning_ledger, model_metrics),
            "activity",
        ),
    }
    title = page.upper()
    lines = [header, _HORIZONTAL_RULE, title, bodies[page], _HORIZONTAL_RULE]
    lines.append("left/right: switch page - q: quit")
    return "\n".join(lines)


# ── interactive loop ─────────────────────────────────────────────────────────


def _read_key_windows() -> str:
    """One keypress via ``msvcrt`` (Windows): poll ``kbhit``, then ``getwch``.

    Arrow keys arrive as a two-character sequence (``\\x00``/``\\xe0`` prefix
    + ``H``/``P``/``K``/``M``); they are translated to the same tokens
    :func:`navigate` understands (``up``/``down``/``left``/``right``).
    """
    import msvcrt  # noqa: PLC0415 - platform module, imported only on win32

    while not msvcrt.kbhit():
        time.sleep(0.05)
    char = msvcrt.getwch()
    if char in ("\x00", "\xe0"):
        extension = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(extension, "")
    return char


def _read_key_posix() -> str:
    """One keypress via ``termios`` (POSIX fallback, documented): cbreak read.

    Puts the terminal into ``cbreak`` mode for a single read and restores it
    in ``finally`` — arrow keys arrive as ESC [ A/B/C/D sequences and map to
    ``up``/``down``/``left``/``right``. POSIX terminals are outside the test
    matrix here; the injected-stream path (:func:`run_cli_dashboard` with an
    ``input_stream``) is the tested behaviour.
    """
    if sys.platform != "posix":  # pragma: no cover - guarded by _read_key
        raise RuntimeError("termios key reading is only available on POSIX")
    import termios  # noqa: PLC0415 - platform module, imported only on POSIX
    import tty  # noqa: PLC0415

    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        char = sys.stdin.read(1)
        if char == "\x1b":
            sequence = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[D": "left", "[C": "right"}.get(sequence, "")
        return char
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)


def _read_key() -> str:
    """Dispatch the raw-key reader to the current platform's implementation."""
    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_posix()


def _key_stream(input_stream: Iterable[str]) -> Iterator[str]:
    """Normalize an injected stream into key tokens (one per line/element).

    Literal ANSI arrow sequences (as a terminal would produce them) are
    translated too, so an injected stream may use either the token names
    (``left``/``right``) or the raw sequences (``\\x1b[D``/``\\x1b[C``).
    """
    for element in input_stream:
        token = element.strip()
        if token.startswith("\x1b"):
            token = {"\x1b[A": "up", "\x1b[B": "down", "\x1b[D": "left", "\x1b[C": "right"}.get(
                token, ""
            )
        yield token


def _raw_key_stream() -> Iterator[str]:
    """Yield platform raw keys forever; the loop's quit break is the exit."""
    while True:
        yield _read_key()


def _visible_page(
    page: str,
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None,
    learning_ledger: LearningMetricsLedger | None,
    model_metrics: ModelMetricsCollector | None,
    writer: Any,
) -> str:
    """Render ``page`` to ``writer`` and return the rendered text."""
    rendered = render_page(page, diagnostics, quality_tracker, learning_ledger, model_metrics)
    writer.write(rendered + "\n")
    writer.flush()
    return rendered


def run_cli_dashboard(
    diagnostics: DiagnosticsSystem,
    quality_tracker: MemoryQualityTracker | None = None,
    learning_ledger: LearningMetricsLedger | None = None,
    model_metrics: ModelMetricsCollector | None = None,
    *,
    input_stream: Iterable[str] | None = None,
    output: Any = None,
) -> str:
    """Run the interactive dashboard loop; return the final page rendered.

    Navigation: ``left``/``right`` (and arrow aliases) switch pages of
    :func:`pages`, ``q``/``quit``/``esc`` exits, unknown keys stay on the
    current page. Key sources:

    - ``input_stream=None`` (default): raw keys — ``msvcrt.kbhit``/
      ``getwch`` on Windows, the documented ``termios`` cbreak fallback on
      POSIX (see :func:`_read_key_posix`);
    - ``input_stream`` given (the TEST path): key tokens read line-by-line
      from the iterable — no raw keys are ever touched; the loop ends at
      ``quit`` or end of stream (EOF behaves like quit).

    Display contract: the first page is rendered before the first key is
    read; every navigation key renders the page now visible; a quit key ends
    the loop without further rendering. ``output`` (default ``sys.stdout``)
    receives each rendered page and the final rendered text is returned, so
    callers (and tests) can inspect the last visible screen.
    """
    writer = output if output is not None else sys.stdout
    page_list = pages()
    current = page_list[0]
    final_render = _visible_page(
        current, diagnostics, quality_tracker, learning_ledger, model_metrics, writer
    )
    raw_keys: Iterable[str] = (
        _raw_key_stream() if input_stream is None else _key_stream(input_stream)
    )
    for key in raw_keys:
        token = key.strip().lower()
        if token in KEYS_QUIT or token == "":
            break  # quit key, blank line or exhausted stream end the session
        current = navigate(page_list, current, token) or current
        final_render = _visible_page(
            current, diagnostics, quality_tracker, learning_ledger, model_metrics, writer
        )
    return final_render


def _main() -> int:  # pragma: no cover - manual entry point, not test-covered
    """``python -m hermes.diagnostics.dashboard_cli`` entry point (default wiring)."""
    run_cli_dashboard(DiagnosticsSystem())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
