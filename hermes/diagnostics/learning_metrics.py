"""hermes-v2 learning loop effectiveness diagnostics (plan P14.T8, plan Phase 14).

Monitors whether the GEPA optimization loop (plan P10.T5) actually improves
the pipeline: before/after eval scores are recorded per optimization run
into a JSONL ledger under
``<hermes_home>/diagnostics/learning_metrics.jsonl`` (one line per run,
atomic append + fsync — the same ledger convention as the P14.T2/P14.T7
siblings), :meth:`LearningMetricsLedger.trend_report` aggregates the
trailing ``window_runs`` runs per component, and
:func:`should_alert` / :func:`alert_payload` fire the
``learning_no_improvement`` alert when a component shows ``threshold``
(default 3) consecutive runs without improvement (plan acceptance: "alert
if no improvement in 3 consecutive runs").

Semantics
---------
- ``improvement`` of a run is ``after_score - before_score``;
- a run is **flat** when ``|improvement| < :data:`FLAT_EPSILON`` (0.005);
- a component's trend is ``improving`` / ``declining`` from the sign of its
  window's average improvement (beyond the flat epsilon), else ``flat``;
- ``consecutive_no_improvement`` counts the trailing window runs whose
  improvement was NOT an improvement (flat or declining) — the streak the
  alert rule thresholds on;
- the report window is the trailing ``window_runs`` (default
  :data:`DEFAULT_WINDOW_RUNS` = 10) runs per component; GEPA runs weekly
  (plan P10.T5), so the default report spans the last 10 weekly runs —
  the "trend report weekly" acceptance reads as: the report covers the
  weekly cadence's trailing window.

Scores are validated as finite numbers but not range-bounded: the adapter's
aggregate metric (``hermes.optimization.adapters.BaseAdapter.evaluate``) is
component-defined, so the ledger accepts whatever the runner measured. The
wall clock is injectable (``now``) so tests stay deterministic; ledger
lines the reader cannot parse are skipped (one bad line never breaks the
report), mirroring the sibling ledgers.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from hermes.config import hermes_home

__all__ = [
    "ALERT_NAME",
    "DEFAULT_WINDOW_RUNS",
    "FLAT_EPSILON",
    "NO_IMPROVEMENT_THRESHOLD",
    "ComponentTrend",
    "LearningMetricsLedger",
    "LearningTrend",
    "LearningTrendReport",
    "alert_payload",
    "default_learning_metrics_path",
    "should_alert",
]

#: An improvement strictly below this magnitude counts as flat (plan P14.T8
#: design: ``flat = |improvement| < 0.005``).
FLAT_EPSILON: Final[float] = 0.005

#: Default report window: the trailing 10 optimization runs per component.
DEFAULT_WINDOW_RUNS: Final[int] = 10

#: Plan acceptance: alert after 3 consecutive runs without improvement.
NO_IMPROVEMENT_THRESHOLD: Final[int] = 3

#: Name of the learning-loop alert (payload key ``alert``).
ALERT_NAME: Final[str] = "learning_no_improvement"

#: Injectable wall clock: returns the current datetime (aware UTC by convention).
Clock = Callable[[], datetime]

#: Trend of a component's trailing window.
LearningTrend = Literal["improving", "flat", "declining"]


def _utcnow() -> datetime:
    """Default clock: current aware UTC time."""
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def default_learning_metrics_path() -> Path:
    """Default ledger path: ``<hermes_home>/diagnostics/learning_metrics.jsonl``."""
    path: Path = hermes_home() / "diagnostics" / "learning_metrics.jsonl"
    return path


def _validate_non_empty(value: str, name: str) -> None:
    """Reject values that are not non-empty strings."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_score(value: float, name: str) -> float:
    """Validate one eval score as a finite number; return it as float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return score


@dataclass(frozen=True)
class ComponentTrend:
    """One component's trailing-window trend (plan P14.T8 design shape).

    ``avg_improvement`` is the mean of the window's per-run improvements;
    ``trend`` classifies that average; ``consecutive_no_improvement`` counts
    the trailing flat/declining runs (the alert streak).
    """

    runs: int
    avg_improvement: float
    trend: LearningTrend
    consecutive_no_improvement: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection (report payload / dashboard, P14.T10)."""
        return {
            "runs": self.runs,
            "avg_improvement": self.avg_improvement,
            "trend": self.trend,
            "consecutive_no_improvement": self.consecutive_no_improvement,
        }


@dataclass
class LearningTrendReport:
    """Trend report over the trailing window (plan P14.T8 acceptance).

    ``components`` maps every recorded component (or just the filtered one)
    to its :class:`ComponentTrend`; ``window_runs`` and ``generated_at``
    document the window the report covers; ``component`` is the filter
    (``None`` = every component).
    """

    generated_at: datetime
    window_runs: int
    component: str | None
    components: dict[str, ComponentTrend] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection for the diagnostics report (plan 5.1 shape)."""
        return {
            "generated_at": _as_utc(self.generated_at).isoformat(),
            "window_runs": self.window_runs,
            "component": self.component,
            "components": {
                name: trend.to_dict() for name, trend in self.components.items()
            },
        }


def _classify(avg_improvement: float) -> LearningTrend:
    """Trend from the window's average improvement (flat within epsilon)."""
    if avg_improvement > FLAT_EPSILON:
        return "improving"
    if avg_improvement < -FLAT_EPSILON:
        return "declining"
    return "flat"


def _consecutive_no_improvement(improvements: list[float]) -> int:
    """Trailing runs (most recent backwards) that did not improve."""
    count = 0
    for improvement in reversed(improvements):
        if improvement >= FLAT_EPSILON:
            break
        count += 1
    return count


class LearningMetricsLedger:
    """Records GEPA run outcomes and reports learning trends (plan P14.T8).

    Args:
        ledger_path: JSONL ledger (default
            ``<hermes_home>/diagnostics/learning_metrics.jsonl``).
        now: injectable wall clock (aware UTC by convention), used to stamp
            runs recorded without an explicit ``finished_at``.
    """

    def __init__(self, ledger_path: Path | None = None, now: Clock | None = None) -> None:
        self.ledger_path = (
            ledger_path if ledger_path is not None else default_learning_metrics_path()
        )
        self._now: Clock = now if now is not None else _utcnow
        self._lock = threading.Lock()

    # ── recording API ─────────────────────────────────────────────────────
    def record_run(
        self,
        component: str,
        run_id: str,
        before_score: float,
        after_score: float,
        deployed: bool,
        finished_at: datetime | None = None,
    ) -> None:
        """Append one optimization run outcome to the ledger (atomic + fsync).

        ``improvement`` is stored precomputed (``after_score -
        before_score``) so readers never re-derive it. ``finished_at``
        defaults to the ledger's clock; naive datetimes are read as UTC.
        """
        _validate_non_empty(component, "component")
        _validate_non_empty(run_id, "run_id")
        before = _validate_score(before_score, "before_score")
        after = _validate_score(after_score, "after_score")
        if not isinstance(deployed, bool):
            raise TypeError(f"deployed must be a bool, got {type(deployed).__name__}")
        moment = _as_utc(finished_at) if finished_at is not None else _as_utc(self._now())
        record = {
            "ts": moment.isoformat(),
            "component": component,
            "run_id": run_id,
            "before_score": before,
            "after_score": after,
            "improvement": after - before,
            "deployed": deployed,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as file_handle:
                file_handle.write(line + "\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())

    # ── reporting API ─────────────────────────────────────────────────────
    def trend_report(
        self, component: str | None = None, window_runs: int = DEFAULT_WINDOW_RUNS
    ) -> LearningTrendReport:
        """Aggregate the trailing ``window_runs`` runs per component.

        ``component=None`` spans every recorded component; a named filter
        with no recorded runs yields an empty ``components`` mapping (the
        report is honest about having nothing to report). Runs are ordered
        by their ledger timestamps (stable: equal timestamps keep ledger
        order) and the window is the trailing slice of that order.
        """
        if component is not None:
            _validate_non_empty(component, "component")
        if isinstance(window_runs, bool) or not isinstance(window_runs, int) or window_runs < 1:
            raise ValueError(f"window_runs must be a positive int, got {window_runs!r}")
        runs = self._read_runs()
        by_component: dict[str, list[dict[str, Any]]] = {}
        for index, run in enumerate(runs):
            if component is not None and run["component"] != component:
                continue
            key = (run["ts"], index)  # stable: ledger order breaks ties
            by_component.setdefault(run["component"], []).append({**run, "_order": key})
        components: dict[str, ComponentTrend] = {}
        for name, component_runs in by_component.items():
            ordered = sorted(component_runs, key=lambda run: run["_order"])
            window = ordered[-window_runs:]
            improvements = [float(run["improvement"]) for run in window]
            avg = sum(improvements) / len(improvements) if improvements else 0.0
            components[name] = ComponentTrend(
                runs=len(window),
                avg_improvement=avg,
                trend=_classify(avg),
                consecutive_no_improvement=_consecutive_no_improvement(improvements),
            )
        return LearningTrendReport(
            generated_at=_as_utc(self._now()),
            window_runs=window_runs,
            component=component,
            components=components,
        )

    # ── internals ─────────────────────────────────────────────────────────
    def _read_runs(self) -> list[dict[str, Any]]:
        """Parsed ledger runs, oldest first; malformed lines are skipped."""
        if not self.ledger_path.exists():
            return []
        runs: list[dict[str, Any]] = []
        with open(self.ledger_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                    run = {
                        "ts": _as_utc(datetime.fromisoformat(str(raw["ts"]))),
                        "component": str(raw["component"]),
                        "run_id": str(raw["run_id"]),
                        "improvement": float(raw["improvement"]),
                        "deployed": bool(raw["deployed"]),
                        "before_score": float(raw["before_score"]),
                        "after_score": float(raw["after_score"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue  # never let one bad line break the report
                runs.append(run)
        return runs


def should_alert(
    report: LearningTrendReport, threshold: int = NO_IMPROVEMENT_THRESHOLD
) -> bool:
    """True when any component shows ``threshold`` consecutive non-improvements.

    Plan P14.T8 acceptance: "alert if no improvement in 3 consecutive runs"
    — the default threshold is 3.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError(f"threshold must be a positive int, got {threshold!r}")
    return any(
        trend.consecutive_no_improvement >= threshold for trend in report.components.values()
    )


def alert_payload(
    report: LearningTrendReport, threshold: int = NO_IMPROVEMENT_THRESHOLD
) -> dict[str, Any] | None:
    """The ``learning_no_improvement`` alert payload, or ``None`` when quiet.

    Payload: alert name (:data:`ALERT_NAME`), severity ``warning``, the
    threshold, the report's generation time and the offending components'
    trend summaries.
    """
    if not should_alert(report, threshold):
        return None
    offending = {
        name: trend.to_dict()
        for name, trend in report.components.items()
        if trend.consecutive_no_improvement >= threshold
    }
    return {
        "alert": ALERT_NAME,
        "severity": "warning",
        "threshold": threshold,
        "generated_at": _as_utc(report.generated_at).isoformat(),
        "components": offending,
    }
