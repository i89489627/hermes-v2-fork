"""Model performance diagnostics ledger (plan P14.T7).

Model-specific metrics — token generation speed (tok/s), output quality
(eval score per model version), error rate and fallback rate — as a JSONL
time series under ``<hermes_home>/diagnostics/model_metrics.jsonl``.

Sibling ledger, NOT a MetricsSnapshot extension
-----------------------------------------------
plan P14.T7 lists ``hermes/diagnostics/metrics.py`` as a file to modify, but
the :class:`~hermes.diagnostics.metrics.MetricsSnapshot` contract is the
5-minute pipeline cadence (latency percentiles, trust distribution,
conflicts) — model events arrive per generation, at a different shape and
cadence. This module therefore keeps a SIBLING ledger file next to
``metrics.jsonl`` (see ``SIBLING_LEDGER_NAME`` in the metrics module) instead
of touching the snapshot contract; nothing existing changes signature. The
dashboard (P14.T10) reads both ledgers.

Event schema (one JSON object per line)
---------------------------------------
``{"ts": <ISO-8601 UTC>, "event": "generation"|"quality"|"error"|"fallback",
"model_id": <str>, ...}`` where generation lines also carry ``tokens``,
``seconds`` and ``tok_per_s``; quality lines carry ``eval_score``. Appends
are atomic (single write + fsync), the wall clock is injectable
(``now: Callable[[], datetime]``) so tests stay deterministic, and the
report aggregates over a trailing window (default 24h) from the ledger on
disk, so reports survive collector restarts.

Report semantics
----------------
- ``tok_per_s`` percentiles over the window's generation events (linear
  interpolation, matching :mod:`hermes.diagnostics.metrics`);
- ``quality_scores`` in chronological order and ``quality_trend`` from the
  least-squares slope sign over the sequence index (flat when fewer than two
  scores or |slope| below :data:`TREND_EPSILON`);
- ``error_rate`` = errors / (generations + errors) — an error is a FAILED
  generation attempt, so they share the denominator;
- ``fallback_rate`` = fallbacks / (generations + errors + fallbacks) — the
  share of operations served by the trust-only/regex fallback path (plan 5.3
  ``fallback_rate_high`` alert rule: >20% of ops in the window).
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

from hermes.config import hermes_home

__all__ = [
    "DEFAULT_WINDOW_HOURS",
    "FALLBACK_ALERT_THRESHOLD",
    "TREND_EPSILON",
    "ModelMetricsCollector",
    "ModelMetricsReport",
    "QualityTrend",
    "alert_payload",
    "default_model_metrics_path",
    "should_alert",
]

#: Default report window (plan P14.T7: trailing day of activity).
DEFAULT_WINDOW_HOURS: Final[float] = 24.0

#: Plan 5.3 alert rule: fallback rate strictly above 20% of operations.
FALLBACK_ALERT_THRESHOLD: Final[float] = 0.2

#: |slope| below this counts as a flat quality trend.
TREND_EPSILON: Final[float] = 1e-9

#: Injectable wall clock: returns the current datetime (aware UTC by convention).
Clock = Callable[[], datetime]

#: Quality trend over the report window.
QualityTrend = Literal["improving", "flat", "declining"]


def _utcnow() -> datetime:
    """Default clock: current aware UTC time."""
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def default_model_metrics_path() -> Path:
    """Default ledger path: ``<hermes_home>/diagnostics/model_metrics.jsonl``."""
    path: Path = hermes_home() / "diagnostics" / "model_metrics.jsonl"
    return path


def _validate_model_id(model_id: str) -> None:
    """Reject model ids that are not non-empty strings."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentile ``q`` (0..1) of already-sorted values, linear interpolation."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample set")
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    if lower >= len(sorted_values) - 1:
        return sorted_values[-1]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[lower + 1] - sorted_values[lower]) * fraction


def _slope_sign(values: list[float]) -> QualityTrend:
    """Trend of ``values`` (chronological) from the least-squares slope sign."""
    if len(values) < 2:
        return "flat"
    count = len(values)
    mean_x = (count - 1) / 2.0
    mean_y = sum(values) / count
    numerator = sum(
        (index - mean_x) * (value - mean_y) for index, value in enumerate(values)
    )
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator == 0.0:  # pragma: no cover - count >= 2 implies positive
        return "flat"
    slope = numerator / denominator
    if slope > TREND_EPSILON:
        return "improving"
    if slope < -TREND_EPSILON:
        return "declining"
    return "flat"


@dataclass
class ModelMetricsReport:
    """Aggregated model metrics over one trailing window (plan P14.T7).

    ``model_id`` is ``None`` when the report spans every model. Percentiles
    are over the window's generation events; ``quality_scores`` is
    chronological (dashboard-reportable, plan P14.T7 acceptance "quality
    trend visible"). Empty windows report zeros and a ``flat`` trend.
    """

    model_id: str | None
    window_hours: float
    tok_per_s_p50: float = 0.0
    tok_per_s_p95: float = 0.0
    quality_scores: list[float] = field(default_factory=list)
    quality_trend: QualityTrend = "flat"
    error_rate: float = 0.0
    fallback_rate: float = 0.0
    generations: int = 0
    errors: int = 0
    fallbacks: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection for the diagnostics dashboard (P14.T10)."""
        return {
            "model_id": self.model_id,
            "window_hours": self.window_hours,
            "tok_per_s_p50": self.tok_per_s_p50,
            "tok_per_s_p95": self.tok_per_s_p95,
            "quality_scores": list(self.quality_scores),
            "quality_trend": self.quality_trend,
            "error_rate": self.error_rate,
            "fallback_rate": self.fallback_rate,
            "generations": self.generations,
            "errors": self.errors,
            "fallbacks": self.fallbacks,
        }


class ModelMetricsCollector:
    """Records per-model events and persists them as a JSONL ledger (plan P14.T7).

    Recording methods append one line each (atomic single write + fsync);
    :meth:`report` aggregates the trailing window by reading the ledger from
    disk, so reports survive restarts and reflect what other collector
    instances appended too. Thread-safe via a lock around appends.
    """

    def __init__(self, ledger_path: Path | None = None, now: Clock | None = None) -> None:
        """Open a collector writing to ``ledger_path`` (default under ``hermes_home()``)."""
        self.ledger_path = ledger_path if ledger_path is not None else default_model_metrics_path()
        self._now: Clock = now if now is not None else _utcnow
        self._lock = threading.Lock()

    # ── recording API (plan P14.T7 description) ──────────────────────────────

    def record_generation(self, model_id: str, tokens: int, seconds: float) -> float:
        """Record one successful generation; returns its tok/s speed.

        ``tokens`` must be a non-negative int and ``seconds`` a finite
        positive number (``TypeError``/``ValueError`` otherwise) — a zero-length
        generation has no meaningful speed.
        """
        _validate_model_id(model_id)
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TypeError(f"tokens must be an int, got {type(tokens).__name__}")
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise TypeError(f"seconds must be a number, got {type(seconds).__name__}")
        seconds_value = float(seconds)
        if not math.isfinite(seconds_value) or seconds_value <= 0.0:
            raise ValueError(f"seconds must be a finite positive number, got {seconds!r}")
        tok_per_s = tokens / seconds_value
        self._append(
            {
                "event": "generation",
                "model_id": model_id,
                "tokens": tokens,
                "seconds": seconds_value,
                "tok_per_s": tok_per_s,
            }
        )
        return tok_per_s

    def record_quality(self, model_id: str, eval_score: float) -> None:
        """Record one eval score for ``model_id`` (valid scores lie in [0, 1])."""
        _validate_model_id(model_id)
        if isinstance(eval_score, bool) or not isinstance(eval_score, int | float):
            raise TypeError(f"eval_score must be a number, got {type(eval_score).__name__}")
        score = float(eval_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"eval_score must be a finite number in [0, 1], got {eval_score!r}")
        self._append({"event": "quality", "model_id": model_id, "eval_score": score})

    def record_error(self, model_id: str) -> None:
        """Record one failed generation attempt for ``model_id``."""
        _validate_model_id(model_id)
        self._append({"event": "error", "model_id": model_id})

    def record_fallback(self, model_id: str) -> None:
        """Record one operation served by the degraded fallback path (plan 5.3)."""
        _validate_model_id(model_id)
        self._append({"event": "fallback", "model_id": model_id})

    # ── reporting API ─────────────────────────────────────────────────────────

    def report(
        self,
        model_id: str | None = None,
        window_hours: float = DEFAULT_WINDOW_HOURS,
    ) -> ModelMetricsReport:
        """Aggregate the trailing ``window_hours`` (plan P14.T7 acceptance).

        ``model_id=None`` spans every model; otherwise only that model's
        events count. Events are read from the ledger on disk; timestamps
        ahead of the clock are treated as "now" (monotonic safety against
        clock skew between the recorder and the reporter).
        """
        if window_hours <= 0.0:
            raise ValueError(f"window_hours must be positive, got {window_hours!r}")
        moment = _as_utc(self._now())
        cutoff = moment - timedelta(hours=window_hours)
        generations: list[float] = []
        quality_scores: list[float] = []
        errors = 0
        fallbacks = 0
        for event in self._read_events():
            if model_id is not None and event.get("model_id") != model_id:
                continue
            timestamp = event["ts"]
            if timestamp < cutoff:
                continue
            kind = event.get("event")
            if kind == "generation":
                generations.append(float(event.get("tok_per_s", 0.0)))
            elif kind == "quality":
                quality_scores.append(float(event.get("eval_score", 0.0)))
            elif kind == "error":
                errors += 1
            elif kind == "fallback":
                fallbacks += 1
        generation_count = len(generations)
        ordered = sorted(generations)
        # quality_scores stays in ledger (chronological) order — the trend
        # slope is computed over that sequence; percentiles sort separately.
        error_rate = errors / (generation_count + errors) if (generation_count + errors) else 0.0
        fallback_denominator = generation_count + errors + fallbacks
        fallback_rate = fallbacks / fallback_denominator if fallback_denominator else 0.0
        trend = _slope_sign(quality_scores)
        return ModelMetricsReport(
            model_id=model_id,
            window_hours=window_hours,
            tok_per_s_p50=_percentile(ordered, 0.50) if ordered else 0.0,
            tok_per_s_p95=_percentile(ordered, 0.95) if ordered else 0.0,
            quality_scores=quality_scores,
            quality_trend=trend,
            error_rate=error_rate,
            fallback_rate=fallback_rate,
            generations=generation_count,
            errors=errors,
            fallbacks=fallbacks,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _append(self, event: dict[str, Any]) -> None:
        """Stamp, serialize and atomically append one event line."""
        record = {"ts": _as_utc(self._now()).isoformat(), **event}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _read_events(self) -> list[dict[str, Any]]:
        """Parsed ledger events, oldest first; malformed lines are skipped."""
        if not self.ledger_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self.ledger_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                    kind = str(raw["event"])
                    timestamp = _as_utc(datetime.fromisoformat(str(raw["ts"])))
                    model = str(raw["model_id"])
                except (KeyError, TypeError, ValueError):
                    continue  # never let one bad line break the report
                event = dict(raw)
                event["ts"] = timestamp
                event["event"] = kind
                event["model_id"] = model
                events.append(event)
        return events


# ── alert rule (plan 5.3: fallback_rate_high) ────────────────────────────────


def should_alert(report: ModelMetricsReport, threshold: float = FALLBACK_ALERT_THRESHOLD) -> bool:
    """True when the report's fallback rate exceeds ``threshold`` (plan 5.3)."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")
    return report.fallback_rate > threshold


def alert_payload(
    report: ModelMetricsReport, threshold: float = FALLBACK_ALERT_THRESHOLD
) -> dict[str, Any] | None:
    """The ``fallback_rate_high`` alert payload, or ``None`` when not alerting.

    Payload shape matches the plan 5.3 alert table entry (``fallback_rate_high``,
    severity ``warning``): alert name, model scope, observed rate, threshold
    and window length.
    """
    if not should_alert(report, threshold):
        return None
    return {
        "alert": "fallback_rate_high",
        "severity": "warning",
        "model_id": report.model_id,
        "fallback_rate": report.fallback_rate,
        "threshold": threshold,
        "window_hours": report.window_hours,
        "fallbacks": report.fallbacks,
        "operations": report.generations + report.errors + report.fallbacks,
    }
