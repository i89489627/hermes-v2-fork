"""hermes-v2 metrics collection (plan P14.T2).

Periodic pipeline metrics as a JSONL time series under
``<hermes_home>/diagnostics/metrics.jsonl`` (one snapshot per line, default
collection cadence 5 minutes — :func:`due_for_collection` is the pure
scheduling helper the pipeline ticks against).

One :class:`MetricsSnapshot` aggregates, per operation name:

- **latency percentiles** — ``p50``/``p95``/``p99`` over the bounded window of
  recorded samples for the op (linear interpolation between neighbouring
  order statistics, matching numpy's default ``linear`` method);
- **token usage** — accumulated token totals per op since the collector
  started (cumulative, not per-interval deltas);
- **trust distribution** — ``avg``/``min``/``max`` plus the share of recorded
  trust scores below the 0.3 low-trust threshold (``count_below_0.3``);
- **conflict events** — entry-id conflicts recorded in the trailing 7-day
  window, and :attr:`MetricsSnapshot.conflict_rate_7d`: the number of those
  conflicts divided by the window length in hours (``7 * 24 = 168``), i.e.
  the unit is **conflicts per hour** over the rolling 7-day window.

Snapshots are appended to the JSONL file atomically (single write + fsync).
The time series is retained for 90 days: :meth:`MetricsCollector.prune`
rewrites the file without older lines, and it runs automatically inside
:meth:`MetricsCollector.append_snapshot` whenever a UTC day boundary is
crossed between consecutive appends. All times are aware UTC; naive datetimes
passed to the API are interpreted as UTC, and the wall clock is injectable
(``now: Callable[[], datetime]``) so tests stay deterministic.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from hermes.config import hermes_home

__all__ = [
    "COLLECTION_INTERVAL_MINUTES",
    "CONFLICT_WINDOW_DAYS",
    "DEFAULT_SAMPLE_WINDOW",
    "LOW_TRUST_THRESHOLD",
    "RETENTION_DAYS",
    "SIBLING_LEDGER_NAME",
    "MetricsCollector",
    "MetricsSnapshot",
    "default_metrics_path",
    "due_for_collection",
]

#: Default cadence between collections (plan P14.T2: every 5 minutes).
COLLECTION_INTERVAL_MINUTES: Final[float] = 5.0

#: Time-series retention (plan P14.T2 acceptance: series kept for 90 days).
RETENTION_DAYS: Final[int] = 90

#: Conflict-rate lookback (plan P14.T2: trailing 7 days).
CONFLICT_WINDOW_DAYS: Final[int] = 7

#: Denominator of ``conflict_rate_7d``: the window length in hours, so the
#: rate unit is conflicts per hour (``7 * 24 = 168``).
CONFLICT_WINDOW_HOURS: Final[float] = float(CONFLICT_WINDOW_DAYS * 24)

#: Bounded number of latency samples per op / trust scores kept in memory.
DEFAULT_SAMPLE_WINDOW: Final[int] = 512

#: Trust score at or above which a memory is considered trustworthy; recorded
#: scores strictly below this are counted by ``count_below_0.3``.
LOW_TRUST_THRESHOLD: Final[float] = 0.3

#: Injectable wall clock: returns the current datetime (aware UTC by convention).
Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    """Default clock: current aware UTC time."""
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def default_metrics_path() -> Path:
    """Default JSONL store path: ``<hermes_home>/diagnostics/metrics.jsonl``."""
    return hermes_home() / "diagnostics" / "metrics.jsonl"


def due_for_collection(
    last: datetime | None,
    now: datetime,
    interval_minutes: float = COLLECTION_INTERVAL_MINUTES,
) -> bool:
    """Pure scheduling helper: is a metrics collection due?

    ``True`` when the pipeline has never collected (``last`` is ``None``) or
    when at least ``interval_minutes`` minutes elapsed between ``last`` and
    ``now``. Naive datetimes are interpreted as UTC.
    """
    if last is None:
        return True
    elapsed = (_as_utc(now) - _as_utc(last)).total_seconds()
    return elapsed >= interval_minutes * 60.0


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentile ``q`` (0..1) of already-sorted values, linear interpolation.

    With ``n`` samples the percentile rank is ``(n - 1) * q``; the fractional
    part of the rank interpolates linearly between the two neighbouring order
    statistics (numpy's default ``linear`` interpolation method).
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must lie in [0, 1], got {q}")
    if not sorted_values:
        raise ValueError("percentile of an empty sample set")
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    if lower >= len(sorted_values) - 1:
        return sorted_values[-1]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[lower + 1] - sorted_values[lower]) * fraction


def _trust_distribution(scores: list[float]) -> dict[str, float]:
    """``avg``/``min``/``max``/``count_below_0.3`` over recorded trust scores.

    An empty distribution reports zeros so snapshots stay JSON-stable.
    ``count_below_0.3`` is an absolute count (float-valued for a uniform
    ``dict[str, float]`` mapping), not a ratio.
    """
    if not scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "count_below_0.3": 0.0}
    return {
        "avg": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "count_below_0.3": float(sum(1 for score in scores if score < LOW_TRUST_THRESHOLD)),
    }


@dataclass
class MetricsSnapshot:
    """One periodic metrics sample (plan P14.T2), JSONL-serializable."""

    timestamp: datetime
    latency_percentiles: dict[str, dict[str, float]] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    trust_scores: list[float] = field(default_factory=list)
    trust_stats: dict[str, float] = field(default_factory=dict)
    conflict_events: list[dict[str, Any]] = field(default_factory=list)
    conflict_rate_7d: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection: timestamps become ISO-8601 strings."""
        events: list[dict[str, Any]] = []
        for event in self.conflict_events:
            moment = event.get("timestamp")
            events.append(
                {
                    "timestamp": (
                        moment.isoformat() if isinstance(moment, datetime) else str(moment)
                    ),
                    "entry_id": str(event.get("entry_id", "")),
                }
            )
        return {
            "timestamp": _as_utc(self.timestamp).isoformat(),
            "latency_percentiles": {
                op: dict(values) for op, values in self.latency_percentiles.items()
            },
            "token_usage": dict(self.token_usage),
            "trust_scores": list(self.trust_scores),
            "trust_stats": dict(self.trust_stats),
            "conflict_events": events,
            "conflict_rate_7d": self.conflict_rate_7d,
            "counters": dict(self.counters),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MetricsSnapshot:
        """Rebuild from :meth:`to_dict` output; malformed records raise ``ValueError``."""
        try:
            timestamp = datetime.fromisoformat(str(raw["timestamp"]))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            percentiles_raw = raw.get("latency_percentiles", {})
            tokens_raw = raw.get("token_usage", {})
            stats_raw = raw.get("trust_stats", {})
            counters_raw = raw.get("counters", {})
            if not all(isinstance(section, Mapping) for section in (
                percentiles_raw, tokens_raw, stats_raw, counters_raw
            )):
                raise TypeError("snapshot sections must be mappings")
            events_raw = raw.get("conflict_events", [])
            if not isinstance(events_raw, list):
                raise TypeError("conflict_events must be a list")
            events: list[dict[str, Any]] = []
            for item in events_raw:
                if not isinstance(item, Mapping):
                    raise TypeError("conflict event must be a mapping")
                moment = datetime.fromisoformat(str(item["timestamp"]))
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                events.append({"timestamp": moment, "entry_id": str(item.get("entry_id", ""))})
            return cls(
                timestamp=timestamp,
                latency_percentiles={
                    str(op): {str(key): float(value) for key, value in mapping.items()}
                    for op, mapping in percentiles_raw.items()
                },
                token_usage={str(op): int(value) for op, value in tokens_raw.items()},
                trust_scores=[float(value) for value in raw.get("trust_scores", [])],
                trust_stats={str(key): float(value) for key, value in stats_raw.items()},
                conflict_events=events,
                conflict_rate_7d=float(raw.get("conflict_rate_7d", 0.0)),
                counters={str(key): int(value) for key, value in counters_raw.items()},
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed metrics snapshot record: {raw!r}") from exc


class MetricsCollector:
    """Records pipeline metrics and persists snapshots as a JSONL time series.

    The collector keeps bounded in-memory windows (latency per op, trust
    scores) so long-running processes stay flat in memory; token totals and
    conflict events accumulate for the lifetime of the collector. Snapshots
    written by :meth:`append_snapshot` survive restarts via
    :meth:`load_history`; the JSONL file is pruned to 90 days (automatically
    on day-boundary-crossing appends, or explicitly via :meth:`prune`).
    """

    def __init__(
        self,
        store_path: Path | None = None,
        now: Clock | None = None,
        sample_window: int = DEFAULT_SAMPLE_WINDOW,
    ) -> None:
        """Open a collector writing to ``store_path`` (default under ``hermes_home()``)."""
        self.store_path: Path = store_path if store_path is not None else default_metrics_path()
        self._now: Clock = now if now is not None else _utcnow
        if sample_window < 1:
            raise ValueError(f"sample_window must be >= 1, got {sample_window}")
        self._sample_window = sample_window
        self._lock = threading.Lock()
        self._latencies: dict[str, deque[float]] = {}
        self._tokens: dict[str, int] = {}
        self._trust_scores: deque[float] = deque(maxlen=sample_window)
        self._conflicts: list[dict[str, Any]] = []
        self._last_append_day: str | None = None

    # ── recording API ─────────────────────────────────────────────────────
    def record_latency(self, op: str, seconds: float) -> None:
        """Record one latency sample (seconds) for ``op`` in its bounded window."""
        _validate_op(op)
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise TypeError(f"seconds must be a number, got {type(seconds).__name__}")
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"seconds must be a finite non-negative number, got {seconds!r}")
        with self._lock:
            window = self._latencies.setdefault(op, deque(maxlen=self._sample_window))
            window.append(value)

    def record_tokens(self, op: str, tokens: int) -> None:
        """Accumulate ``tokens`` onto ``op``'s running total."""
        _validate_op(op)
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TypeError(f"tokens must be an int, got {type(tokens).__name__}")
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        with self._lock:
            self._tokens[op] = self._tokens.get(op, 0) + tokens

    def record_trust_score(self, score: float) -> None:
        """Record one memory trust score; valid scores lie in ``[0, 1]``."""
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise TypeError(f"score must be a number, got {type(score).__name__}")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"score must be a finite number in [0, 1], got {score!r}")
        with self._lock:
            self._trust_scores.append(value)

    def record_conflict(self, entry_id: str, now: datetime | None = None) -> None:
        """Record a memory conflict on ``entry_id`` at ``now`` (default: clock)."""
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("entry_id must be a non-empty string")
        moment = _as_utc(now) if now is not None else _as_utc(self._now())
        with self._lock:
            self._conflicts.append({"timestamp": moment, "entry_id": entry_id})

    # ── snapshot API ──────────────────────────────────────────────────────
    def collect_snapshot(self, now: datetime | None = None) -> MetricsSnapshot:
        """Aggregate the current windows into a :class:`MetricsSnapshot`.

        Percentiles use linear interpolation over each op's bounded window;
        the conflict rate counts conflicts in the trailing 7 days relative to
        ``now`` divided by 168 (conflicts/hour — see module docstring).
        """
        moment = _as_utc(now) if now is not None else _as_utc(self._now())
        with self._lock:
            percentiles: dict[str, dict[str, float]] = {}
            for op, window in self._latencies.items():
                if not window:
                    continue
                ordered = sorted(window)
                percentiles[op] = {
                    "p50": _percentile(ordered, 0.50),
                    "p95": _percentile(ordered, 0.95),
                    "p99": _percentile(ordered, 0.99),
                }
            token_usage = dict(self._tokens)
            trust_scores = [float(score) for score in self._trust_scores]
            trust_stats = _trust_distribution(trust_scores)
            cutoff = moment - timedelta(days=CONFLICT_WINDOW_DAYS)
            conflict_events = [
                {"timestamp": event["timestamp"], "entry_id": str(event["entry_id"])}
                for event in self._conflicts
                if isinstance(event["timestamp"], datetime)
                and _as_utc(event["timestamp"]) >= cutoff
            ]
            counters = {"conflicts_total": len(self._conflicts)}
        return MetricsSnapshot(
            timestamp=moment,
            latency_percentiles=percentiles,
            token_usage=token_usage,
            trust_scores=trust_scores,
            trust_stats=trust_stats,
            conflict_events=conflict_events,
            conflict_rate_7d=len(conflict_events) / CONFLICT_WINDOW_HOURS,
            counters=counters,
        )

    def append_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Persist ``snapshot`` as one JSON line (single atomic append + fsync).

        When a UTC day boundary is crossed since the previous append, the
        90-day retention prune runs first, so the new line is never dropped.
        """
        if not isinstance(snapshot, MetricsSnapshot):
            raise TypeError(f"snapshot must be a MetricsSnapshot, got {type(snapshot).__name__}")
        line = json.dumps(snapshot.to_dict(), ensure_ascii=False)
        day = _day_stamp(_as_utc(snapshot.timestamp))
        with self._lock:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            if self._last_append_day is not None and day != self._last_append_day:
                self._prune_locked(RETENTION_DAYS, _as_utc(snapshot.timestamp))
            with open(self.store_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._last_append_day = day

    def load_history(self) -> list[MetricsSnapshot]:
        """Read back every snapshot from the JSONL file (oldest first).

        Missing file -> ``[]``. Malformed lines raise ``ValueError``.
        """
        with self._lock:
            if not self.store_path.exists():
                return []
            history: list[MetricsSnapshot] = []
            with open(self.store_path, encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    history.append(MetricsSnapshot.from_dict(json.loads(stripped)))
            return history

    def prune(
        self, retention_days: int = RETENTION_DAYS, now: datetime | None = None
    ) -> int:
        """Drop JSONL lines older than ``retention_days``; returns the count.

        The file is rewritten atomically (tmp file + ``os.replace``). Lines
        that cannot be parsed as snapshots are preserved as-is — pruning never
        destroys data it cannot interpret.
        """
        moment = _as_utc(now) if now is not None else _as_utc(self._now())
        with self._lock:
            return self._prune_locked(retention_days, moment)

    # ── internals ─────────────────────────────────────────────────────────
    def _prune_locked(self, retention_days: int, moment: datetime) -> int:
        """Prune implementation; caller holds ``self._lock``."""
        if retention_days < 0:
            raise ValueError(f"retention_days must be >= 0, got {retention_days}")
        cutoff = moment - timedelta(days=retention_days)
        if not self.store_path.exists():
            return 0
        kept: list[str] = []
        removed = 0
        with open(self.store_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    snapshot = MetricsSnapshot.from_dict(json.loads(stripped))
                except (TypeError, ValueError):
                    kept.append(stripped)  # preserve unparseable lines verbatim
                    continue
                if _as_utc(snapshot.timestamp) >= cutoff:
                    kept.append(stripped)
                else:
                    removed += 1
        tmp = self.store_path.with_name(f"{self.store_path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as out:
                out.write("".join(entry + "\n" for entry in kept))
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, self.store_path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return removed


def _validate_op(op: str) -> None:
    """Reject operation names that are not non-empty strings."""
    if not isinstance(op, str) or not op:
        raise ValueError("op must be a non-empty string")


# ── Sibling model-metrics ledger (appended additively by plan P14.T7) ────────
#
# Model-specific metrics (tok/s, eval scores, error/fallback rates) arrive
# per generation — a different event shape and cadence than the 5-minute
# MetricsSnapshot — so they live in a SIBLING JSONL ledger next to
# ``metrics.jsonl`` instead of extending the snapshot contract (nothing above
# changes signature). :mod:`hermes.diagnostics.model_metrics` owns that
# ledger; this constant lets dashboard/collect code discover the sibling file
# relative to the main ledger path.

#: File name of the sibling model-metrics ledger in the same directory as
#: :func:`default_metrics_path` (plan P14.T7).
SIBLING_LEDGER_NAME: Final[str] = "model_metrics.jsonl"


def _day_stamp(moment: datetime) -> str:
    """UTC calendar day (``YYYY-MM-DD``) of a moment."""
    return moment.strftime("%Y-%m-%d")
