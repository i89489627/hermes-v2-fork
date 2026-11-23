"""hermes-v2 memory quality trend diagnostics (plan P14.T9).

Weekly time series of memory-layer quality (plan P14.T9: "Track memory quality
trends over time") as a JSONL ledger under
``<hermes_home>/diagnostics/memory_quality.jsonl`` — one
:class:`QualityPoint` per week bucket:

- **trust distribution** — ``avg``/``p10``/``p50``/``p90`` (linear
  interpolation, matching :mod:`hermes.diagnostics.metrics`) over the live
  entries' ``trust_score`` values (plan: "Trust score distribution (avg, p10,
  p50, p90) over time"); an empty store reports zeros;
- **conflict rate** — ``conflict_rate_7d``, the trailing 7-day rate in the
  same unit as :attr:`~hermes.diagnostics.metrics.MetricsSnapshot.
  conflict_rate_7d` (conflicts per hour, ``7 * 24`` denominator): from the
  wired :class:`~hermes.diagnostics.metrics.MetricsCollector` when one is
  injected (the pipeline records conflicts there), otherwise computed from
  the store's verification events — a negative anchor (``value == 0.0``,
  e.g. a red test disproving the claim) in the trailing window counts as a
  conflict;
- **archive rate** — ``archive_count_this_week``: entries in
  ``<memory-repo>/archive/`` (plan P8.T6) whose ``archived_at`` frontmatter
  stamp falls inside the ISO week bucket (Monday 00:00 UTC based) of the
  snapshot;
- **retrieval recall** — ``recall_at_5`` needs an evaluation to produce, so
  it is injected: ``recall_probe: Callable[[], float] | None`` (``None`` ->
  ``None`` — the plan's "retrieval recall@5 over time" stays honest about
  not being measured until an eval is wired).

Public surface: :meth:`MemoryQualityTracker.record_weekly` appends one point
(the wall clock is injectable via ``now=`` so tests stay deterministic),
:meth:`MemoryQualityTracker.load_history` reads the persisted series back,
:meth:`MemoryQualityTracker.trend` / :meth:`MemoryQualityTracker.trends`
classify each tracked metric's direction (``rising``/``flat``/``falling``,
least-squares slope sign — rising slope means the metric goes up, which is
good for trust and bad for conflicts; the direction is neutral), and
:func:`should_alert` / :func:`alert_payload` implement the plan's "declining
trust score trend triggers alert" acceptance: trust avg slope negative over
>= 3 points AND last point < first * :data:`TRUST_DECLINE_RATIO` -> the
``trust_declining`` payload. :func:`due_for_weekly_snapshot` is the pure
7-day-cadence scheduling helper the pipeline ticks against (mirroring
:func:`hermes.diagnostics.metrics.due_for_collection`).

Trends are dashboard-reportable (plan acceptance: "trends visible in
dashboard"): :meth:`Trend.values` is the chronological series and
:meth:`QualityPoint.to_dict` the JSON projection the P14.T10 dashboard
renders.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

from hermes.config import hermes_home
from hermes.diagnostics.metrics import MetricsCollector
from hermes.memory.store import ARCHIVE_DIRNAME, MemoryStore

__all__ = [
    "ARCHIVE_STAMP_KEY",
    "CONFLICT_WINDOW_DAYS",
    "CONFLICT_WINDOW_HOURS",
    "TRACKED_METRICS",
    "TRUST_DECLINE_RATIO",
    "TRUST_DECLINING_ALERT",
    "TREND_EPSILON",
    "WEEK_CADENCE_DAYS",
    "MemoryQualityTracker",
    "QualityPoint",
    "Trend",
    "TrendDirection",
    "alert_payload",
    "default_memory_quality_path",
    "due_for_weekly_snapshot",
    "should_alert",
]

#: Weekly cadence (plan P14.T9: one point per week).
WEEK_CADENCE_DAYS: Final[int] = 7

#: Conflict-rate lookback and its hour denominator — same trailing-7-day
#: window and conflicts/hour unit as
#: :attr:`~hermes.diagnostics.metrics.MetricsSnapshot.conflict_rate_7d`.
CONFLICT_WINDOW_DAYS: Final[int] = 7
CONFLICT_WINDOW_HOURS: Final[float] = float(CONFLICT_WINDOW_DAYS * 24)

#: |slope| below this counts as a flat trend (mirrors P14.T7's epsilon).
TREND_EPSILON: Final[float] = 1e-9

#: Trust-decline alert threshold (plan P14.T9: last < first * 0.9).
TRUST_DECLINE_RATIO: Final[float] = 0.9

#: Alert name for the declining-trust rule (plan P14.T9 acceptance).
TRUST_DECLINING_ALERT: Final[str] = "trust_declining"

#: The metrics :meth:`MemoryQualityTracker.trend` tracks — the QualityPoint
#: field names, in dashboard order (trust distribution, conflict, archive,
#: recall).
TRACKED_METRICS: Final[tuple[str, ...]] = (
    "trust_avg",
    "trust_p10",
    "trust_p50",
    "trust_p90",
    "conflict_rate_7d",
    "archive_count_this_week",
    "recall_at_5",
)

#: Frontmatter key the P8.T6 archiver stamps onto archived files.
ARCHIVE_STAMP_KEY: Final[str] = "archived_at"

#: Injectable wall clock: returns the current datetime (aware UTC by convention).
Clock = Callable[[], datetime]

#: Trend vocabulary: the least-squares slope sign of the chronological series.
TrendDirection = Literal["rising", "flat", "falling"]


def _utcnow() -> datetime:
    """Default clock: current aware UTC time."""
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def default_memory_quality_path() -> Path:
    """Default ledger path: ``<hermes_home>/diagnostics/memory_quality.jsonl``."""
    path: Path = hermes_home() / "diagnostics" / "memory_quality.jsonl"
    return path


def _week_start(moment: datetime) -> datetime:
    """Monday 00:00 UTC of ``moment``'s ISO week (the weekly bucket start)."""
    utc = _as_utc(moment)
    monday = utc.date() - timedelta(days=utc.date().weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=UTC)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentile ``q`` (0..1) of already-sorted values; ``0.0`` when empty.

    Linear interpolation between neighbouring order statistics (numpy's
    default ``linear`` method), matching
    :mod:`hermes.diagnostics.metrics` and P14.T7. An empty sample reports
    ``0.0`` so trust stats stay JSON-stable on an empty store.
    """
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    if lower >= len(sorted_values) - 1:
        return sorted_values[-1]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[lower + 1] - sorted_values[lower]) * fraction


def _slope_sign(values: list[float]) -> TrendDirection:
    """Direction of ``values`` (chronological) from the least-squares slope sign."""
    if len(values) < 2:
        return "flat"
    count = len(values)
    mean_x = (count - 1) / 2.0
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator == 0.0:  # pragma: no cover - count >= 2 implies positive
        return "flat"
    slope = numerator / denominator
    if slope > TREND_EPSILON:
        return "rising"
    if slope < -TREND_EPSILON:
        return "falling"
    return "flat"


def due_for_weekly_snapshot(last: datetime | None, now: datetime) -> bool:
    """Pure scheduling helper: is a weekly quality snapshot due?

    ``True`` when no snapshot was taken yet (``last`` is ``None``) or when at
    least :data:`WEEK_CADENCE_DAYS` days elapsed between ``last`` and ``now``.
    A ``last`` in the future relative to ``now`` is "not due" (no catch-up
    storms from clock skew). Naive datetimes are interpreted as UTC.
    """
    if last is None:
        return True
    elapsed = (_as_utc(now) - _as_utc(last)).total_seconds()
    return elapsed >= WEEK_CADENCE_DAYS * 86400.0


def _parse_archived_at(path: Path) -> datetime | None:
    """The ``archived_at`` frontmatter stamp of an archived fact file.

    Reads only the leading YAML frontmatter block and the
    :data:`ARCHIVE_STAMP_KEY` line the P8.T6 archiver writes
    (``archived_at: <ISO-Z>``); unreadable or stamp-less files yield ``None``
    (they never count toward a week and never raise).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    closing = text.find("\n---", 3)
    if closing < 0:
        return None
    for line in text[3:closing].splitlines():
        stripped = line.strip()
        if not stripped.startswith(f"{ARCHIVE_STAMP_KEY}:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        try:
            moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _as_utc(moment)
    return None


@dataclass
class QualityPoint:
    """One weekly memory-quality sample (plan P14.T9), JSONL-serializable.

    ``week_start`` is the ISO week bucket (Monday 00:00 UTC); the trust
    percentiles span the live entries; ``recall_at_5`` is ``None`` when no
    recall probe is wired (not measured, not zero).
    """

    week_start: datetime
    trust_avg: float = 0.0
    trust_p10: float = 0.0
    trust_p50: float = 0.0
    trust_p90: float = 0.0
    conflict_rate_7d: float = 0.0
    archive_count_this_week: int = 0
    recall_at_5: float | None = None
    #: Metrics recorded alongside but not part of the plan's four trends.
    entry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection (one ledger line / dashboard input)."""
        return {
            "week_start": _as_utc(self.week_start).isoformat(),
            "trust_avg": self.trust_avg,
            "trust_p10": self.trust_p10,
            "trust_p50": self.trust_p50,
            "trust_p90": self.trust_p90,
            "conflict_rate_7d": self.conflict_rate_7d,
            "archive_count_this_week": self.archive_count_this_week,
            "recall_at_5": self.recall_at_5,
            "entry_count": self.entry_count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QualityPoint:
        """Rebuild from :meth:`to_dict` output; malformed records raise ``ValueError``."""
        try:
            week_start = datetime.fromisoformat(str(raw["week_start"]))
            if week_start.tzinfo is None:
                week_start = week_start.replace(tzinfo=UTC)
            recall_raw = raw.get("recall_at_5")
            recall = None if recall_raw is None else float(recall_raw)
            return cls(
                week_start=week_start,
                trust_avg=float(raw["trust_avg"]),
                trust_p10=float(raw["trust_p10"]),
                trust_p50=float(raw["trust_p50"]),
                trust_p90=float(raw["trust_p90"]),
                conflict_rate_7d=float(raw["conflict_rate_7d"]),
                archive_count_this_week=int(raw["archive_count_this_week"]),
                recall_at_5=recall,
                entry_count=int(raw.get("entry_count", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed quality point record: {raw!r}") from exc


@dataclass
class Trend:
    """Direction + chronological values of one tracked metric (dashboard-ready).

    ``direction`` is the neutral slope sign (``rising``/``flat``/``falling``):
    a rising conflict rate is bad while rising trust is good — interpretation
    belongs to the reader/alert rule, not the trend itself.
    """

    direction: TrendDirection
    values: list[float] = field(default_factory=list)


def should_alert(history: Sequence[QualityPoint]) -> bool:
    """Declining-trust rule: avg slope negative over >= 3 points AND last < first * ratio.

    ``history`` must be chronological (oldest first, as
    :meth:`MemoryQualityTracker.load_history` returns). Fewer than three
    points never alert — a two-point drop has no confirmed trend.
    """
    trust_values = [point.trust_avg for point in history]
    if len(trust_values) < 3:
        return False
    if _slope_sign(trust_values) != "falling":
        return False
    first = trust_values[0]
    last = trust_values[-1]
    return last < first * TRUST_DECLINE_RATIO


def alert_payload(history: Sequence[QualityPoint]) -> dict[str, Any] | None:
    """The ``trust_declining`` alert payload, or ``None`` when not alerting.

    Payload shape mirrors the plan 5.3 alert entries (alert name, severity,
    observed values, threshold) so the P14.T10 dashboard can render it
    alongside the engine's alerts.
    """
    if not should_alert(history):
        return None
    trust_values = [point.trust_avg for point in history]
    return {
        "alert": TRUST_DECLINING_ALERT,
        "severity": "warning",
        "trust_values": trust_values,
        "first": trust_values[0],
        "last": trust_values[-1],
        "decline_ratio_threshold": TRUST_DECLINE_RATIO,
        "week_start": _as_utc(history[-1].week_start).isoformat(),
    }


class MemoryQualityTracker:
    """Records weekly memory-quality points and persists them (plan P14.T9).

    Args:
        store: the git-backed memory store whose live entries (and archive
            directory) are sampled.
        ledger_path: JSONL ledger (default
            :func:`default_memory_quality_path`, i.e.
            ``<hermes_home>/diagnostics/memory_quality.jsonl``).
        recall_probe: injectable recall@5 evaluation; ``None`` (default)
            records ``recall_at_5=None`` — recall needs an eval to exist.
        collector: optional wired
            :class:`~hermes.diagnostics.metrics.MetricsCollector`; when
            present, ``conflict_rate_7d`` comes from its snapshot, otherwise
            it is computed from the store's negative verification events.
    """

    def __init__(
        self,
        store: MemoryStore,
        ledger_path: Path | None = None,
        recall_probe: Callable[[], float] | None = None,
        collector: MetricsCollector | None = None,
    ) -> None:
        self.store = store
        self.ledger_path: Path = (
            ledger_path if ledger_path is not None else default_memory_quality_path()
        )
        self.recall_probe = recall_probe
        self.collector = collector
        self._lock = threading.Lock()

    # ── recording API ─────────────────────────────────────────────────────
    def record_weekly(self, now: datetime | None = None) -> QualityPoint:
        """Sample the store into one :class:`QualityPoint` and append it.

        ``now`` (default: the wall clock) selects the ISO week bucket and the
        trailing conflict window. Recording is unconditional — callers gate
        the cadence with :func:`due_for_weekly_snapshot` (the
        :func:`hermes.diagnostics.metrics.due_for_collection` pattern), so a
        missed week stays missed instead of being back-filled with
        fabricated data.
        """
        moment = _as_utc(now) if now is not None else _utcnow()
        week_start = _week_start(moment)
        entries = self.store.list()
        scores = sorted(float(entry.trust_score) for entry in entries)
        point = QualityPoint(
            week_start=week_start,
            trust_avg=(sum(scores) / len(scores)) if scores else 0.0,
            trust_p10=_percentile(scores, 0.10),
            trust_p50=_percentile(scores, 0.50),
            trust_p90=_percentile(scores, 0.90),
            conflict_rate_7d=self._conflict_rate(moment),
            archive_count_this_week=self._archived_in_week(week_start),
            recall_at_5=self._recall(),
            entry_count=len(entries),
        )
        self._append(point)
        return point

    # ── reading API ───────────────────────────────────────────────────────
    def load_history(self) -> list[QualityPoint]:
        """Read back every point from the ledger (oldest first).

        Missing file -> ``[]``; malformed lines raise ``ValueError``.
        """
        with self._lock:
            if not self.ledger_path.exists():
                return []
            history: list[QualityPoint] = []
            with open(self.ledger_path, encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    history.append(QualityPoint.from_dict(json.loads(stripped)))
            return history

    def trend(self, metric: str) -> Trend:
        """Direction + values of ``metric`` over the persisted history.

        ``metric`` is a :data:`TRACKED_METRICS` name (unknown names raise
        ``ValueError``). ``None`` samples (recall without a probe) are skipped
        so the trend only spans measured points.
        """
        if metric not in TRACKED_METRICS:
            raise ValueError(f"unknown metric: {metric!r}; tracked: {list(TRACKED_METRICS)}")
        values: list[float] = []
        for point in self.load_history():
            raw = getattr(point, metric)
            if raw is not None:
                values.append(float(raw))
        return Trend(direction=_slope_sign(values), values=values)

    def trends(self) -> dict[str, Trend]:
        """Every tracked metric's trend (the dashboard-reportable view)."""
        return {metric: self.trend(metric) for metric in TRACKED_METRICS}

    # ── internals ─────────────────────────────────────────────────────────
    def _conflict_rate(self, moment: datetime) -> float:
        """Trailing 7-day conflict rate (conflicts/hour) at ``moment``.

        The wired collector's snapshot is authoritative when present;
        otherwise a negative verification anchor (``value == 0.0``) within the
        window counts as a conflict.
        """
        if self.collector is not None:
            return float(self.collector.collect_snapshot(moment).conflict_rate_7d)
        cutoff = moment - timedelta(days=CONFLICT_WINDOW_DAYS)
        count = 0
        for entry in self.store.list():
            for event in entry.verification_events:
                stamp = _as_utc(event.timestamp)
                if cutoff <= stamp <= moment and event.value == 0.0:
                    count += 1
        return count / CONFLICT_WINDOW_HOURS

    def _archived_in_week(self, week_start: datetime) -> int:
        """Entries archived inside ``[week_start, week_start + 7 days)``."""
        archive_dir = self.store.root / ARCHIVE_DIRNAME
        if not archive_dir.is_dir():
            return 0
        window_end = week_start + timedelta(days=WEEK_CADENCE_DAYS)
        count = 0
        for path in archive_dir.glob("*.md"):
            stamp = _parse_archived_at(path)
            if stamp is not None and week_start <= stamp < window_end:
                count += 1
        return count

    def _recall(self) -> float | None:
        """The injected probe's recall@5, or ``None`` when not wired."""
        if self.recall_probe is None:
            return None
        return float(self.recall_probe())

    def _append(self, point: QualityPoint) -> None:
        """Persist ``point`` as one JSON line (single atomic append + fsync)."""
        line = json.dumps(point.to_dict(), ensure_ascii=False)
        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
