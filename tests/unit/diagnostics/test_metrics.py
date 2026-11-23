"""P14.T2 acceptance: metrics collection (plan Phase 14).

Latency percentiles are computed with linear interpolation over each op's
bounded window; token usage accumulates per op and survives a JSONL
append/load roundtrip; trust distribution stats cover avg/min/max and the
count below the 0.3 low-trust threshold; the conflict rate rolls over a
trailing 7-day window (unit: conflicts/hour). Collection cadence (every
5 minutes) and the 90-day series retention are exercised through the
:func:`due_for_collection` helper and the day-boundary prune.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.diagnostics.metrics import (
    MetricsCollector,
    MetricsSnapshot,
    due_for_collection,
)

_T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


@pytest.mark.unit
def test_latency_percentiles_computed(tmp_path: Path) -> None:
    """p50/p95/p99 use linear interpolation; ops are isolated; windows are bounded."""
    # Default store path derives from hermes_home() so tests stay hermetic.
    assert MetricsCollector().store_path == hermes_home() / "diagnostics" / "metrics.jsonl"

    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    for value in (0.1, 0.2, 0.3, 0.4):
        collector.record_latency("extract", value)
    collector.record_latency("inject", 0.5)
    snapshot = collector.collect_snapshot(now=_T0)
    assert snapshot.timestamp == _T0

    extract = snapshot.latency_percentiles["extract"]
    # rank(n=4, q) = 3 * q: p50 -> 1.5 -> 0.2 + 0.5 * 0.1; p95 -> 2.85; p99 -> 2.97
    assert extract["p50"] == pytest.approx(0.25)
    assert extract["p95"] == pytest.approx(0.385)
    assert extract["p99"] == pytest.approx(0.397)
    # a single sample makes every percentile that sample
    inject = snapshot.latency_percentiles["inject"]
    assert inject["p50"] == pytest.approx(0.5)
    assert inject["p95"] == pytest.approx(0.5)
    assert inject["p99"] == pytest.approx(0.5)

    # bounded window: only the newest sample_window samples are kept per op
    bounded = MetricsCollector(store_path=tmp_path / "bounded.jsonl", sample_window=3)
    for value in (10.0, 20.0, 30.0, 40.0):
        bounded.record_latency("extract", value)
    window = bounded.collect_snapshot(now=_T0).latency_percentiles["extract"]
    assert window["p50"] == pytest.approx(30.0)
    assert window["p95"] == pytest.approx(39.0)
    assert window["p99"] == pytest.approx(39.8)

    # negative / non-finite samples are rejected before they can skew percentiles
    with pytest.raises(ValueError):
        collector.record_latency("extract", -1.0)
    with pytest.raises(ValueError):
        collector.record_latency("extract", float("nan"))


@pytest.mark.unit
def test_token_usage_accumulated(tmp_path: Path) -> None:
    """Token totals accumulate per op and roundtrip through the JSONL store."""
    path = tmp_path / "diagnostics" / "metrics.jsonl"
    collector = MetricsCollector(store_path=path)
    collector.record_tokens("extract", 120)
    collector.record_tokens("extract", 80)
    collector.record_tokens("inject", 45)
    snapshot = collector.collect_snapshot(now=_T0)
    assert snapshot.token_usage == {"extract": 200, "inject": 45}

    # JSONL export: one line per snapshot, reloadable into equivalent objects
    collector.append_snapshot(snapshot)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["token_usage"] == {"extract": 200, "inject": 45}
    restored = MetricsCollector(store_path=path).load_history()
    assert len(restored) == 1
    assert restored[0].token_usage == {"extract": 200, "inject": 45}
    assert restored[0].timestamp == snapshot.timestamp

    # negative / non-int token counts are rejected
    with pytest.raises(ValueError):
        collector.record_tokens("extract", -1)
    with pytest.raises(TypeError):
        collector.record_tokens("extract", 1.5)


@pytest.mark.unit
def test_trust_distribution(tmp_path: Path) -> None:
    """trust_stats reports avg/min/max and the count below the 0.3 threshold."""
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    for score in (0.9, 0.2, 0.5):
        collector.record_trust_score(score)
    snapshot = collector.collect_snapshot(now=_T0)
    assert snapshot.trust_scores == [0.9, 0.2, 0.5]
    stats = snapshot.trust_stats
    assert stats["avg"] == pytest.approx((0.9 + 0.2 + 0.5) / 3)
    assert stats["min"] == pytest.approx(0.2)
    assert stats["max"] == pytest.approx(0.9)
    assert stats["count_below_0.3"] == pytest.approx(1.0)

    # an empty collector yields a zeroed, JSON-stable distribution
    empty = MetricsCollector(store_path=tmp_path / "empty.jsonl").collect_snapshot(now=_T0)
    assert empty.trust_scores == []
    assert empty.trust_stats == {
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
        "count_below_0.3": 0.0,
    }

    # scores outside [0, 1] are rejected
    with pytest.raises(ValueError):
        collector.record_trust_score(1.5)
    with pytest.raises(ValueError):
        collector.record_trust_score(-0.1)


@pytest.mark.unit
def test_conflict_rate_7d_rolling(tmp_path: Path) -> None:
    """Conflict rate counts only the trailing 7 days (unit: conflicts/hour)."""
    path = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(store_path=path, now=lambda: _T0)
    collector.record_conflict("entry-old", now=_T0 - timedelta(days=8))
    collector.record_conflict("entry-recent", now=_T0 - timedelta(days=2, hours=1))

    snapshot = collector.collect_snapshot(now=_T0)
    assert [event["entry_id"] for event in snapshot.conflict_events] == ["entry-recent"]
    # 1 conflict / (7 * 24h) — the documented unit is conflicts per hour
    assert snapshot.conflict_rate_7d == pytest.approx(1 / 168)
    assert snapshot.counters["conflicts_total"] == 2

    # the window rolls forward with `now`: still inside after 3 days, gone after 10
    later = collector.collect_snapshot(now=_T0 + timedelta(days=3))
    assert [event["entry_id"] for event in later.conflict_events] == ["entry-recent"]
    rolled_out = collector.collect_snapshot(now=_T0 + timedelta(days=10))
    assert rolled_out.conflict_events == []
    assert rolled_out.conflict_rate_7d == 0.0

    # cadence helper: collections are due every 5 minutes
    assert due_for_collection(None, now=_T0)
    assert not due_for_collection(_T0, now=_T0 + timedelta(minutes=4))
    assert due_for_collection(_T0, now=_T0 + timedelta(minutes=5))

    # 90-day retention: crossing a day boundary between appends prunes old lines
    series = MetricsCollector(store_path=path)
    stale = MetricsSnapshot(timestamp=_T0 - timedelta(days=100))
    fresh = MetricsSnapshot(timestamp=_T0 - timedelta(days=1))
    series.append_snapshot(stale)
    series.append_snapshot(fresh)  # different UTC day -> automatic prune
    history = series.load_history()
    assert [entry.timestamp for entry in history] == [fresh.timestamp]
