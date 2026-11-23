"""P14.T9 acceptance: memory quality trend diagnostics (plan Phase 14).

Covers the spec behaviours: the weekly trust distribution (avg/p10/p50/p90
over the live entries) plus the declining-trust alert rule, the trailing
7-day conflict rate (wired collector and store-event fallback), the weekly
archive count from the P8.T6 archive directory, and the injectable recall@5
probe trend. Real git-backed MemoryStore + real MetricsCollector; clocks are
fake (``record_weekly(now=...)``) so weeks are deterministic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.diagnostics.memory_quality_metrics import (
    TRACKED_METRICS,
    MemoryQualityTracker,
    QualityPoint,
    alert_payload,
    default_memory_quality_path,
    due_for_weekly_snapshot,
    should_alert,
)
from hermes.diagnostics.metrics import MetricsCollector
from hermes.memory.decay import archive_stale
from hermes.memory.store import MemoryStore
from hermes.memory.types import VerificationEvent

_BASE = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)  # a Monday, noon UTC


@pytest.mark.unit
def test_trust_distribution_tracked(tmp_path: Path) -> None:
    """Weekly trust distribution (avg/p10/p50/p90) tracked + trust_declining alert."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    low = store.add("low trust note", type="Reference", source="agent-inferred")
    mid = store.add("mid trust note", type="Feedback", source="agent-inferred")
    high = store.add("high trust note", type="Feedback", source="user-explicit")
    store.update(low.id, trust_score=0.3)
    store.update(mid.id, trust_score=0.5)
    store.update(high.id, trust_score=1.0)

    tracker = MemoryQualityTracker(
        store, ledger_path=tmp_path / "memory_quality.jsonl", recall_probe=lambda: 0.8
    )
    point = tracker.record_weekly(now=_BASE)

    # the point is bucketed to the ISO week (Monday 00:00 UTC)
    assert point.week_start == datetime(2026, 11, 2, tzinfo=UTC)
    assert point.entry_count == 3
    assert point.trust_avg == pytest.approx((0.3 + 0.5 + 1.0) / 3)
    # linear interpolation over the sorted scores [0.3, 0.5, 1.0]
    assert point.trust_p10 == pytest.approx(0.34)  # rank 0.2
    assert point.trust_p50 == pytest.approx(0.5)
    assert point.trust_p90 == pytest.approx(0.9)  # rank 1.8

    # two more weekly points with sinking trust: the trend falls and the
    # plan's declining-trust alert rule fires on the tracker's own history
    store.update(low.id, trust_score=0.4)
    store.update(mid.id, trust_score=0.45)
    store.update(high.id, trust_score=0.5)
    tracker.record_weekly(now=_BASE + timedelta(days=7))  # avg 0.45
    store.update(low.id, trust_score=0.3)
    store.update(mid.id, trust_score=0.35)
    store.update(high.id, trust_score=0.4)
    third = tracker.record_weekly(now=_BASE + timedelta(days=14))  # avg 0.35

    history = tracker.load_history()
    assert [entry.trust_avg for entry in history] == pytest.approx([0.6, 0.45, 0.35])
    assert third.week_start == datetime(2026, 11, 16, tzinfo=UTC)  # next ISO weeks
    trust_trend = tracker.trend("trust_avg")
    assert trust_trend.direction == "falling"
    assert trust_trend.values == pytest.approx([0.6, 0.45, 0.35])

    assert should_alert(history) is True
    payload = alert_payload(history)
    assert payload is not None
    assert payload["alert"] == "trust_declining"
    assert payload["severity"] == "warning"
    assert payload["first"] == pytest.approx(0.6)
    assert payload["last"] == pytest.approx(0.35)

    # the alert rule's documented edges: <3 points never alert, a flat or
    # rising slope never alerts, and the last<first*0.9 ratio is required
    assert should_alert(history[:2]) is False
    flat = [
        QualityPoint(week_start=_BASE + timedelta(weeks=index), trust_avg=0.5)
        for index in range(3)
    ]
    assert should_alert(flat) is False
    improving = [
        QualityPoint(week_start=_BASE + timedelta(weeks=index), trust_avg=0.4 + 0.2 * index)
        for index in range(3)
    ]
    assert should_alert(improving) is False
    mild = [
        QualityPoint(week_start=_BASE + timedelta(weeks=index), trust_avg=0.8 - 0.035 * index)
        for index in range(3)
    ]  # falling slope, but last (0.73) >= first * 0.9 (0.72)
    assert should_alert(mild) is False
    assert alert_payload(mild) is None

    # every tracked metric has a trend (dashboard-reportable view)
    assert list(tracker.trends()) == list(TRACKED_METRICS)
    for metric in TRACKED_METRICS:
        single = tracker.trend(metric)
        assert single.direction in ("rising", "flat", "falling")

    # points persist as one JSON line each and round-trip through from_dict
    lines = [
        json.loads(line)
        for line in tracker.ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    assert lines[0]["week_start"] == "2026-11-02T00:00:00+00:00"
    reloaded = MemoryQualityTracker(store, ledger_path=tracker.ledger_path)
    assert [entry.to_dict() for entry in reloaded.load_history()] == [
        entry.to_dict() for entry in history
    ]

    # weekly cadence helper: 7 days, exactly
    assert due_for_weekly_snapshot(None, _BASE) is True
    assert due_for_weekly_snapshot(_BASE, _BASE + timedelta(days=6, hours=23)) is False
    assert due_for_weekly_snapshot(_BASE, _BASE + timedelta(days=7)) is True
    assert due_for_weekly_snapshot(_BASE + timedelta(days=1), _BASE) is False

    # default ledger path derives from hermes_home; an empty store reports zeros
    assert default_memory_quality_path() == hermes_home() / "diagnostics" / "memory_quality.jsonl"
    empty_tracker = MemoryQualityTracker(MemoryStore(root=tmp_path / "empty-repo"))
    assert empty_tracker.ledger_path == hermes_home() / "diagnostics" / "memory_quality.jsonl"
    empty_point = empty_tracker.record_weekly(now=_BASE)
    assert (
        empty_point.trust_avg,
        empty_point.trust_p10,
        empty_point.trust_p50,
        empty_point.trust_p90,
    ) == (0.0, 0.0, 0.0, 0.0)

    # malformed ledger lines are rejected loudly, never silently skipped
    # (JSONDecodeError is a ValueError, same contract as the metrics ledger)
    with open(tracker.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("{not json at all}\n")
    with pytest.raises(ValueError):
        tracker.load_history()


@pytest.mark.unit
def test_conflict_rate_trend(tmp_path: Path) -> None:
    """conflict_rate_7d tracked weekly: wired collector first, store events fallback."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("conflicted note", type="Feedback", source="agent-inferred")

    # wired collector: conflicts land there, the tracker reads its snapshot
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")
    tracker = MemoryQualityTracker(
        store, ledger_path=tmp_path / "memory_quality.jsonl", collector=collector
    )
    first = tracker.record_weekly(now=_BASE)
    assert first.conflict_rate_7d == pytest.approx(0.0)

    collector.record_conflict("entry-a", now=_BASE + timedelta(days=8))
    collector.record_conflict("entry-b", now=_BASE + timedelta(days=9))
    second = tracker.record_weekly(now=_BASE + timedelta(days=9))
    assert second.conflict_rate_7d == pytest.approx(2 / (7 * 24))

    conflict_trend = tracker.trend("conflict_rate_7d")
    assert conflict_trend.direction == "rising"
    assert conflict_trend.values == pytest.approx([0.0, 2 / (7 * 24)])

    # fallback (no collector): a negative verification anchor (value == 0.0,
    # e.g. a red test disproving the claim) in the trailing window counts
    fallback_tracker = MemoryQualityTracker(store, ledger_path=tmp_path / "fallback.jsonl")
    pre = fallback_tracker.record_weekly(now=_BASE - timedelta(days=1))
    assert pre.conflict_rate_7d == pytest.approx(0.0)
    entry = store.list()[0]
    store.record_event(
        entry.id,
        VerificationEvent(
            event_id="ev-negative",
            timestamp=_BASE,
            source="llm",
            verifier="test",
            evidence="red test disproves the claim",
            value=0.0,
        ),
    )
    post = fallback_tracker.record_weekly(now=_BASE + timedelta(days=1))
    assert post.conflict_rate_7d == pytest.approx(1 / (7 * 24))
    fallback_trend = fallback_tracker.trend("conflict_rate_7d")
    assert fallback_trend.values == pytest.approx([0.0, 1 / (7 * 24)])
    assert fallback_trend.direction == "rising"


@pytest.mark.unit
def test_archive_rate_tracked(tmp_path: Path) -> None:
    """Archive rate tracked weekly from the P8.T6 archive directory stamps."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    stale = store.add("stale low-trust note", type="Reference", source="agent-inferred")
    store.add("healthy note", type="Feedback", source="user-explicit")
    store.update(stale.id, trust_score=0.1)  # below the 0.2 archive threshold

    tracker = MemoryQualityTracker(store, ledger_path=tmp_path / "memory_quality.jsonl")

    # week 1: nothing archived yet
    archival_day = datetime.now(UTC) + timedelta(days=400)  # stale: >1y unverified
    before = tracker.record_weekly(now=archival_day - timedelta(days=7))
    assert before.archive_count_this_week == 0

    # the P8.T6 archiver moves the file and stamps archived_at = the given now
    assert archive_stale(store, now=archival_day) == 1
    assert store.get(stale.id) is None  # archived entries leave the live surface
    assert len(store.archived_entries()) == 1

    after = tracker.record_weekly(now=archival_day)
    assert after.archive_count_this_week == 1

    archive_trend = tracker.trend("archive_count_this_week")
    assert archive_trend.values == [0.0, 1.0]
    assert archive_trend.direction == "rising"

    # the archived stamp belongs to that week only: a point in the next
    # week bucket does not count it again
    later = tracker.record_weekly(now=archival_day + timedelta(days=7))
    assert later.archive_count_this_week == 0
    assert tracker.trend("archive_count_this_week").values == [0.0, 1.0, 0.0]

    # cadence check the pipeline ticks against
    assert due_for_weekly_snapshot(before.week_start, archival_day) is True


@pytest.mark.unit
def test_recall_trend(tmp_path: Path) -> None:
    """recall_at_5 tracked from the injected probe; unmeasured weeks stay None."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add("recalled note", type="Reference", source="agent-inferred")

    probe_values = [0.7, 0.8, 0.9]
    tracker = MemoryQualityTracker(
        store,
        ledger_path=tmp_path / "memory_quality.jsonl",
        recall_probe=lambda: probe_values.pop(0),
    )
    for weeks in range(3):
        expected = probe_values[0]  # the next value the probe will hand out
        point = tracker.record_weekly(now=_BASE + timedelta(days=7 * weeks))
        assert point.recall_at_5 == pytest.approx(expected)

    recall_trend = tracker.trend("recall_at_5")
    assert recall_trend.direction == "rising"
    assert recall_trend.values == pytest.approx([0.7, 0.8, 0.9])

    # without a probe recall is not measured (None, not zero) and the trend
    # spans only the measured points
    unprobed = MemoryQualityTracker(store, ledger_path=tmp_path / "unprobed.jsonl")
    assert unprobed.record_weekly(now=_BASE).recall_at_5 is None
    assert unprobed.trend("recall_at_5").values == []
    assert unprobed.trend("recall_at_5").direction == "flat"

    # unknown metrics are rejected, they never return a silent flat trend
    with pytest.raises(ValueError, match="unknown metric"):
        tracker.trend("not_a_metric")
