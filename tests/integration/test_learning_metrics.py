"""P14.T8 integration: the trend report over a real ledger (plan Phase 14).

A ledger recording 4 improving GEPA runs followed by 3 flat runs produces
the per-component trend report (windowed, average improvement, trend,
consecutive-no-improvement streak) and the learning-loop alert condition
goes true — the exact plan acceptance scenario ("alert if no improvement in
3 consecutive runs").
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.diagnostics.learning_metrics import (
    LearningMetricsLedger,
    alert_payload,
    should_alert,
)

_T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)  # a Sunday 03:00 UTC weekly slot


@pytest.mark.integration
def test_trend_report_generated(tmp_path: Path) -> None:
    """4 improving runs then 3 flat runs: report generated, alert condition true."""
    ledger_path = tmp_path / "diagnostics" / "learning_metrics.jsonl"
    ledger = LearningMetricsLedger(ledger_path=ledger_path, now=lambda: _T0)

    score = 0.50
    for week in range(7):
        improvement = 0.02 if week < 4 else 0.001  # improving, then flat
        after = score + improvement
        ledger.record_run(
            "extractor",
            f"gepa-week-{week}",
            score,
            after,
            deployed=improvement >= 0.005,
            finished_at=_T0 + timedelta(weeks=week),
        )
        score = after

    assert ledger_path.is_file()  # the ledger really persisted the 7 runs
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 7

    report = ledger.trend_report()  # default window (10) spans all weekly runs
    assert report.window_runs == 10
    trend = report.components["extractor"]
    assert trend.runs == 7
    assert trend.avg_improvement == pytest.approx((4 * 0.02 + 3 * 0.001) / 7)
    assert trend.trend == "improving"  # the average still improves
    assert trend.consecutive_no_improvement == 3  # the trailing flat streak

    assert should_alert(report) is True
    payload = alert_payload(report)
    assert payload is not None
    assert payload["alert"] == "learning_no_improvement"
    assert payload["components"]["extractor"]["consecutive_no_improvement"] == 3
    assert json.dumps(report.to_dict(), ensure_ascii=False)  # report is JSON-safe

    # a tighter window sees only the trailing flat runs
    recent = ledger.trend_report(window_runs=3)
    recent_trend = recent.components["extractor"]
    assert recent_trend.runs == 3
    assert recent_trend.trend == "flat"
    assert recent_trend.avg_improvement == pytest.approx(0.001)
    assert should_alert(recent) is True
