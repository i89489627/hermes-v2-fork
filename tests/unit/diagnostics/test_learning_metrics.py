"""P14.T8 acceptance: learning loop effectiveness diagnostics (plan Phase 14).

Fast single-component tests: GEPA run outcomes persist to the JSONL ledger
and their before/after improvements aggregate into the per-component trend
report, and the ``learning_no_improvement`` alert fires exactly when a
component shows 3 consecutive runs without improvement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.diagnostics.learning_metrics import (
    ALERT_NAME,
    LearningMetricsLedger,
    alert_payload,
    should_alert,
)

_T0 = datetime(2026, 11, 20, 3, 0, 0, tzinfo=UTC)


@pytest.mark.unit
def test_gepa_improvement_tracked(tmp_path: Path) -> None:
    """Before/after scores persist per run and aggregate into the trend."""
    ledger_path = tmp_path / "diagnostics" / "learning_metrics.jsonl"
    ledger = LearningMetricsLedger(ledger_path=ledger_path, now=lambda: _T0)
    ledger.record_run("extractor", "gepa-20261101", 0.50, 0.55, True,
                      finished_at=_T0 + timedelta(days=1))
    ledger.record_run("extractor", "gepa-20261108", 0.55, 0.60, True,
                      finished_at=_T0 + timedelta(days=8))
    ledger.record_run("extractor", "gepa-20261115", 0.60, 0.66, False,
                      finished_at=_T0 + timedelta(days=15))
    ledger.record_run("reviewer", "gepa-20261115", 0.40, 0.42, True,
                      finished_at=_T0 + timedelta(days=15))

    # every run is on the ledger with its derived improvement
    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    assert [record["run_id"] for record in lines][:3] == [
        "gepa-20261101",
        "gepa-20261108",
        "gepa-20261115",
    ]
    assert lines[0]["improvement"] == pytest.approx(0.05)
    assert lines[2]["deployed"] is False

    # the trend report aggregates per component over the trailing window
    report = ledger.trend_report("extractor")
    trend = report.components["extractor"]
    assert trend.runs == 3
    assert trend.avg_improvement == pytest.approx((0.05 + 0.05 + 0.06) / 3)
    assert trend.trend == "improving"
    assert trend.consecutive_no_improvement == 0
    assert report.components.get("reviewer") is None  # component filter respected

    everything = ledger.trend_report()
    assert set(everything.components) == {"extractor", "reviewer"}

    # a fresh instance over the same path reads the same runs (ledger survives)
    reopened = LearningMetricsLedger(ledger_path=ledger_path)
    assert reopened.trend_report("extractor").components["extractor"].runs == 3


@pytest.mark.unit
def test_no_improvement_alert(tmp_path: Path) -> None:
    """3 consecutive runs without improvement fire the alert; fewer do not."""
    ledger = LearningMetricsLedger(ledger_path=tmp_path / "learning_metrics.jsonl")
    base = _T0
    day = 0

    def record(improvements: list[float]) -> None:
        """Record runs with the given improvement sequence (later timestamps)."""
        nonlocal day
        score = 0.50
        for improvement in improvements:
            day += 1
            after = score + improvement
            ledger.record_run(
                "extractor",
                f"gepa-{day}",
                score,
                after,
                improvement >= 0.005,
                finished_at=base + timedelta(days=day),
            )
            score = after

    # two consecutive flats: below the threshold -> no alert
    record([0.02, 0.001, 0.0])
    report = ledger.trend_report()
    assert should_alert(report) is False
    assert alert_payload(report) is None

    # a third non-improving run (declining) reaches 3 in a row -> alert
    record([0.02, 0.001, 0.0, -0.01])
    report = ledger.trend_report()
    trend = report.components["extractor"]
    assert trend.consecutive_no_improvement == 3
    assert should_alert(report) is True
    payload = alert_payload(report)
    assert payload is not None
    assert payload["alert"] == ALERT_NAME == "learning_no_improvement"
    assert payload["severity"] == "warning"
    assert payload["threshold"] == 3
    assert "extractor" in payload["components"]

    # an improving run resets the streak
    record([0.02, 0.001, 0.0, -0.01, 0.03])
    report = ledger.trend_report()
    assert report.components["extractor"].consecutive_no_improvement == 0
    assert should_alert(report) is False

    # the payload is JSON-safe: with threshold 1 a single flat run fires it
    record([0.001])
    low = alert_payload(ledger.trend_report(), threshold=1)
    low = alert_payload(ledger.trend_report(), threshold=1)
    assert low is not None
    assert json.dumps(low, ensure_ascii=False)
