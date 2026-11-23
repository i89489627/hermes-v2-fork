"""P14.T7 acceptance tests: model performance diagnostics (plan Phase 14).

Covers the spec behaviours: tok/s tracking from recorded generations (p50/p95
over the window), quality-score tracking with trend classification, error
rate computation, and the plan 5.3 ``fallback_rate_high`` alert rule (fallback
rate >20% triggers an alert payload). Uses an injected clock so windows are
deterministic, and the default ledger path under the per-test HERMES_HOME.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.diagnostics.model_metrics import (
    DEFAULT_WINDOW_HOURS,
    ModelMetricsCollector,
    default_model_metrics_path,
)


def _collector(
    tmp_path: Path, start: datetime | None = None
) -> tuple[ModelMetricsCollector, datetime]:
    """Collector on a tmp ledger + the fixed base time its clock ticks from."""
    base = start or datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    return ModelMetricsCollector(
        ledger_path=tmp_path / "model_metrics.jsonl",
        now=lambda: base,
    ), base


@pytest.mark.unit
def test_token_speed_tracked(tmp_path: Path) -> None:
    """Generations record tok/s; report aggregates p50/p95 over the window."""
    collector, base = _collector(tmp_path)

    speeds = [
        collector.record_generation("main-model", 300, 10.0),  # 30.0 tok/s
        collector.record_generation("main-model", 210, 3.0),  # 70.0 tok/s
        collector.record_generation("main-model", 45, 1.5),  # 30.0 tok/s
    ]
    assert speeds == [30.0, 70.0, 30.0]

    report = collector.report("main-model", window_hours=DEFAULT_WINDOW_HOURS)
    assert report.generations == 3
    assert report.tok_per_s_p50 == 30.0  # middle of {30, 30, 70}
    # p95 with linear interpolation: rank 1.9 -> 30 + 0.9*(70-30)
    assert report.tok_per_s_p95 == pytest.approx(66.0)

    # Per-model scoping: another model's generations do not leak in.
    collector.record_generation("fallback-model", 10, 1.0)
    scoped = collector.report("main-model")
    assert scoped.generations == 3
    overall = collector.report(None)
    assert overall.generations == 4

    # Events persist as JSONL under the default sibling-ledger location.
    lines = (tmp_path / "model_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["event"] == "generation"
    assert first["model_id"] == "main-model"
    assert first["tokens"] == 300
    assert first["tok_per_s"] == 30.0

    # Windowing: only the trailing 24h counts.
    later = collector.__class__(
        ledger_path=collector.ledger_path,
        now=lambda: base + timedelta(hours=30),
    )
    aged_out = later.report("main-model")
    assert aged_out.generations == 0
    assert aged_out.tok_per_s_p50 == 0.0

    # Invalid inputs are rejected eagerly.
    with pytest.raises(ValueError, match="seconds"):
        collector.record_generation("main-model", 10, 0.0)
    with pytest.raises(TypeError, match="tokens"):
        collector.record_generation("main-model", "300", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id"):
        collector.record_generation("", 10, 1.0)


@pytest.mark.unit
def test_quality_score_tracked(tmp_path: Path) -> None:
    """Eval scores accumulate chronologically; trend is slope-sign classified."""
    collector, _base = _collector(tmp_path)

    for score in (0.60, 0.65, 0.70, 0.78):
        collector.record_quality("main-model", score)
    report = collector.report("main-model")
    assert report.quality_scores == [0.60, 0.65, 0.70, 0.78]  # chronological
    assert report.quality_trend == "improving"

    declining = ModelMetricsCollector(
        ledger_path=tmp_path / "declining.jsonl", now=lambda: datetime.now(UTC)
    )
    for score in (0.80, 0.70, 0.62, 0.55):
        declining.record_quality("main-model", score)
    assert declining.report("main-model").quality_trend == "declining"

    flat = ModelMetricsCollector(
        ledger_path=tmp_path / "flat.jsonl", now=lambda: datetime.now(UTC)
    )
    for score in (0.70, 0.70, 0.70):
        flat.record_quality("main-model", score)
    assert flat.report("main-model").quality_trend == "flat"

    single = ModelMetricsCollector(
        ledger_path=tmp_path / "single.jsonl", now=lambda: datetime.now(UTC)
    )
    single.record_quality("main-model", 0.9)
    assert single.report("main-model").quality_trend == "flat"  # <2 scores

    with pytest.raises(ValueError, match="eval_score"):
        collector.record_quality("main-model", 1.5)

    # The report is dashboard-reportable (plan P14.T7 acceptance).
    payload = report.to_dict()
    assert payload["quality_scores"] == [0.60, 0.65, 0.70, 0.78]
    assert payload["quality_trend"] == "improving"


@pytest.mark.unit
def test_error_rate_computed(tmp_path: Path) -> None:
    """error_rate = errors / (generations + errors) over the window."""
    collector, _base = _collector(tmp_path)

    assert collector.report("main-model").error_rate == 0.0  # empty window

    collector.record_generation("main-model", 100, 2.0)
    collector.record_generation("main-model", 100, 2.0)
    collector.record_generation("main-model", 100, 2.0)
    assert collector.report("main-model").error_rate == 0.0

    collector.record_error("main-model")
    assert collector.report("main-model").error_rate == pytest.approx(1 / 4)

    collector.record_error("main-model")
    assert collector.report("main-model").error_rate == pytest.approx(2 / 5)

    # Errors from other models stay out of a scoped report.
    collector.record_error("backup-model")
    assert collector.report("main-model").error_rate == pytest.approx(2 / 5)
    assert collector.report("backup-model").error_rate == pytest.approx(1.0)


@pytest.mark.unit
def test_fallback_rate_alert(tmp_path: Path) -> None:
    """Fallback rate >20% triggers the plan 5.3 fallback_rate_high alert."""
    collector, _base = _collector(tmp_path)
    from hermes.diagnostics.model_metrics import alert_payload, should_alert

    healthy = collector.report("main-model")
    assert healthy.fallback_rate == 0.0
    assert should_alert(healthy) is False
    assert alert_payload(healthy) is None

    # 9 generations, 1 fallback -> exactly 10%: below the 20% bar.
    for _ in range(9):
        collector.record_generation("main-model", 100, 1.0)
    collector.record_fallback("main-model")
    calm = collector.report("main-model")
    assert calm.fallback_rate == pytest.approx(0.1)
    assert should_alert(calm) is False

    # Push fallbacks past 20%: 9 gens + 1 error + 3 fallbacks -> 3/13 ~ 23%.
    collector.record_error("main-model")
    collector.record_fallback("main-model")
    collector.record_fallback("main-model")
    hot = collector.report("main-model")
    assert hot.fallback_rate > 0.2
    assert should_alert(hot) is True
    payload = alert_payload(hot)
    assert payload is not None
    assert payload["alert"] == "fallback_rate_high"
    assert payload["severity"] == "warning"
    assert payload["model_id"] == "main-model"
    assert payload["fallback_rate"] == hot.fallback_rate
    assert payload["threshold"] == 0.2

    # The >20% rule is strict: exactly 20% does not alert.
    strict = ModelMetricsCollector(ledger_path=tmp_path / "strict.jsonl")
    strict.record_generation("main-model", 1, 1.0)
    strict.record_generation("main-model", 1, 1.0)
    strict.record_generation("main-model", 1, 1.0)
    strict.record_generation("main-model", 1, 1.0)
    strict.record_fallback("main-model")
    boundary = strict.report("main-model")
    assert boundary.fallback_rate == pytest.approx(0.2)
    assert should_alert(boundary) is False

    with pytest.raises(ValueError, match="threshold"):
        should_alert(boundary, threshold=1.5)


@pytest.mark.unit
def test_default_ledger_path_and_persistence(tmp_path: Path) -> None:
    """Default path is <hermes_home>/diagnostics/model_metrics.jsonl; restart-safe."""
    collector = ModelMetricsCollector()
    assert collector.ledger_path == hermes_home() / "diagnostics" / "model_metrics.jsonl"
    assert default_model_metrics_path() == hermes_home() / "diagnostics" / "model_metrics.jsonl"

    collector.record_generation("main-model", 500, 2.0)
    collector.record_fallback("main-model")

    # A NEW collector instance over the same default path sees the history.
    reopened = ModelMetricsCollector()
    report = reopened.report("main-model")
    assert report.generations == 1
    assert report.fallbacks == 1
    assert report.fallback_rate == pytest.approx(0.5)

    # The sibling ledger is named in the metrics module (plan P14.T7 wiring).
    from hermes.diagnostics.metrics import SIBLING_LEDGER_NAME

    assert SIBLING_LEDGER_NAME == "model_metrics.jsonl"
