"""P19.T2 acceptance tests: drift detection on version change (plan Phase 19).

Covers the spec behaviours: the eval harness runs on every version change,
a >2% regression fires a ``model_regression`` alert (2.5% case), and a >5%
regression (6% case) auto-rolls back by re-registering the previous version
as current. Deterministic eval fakes and a capturing notifier; the registry
lives under the per-test HERMES_HOME / tmp paths.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.models.drift_detector import (
    ALERT_MODEL_REGRESSION,
    ALERT_MODEL_ROLLBACK,
    DriftDetector,
    DriftResult,
)
from hermes.models.version_registry import ModelVersionRegistry


def _fake_eval(scores: dict[str, float], calls: list[str]) -> Callable[[str], float]:
    """Deterministic eval runner: scores by version id, recording every call."""

    def runner(version_id: str) -> float:
        calls.append(version_id)
        if version_id not in scores:
            raise KeyError(f"no eval fixture for version {version_id!r}")
        return scores[version_id]

    return runner


@pytest.mark.unit
def test_eval_runs_on_version_change(tmp_path: Path) -> None:
    """Every version change runs the eval harness on the new version."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "v1", eval_score=0.90)
    registry.register("main-model", "v2", eval_score=0.90)

    calls: list[str] = []
    detector = DriftDetector(
        registry,
        _fake_eval({"v1": 0.90, "v2": 0.90, "v3": 0.90, "v4": 0.90, "v5": 0.95}, calls),
    )

    result = detector.on_version_change("v1", "v2")
    assert isinstance(result, DriftResult)
    assert result.ran_eval is True
    assert "v2" in calls  # eval ran on the NEW version
    assert result.old_score == 0.90  # recorded baselines need no re-eval
    assert result.new_score == 0.90
    assert result.regression_pct == 0.0
    assert result.action == "none"

    # Old version with no recorded score: the harness evals it too.
    registry.register("main-model", "v3")  # deployed without a recorded score
    registry.register("main-model", "v4")
    calls.clear()
    unrecorded = detector.on_version_change("v3", "v4")
    assert unrecorded.ran_eval is True
    assert calls == ["v4", "v3"]  # new first, then the unscored baseline
    assert unrecorded.old_score == 0.90
    assert unrecorded.new_score == 0.90

    # Improvement is not a regression.
    registry.register("main-model", "v5", eval_score=0.95)
    calls.clear()
    better = detector.on_version_change("v4", "v5")
    assert better.action == "none"
    assert better.regression_pct == 0.0
    assert better.new_score == 0.95


@pytest.mark.unit
def test_alert_on_2_percent_regression(tmp_path: Path) -> None:
    """A 2.5% regression fires model_regression and keeps the new version."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "v1", eval_score=0.80)
    registry.register("main-model", "v2", eval_score=0.78)  # 2.5% regression

    alerts: list[tuple[str, str]] = []
    detector = DriftDetector(
        registry,
        _fake_eval({"v1": 0.80, "v2": 0.78}, []),
        notifier=lambda name, message: alerts.append((name, message)),
    )

    result = detector.on_version_change("v1", "v2")

    assert result.ran_eval is True
    assert result.regression_pct == pytest.approx(2.5)
    assert result.action == "alert"
    assert len(alerts) == 1
    name, message = alerts[0]
    assert name == ALERT_MODEL_REGRESSION
    assert "v2" in message and "v1" in message  # the suggestion names both versions
    assert "rollback" in message.lower()  # plan: alert + SUGGEST rollback

    # Alerting does NOT change the current version.
    current = registry.get_current("main-model")
    assert current is not None
    assert current.version_id == "v2"


@pytest.mark.unit
def test_auto_rollback_on_5_percent_regression(tmp_path: Path) -> None:
    """A 6% regression auto-rolls back, restoring the previous version id."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register(
        "main-model", "v1", provider="openai", model_id="m-1", eval_score=0.80
    )
    registry.register(
        "main-model", "v2", provider="openai", model_id="m-2", eval_score=0.752
    )  # (0.80 - 0.752) / 0.80 * 100 = 6% regression

    alerts: list[tuple[str, str]] = []
    detector = DriftDetector(
        registry,
        _fake_eval({"v1": 0.80, "v2": 0.752}, []),
        notifier=lambda name, message: alerts.append((name, message)),
    )

    result = detector.on_version_change("v1", "v2")

    assert result.ran_eval is True
    assert result.regression_pct == pytest.approx(6.0)
    assert result.action == "rollback"

    # The previous version is current again (registry-append rollback).
    current = registry.get_current("main-model")
    assert current is not None
    assert current.version_id == "v1"
    assert current.provider == "openai"  # original deployment metadata preserved
    assert current.model_id == "m-1"
    assert current.eval_score == pytest.approx(0.80)

    # History is never rewritten: both versions stay addressable.
    assert [entry.version_id for entry in registry.get_history("main-model")] == [
        "v1",
        "v2",
        "v1",
    ]

    assert [name for name, _message in alerts] == [ALERT_MODEL_ROLLBACK]
    assert "auto-rolled back" in alerts[0][1]

    # Persisted: a reloaded registry still routes to the rolled-back version.
    reloaded = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    assert reloaded.get_current("main-model") is not None
    assert reloaded.get_current("main-model").version_id == "v1"  # type: ignore[union-attr]


@pytest.mark.unit
def test_threshold_validation_and_default_paths() -> None:
    """Defaults follow the plan (2%/5%); misordered thresholds are rejected."""
    registry = ModelVersionRegistry()  # default path under per-test HERMES_HOME
    assert registry.path == hermes_home() / "model_versions.json"

    detector = DriftDetector(registry, lambda _version: 1.0)
    assert detector.alert_threshold_pct == 2.0
    assert detector.rollback_threshold_pct == 5.0

    with pytest.raises(ValueError, match="rollback_threshold_pct"):
        DriftDetector(registry, lambda _version: 1.0, alert_threshold_pct=5.0,
                      rollback_threshold_pct=5.0)
    with pytest.raises(ValueError, match="alert_threshold_pct"):
        DriftDetector(registry, lambda _version: 1.0, alert_threshold_pct=-1.0)

    # record-only mode: no notifier, but the action is still reported.
    registry.register("main-model", "base", eval_score=1.0)
    registry.register("main-model", "bad", eval_score=0.9)  # 10% regression
    silent = DriftDetector(registry, lambda version: {"base": 1.0, "bad": 0.9}[version])
    result = silent.on_version_change("base", "bad")
    assert result.action == "rollback"
    assert registry.get_current("main-model") is not None
    assert registry.get_current("main-model").version_id == "base"  # type: ignore[union-attr]


@pytest.mark.unit
def test_rollback_threshold_exactly_5_percent(tmp_path: Path) -> None:
    """Rollback boundary is exact (P20.T6): strictly MORE than 5% rolls back.

    Mutation-gate boundary test for the plan's 2%/5% thresholds (plan P19.T2:
    "worse by MORE than 5%: auto-rollback" — the comparison is strict). The
    score pairs are chosen so the computed regression percentage is
    float-exact on the boundary:

    - exactly 5.0% (100.0 -> 95.0): alert ONLY — no rollback, new version
      stays current;
    - 4.99% (5.0 -> 4.7505): still alert-only (above the 2% alert bar);
    - just above 5% (0.8 -> 0.76, 5.000000000000004%): auto-rollback fires;
    - exactly 2.0% (100.0 -> 98.0): no action at all (strict alert bar).
    """
    alerts: list[tuple[str, str]] = []

    def _scenario(
        name: str, pairs: list[tuple[str, float]]
    ) -> tuple[DriftDetector, ModelVersionRegistry]:
        """One registry holding the scenario's versions, wired to the notifier."""
        registry = ModelVersionRegistry(path=tmp_path / f"{name}.json")
        for version_id, score in pairs:
            registry.register("main-model", version_id, eval_score=score)
        detector = DriftDetector(
            registry,
            _fake_eval({version_id: score for version_id, score in pairs}, []),
            notifier=lambda alert_name, message: alerts.append((alert_name, message)),
        )
        return detector, registry

    # exactly 5.0% regression: alert-only, the new version stays current and
    # the history is NOT extended by a rollback re-registration
    detector, registry = _scenario(
        "exactly_5", [("v1", 100.0), ("v2", 95.0)]
    )
    at_boundary = detector.on_version_change("v1", "v2")
    assert at_boundary.ran_eval is True
    assert at_boundary.regression_pct == pytest.approx(5.0)
    assert at_boundary.action == "alert"
    current = registry.get_current("main-model")
    assert current is not None and current.version_id == "v2"
    assert [entry.version_id for entry in registry.get_history("main-model")] == [
        "v1",
        "v2",
    ]
    assert [name for name, _message in alerts] == [ALERT_MODEL_REGRESSION]

    # 4.99% regression: still alert-only, no rollback
    detector_below, registry_below = _scenario(
        "below_5", [("v3", 5.0), ("v4", 4.7505)]
    )
    below = detector_below.on_version_change("v3", "v4")
    assert below.regression_pct == pytest.approx(4.99)
    assert below.action == "alert"
    current = registry_below.get_current("main-model")
    assert current is not None and current.version_id == "v4"
    assert [name for name, _message in alerts] == [
        ALERT_MODEL_REGRESSION,
        ALERT_MODEL_REGRESSION,
    ]

    # just above 5%: auto-rollback restores the previous version
    detector_above, registry_above = _scenario(
        "above_5", [("v5", 0.8), ("v6", 0.76)]
    )
    above = detector_above.on_version_change("v5", "v6")
    assert above.regression_pct > 5.0
    assert above.action == "rollback"
    current = registry_above.get_current("main-model")
    assert current is not None and current.version_id == "v5"
    assert [name for name, _message in alerts][-1] == ALERT_MODEL_ROLLBACK

    # exactly 2.0% regression: below the strict alert bar — no action, no alert
    quiet_registry = ModelVersionRegistry(path=tmp_path / "alert_boundary.json")
    quiet_registry.register("main-model", "a1", eval_score=100.0)
    quiet_registry.register("main-model", "a2", eval_score=98.0)
    quiet = DriftDetector(
        quiet_registry,
        _fake_eval({"a1": 100.0, "a2": 98.0}, []),
        notifier=lambda name, message: alerts.append((name, message)),
    )
    alerts_before = len(alerts)
    at_alert_bar = quiet.on_version_change("a1", "a2")
    assert at_alert_bar.regression_pct == pytest.approx(2.0)
    assert at_alert_bar.action == "none"
    assert len(alerts) == alerts_before  # no alert delivered
