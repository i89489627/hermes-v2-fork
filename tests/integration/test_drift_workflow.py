"""P19.T2 integration test: drift workflow over the default registry (plan Phase 19).

Drives the DEFAULT persistence layout (``<hermes_home>/model_versions.json``
via ``hermes.config.hermes_home``) with a deterministic eval fake and a
capturing notifier: a version change must trigger the eval run, a 2.5%
regression must fire the ``model_regression`` alert, and a 6% regression
must auto-rollback — restoring the previous version id through a reloaded
registry (the rollback is persisted, satisfying the plan's "rollback within
5 minutes" budget as a single atomic write).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hermes.config import hermes_home
from hermes.models.drift_detector import (
    ALERT_MODEL_REGRESSION,
    ALERT_MODEL_ROLLBACK,
    DriftDetector,
)
from hermes.models.router import ModelRouter
from hermes.models.version_registry import ModelVersionRegistry


def _eval(scores: dict[str, float], calls: list[str]) -> Callable[[str], float]:
    def runner(version_id: str) -> float:
        calls.append(version_id)
        return scores[version_id]

    return runner


@pytest.mark.integration
def test_version_change_triggers_eval() -> None:
    """Default-path registry: eval runs on the change; 2.5% regression alerts."""
    registry = ModelVersionRegistry()  # default path under per-test HERMES_HOME
    assert registry.path == hermes_home() / "model_versions.json"
    registry.register("main-model", "deploy-a", eval_score=0.80)
    registry.register("main-model", "deploy-b", eval_score=0.78)  # 2.5% regression

    calls: list[str] = []
    alerts: list[tuple[str, str]] = []
    detector = DriftDetector(
        registry,
        _eval({"deploy-a": 0.80, "deploy-b": 0.78}, calls),
        notifier=lambda name, message: alerts.append((name, message)),
    )

    result = detector.on_version_change("deploy-a", "deploy-b")

    assert result.ran_eval is True
    assert "deploy-b" in calls  # the eval set ran on the new deployment
    assert result.regression_pct == pytest.approx(2.5)
    assert result.action == "alert"
    assert [name for name, _message in alerts] == [ALERT_MODEL_REGRESSION]

    router = ModelRouter(registry)
    assert router.route() == "deploy-b"  # alerting does not switch the version


@pytest.mark.integration
def test_auto_rollback_restores_previous() -> None:
    """Default-path registry: 6% regression rolls back and the restore persists."""
    registry = ModelVersionRegistry()
    registry.register("main-model", "stable-1", provider="openai", model_id="m-1",
                      eval_score=0.80)
    registry.register("main-model", "candidate-2", provider="openai", model_id="m-2",
                      eval_score=0.752)  # 6% regression

    alerts: list[tuple[str, str]] = []
    detector = DriftDetector(
        registry,
        _eval({"stable-1": 0.80, "candidate-2": 0.752}, []),
        notifier=lambda name, message: alerts.append((name, message)),
    )

    result = detector.on_version_change("stable-1", "candidate-2")

    assert result.action == "rollback"
    assert result.regression_pct == pytest.approx(6.0)
    assert [name for name, _message in alerts] == [ALERT_MODEL_ROLLBACK]

    # Current version restored, persisted, and routed — reloaded from disk.
    reloaded = ModelVersionRegistry()
    router = ModelRouter(reloaded)
    assert router.route() == "stable-1"
    current = reloaded.get_current("main-model")
    assert current is not None
    assert current.provider == "openai"
    assert current.model_id == "m-1"
    assert current.eval_score == pytest.approx(0.80)

    # The split primitive also resolves deterministically post-rollback.
    assert router.route_with_weight(request_key="conversation-17") == "stable-1"
