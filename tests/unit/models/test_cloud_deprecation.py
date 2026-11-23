"""P19.T3 acceptance tests: cloud model deprecation detection (plan Phase 19).

Covers: deprecation-warning detection with an alert that suggests the
configured alternatives (registry untouched while the model still works),
auto-fallback on removal (first available alternative registered as the new
current ``main-model`` version, full history retained), the periodic-run
gate (:func:`due_for_check`, 24h cadence), and the loud-failure contract of
the unwired :class:`CloudModelClient`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.models.cloud_client import AvailabilityResult, CloudModelClient
from hermes.models.cloud_deprecation import (
    ALERT_MODEL_DEPRECATION_WARNING,
    ALERT_MODEL_REMOVED_FALLBACK,
    ALERT_MODEL_REMOVED_NO_ALTERNATIVE,
    CHECK_OK,
    DeprecationDetector,
    due_for_check,
)
from hermes.models.version_registry import ModelVersionRegistry

LEGACY = "gpt-legacy"
NEXT = "gpt-next"
ULTRA = "gpt-ultra"


def _probe(
    results: dict[str, AvailabilityResult],
) -> CloudModelClient:
    """Client backed by an injected fake probe over a fixed result table."""

    def probe(model_id: str) -> AvailabilityResult:
        return results[model_id]

    return CloudModelClient(check_availability=probe)


def _registry_with_legacy(tmp_path: Path) -> ModelVersionRegistry:
    """Registry with the legacy model as the current main-model deployment."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "deploy-1", provider="openai", model_id=LEGACY)
    return registry


@pytest.mark.unit
def test_detects_deprecation_warning(tmp_path: Path) -> None:
    registry = _registry_with_legacy(tmp_path)
    alerts: list[tuple[str, str]] = []
    detector = DeprecationDetector(
        client=_probe(
            {
                LEGACY: AvailabilityResult(
                    model_id=LEGACY,
                    available=True,
                    deprecation_warning="retirement scheduled by the provider",
                    estimated_removal="2027-01-15",
                ),
                NEXT: AvailabilityResult(model_id=NEXT, available=True),
                ULTRA: AvailabilityResult(model_id=ULTRA, available=True),
            }
        ),
        alternatives={LEGACY: [NEXT, ULTRA]},
        registry=registry,
        notifier=lambda alert, message: alerts.append((alert, message)),
    )

    # check_model_availability reports the current model with the warning.
    result = detector.check_model_availability()
    assert result.model_id == LEGACY
    assert result.available is True
    assert result.deprecation_warning == "retirement scheduled by the provider"
    assert result.estimated_removal == "2027-01-15"

    # Unwired clients fail loudly instead of pretending availability.
    with pytest.raises(NotImplementedError):
        CloudModelClient().check_availability(LEGACY)
    # A detector without a registry cannot know what to probe.
    with pytest.raises(ValueError):
        DeprecationDetector(
            client=_probe({}), alternatives={}, registry=None
        ).current_model_id()

    outcome = detector.run_check()
    assert outcome.model_id == LEGACY
    assert outcome.status == ALERT_MODEL_DEPRECATION_WARNING
    assert outcome.alert == ALERT_MODEL_DEPRECATION_WARNING
    assert outcome.fallback_model_id is None
    assert outcome.checked_at

    # The alert carries the warning text AND the configured alternatives,
    # in the configured priority order.
    assert len(alerts) == 1
    alert_name, message = alerts[0]
    assert alert_name == ALERT_MODEL_DEPRECATION_WARNING
    assert "retirement scheduled by the provider" in message
    assert "estimated removal 2027-01-15" in message
    assert NEXT in message and ULTRA in message
    assert message.index(NEXT) < message.index(ULTRA)  # priority order preserved

    # A warning alone must NOT switch the registry: the model still works.
    current = registry.get_current("main-model")
    assert current is not None
    assert current.model_id == LEGACY

    # A healthy model yields the no-alert status.
    healthy_registry = ModelVersionRegistry(path=tmp_path / "healthy.json")
    healthy_registry.register("main-model", "deploy-ok", model_id=NEXT)
    healthy = DeprecationDetector(
        client=_probe({NEXT: AvailabilityResult(model_id=NEXT, available=True)}),
        alternatives={},
        registry=healthy_registry,
    )
    ok_outcome = healthy.run_check()
    assert ok_outcome.status == CHECK_OK
    assert ok_outcome.alert is None

    # Periodic-run gate: due_for_check drives the <=24h detection cadence.
    now = datetime.now(UTC)
    assert due_for_check(None, now=now) is True  # never checked -> run now
    assert due_for_check(now - timedelta(hours=23), now=now) is False
    assert due_for_check(now - timedelta(hours=24), now=now) is True
    assert due_for_check(now - timedelta(hours=25), now=now) is True
    assert due_for_check(now - timedelta(minutes=30), interval_hours=0.5, now=now) is True
    # Naive datetimes are treated as UTC (pure helper, no clock surprises).
    naive = (now - timedelta(hours=25)).replace(tzinfo=None)
    assert due_for_check(naive, now=now) is True


@pytest.mark.unit
def test_auto_fallback_on_removal(tmp_path: Path) -> None:
    registry = _registry_with_legacy(tmp_path)
    alerts: list[tuple[str, str]] = []
    detector = DeprecationDetector(
        client=_probe(
            {
                LEGACY: AvailabilityResult(model_id=LEGACY, available=False),
                NEXT: AvailabilityResult(model_id=NEXT, available=True),
                ULTRA: AvailabilityResult(model_id=ULTRA, available=True),
            }
        ),
        alternatives={LEGACY: [NEXT, ULTRA]},
        registry=registry,
        notifier=lambda alert, message: alerts.append((alert, message)),
    )

    outcome = detector.run_check()
    assert outcome.model_id == LEGACY
    assert outcome.status == ALERT_MODEL_REMOVED_FALLBACK
    assert outcome.alert == ALERT_MODEL_REMOVED_FALLBACK
    assert outcome.fallback_model_id == NEXT  # first AVAILABLE alternative, in order
    assert "no longer available" in outcome.message
    assert NEXT in outcome.message

    # The fallback is registered as the new current main-model version.
    current = registry.get_current("main-model")
    assert current is not None
    assert current.model_id == NEXT
    assert current.version_id == NEXT
    # History is retained: the legacy deployment is still in the registry.
    history = registry.get_history("main-model")
    assert [record.model_id for record in history] == [LEGACY, NEXT]

    # Alert went through the notifier seam exactly once.
    assert len(alerts) == 1
    assert alerts[0][0] == ALERT_MODEL_REMOVED_FALLBACK

    # Removal with NO available alternative: registry stays put, alert fires.
    stuck_registry = _registry_with_legacy(tmp_path / "stuck")
    stuck_alerts: list[tuple[str, str]] = []
    stuck = DeprecationDetector(
        client=_probe(
            {
                LEGACY: AvailabilityResult(model_id=LEGACY, available=False),
                NEXT: AvailabilityResult(model_id=NEXT, available=False),
                ULTRA: AvailabilityResult(model_id=ULTRA, available=False),
            }
        ),
        alternatives={LEGACY: [NEXT, ULTRA]},
        registry=stuck_registry,
        notifier=lambda alert, message: stuck_alerts.append((alert, message)),
    )
    stuck_outcome = stuck.run_check()
    assert stuck_outcome.status == ALERT_MODEL_REMOVED_NO_ALTERNATIVE
    assert stuck_outcome.fallback_model_id is None
    still = stuck_registry.get_current("main-model")
    assert still is not None
    assert still.model_id == LEGACY
    assert stuck_alerts[0][0] == ALERT_MODEL_REMOVED_NO_ALTERNATIVE

    # Removal with no alternatives configured at all takes the same path.
    bare = DeprecationDetector(
        client=_probe({LEGACY: AvailabilityResult(model_id=LEGACY, available=False)}),
        alternatives={},
        registry=_registry_with_legacy(tmp_path / "bare"),
    )
    bare_outcome = bare.run_check()
    assert bare_outcome.status == ALERT_MODEL_REMOVED_NO_ALTERNATIVE
