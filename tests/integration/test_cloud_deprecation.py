"""P19.T3 integration test: end-to-end deprecation alert flow (plan Phase 19).

Drives the detector against the DEFAULT persistence layout (registry file
under the per-test ``HERMES_HOME`` via ``hermes.config.hermes_home``), with
an injected fake provider probe and a capturing notifier: a deprecated
current model must produce a ``model_deprecation_warning`` alert whose
message suggests the configured alternatives (priority order), and a removed
model must auto-fallback by registering the first available alternative as
the new current ``main-model`` version — persisted to the default registry
file across a detector/registry restart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.models.cloud_client import AvailabilityResult, CloudModelClient
from hermes.models.cloud_deprecation import (
    ALERT_MODEL_DEPRECATION_WARNING,
    ALERT_MODEL_REMOVED_FALLBACK,
    DeprecationDetector,
)
from hermes.models.version_registry import ModelVersionRegistry

LEGACY = "gpt-legacy"
NEXT = "gpt-next"
ULTRA = "gpt-ultra"


def _probe(results: dict[str, AvailabilityResult]) -> CloudModelClient:
    def probe(model_id: str) -> AvailabilityResult:
        return results[model_id]

    return CloudModelClient(check_availability=probe)


@pytest.mark.integration
def test_alert_on_deprecation() -> None:
    registry = ModelVersionRegistry()  # default path under the per-test HERMES_HOME
    assert registry.path == hermes_home() / "model_versions.json"
    registry.register("main-model", "deploy-legacy", provider="openai", model_id=LEGACY)

    alerts: list[tuple[str, str]] = []
    detector = DeprecationDetector(
        client=_probe(
            {
                LEGACY: AvailabilityResult(
                    model_id=LEGACY,
                    available=True,
                    deprecation_warning="provider notice: capacity tier retiring",
                    estimated_removal="2027-02-01",
                ),
                NEXT: AvailabilityResult(model_id=NEXT, available=True),
                ULTRA: AvailabilityResult(model_id=ULTRA, available=True),
            }
        ),
        alternatives={LEGACY: [NEXT, ULTRA]},
        registry=registry,
        notifier=lambda alert, message: alerts.append((alert, message)),
    )

    outcome = detector.run_check()
    assert outcome.status == ALERT_MODEL_DEPRECATION_WARNING
    assert len(alerts) == 1
    alert_name, message = alerts[0]
    assert alert_name == ALERT_MODEL_DEPRECATION_WARNING
    # The alert tells the operator what to switch to, in priority order.
    assert "provider notice: capacity tier retiring" in message
    assert "estimated removal 2027-02-01" in message
    assert NEXT in message and ULTRA in message
    assert message.index(NEXT) < message.index(ULTRA)
    # Still-servable model: the registry is deliberately NOT switched.
    current = registry.get_current("main-model")
    assert current is not None
    assert current.model_id == LEGACY

    # Later the provider completes the removal: the same wired detector
    # auto-fallbacks and the switch persists at the default location.
    removed_probe = _probe(
        {
            LEGACY: AvailabilityResult(model_id=LEGACY, available=False),
            NEXT: AvailabilityResult(model_id=NEXT, available=True),
            ULTRA: AvailabilityResult(model_id=ULTRA, available=True),
        }
    )
    fallback_alerts: list[tuple[str, str]] = []
    restarted_detector = DeprecationDetector(
        client=removed_probe,
        alternatives={LEGACY: [NEXT, ULTRA]},
        registry=ModelVersionRegistry(),  # reloaded from the default file
        notifier=lambda alert, message: fallback_alerts.append((alert, message)),
    )
    fallback_outcome = restarted_detector.run_check()
    assert fallback_outcome.status == ALERT_MODEL_REMOVED_FALLBACK
    assert fallback_outcome.fallback_model_id == NEXT
    assert fallback_alerts[0][0] == ALERT_MODEL_REMOVED_FALLBACK

    persisted = ModelVersionRegistry()  # fresh reader, same default file
    current = persisted.get_current("main-model")
    assert current is not None
    assert current.model_id == NEXT
    history = persisted.get_history("main-model")
    assert [record.model_id for record in history] == [LEGACY, NEXT]
