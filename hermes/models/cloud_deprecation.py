"""Cloud model deprecation detection (plan P19.T3).

The v2 pipeline is single-model (plan Section 2.1): every turn is served by
the model currently registered under the ``main-model`` component of
:class:`~hermes.models.version_registry.ModelVersionRegistry`. When a cloud
provider retires that model, the pipeline must notice and recover without a
human reading release notes. This module closes that loop:

* :meth:`DeprecationDetector.check_model_availability` probes the current
  ``main-model`` through :class:`~hermes.models.cloud_client.CloudModelClient`;
* :meth:`DeprecationDetector.run_check` classifies the probe and acts:
  - still available, no warning -> ``"ok"``, nothing happens;
  - still available WITH a deprecation warning -> alert
    ``"model_deprecation_warning"`` whose message suggests the alternatives
    configured in the ``alternatives`` map (registry is NOT switched: the
    model still works, only the human/tick loop decides);
  - removed -> AUTO-FALLBACK: alternatives are probed in configured order and
    the first available one is registered as the new current ``main-model``
    version (full history retained), alert ``"model_removed_fallback"``;
  - removed with no reachable alternative -> alert
    ``"model_removed_no_alternative"`` (registry unchanged).

Alerts go through the ``notifier`` seam (``Callable[[alert, message], None]``);
callers bind it to diagnostics or a gateway-message fallback — the detector
never owns a transport, mirroring the watchdog convention (plan P21.T5).

Periodic scheduling design: the checker is a pure class with no threads; the
cron layer (plan P21) or a long-lived loop calls :func:`due_for_check` with
the previous check time (24h cadence per the acceptance criteria) and runs
:meth:`DeprecationDetector.run_check` when it returns ``True``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hermes.models.cloud_client import AvailabilityResult, CloudModelClient
from hermes.models.version_registry import ModelVersionRegistry

__all__ = [
    "ALERT_MODEL_DEPRECATION_WARNING",
    "ALERT_MODEL_REMOVED_FALLBACK",
    "ALERT_MODEL_REMOVED_NO_ALTERNATIVE",
    "CHECK_OK",
    "DEFAULT_CHECK_INTERVAL_HOURS",
    "CheckOutcome",
    "DeprecationDetector",
    "due_for_check",
]

#: Registry component the detector watches (the one model serving all turns).
MAIN_MODEL_COMPONENT = "main-model"

#: Status/alert vocabulary of :class:`CheckOutcome`.
CHECK_OK = "ok"
ALERT_MODEL_DEPRECATION_WARNING = "model_deprecation_warning"
ALERT_MODEL_REMOVED_FALLBACK = "model_removed_fallback"
ALERT_MODEL_REMOVED_NO_ALTERNATIVE = "model_removed_no_alternative"

#: Default cadence for periodic availability checks (acceptance: within 24h).
DEFAULT_CHECK_INTERVAL_HOURS = 24.0


@dataclass(frozen=True)
class CheckOutcome:
    """Result of one :meth:`DeprecationDetector.run_check` pass.

    ``status`` is one of the module's alert constants (or ``CHECK_OK``);
    ``alert`` repeats the alert name actually surfaced through the notifier
    (``None`` only for ``CHECK_OK``); ``fallback_model_id`` is set when a
    removed model triggered the auto-fallback registration.
    """

    model_id: str
    status: str
    alert: str | None
    message: str
    fallback_model_id: str | None = None
    checked_at: str = ""


def due_for_check(
    last_check: datetime | None,
    interval_hours: float = DEFAULT_CHECK_INTERVAL_HOURS,
    now: datetime | None = None,
) -> bool:
    """Pure periodic-run gate: ``True`` when another check is due.

    ``last_check=None`` (never checked) is always due. Naive datetimes are
    assumed UTC. ``now`` is injectable for deterministic callers; it defaults
    to the current UTC time. With the default ``interval_hours=24`` this is
    the 24-hour cadence the acceptance criteria require.
    """
    if last_check is not None and last_check.tzinfo is None:
        last_check = last_check.replace(tzinfo=UTC)
    current = now if now is not None else datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    if last_check is None:
        return True
    return current - last_check >= timedelta(hours=interval_hours)


class DeprecationDetector:
    """Detects provider deprecation/removal of the current main model.

    ``alternatives`` maps the current model id to an ORDERED list of fallback
    model ids — order is the fallback priority. ``registry`` resolves the
    current model and receives the auto-fallback registration; it is
    effectively required (a detector without one cannot know what to check
    and raises on use). ``notifier`` is the alert seam; without it alerts are
    dropped, but the registry side effects (fallback) still happen.
    """

    def __init__(
        self,
        client: CloudModelClient,
        alternatives: dict[str, list[str]],
        registry: ModelVersionRegistry | None = None,
        notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self.client = client
        self.alternatives = alternatives
        self.registry = registry
        self.notifier = notifier

    # -- queries -------------------------------------------------------------

    def current_model_id(self) -> str:
        """Model id of the current ``main-model`` registration.

        Raises ``ValueError`` when no registry is wired, nothing is
        registered under ``main-model``, or the record carries no model id —
        the detector must fail loudly rather than probe a guessed model.
        """
        if self.registry is None:
            raise ValueError(
                "DeprecationDetector requires a ModelVersionRegistry to resolve "
                "the current main model"
            )
        record = self.registry.get_current(MAIN_MODEL_COMPONENT)
        if record is None:
            raise ValueError(
                f"no model registered under component {MAIN_MODEL_COMPONENT!r}; "
                "register the deployed model before running deprecation checks"
            )
        if not record.model_id:
            raise ValueError(
                f"current {MAIN_MODEL_COMPONENT!r} registration "
                f"({record.version_id!r}) carries no model_id to probe"
            )
        return record.model_id

    def check_model_availability(self) -> AvailabilityResult:
        """Probe the current model's availability (no side effects)."""
        return self.client.check_availability(self.current_model_id())

    # -- one check pass --------------------------------------------------------

    def run_check(self, now: datetime | None = None) -> CheckOutcome:
        """Classify the current model's availability and act on it.

        See the module docstring for the four outcomes. ``now`` stamps
        :attr:`CheckOutcome.checked_at` (deterministic for callers who pass
        it); it defaults to the current UTC time.
        """
        checked_at = now.isoformat() if now is not None else datetime.now(UTC).isoformat()
        model_id = self.current_model_id()
        result = self.client.check_availability(model_id)

        if result.available and result.deprecation_warning is None:
            return CheckOutcome(
                model_id=model_id,
                status=CHECK_OK,
                alert=None,
                message=f"{model_id} is available",
                checked_at=checked_at,
            )

        if result.available:
            message = self._deprecation_message(model_id, result)
            self._notify(ALERT_MODEL_DEPRECATION_WARNING, message)
            return CheckOutcome(
                model_id=model_id,
                status=ALERT_MODEL_DEPRECATION_WARNING,
                alert=ALERT_MODEL_DEPRECATION_WARNING,
                message=message,
                checked_at=checked_at,
            )

        return self._fallback_after_removal(model_id, checked_at)

    # -- internals -------------------------------------------------------------

    def _deprecation_message(self, model_id: str, result: AvailabilityResult) -> str:
        """Build the deprecation-warning alert text, suggesting alternatives."""
        parts = [f"{model_id} is scheduled for deprecation"]
        if result.deprecation_warning:
            parts.append(result.deprecation_warning)
        if result.estimated_removal:
            parts.append(f"estimated removal {result.estimated_removal}")
        suggestions = self.alternatives.get(model_id, [])
        parts.append(
            "suggested alternatives: " + ", ".join(suggestions)
            if suggestions
            else "no alternatives configured"
        )
        return " - ".join(parts)

    def _fallback_after_removal(self, model_id: str, checked_at: str) -> CheckOutcome:
        """Removed model: probe alternatives in order, register the first live one."""
        for alternative in self.alternatives.get(model_id, []):
            probe = self.client.check_availability(alternative)
            if not probe.available:
                continue
            self._register_fallback(alternative)
            message = (
                f"{model_id} is no longer available; auto-fallback to "
                f"{alternative} (first available alternative, registered as the "
                f"current {MAIN_MODEL_COMPONENT!r} version)"
            )
            self._notify(ALERT_MODEL_REMOVED_FALLBACK, message)
            return CheckOutcome(
                model_id=model_id,
                status=ALERT_MODEL_REMOVED_FALLBACK,
                alert=ALERT_MODEL_REMOVED_FALLBACK,
                message=message,
                fallback_model_id=alternative,
                checked_at=checked_at,
            )
        message = (
            f"{model_id} is no longer available and none of the configured "
            f"alternatives ({', '.join(self.alternatives.get(model_id, [])) or 'none'}) "
            "is available; manual intervention required"
        )
        self._notify(ALERT_MODEL_REMOVED_NO_ALTERNATIVE, message)
        return CheckOutcome(
            model_id=model_id,
            status=ALERT_MODEL_REMOVED_NO_ALTERNATIVE,
            alert=ALERT_MODEL_REMOVED_NO_ALTERNATIVE,
            message=message,
            checked_at=checked_at,
        )

    def _register_fallback(self, alternative: str) -> None:
        """Register ``alternative`` as the new current ``main-model`` version."""
        if self.registry is None:  # unreachable (current_model_id validated) — keeps types honest
            raise ValueError("cannot register a fallback without a ModelVersionRegistry")
        self.registry.register(MAIN_MODEL_COMPONENT, alternative, model_id=alternative)

    def _notify(self, alert: str, message: str) -> None:
        """Surface one alert through the injected notifier seam (no-op if unset)."""
        if self.notifier is not None:
            self.notifier(alert, message)
