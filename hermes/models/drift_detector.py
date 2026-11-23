"""Drift detection: eval on model version change (plan P19.T2).

When the model version of a component changes (manual redeploy or a
GEPA-deployed prompt update), the eval set runs on the new version and the
score is compared against the previous version's baseline (plan Phase 19):

- worse by more than :data:`ALERT_THRESHOLD_PCT` (2%): a
  ``model_regression`` alert fires, suggesting rollback;
- worse by more than :data:`ROLLBACK_THRESHOLD_PCT` (5%): auto-rollback —
  the previous version is re-registered as the component's current version
  (registry-append rollback; history is never rewritten, mirroring
  :class:`~hermes.skills.registry.SkillRegistry.rollback` semantics) and a
  ``model_rollback`` alert fires. Re-registration makes the rollback
  immediate (single atomic JSON write — well within the plan's "rollback
  within 5 minutes" budget).

The eval runner is injected (``version_id -> eval score``), so production
wires the real eval-set harness (plan Phase 10 fixtures) and tests inject
deterministic fakes. Alert delivery goes through an injected notifier
``Callable[[alert_name, message], None]``; with ``notifier=None`` the alerts
are still reflected in the returned :class:`DriftResult` ``action`` but not
delivered anywhere (the diagnostics layer owns the channel choice, plan 5.3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from hermes.models.version_registry import ModelVersion, ModelVersionRegistry

__all__ = [
    "ALERT_MODEL_REGRESSION",
    "ALERT_MODEL_ROLLBACK",
    "ALERT_THRESHOLD_PCT",
    "DriftAction",
    "DriftDetector",
    "DriftResult",
    "ROLLBACK_THRESHOLD_PCT",
]

#: Alert name for a >2% regression (plan P19.T2 / plan 5.3 channel contract).
ALERT_MODEL_REGRESSION: Final[str] = "model_regression"

#: Alert name for the auto-rollback action (plan P19.T2).
ALERT_MODEL_ROLLBACK: Final[str] = "model_rollback"

#: Regression percentage above which an alert fires (plan P19.T2: >2%).
ALERT_THRESHOLD_PCT: Final[float] = 2.0

#: Regression percentage above which auto-rollback fires (plan P19.T2: >5%).
ROLLBACK_THRESHOLD_PCT: Final[float] = 5.0

#: Outcome of :meth:`DriftDetector.on_version_change`.
DriftAction = Literal["none", "alert", "rollback"]

#: Version-id -> eval score (the injected eval-set harness, plan Phase 10).
EvalRunner = Callable[[str], float]

#: Alert sink: ``(alert_name, message)`` (plan 5.3 single alerting channel).
Notifier = Callable[[str, str], None]


@dataclass(frozen=True)
class DriftResult:
    """Outcome of one version-change check (plan P19.T2).

    ``ran_eval`` is True whenever the eval harness ran (it always runs on the
    new version; it also runs for the old version when the registry has no
    recorded ``eval_score`` for it). ``regression_pct`` is
    ``(old - new) / old * 100`` — 0.0 when the old baseline is not positive
    (no meaningful baseline to regress from) or the new version improved.
    """

    ran_eval: bool
    old_score: float | None
    new_score: float | None
    regression_pct: float | None
    action: DriftAction


class DriftDetector:
    """Runs the eval set on version changes and mitigates regressions (plan P19.T2).

    ``registry`` is the :class:`~hermes.models.version_registry.ModelVersionRegistry`
    (plan P19.T1); ``eval_runner`` is the injected eval harness; ``notifier``
    receives ``(alert_name, message)`` for alerts (None = record-only).
    Thresholds are overridable for testing; production uses the plan's 2%/5%.
    """

    def __init__(
        self,
        registry: ModelVersionRegistry,
        eval_runner: EvalRunner,
        notifier: Notifier | None = None,
        alert_threshold_pct: float = ALERT_THRESHOLD_PCT,
        rollback_threshold_pct: float = ROLLBACK_THRESHOLD_PCT,
    ) -> None:
        if alert_threshold_pct < 0.0:
            raise ValueError(f"alert_threshold_pct must be >= 0, got {alert_threshold_pct!r}")
        if rollback_threshold_pct <= alert_threshold_pct:
            raise ValueError(
                f"rollback_threshold_pct ({rollback_threshold_pct!r}) must be strictly "
                f"greater than alert_threshold_pct ({alert_threshold_pct!r})"
            )
        self.registry = registry
        self.eval_runner = eval_runner
        self.notifier = notifier
        self.alert_threshold_pct = alert_threshold_pct
        self.rollback_threshold_pct = rollback_threshold_pct

    def on_version_change(
        self,
        old_version_id: str,
        new_version_id: str,
        component: str = "main-model",
    ) -> DriftResult:
        """Evaluate ``new_version_id`` and act on the regression (plan P19.T2).

        The eval harness runs on the new version (always) and on the old
        version only when the registry has no recorded ``eval_score`` for it.
        A >2% regression alerts (``model_regression``), a >5% regression
        additionally auto-rolls back by re-registering the previous version
        as current (``model_rollback``).
        """
        new_score = float(self.eval_runner(new_version_id))
        old_record = self._find_record(component, old_version_id)
        if old_record is not None and old_record.eval_score is not None:
            old_score = float(old_record.eval_score)
        else:
            old_score = float(self.eval_runner(old_version_id))

        regression_pct = self._regression_pct(old_score, new_score)

        if regression_pct > self.rollback_threshold_pct:
            self._rollback(component, old_version_id, old_record, old_score)
            self._notify(
                ALERT_MODEL_ROLLBACK,
                f"{component}: {new_version_id} regressed {regression_pct:.2f}% vs "
                f"{old_version_id} (old={old_score:.4f}, new={new_score:.4f}) — "
                f"auto-rolled back to {old_version_id}",
            )
            return DriftResult(
                ran_eval=True,
                old_score=old_score,
                new_score=new_score,
                regression_pct=regression_pct,
                action="rollback",
            )
        if regression_pct > self.alert_threshold_pct:
            self._notify(
                ALERT_MODEL_REGRESSION,
                f"{component}: {new_version_id} regressed {regression_pct:.2f}% vs "
                f"{old_version_id} (old={old_score:.4f}, new={new_score:.4f}) — "
                f"rollback to {old_version_id} recommended",
            )
            return DriftResult(
                ran_eval=True,
                old_score=old_score,
                new_score=new_score,
                regression_pct=regression_pct,
                action="alert",
            )
        return DriftResult(
            ran_eval=True,
            old_score=old_score,
            new_score=new_score,
            regression_pct=regression_pct,
            action="none",
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _find_record(self, component: str, version_id: str) -> ModelVersion | None:
        """Most recent registry record of ``component`` with ``version_id``."""
        for record in reversed(self.registry.get_history(component)):
            if record.version_id == version_id:
                return record
        return None

    @staticmethod
    def _regression_pct(old_score: float, new_score: float) -> float:
        """``(old - new) / old * 100``, floored at 0 for non-positive baselines."""
        if old_score <= 0.0:
            return 0.0
        return max(0.0, (old_score - new_score) / old_score * 100.0)

    def _rollback(
        self,
        component: str,
        old_version_id: str,
        old_record: ModelVersion | None,
        old_score: float,
    ) -> None:
        """Re-register the previous version as current (registry-append rollback)."""
        self.registry.register(
            component,
            old_version_id,
            provider=old_record.provider if old_record is not None else None,
            model_id=old_record.model_id if old_record is not None else None,
            eval_score=old_score,
            gepa_run_id=old_record.gepa_run_id if old_record is not None else None,
        )

    def _notify(self, alert_name: str, message: str) -> None:
        """Deliver an alert through the injected notifier (None = record-only)."""
        if self.notifier is not None:
            self.notifier(alert_name, message)
