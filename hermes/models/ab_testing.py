"""A/B testing framework for model versions (plan P19.T4).

Runs two model versions in parallel and decides whether the candidate
("treatment") replaces the current version ("control"):

- **Deterministic 80/20 traffic split** (configurable) — bucketing reuses
  the router's split primitive :func:`~hermes.models.router.bucket_for_request`
  (plan P19.T2), so the SAME ``request_key`` maps to the SAME side of the
  split as ``ModelRouter.route_with_weight(component, traffic_split,
  request_key)`` whenever the experiment's control is the registry's current
  version (and the treatment is the previous registration on the router
  side). Stable across processes and restarts — a conversation never flips
  variants mid-experiment.
- **Metric comparison after N requests** — per-request results
  (``eval_score`` / ``latency_ms`` / ``error``) are recorded per variant;
  :meth:`ABTest.compare` aggregates the PRIMARY metric (``eval_score``) once
  both variants have at least ``Experiment.min_requests`` recorded results.
- **Auto-promotion** — when the treatment is better by more than
  ``promote_threshold_pct`` (default 3%) on the primary metric,
  :meth:`ABTest.auto_promote` re-registers the treatment as the component's
  current version and fires the ``check`` callback; worse by more than
  ``rollback_threshold_pct`` (default 2%, mirroring the drift alert
  threshold, plan P19.T2) recommends ``rollback``; anything else ``keep``.

Promotion follows the registry-append pattern
(:class:`~hermes.models.version_registry.ModelVersionRegistry.register`
re-registers the treatment as current — history is never rewritten, the same
semantics the drift detector's auto-rollback uses, plan P19.T2). Tests drive
it with synthetic results; production feeds it the eval-score harness
outputs (plan Phase 10 fixtures).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal

from hermes.models.router import bucket_for_request
from hermes.models.version_registry import ModelVersionRegistry

__all__ = [
    "ABComparison",
    "ABTest",
    "Assignment",
    "DEFAULT_MIN_REQUESTS",
    "DEFAULT_PROMOTE_THRESHOLD_PCT",
    "DEFAULT_ROLLBACK_THRESHOLD_PCT",
    "Experiment",
    "PRIMARY_METRIC",
]

#: Primary comparison metric (plan P19.T4: eval scores decide promotion).
PRIMARY_METRIC: Final[str] = "eval_score"

#: Default minimum recorded requests per variant before comparing (plan P19.T4).
DEFAULT_MIN_REQUESTS: Final[int] = 100

#: Improvement percentage above which the treatment is promoted (plan P19.T4: >3%).
DEFAULT_PROMOTE_THRESHOLD_PCT: Final[float] = 3.0

#: Regression percentage above which a rollback is recommended (plan P19.T2 mirror: >2%).
DEFAULT_ROLLBACK_THRESHOLD_PCT: Final[float] = 2.0

Variant = Literal["control", "treatment"]
Recommendation = Literal["promote", "rollback", "keep", "insufficient_data"]

#: Per-request metrics: ``eval_score`` (float), ``latency_ms`` (float),
#: ``error`` (bool); extra numeric/boolean dimensions are allowed.
Metrics = dict[str, float | bool]

#: Sink for every recorded result: ``(experiment_id, variant, metrics)``.
MetricsRecorder = Callable[[str, str, Mapping[str, float | bool]], None]

#: Post-promotion callback: ``(component, promoted_version_id)``.
PromoteCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class Assignment:
    """One request's A/B decision (plan P19.T4).

    ``version_id`` is the model version to serve; ``variant`` records which
    side of the split the request landed on.
    """

    version_id: str
    variant: Variant


@dataclass(frozen=True)
class Experiment:
    """One A/B experiment's static configuration (plan P19.T4).

    ``control_version`` is the currently deployed version; ``treatment_version``
    the candidate. ``traffic_split`` is the FRACTION SERVED BY THE CONTROL
    (0.8 = the plan's 80/20 current/new split). ``min_requests`` gates the
    comparison ("metrics compared after N requests"). ``promote_threshold_pct``
    mirrors the plan's auto-promotion criterion (>3% better); the
    ``rollback_threshold_pct`` mirrors the drift alert threshold (>2% worse).
    """

    experiment_id: str
    component: str
    control_version: str
    treatment_version: str
    traffic_split: float = 0.8
    min_requests: int = DEFAULT_MIN_REQUESTS
    promote_threshold_pct: float = DEFAULT_PROMOTE_THRESHOLD_PCT
    rollback_threshold_pct: float = DEFAULT_ROLLBACK_THRESHOLD_PCT

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "component", "control_version", "treatment_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
        if self.control_version == self.treatment_version:
            raise ValueError("control_version and treatment_version must differ")
        if isinstance(self.traffic_split, bool) or not 0.0 <= self.traffic_split <= 1.0:
            raise ValueError(f"traffic_split must lie in [0, 1], got {self.traffic_split!r}")
        if isinstance(self.min_requests, bool) or self.min_requests < 1:
            raise ValueError(f"min_requests must be >= 1, got {self.min_requests!r}")
        for field_name in ("promote_threshold_pct", "rollback_threshold_pct"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value <= 0.0:
                raise ValueError(f"{field_name} must be > 0, got {value!r}")

    def assign(self, request_key: str) -> Assignment:
        """Deterministic variant for ``request_key`` (plan P19.T4).

        Bucketing uses the router's split primitive
        (:func:`~hermes.models.router.bucket_for_request`): buckets below
        ``traffic_split`` get the control, the rest the treatment — the same
        key maps identically under
        ``ModelRouter.route_with_weight(component, traffic_split,
        request_key)`` when the control is the registry's current version.
        """
        if not isinstance(request_key, str):
            raise TypeError(
                f"request_key must be str, got {type(request_key).__name__}"
            )
        bucket = bucket_for_request(request_key)
        variant: Variant = "control" if bucket < self.traffic_split else "treatment"
        version_id = (
            self.control_version if variant == "control" else self.treatment_version
        )
        return Assignment(version_id=version_id, variant=variant)


@dataclass(frozen=True)
class ABComparison:
    """Outcome of :meth:`ABTest.compare` / :meth:`ABTest.auto_promote`.

    Means and deltas are computed over recorded results carrying the primary
    metric (``eval_score``); ``improvement_pct`` is
    ``(treatment - control) / control * 100``. ``recommendation`` is
    ``insufficient_data`` before ``min_requests`` per variant (or without
    scored records), then ``promote`` / ``rollback`` / ``keep`` per the plan
    thresholds. ``promoted`` is set by :meth:`ABTest.auto_promote` when the
    treatment was actually re-registered as current.
    """

    experiment_id: str
    component: str
    n_control: int
    n_treatment: int
    mean_control: float | None
    mean_treatment: float | None
    delta: float | None
    improvement_pct: float | None
    recommendation: Recommendation
    promoted: bool = False


class ABTest:
    """Runs one experiment against the model version registry (plan P19.T4).

    ``registry`` resolves and re-registers versions; ``metrics_recorder``
    (optional) observes every recorded result — ``(experiment_id, variant,
    metrics)`` — for the diagnostics/telemetry layer.
    """

    def __init__(
        self,
        registry: ModelVersionRegistry,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.metrics_recorder = metrics_recorder
        self._experiment: Experiment | None = None
        # One recorded result per request key (last write wins on re-record).
        self._results: dict[str, tuple[Variant, Metrics]] = {}

    @property
    def experiment(self) -> Experiment | None:
        """The active experiment (``None`` before :meth:`begin`)."""
        return self._experiment

    # ── experiment lifecycle ──────────────────────────────────────────────
    def begin(self, experiment: Experiment) -> None:
        """Start ``experiment``: the control must be the component's current version.

        Mirrors the plan's 80/20 "current/new" semantics (and the router's):
        the control side of the split IS the deployed version. Clears any
        previously recorded results.
        """
        current = self.registry.get_current(experiment.component)
        if current is None:
            raise ValueError(
                f"component {experiment.component!r} has no current version in the "
                f"registry; register the control version {experiment.control_version!r} first"
            )
        if current.version_id != experiment.control_version:
            raise ValueError(
                f"experiment {experiment.experiment_id!r} control "
                f"{experiment.control_version!r} is not the current version of "
                f"{experiment.component!r} (current: {current.version_id!r})"
            )
        self._experiment = experiment
        self._results = {}

    def assign(self, request_key: str) -> Assignment:
        """Deterministic split for ``request_key`` under the active experiment."""
        return self._require_experiment().assign(request_key)

    # ── metric collection ─────────────────────────────────────────────────
    def record_result(
        self,
        request_key: str,
        variant: Variant,
        metrics: Mapping[str, float | bool],
    ) -> None:
        """Record one served request's outcome for ``variant``.

        ``variant`` is what was ACTUALLY served (normally
        ``assign(request_key).variant``; a failover may legitimately differ
        and is recorded as served). Results are deduplicated per
        ``request_key`` (one request, one result; re-recording replaces).
        ``metrics`` values must be numbers or booleans. The recorder sink (if
        wired) fires with ``(experiment_id, variant, metrics)``.
        """
        experiment = self._require_experiment()
        if variant not in ("control", "treatment"):
            raise ValueError(
                f"variant must be 'control' or 'treatment', got {variant!r}"
            )
        if not isinstance(request_key, str):
            raise TypeError(
                f"request_key must be str, got {type(request_key).__name__}"
            )
        metrics_dict: Metrics = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError(f"metric keys must be str, got {type(key).__name__}")
            if isinstance(value, bool):
                metrics_dict[key] = value
            elif isinstance(value, int | float):
                metrics_dict[key] = float(value)
            else:
                raise TypeError(
                    f"metric {key!r} must be a number or bool, got {type(value).__name__}"
                )
        self._results[request_key] = (variant, metrics_dict)
        if self.metrics_recorder is not None:
            self.metrics_recorder(experiment.experiment_id, variant, metrics_dict)

    # ── comparison & promotion ────────────────────────────────────────────
    def compare(self) -> ABComparison:
        """Aggregate the primary metric and recommend an action (plan P19.T4).

        After >= ``min_requests`` results on BOTH variants: treatment better
        by > ``promote_threshold_pct`` -> ``promote``; worse by >
        ``rollback_threshold_pct`` -> ``rollback``; else ``keep``.
        """
        experiment = self._require_experiment()
        control_scores: list[float] = []
        treatment_scores: list[float] = []
        n_control = 0
        n_treatment = 0
        for variant, metrics in self._results.values():
            if variant == "control":
                n_control += 1
                score = metrics.get(PRIMARY_METRIC)
                if isinstance(score, float):
                    control_scores.append(score)
            else:
                n_treatment += 1
                score = metrics.get(PRIMARY_METRIC)
                if isinstance(score, float):
                    treatment_scores.append(score)

        mean_control = (
            sum(control_scores) / len(control_scores) if control_scores else None
        )
        mean_treatment = (
            sum(treatment_scores) / len(treatment_scores) if treatment_scores else None
        )
        delta: float | None = None
        improvement_pct: float | None = None
        if mean_control is not None and mean_treatment is not None:
            delta = mean_treatment - mean_control
            if mean_control > 0.0:
                improvement_pct = delta / mean_control * 100.0

        recommendation: Recommendation = "insufficient_data"
        if (
            n_control >= experiment.min_requests
            and n_treatment >= experiment.min_requests
            and improvement_pct is not None
        ):
            if improvement_pct > experiment.promote_threshold_pct:
                recommendation = "promote"
            elif improvement_pct < -experiment.rollback_threshold_pct:
                recommendation = "rollback"
            else:
                recommendation = "keep"
        return ABComparison(
            experiment_id=experiment.experiment_id,
            component=experiment.component,
            n_control=n_control,
            n_treatment=n_treatment,
            mean_control=mean_control,
            mean_treatment=mean_treatment,
            delta=delta,
            improvement_pct=improvement_pct,
            recommendation=recommendation,
        )

    def auto_promote(self, check: PromoteCallback | None = None) -> ABComparison:
        """Compare, and on ``promote`` re-register the treatment as current.

        Promotion appends a new registration for the treatment version (with
        the treatment's mean primary metric as its ``eval_score``) — the same
        registry-append pattern the drift detector's auto-rollback uses;
        history is never rewritten. ``check`` (optional) fires after the
        promotion with ``(component, promoted_version_id)``. Returns the
        comparison (with ``promoted=True`` when promotion happened).
        """
        comparison = self.compare()
        experiment = self._require_experiment()
        if comparison.recommendation == "promote":
            self.registry.register(
                experiment.component,
                experiment.treatment_version,
                eval_score=comparison.mean_treatment,
            )
            comparison = replace(comparison, promoted=True)
            if check is not None:
                check(experiment.component, experiment.treatment_version)
        return comparison

    # ── internals ─────────────────────────────────────────────────────────
    def _require_experiment(self) -> Experiment:
        """The active experiment; raises when :meth:`begin` was not called."""
        if self._experiment is None:
            raise RuntimeError("no active experiment: call begin() first")
        return self._experiment
