"""P19.T4 acceptance tests: A/B testing framework (plan Phase 19).

Covers the deterministic, router-aligned 80/20 traffic split and the metric
comparison state machine (insufficient data -> keep / promote / rollback per
the plan thresholds: >3% better promotes, >2% worse rolls back — mirroring
the drift alert threshold). Auto-promotion against the real registry
(default persistence layout) lives in the integration test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hermes.models.ab_testing import ABTest, Assignment, Experiment
from hermes.models.router import DEFAULT_TRAFFIC_SPLIT, ModelRouter, bucket_for_request
from hermes.models.version_registry import ModelVersionRegistry


@pytest.mark.unit
def test_traffic_split_80_20(tmp_path: Path) -> None:
    """200 deterministic keys split ~80/20, aligned with the router's bucketing."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "candidate-v9")  # previous registration
    registry.register("main-model", "current-v1")  # current = the control side

    experiment = Experiment(
        experiment_id="exp-80-20",
        component="main-model",
        control_version="current-v1",
        treatment_version="candidate-v9",
        traffic_split=0.8,
        min_requests=10,
    )
    assert DEFAULT_TRAFFIC_SPLIT == 0.8
    assert experiment.traffic_split == 0.8

    ab = ABTest(registry)
    ab.begin(experiment)

    keys = [f"request-{index}" for index in range(200)]
    assignments = {key: ab.assign(key) for key in keys}

    # deterministic: the same key always lands on the same side
    assert ab.assign("request-7") == assignments["request-7"]
    assert isinstance(ab.assign("request-7"), Assignment)

    # both sides reachable, split within tolerance of 80/20
    to_control = sum(1 for a in assignments.values() if a.variant == "control")
    assert 0.70 <= to_control / len(keys) <= 0.90
    to_treatment = len(keys) - to_control
    assert to_treatment > 0

    # variant -> version mapping
    for assignment in assignments.values():
        if assignment.variant == "control":
            assert assignment.version_id == "current-v1"
        else:
            assert assignment.version_id == "candidate-v9"

    # bucketing alignment: the same request_key maps identically under
    # ModelRouter.route_with_weight (control = current, treatment = previous
    # registration) and under the shared bucket primitive.
    router = ModelRouter(registry)
    for key in keys:
        assert router.route_with_weight(
            "main-model", traffic_split=0.8, request_key=key
        ) == ab.assign(key).version_id
        assert (bucket_for_request(key) < 0.8) == (
            ab.assign(key).variant == "control"
        )
    # and the router's route_ab delegation agrees as well
    assert router.route_ab("main-model", ab, "request-3") == ab.assign(
        "request-3"
    ).version_id
    with pytest.raises(ValueError, match="covers component"):
        router.route_ab("other-component", ab, "request-3")

    # the split is configurable: 50/50, all-treatment, all-control
    fifty = Experiment(
        experiment_id="exp-50-50",
        component="main-model",
        control_version="current-v1",
        treatment_version="candidate-v9",
        traffic_split=0.5,
    )
    fifty_ab = ABTest(registry)
    fifty_ab.begin(fifty)
    half = sum(
        1 for key in keys if fifty_ab.assign(key).variant == "control"
    ) / len(keys)
    assert 0.35 <= half <= 0.65

    extremes = Experiment(
        experiment_id="exp-extreme",
        component="main-model",
        control_version="current-v1",
        treatment_version="candidate-v9",
        traffic_split=1.0,
    )
    extremes_ab = ABTest(registry)
    extremes_ab.begin(extremes)
    assert all(extremes_ab.assign(key).variant == "control" for key in keys)
    all_treatment = Experiment(
        experiment_id="exp-all-treatment",
        component="main-model",
        control_version="current-v1",
        treatment_version="candidate-v9",
        traffic_split=0.0,
    )
    all_treatment_ab = ABTest(registry)
    all_treatment_ab.begin(all_treatment)
    assert all(all_treatment_ab.assign(key).variant == "treatment" for key in keys)

    # invalid configurations rejected eagerly
    with pytest.raises(ValueError, match="traffic_split"):
        Experiment(
            experiment_id="bad", component="main-model", control_version="a",
            treatment_version="b", traffic_split=1.5,
        )
    with pytest.raises(ValueError, match="must differ"):
        Experiment(
            experiment_id="bad", component="main-model", control_version="a",
            treatment_version="a",
        )

    # begin() requires the control to be the registry's current version
    with pytest.raises(ValueError, match="not the current version"):
        ABTest(registry).begin(
            Experiment(
                experiment_id="wrong-control",
                component="main-model",
                control_version="candidate-v9",
                treatment_version="current-v1",
            )
        )
    with pytest.raises(ValueError, match="no current version"):
        ABTest(registry).begin(
            Experiment(
                experiment_id="unregistered",
                component="never-registered",
                control_version="x",
                treatment_version="y",
            )
        )
    with pytest.raises(RuntimeError, match="no active experiment"):
        ABTest(registry).assign("request-1")


@pytest.mark.unit
def test_metric_comparison(tmp_path: Path) -> None:
    """Metrics compare after N requests; thresholds yield keep/promote/rollback."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "baseline", eval_score=0.70)
    experiment = Experiment(
        experiment_id="exp-metrics",
        component="main-model",
        control_version="baseline",
        treatment_version="candidate",
        traffic_split=0.5,
        min_requests=10,
        promote_threshold_pct=3.0,
        rollback_threshold_pct=2.0,
    )
    recorded: list[tuple[str, str, dict[str, Any]]] = []
    ab = ABTest(
        registry,
        metrics_recorder=lambda exp_id, variant, metrics: recorded.append(
            (exp_id, variant, dict(metrics))
        ),
    )
    ab.begin(experiment)
    assert ab.experiment == experiment

    def record(index: int, variant: str, score: float) -> None:
        # one request key per served request: a request is served by exactly
        # one variant, so keys must not collide across variants
        ab.record_result(
            f"req-{variant}-{index}",
            variant,  # type: ignore[arg-type]
            {"eval_score": score, "latency_ms": 900.0 if variant == "control" else 850.0,
             "error": False},
        )

    # before min_requests on both sides: insufficient data
    for index in range(6):
        record(index, "control", 0.70)
        record(index, "treatment", 0.77)
    comparison = ab.compare()
    assert comparison.n_control == 6
    assert comparison.n_treatment == 6
    assert comparison.recommendation == "insufficient_data"
    assert comparison.mean_control == pytest.approx(0.70)
    assert comparison.mean_treatment == pytest.approx(0.77)
    assert comparison.improvement_pct == pytest.approx(10.0)
    assert comparison.delta == pytest.approx(0.07)
    assert comparison.promoted is False

    # reach min_requests on both sides: +10% -> promote
    for index in range(6, 12):
        record(index, "control", 0.70)
        record(index, "treatment", 0.77)
    comparison = ab.compare()
    assert comparison.n_control == 12
    assert comparison.n_treatment == 12
    assert comparison.recommendation == "promote"
    assert comparison.improvement_pct == pytest.approx(10.0)

    # +1% (inside the 3% promote / 2% rollback band) -> keep
    keep_ab = ABTest(registry)
    keep_ab.begin(experiment)
    for index in range(12):
        keep_ab.record_result(
            f"k-c-{index}", "control", {"eval_score": 0.70, "error": False}
        )
        keep_ab.record_result(
            f"k-t-{index}",
            "treatment",
            {"eval_score": 0.7070, "error": False},
        )
    keep_comparison = keep_ab.compare()
    assert keep_comparison.improvement_pct == pytest.approx(1.0)
    assert keep_comparison.recommendation == "keep"

    # -6% (beyond the 2% drift-mirror threshold) -> rollback
    worse_ab = ABTest(registry)
    worse_ab.begin(experiment)
    for index in range(12):
        worse_ab.record_result(
            f"w-c-{index}", "control", {"eval_score": 0.70, "error": False}
        )
        worse_ab.record_result(
            f"w-t-{index}", "treatment", {"eval_score": 0.658, "error": True}
        )
    worse_comparison = worse_ab.compare()
    assert worse_comparison.improvement_pct == pytest.approx(-6.0)
    assert worse_comparison.recommendation == "rollback"

    # the recorder observed every recorded result
    assert recorded, "metrics_recorder was not invoked"
    assert recorded[0] == (
        "exp-metrics",
        "control",
        {"eval_score": 0.70, "latency_ms": 900.0, "error": False},
    )
    assert len(recorded) == 24  # 12 requests x 2 variants on the primary ab

    # validation: bad variant, bad metric value, and missing begin()
    with pytest.raises(ValueError, match="variant"):
        ab.record_result("req-x", "canary", {"eval_score": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metric"):
        ab.record_result("req-x", "control", {"eval_score": "high"})
    with pytest.raises(RuntimeError, match="no active experiment"):
        ABTest(registry).compare()

    # auto_promote on a non-promote comparison must NOT touch the registry
    keep_result = keep_ab.auto_promote()
    assert keep_result.recommendation == "keep"
    assert keep_result.promoted is False
    assert registry.get_current("main-model") is not None
    assert registry.get_current("main-model").version_id == "baseline"
