"""P19.T4 integration test: auto-promotion over the default registry (plan Phase 19).

Drives the DEFAULT persistence layout (``<hermes_home>/model_versions.json``
via ``hermes.config.hermes_home``) end-to-end: an experiment serving a
treatment 10% better than control on the primary metric (``eval_score``)
must — after enough recorded requests — compare to ``promote``,
re-register the treatment as the component's current version (registry-append
promotion, the same pattern the drift detector's auto-rollback uses), fire
the post-promotion check, and survive a registry reload. The router's
current-version routing then resolves to the promoted treatment.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.config import hermes_home
from hermes.models.ab_testing import ABTest, Experiment
from hermes.models.router import ModelRouter
from hermes.models.version_registry import ModelVersionRegistry


@pytest.mark.integration
def test_auto_promote_on_better_metrics() -> None:
    """Treatment +10%: comparison promotes, registry re-registers, check fires."""
    registry = ModelVersionRegistry()  # default path under per-test HERMES_HOME
    assert registry.path == hermes_home() / "model_versions.json"
    registry.register("main-model", "baseline-v1", eval_score=0.70)

    experiment = Experiment(
        experiment_id="ab-weekly-202611",
        component="main-model",
        control_version="baseline-v1",
        treatment_version="candidate-v2",
        traffic_split=0.8,  # the plan's 80/20 — treatment still gets enough traffic
        min_requests=10,
    )
    recorded: list[tuple[str, str, dict[str, Any]]] = []
    ab = ABTest(
        registry,
        metrics_recorder=lambda exp_id, variant, metrics: recorded.append(
            (exp_id, variant, dict(metrics))
        ),
    )
    ab.begin(experiment)

    # 200 deterministic requests through the experiment's split; the
    # treatment is 10% better on the primary metric (0.77 vs 0.70) and
    # slightly faster.
    served: dict[str, str] = {}
    for index in range(200):
        key = f"conversation-{index}"
        assignment = ab.assign(key)
        served[key] = assignment.variant
        if assignment.variant == "control":
            ab.record_result(
                key,
                assignment.variant,
                {"eval_score": 0.70, "latency_ms": 900.0, "error": False},
            )
        else:
            ab.record_result(
                key,
                assignment.variant,
                {"eval_score": 0.77, "latency_ms": 850.0, "error": False},
            )

    # the 80/20 split held over the real request stream
    n_control = sum(1 for variant in served.values() if variant == "control")
    assert 0.70 <= n_control / len(served) <= 0.90
    assert len(recorded) == len(served)
    assert recorded[0][0] == "ab-weekly-202611"

    # routing before promotion: route_ab delegates to the experiment split
    router = ModelRouter(registry)
    assert router.route("main-model") == "baseline-v1"
    for index in range(5):
        key = f"conversation-{index}"
        assert router.route_ab("main-model", ab, key) == ab.assign(key).version_id

    # after >= min_requests per variant: treatment is >3% better -> promote
    calls: list[tuple[str, str]] = []
    comparison = ab.auto_promote(
        check=lambda component, version: calls.append((component, version))
    )

    assert comparison.recommendation == "promote"
    assert comparison.improvement_pct == pytest.approx(10.0)
    assert comparison.mean_control == pytest.approx(0.70)
    assert comparison.mean_treatment == pytest.approx(0.77)
    assert comparison.promoted is True
    assert calls == [("main-model", "candidate-v2")]

    # the promotion persisted: a reloaded registry sees the treatment current
    reloaded = ModelVersionRegistry()
    current = reloaded.get_current("main-model")
    assert current is not None
    assert current.version_id == "candidate-v2"
    assert current.eval_score == pytest.approx(0.77)
    # append-only history: the baseline is retained, never rewritten
    assert [record.version_id for record in reloaded.get_history("main-model")] == [
        "baseline-v1",
        "candidate-v2",
    ]
    # routing now resolves to the promoted version
    assert ModelRouter(reloaded).route("main-model") == "candidate-v2"
