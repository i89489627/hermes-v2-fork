"""P19.T2 acceptance tests: ModelRouter (plan Phase 19).

Covers the routing side of the drift task: current-version routing with a
clear error for unregistered components, and the deterministic hash-based
traffic-split primitive for the 80/20 A/B comparison (the A/B framework
itself is P19.T4, building on this primitive).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.models.router import DEFAULT_TRAFFIC_SPLIT, ModelRouter
from hermes.models.version_registry import ModelVersionRegistry


@pytest.mark.unit
def test_route_returns_current_version(tmp_path: Path) -> None:
    """route() resolves the component's current registration; unknowns fail."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    router = ModelRouter(registry)

    with pytest.raises(KeyError, match="no registered version"):
        router.route("main-model")

    registry.register("main-model", "v1")
    registry.register("prompt.compaction", "p1")
    registry.register("main-model", "v2")

    assert router.route() == "v2"  # default component
    assert router.route("main-model") == "v2"
    assert router.route("prompt.compaction") == "p1"

    # Explicit registry / config wiring is preserved.
    configured = ModelRouter(registry, config={"traffic_split": 0.5})
    assert configured.config["traffic_split"] == 0.5
    assert configured.route("main-model") == "v2"


@pytest.mark.unit
def test_route_with_weight_split(tmp_path: Path) -> None:
    """The split primitive buckets request keys deterministically 80/20."""
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    registry.register("main-model", "v1")
    registry.register("main-model", "v2")
    router = ModelRouter(registry)
    assert DEFAULT_TRAFFIC_SPLIT == 0.8

    # Deterministic: the same key always lands on the same side.
    first = router.route_with_weight(request_key="conversation-42")
    assert router.route_with_weight(request_key="conversation-42") == first
    assert first in {"v1", "v2"}

    # Both sides are reachable and the split is respected over many keys.
    keys = [f"request-{index}" for index in range(200)]
    control = {key: router.route_with_weight(request_key=key) for key in keys}
    assert set(control.values()) == {"v1", "v2"}  # candidate traffic exists
    to_current = sum(1 for version in control.values() if version == "v2")
    assert 0.70 <= to_current / len(keys) <= 0.90  # ~80% to the current version

    # Custom split: 0.0 -> always candidate; 1.0 -> always current.
    assert router.route_with_weight(traffic_split=0.0, request_key="a") == "v1"
    assert router.route_with_weight(traffic_split=1.0, request_key="a") == "v2"
    # Config default wins when the call omits the split.
    configured = ModelRouter(registry, config={"traffic_split": 0.0})
    assert configured.route_with_weight(request_key="b") == "v1"

    with pytest.raises(ValueError, match="traffic_split"):
        router.route_with_weight(traffic_split=1.5)
    with pytest.raises(KeyError, match="no registered version"):
        router.route_with_weight("never-registered")

    # No previous registration: the current version serves everything.
    solo_registry = ModelVersionRegistry(path=tmp_path / "solo.json")
    solo_registry.register("main-model", "only")
    solo = ModelRouter(solo_registry)
    assert solo.route_with_weight(traffic_split=0.0, request_key="x") == "only"
