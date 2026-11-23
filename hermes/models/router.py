"""Model router: current-version routing + deterministic A/B split primitive (plan P19.T2).

Routing always resolves through the
:class:`~hermes.models.version_registry.ModelVersionRegistry` (plan P19.T1):
:meth:`ModelRouter.route` returns the component's current version id.
:meth:`ModelRouter.route_with_weight` is the traffic-split helper for the
80/20 current/new A/B comparison (plan Phase 19): a request key is hashed
into a deterministic bucket in ``[0, 1)`` — requests below the split get the
current version, the rest get the previous registration (the candidate).
Bucketing is stable across processes and restarts (SHA-256, no Python hash
randomization), so a conversation always lands on the same side of the split.

The A/B testing framework itself (metric comparison, auto-promotion) is plan
P19.T4 (``hermes/models/ab_testing.py``); this module contributes the shared
split primitive :func:`bucket_for_request` and the :meth:`ModelRouter.route_ab`
delegation helper so a request can be routed straight through an active
experiment.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Final

from hermes.models.version_registry import ModelVersionRegistry

if TYPE_CHECKING:
    from hermes.models.ab_testing import ABTest

__all__ = ["DEFAULT_TRAFFIC_SPLIT", "ModelRouter", "bucket_for_request"]

#: Default share of requests served by the current version (plan P19.T4: 80/20).
DEFAULT_TRAFFIC_SPLIT: Final[float] = 0.8


def bucket_for_request(request_key: str) -> float:
    """Stable ``[0, 1)`` bucket for ``request_key`` (SHA-256, first 8 bytes).

    THE split primitive shared by :meth:`ModelRouter.route_with_weight` and
    the P19.T4 A/B framework: bucketing is deterministic across processes
    and restarts (no Python hash randomization), so a conversation always
    lands on the same side of a split.
    """
    digest = hashlib.sha256(request_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class ModelRouter:
    """Resolves component requests to model version ids (plan P19.T2).

    ``registry`` defaults to a registry at the default persistence location;
    ``config`` (optional) carries router settings — currently
    ``config["traffic_split"]`` as the default split for
    :meth:`route_with_weight` when the call does not pass one explicitly.
    """

    def __init__(
        self,
        registry: ModelVersionRegistry | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ModelVersionRegistry()
        self.config: dict[str, Any] = dict(config) if config is not None else {}

    def route(self, component: str = "main-model") -> str:
        """Current version id of ``component`` (plan P19.T2).

        Raises :class:`KeyError` when the component has never been registered.
        """
        current = self.registry.get_current(component)
        if current is None:
            raise KeyError(f"no registered version for component {component!r}")
        return current.version_id

    def route_with_weight(
        self,
        component: str = "main-model",
        traffic_split: float | None = None,
        request_key: str = "default",
    ) -> str:
        """Deterministic 80/20-style split between current and candidate (plan P19.T2).

        ``traffic_split`` (share served by the CURRENT version) defaults to
        ``config["traffic_split"]`` and then to :data:`DEFAULT_TRAFFIC_SPLIT`
        (0.8, the plan's 80/20). The ``request_key`` is hashed (SHA-256) into
        a bucket in ``[0, 1)``: buckets below the split resolve to the current
        version, the rest to the component's PREVIOUS registration (the A/B
        candidate); with no previous registration the current version serves
        everything. Deterministic across processes and restarts.
        """
        if traffic_split is None:
            configured = self.config.get("traffic_split", DEFAULT_TRAFFIC_SPLIT)
            traffic_split = float(configured)
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError(f"traffic_split must lie in [0, 1], got {traffic_split!r}")

        current = self.registry.get_current(component)
        if current is None:
            raise KeyError(f"no registered version for component {component!r}")

        bucket = self._bucket(request_key)
        if bucket < traffic_split:
            return current.version_id
        candidate = self._candidate(component)
        return candidate if candidate is not None else current.version_id

    def route_ab(
        self, component: str, experiment: ABTest, request_key: str
    ) -> str:
        """Route one request through an active A/B experiment (plan P19.T4).

        Delegates the decision to ``experiment.assign(request_key)`` — the
        experiment owns the split and the two versions; the component argument
        is cross-checked against the experiment's component so a misrouted
        call fails loudly. Returns the assigned version id (control or
        treatment), consistent with :meth:`route_with_weight` bucketing (both
        sides share :func:`bucket_for_request`).
        """
        active = experiment.experiment
        if active is None:
            raise RuntimeError("no active experiment: call begin() first")
        if active.component != component:
            raise ValueError(
                f"experiment {active.experiment_id!r} covers component "
                f"{active.component!r}, not {component!r}"
            )
        return experiment.assign(request_key).version_id

    @staticmethod
    def _bucket(request_key: str) -> float:
        """Stable ``[0, 1)`` bucket for ``request_key`` (SHA-256, first 8 bytes)."""
        return bucket_for_request(request_key)

    def _candidate(self, component: str) -> str | None:
        """Previous registration of ``component`` (the A/B candidate), or None."""
        history = self.registry.get_history(component)
        if len(history) < 2:
            return None
        return history[-2].version_id
