"""Model version registry (plan P19.T1).

Phase 19 closes the loop between GEPA prompt optimization and deployment:
every model deployment — the main model as well as each optimized prompt
component — is registered here so that (a) the full history is retained
and (b) the current version of any component is queryable at any time.

A :class:`ModelVersion` is one immutable deployment record.
``ModelVersionRegistry.register()`` appends it to the registry, making it
the current version of its component. The registry persists atomically
(tmp file + ``os.replace``) after every mutation to ``path`` (default
``<hermes_home>/model_versions.json``) and reloads on construction when
the file exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes.config import hermes_home

__all__ = ["ModelVersion", "ModelVersionRegistry"]


@dataclass(frozen=True)
class ModelVersion:
    """One registered deployment of a model component (plan P19.T1).

    ``component`` is ``"main-model"`` or a prompt component name.
    ``deployed_at`` is stamped by :meth:`ModelVersionRegistry.register`;
    ``eval_score`` records the evaluation score that motivated the
    deployment, and ``gepa_run_id`` links prompt-component deployments to
    the GEPA run that produced them.
    """

    component: str
    version_id: str
    provider: str | None = None
    model_id: str | None = None
    eval_score: float | None = None
    deployed_at: str = ""
    gepa_run_id: str | None = None


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _default_path() -> Path:
    """Default registry location under the Hermes home (plan P19.T1)."""
    return hermes_home() / "model_versions.json"


def _from_dict(raw: dict[str, Any]) -> ModelVersion:
    """Rebuild a record from its persisted form (inverse of ``asdict``)."""
    score = raw.get("eval_score")
    provider = raw.get("provider")
    model_id = raw.get("model_id")
    gepa_run_id = raw.get("gepa_run_id")
    return ModelVersion(
        component=str(raw["component"]),
        version_id=str(raw["version_id"]),
        provider=None if provider is None else str(provider),
        model_id=None if model_id is None else str(model_id),
        eval_score=None if score is None else float(score),
        deployed_at=str(raw.get("deployed_at", "")),
        gepa_run_id=None if gepa_run_id is None else str(gepa_run_id),
    )


class ModelVersionRegistry:
    """Append-only registry of model deployments with atomic JSON persistence.

    Registration order is the history order: the most recently registered
    record for a component is its current version.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = path if path is not None else _default_path()
        self._entries: list[ModelVersion] = []
        self._load()

    def register(
        self,
        component: str,
        version_id: str,
        provider: str | None = None,
        model_id: str | None = None,
        eval_score: float | None = None,
        gepa_run_id: str | None = None,
    ) -> ModelVersion:
        """Record a deployment; it becomes the component's current version.

        ``component`` and ``version_id`` must be non-empty (``ValueError``
        otherwise). The record is stamped with the deployment time and
        persisted atomically.
        """
        if not component:
            raise ValueError("component must be a non-empty string")
        if not version_id:
            raise ValueError("version_id must be a non-empty string")
        record = ModelVersion(
            component=component,
            version_id=version_id,
            provider=provider,
            model_id=model_id,
            eval_score=eval_score,
            deployed_at=_now(),
            gepa_run_id=gepa_run_id,
        )
        self._entries.append(record)
        self._persist()
        return record

    def get_current(self, component: str) -> ModelVersion | None:
        """Most recently registered version for ``component``.

        ``None`` when the component has never been registered.
        """
        for record in reversed(self._entries):
            if record.component == component:
                return record
        return None

    def get_history(self, component: str | None = None) -> list[ModelVersion]:
        """Registration history in order; all components when ``component`` is None."""
        if component is None:
            return list(self._entries)
        return [record for record in self._entries if record.component == component]

    # -- internals -----------------------------------------------------------

    def _load(self) -> None:
        """Load the registry from disk when the store file exists."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            doc: dict[str, Any] = json.load(fh)
        entries_raw: list[Any] = doc.get("entries", [])
        self._entries = [_from_dict(raw) for raw in entries_raw]

    def _persist(self) -> None:
        """Write the whole registry atomically (tmp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": [asdict(record) for record in self._entries],
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
