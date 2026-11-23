"""hermes-v2 Pareto front store (plan P10.T4).

Keeps, per pipeline component (extraction, gating, injection, ...), the set
of prompt candidates that are non-dominated across all score dimensions (all
dimensions maximized). The v2 pipeline is single-model (plan Section 2.1), so
there is deliberately no model dimension — a component's front is keyed by
the component name alone.

Definitions
-----------
- Candidate ``a`` dominates candidate ``b`` when ``a`` scores ``>=`` ``b`` on
  every dimension and ``>`` ``b`` on at least one. A dimension missing from a
  candidate's score mapping counts as ``0.0`` (the candidate did not attempt
  that metric).
- ``get_front(component)`` returns the non-dominated candidates over the
  whole pool recorded for that component (all contexts), sorted by mean score
  (descending) then creation time.
- ``get_best(component, context=None)`` routes by context: when a context is
  given and the component has candidates recorded for it, the best candidate
  is chosen from that context-specific front; otherwise the global pool front
  is used. Selection is by highest mean score; ties go to the newest
  candidate.

Persistence
-----------
The whole candidate set is written atomically (tmp file + ``os.replace``) as
JSON to ``<hermes_home>/memory-repo/optimization/pareto-front.json`` by
default, so a crash mid-write can never corrupt the previously stored front.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from hermes.config import hermes_home

__all__ = ["Candidate", "ParetoFrontStore", "default_pareto_path"]

_FORMAT_VERSION: Final[int] = 1


def default_pareto_path() -> Path:
    """Default persistence path under ``hermes_home()`` (hermetic via env)."""
    return hermes_home() / "memory-repo" / "optimization" / "pareto-front.json"


def _coerce_scores(raw_scores: Mapping[str, Any]) -> dict[str, float]:
    """Validate a score mapping and copy it into a plain ``dict[str, float]``."""
    if isinstance(raw_scores, str) or not isinstance(raw_scores, Mapping):
        raise TypeError(
            f"scores must be a mapping of dimension -> number, "
            f"got {type(raw_scores).__name__}"
        )
    if not raw_scores:
        raise ValueError("scores must contain at least one dimension")
    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        if not isinstance(key, str):
            raise TypeError(f"score keys must be str, got {type(key).__name__}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"score {key!r} must be a number, got {type(value).__name__}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"score {key!r} must be finite, got {value!r}")
        scores[key] = numeric
    return scores


@dataclass
class Candidate:
    """One scored prompt variant belonging to a pipeline component."""

    component: str
    prompt: str
    scores: dict[str, float]
    context: str | None = None
    created_at: float = field(default_factory=time.time)

    def mean_score(self) -> float:
        """Unweighted mean over all score dimensions (routing/tie-break key)."""
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable projection (field order as declared)."""
        return {
            "component": self.component,
            "prompt": self.prompt,
            "scores": dict(self.scores),
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Candidate:
        """Rebuild from :meth:`to_dict` output; malformed records raise ValueError."""
        try:
            component = raw["component"]
            prompt = raw["prompt"]
            context = raw.get("context")
            created_at = raw.get("created_at", time.time())
            if not isinstance(component, str) or not component.strip():
                raise TypeError("component must be a non-empty str")
            if not isinstance(prompt, str) or not prompt.strip():
                raise TypeError("prompt must be a non-empty str")
            if context is not None and not isinstance(context, str):
                raise TypeError("context must be str or None")
            return cls(
                component=component,
                prompt=prompt,
                scores=_coerce_scores(raw["scores"]),
                context=context,
                created_at=float(created_at),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed candidate record: {raw!r}") from exc


class ParetoFrontStore:
    """Persistent per-component Pareto front of prompt candidates."""

    def __init__(self, path: Path | None = None) -> None:
        """Open (and lazily create) the front store backed by ``path``."""
        self.path: Path = path if path is not None else default_pareto_path()
        self._candidates: list[Candidate] = []
        self._lock = threading.Lock()
        self._load()

    # ── public API ────────────────────────────────────────────────────────
    def add_candidate(
        self,
        component: str,
        prompt: str,
        scores: Mapping[str, float],
        context: str | None = None,
    ) -> Candidate:
        """Validate, record and persist a new candidate; returns the stored one.

        ``scores`` maps metric dimension -> value, all maximized. ``context``
        tags the candidate with the situation it was optimized for (``None``
        = general-purpose). The candidate is persisted immediately.
        """
        if not isinstance(component, str) or not component.strip():
            raise ValueError("component must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if context is not None and not isinstance(context, str):
            raise TypeError("context must be a string or None")
        candidate = Candidate(
            component=component,
            prompt=prompt,
            scores=_coerce_scores(scores),
            context=context,
        )
        with self._lock:
            self._candidates.append(candidate)
            self._save()
        return candidate

    def get_front(self, component: str) -> list[Candidate]:
        """Non-dominated candidates for ``component`` (all contexts pooled).

        The front is sorted deterministically: mean score descending, then
        creation time ascending, then prompt text. Unknown components yield
        an empty list.
        """
        with self._lock:
            pool = [c for c in self._candidates if c.component == component]
            front = [c for c in pool if not self._is_dominated(c, pool)]
        return sorted(front, key=lambda c: (-c.mean_score(), c.created_at, c.prompt))

    def get_best(self, component: str, context: str | None = None) -> Candidate:
        """Best candidate for ``component``, routed by ``context``.

        When ``context`` is given and the component has candidates tagged with
        it, the context-specific front is used; otherwise (or when the
        context has no candidates) the global pool front is used. Selection
        within the front is by highest mean score, ties broken by the newest
        candidate. Raises ``KeyError`` when the component has no candidates.
        """
        with self._lock:
            pool: list[Candidate] = []
            if context is not None:
                pool = [
                    c
                    for c in self._candidates
                    if c.component == component and c.context == context
                ]
            if not pool:
                pool = [c for c in self._candidates if c.component == component]
            if not pool:
                raise KeyError(f"no candidates stored for component {component!r}")
            front = [c for c in pool if not self._is_dominated(c, pool)]
            best = max(front, key=lambda c: (c.mean_score(), c.created_at))
        return best

    # ── internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _dominates(a: Candidate, b: Candidate) -> bool:
        """True when ``a`` maximizes over ``b`` on every dimension and wins one."""
        dims = set(a.scores) | set(b.scores)
        weakly_better = all(a.scores.get(dim, 0.0) >= b.scores.get(dim, 0.0) for dim in dims)
        strictly_better = any(a.scores.get(dim, 0.0) > b.scores.get(dim, 0.0) for dim in dims)
        return weakly_better and strictly_better

    def _is_dominated(self, candidate: Candidate, pool: list[Candidate]) -> bool:
        return any(
            self._dominates(other, candidate) for other in pool if other is not candidate
        )

    def _load(self) -> None:
        """Load candidates from ``self.path``; missing file -> empty store."""
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                doc: Any = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt Pareto front store: {self.path}") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"Pareto front store must be a JSON object: {self.path}")
        if doc.get("version") != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported Pareto front store version {doc.get('version')!r}: "
                f"{self.path}"
            )
        raw_candidates = doc.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError(f"candidates must be a list: {self.path}")
        self._candidates = [Candidate.from_dict(raw) for raw in raw_candidates]

    def _save(self) -> None:
        """Persist all candidates atomically (tmp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": _FORMAT_VERSION,
            "candidates": [candidate.to_dict() for candidate in self._candidates],
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
