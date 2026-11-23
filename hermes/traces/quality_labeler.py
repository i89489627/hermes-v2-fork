"""hermes-v2 auto-quality labeling for turn traces (plan P11.T3).

Assigns the ``quality_label`` of a :class:`~hermes.traces.schema.Trace` from
its recorded pipeline outcome — deterministic, rule-based, no model call —
and applies it through :meth:`~hermes.traces.store.TraceStore.label_quality`
so the JSONL source of truth and the SQLite/FTS5 index stay in sync.

Rule table (plan P11: "skip=bad, add=good, conflict=needs-review", refined
honestly against the pipeline's actual degradation contract — see
``hermes/memory/pipeline.py``; evaluated top-down, first match wins):

1. ``"bad"`` — the turn's model/store glue failed somewhere: a pipeline
   degradation marker rides on a skip action's rationale (``reviewer
   failed: ...`` / ``conflict re-review failed: ...`` — the pipeline's
   conservative skip when a reviewer model call raises, plan 5.5;
   ``unparseable reviewer output`` — the reviewer's own fallback after
   malformed model JSON survived its corrective retry), or the store-error
   signature holds: more ``add``/``update`` decisions than ``add``/``update``
   writes in ``review.applied`` (a decided write never landed — store
   failures propagate out of ``apply_actions`` and abort the run; the same
   signature covers the malformed ``update`` without ``merged_text`` that
   ``apply_actions`` logs and skips). Plan P11's "skip = bad" refined:
   only CONSERVATIVE degradation skips are bad — a deliberate reviewer
   dedup-skip is a correct decision, not a failure (distinguished by
   rationale, per the pipeline's action-rationale channel).
2. ``"needs-review"`` — any final review action is ``conflict`` (plan P11:
   conflict = needs-review). The pipeline's termination rule drops
   still-conflicting drafts without writing, and the P8.T2 protocol's skip
   resolutions leave the target untouched — a human must look at both.
3. ``"good"`` — at least one ``add``/``update`` write actually applied
   (plan P11: add = good; a conflict resolved as merge/replace surfaces as
   an ``update`` write).
4. ``"acceptable"`` — nothing failed and nothing was learned or written:
   zero drafts and zero review decisions (empty turn, or the extractor
   returned nothing after malformed model output — the trace payload cannot
   distinguish "empty turn" from "extraction parse failure" because the
   extractor degrades to ``[]`` with a log-only warning, so both land here
   and are retried next turn per plan 5.5), or only deliberate reviewer
   skips (duplicates etc.). Stop-hook marker traces (extraction marker
   payload, no review stage) land here too.

Failure markers win over conflicts and successes: a turn whose model/store
glue failed anywhere must never enter a GEPA eval set as a positive example.

Label source provenance
------------------------------------------------------------------------------
The Trace schema has no label-source field and
:class:`~hermes.traces.store.TraceStore` offers no payload-extension hook
(P11.T3 must not modify the store), while rewriting the store's JSONL from
outside would race the store's own atomic relabeling. The source is
therefore persisted in a small sidecar JSON file the labeler exclusively
owns — ``<trace_store.root>/label_sources.json`` (``trace_id`` ->
``"auto" | "manual"``), written atomically (temp file + ``os.replace``)
under a module lock. :func:`label_trace` refuses to relabel a trace whose
source is ``manual`` — a manual override always wins over auto-labeling.
The sidecar is a provenance record only: the quality label itself lives in
the store (JSONL + db) and survives without it.

``manual_override`` validates the label against ``good | bad | needs-review``
(plan P11.T3 acceptance: "Manual override via /trace label <id>
<good|bad|needs-review>"; ``acceptable`` is an auto-eligible outcome only —
a human who looked at a trace either confirms it good, condemns it bad or
escalates it) and is the mechanism behind that command surface.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

__all__ = [
    "MANUAL_LABELS",
    "auto_label",
    "label_source",
    "label_trace",
    "manual_override",
]

# ── quality labels (hermes.traces.schema.QUALITY_LABELS values) ─────────────

GOOD: Final[str] = "good"
ACCEPTABLE: Final[str] = "acceptable"
BAD: Final[str] = "bad"
NEEDS_REVIEW: Final[str] = "needs-review"

#: Labels a manual override may set (plan P11.T3: ``/trace label <id>
#: <good|bad|needs-review>``). ``acceptable`` is deliberately excluded: it
#: is an auto-eligible outcome ("nothing learned"), never a human verdict.
MANUAL_LABELS: Final[frozenset[str]] = frozenset({GOOD, BAD, NEEDS_REVIEW})

#: Actions that produce a store write when applied (plan P3.T4 ``apply_actions``).
_WRITE_ACTIONS: Final[frozenset[str]] = frozenset({"add", "update"})

# ── conservative-degradation rationale markers ──────────────────────────────
# Provenance of each literal (same-package, documented channels — the same
# convention the pipeline uses for its structured rationale prefixes):
# - "reviewer failed: {exc}" / "conflict re-review failed: {exc}" are built
#   inline by MemoryPipeline._review_one / _resolve_conflict when a reviewer
#   model call raises (plan 5.5: degrade that draft to a conservative skip).
# - "unparseable reviewer output" is hermes.memory.reviewer's
#   _UNPARSEABLE_RATIONALE for model JSON that stayed malformed after the
#   corrective retry.
_REVIEWER_FAILED_PREFIX: Final[str] = "reviewer failed:"
_CONFLICT_REREVIEW_FAILED_PREFIX: Final[str] = "conflict re-review failed:"
_UNPARSEABLE_REVIEWER_RATIONALE: Final[str] = "unparseable reviewer output"

# ── label-source sidecar (provenance record; see module docstring) ──────────

_SOURCES_FILENAME: Final[str] = "label_sources.json"
_AUTO_SOURCE: Final[str] = "auto"
_MANUAL_SOURCE: Final[str] = "manual"
_SOURCE_VALUES: Final[frozenset[str]] = frozenset({_AUTO_SOURCE, _MANUAL_SOURCE})
_SOURCES_LOCK: Final[threading.Lock] = threading.Lock()


def auto_label(trace: Trace) -> str:
    """Deterministic quality label for ``trace`` (see the rule table above).

    Reads only the trace's ``review`` stage payload (the pipeline's action
    rationales and applied writes). Never inspects model outputs or the
    store — the recorded outcome is the single input. Raises ``TypeError``
    for a non-:class:`Trace` argument; any unexpected payload shape degrades
    to :data:`ACCEPTABLE` (a trace the rule table cannot read must never be
    condemned by default).
    """
    if not isinstance(trace, Trace):
        raise TypeError(f"trace must be a Trace, got {type(trace).__name__}")
    review = trace.review
    actions = _stage_items(review, "actions")
    applied = _stage_items(review, "applied")
    if _has_degradation_marker(actions) or _decided_writes_never_landed(
        actions, applied
    ):
        return BAD
    if any(action.get("action") == "conflict" for action in actions):
        return NEEDS_REVIEW
    if any(write.get("action") in _WRITE_ACTIONS for write in applied):
        return GOOD
    return ACCEPTABLE


def label_trace(trace_store: TraceStore, trace_id: str) -> str:
    """Auto-label the stored trace and persist the ``auto`` source marker.

    Applies :func:`auto_label` via
    :meth:`~hermes.traces.store.TraceStore.label_quality` (JSONL rewrite +
    db update) and records ``"auto"`` in the label-source sidecar. A trace
    whose source is ``manual`` is NEVER relabeled — the manual override wins
    and its stored label is returned unchanged. Raises ``TypeError`` for
    non-str arguments, ``KeyError`` for an unknown ``trace_id`` (including
    a trace missing from the store's JSONL), and propagates sidecar write
    errors (the label is already applied at that point; the source record
    is the only thing lost).
    """
    if not isinstance(trace_store, TraceStore):
        raise TypeError(
            f"trace_store must be a TraceStore, got {type(trace_store).__name__}"
        )
    if not isinstance(trace_id, str):
        raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
    trace = trace_store.get(trace_id)
    if trace is None:
        raise KeyError(f"unknown trace_id: {trace_id}")
    if label_source(trace_store, trace_id) == _MANUAL_SOURCE:
        label = trace.quality_label
        if label is None:
            # Desync (manual source recorded, label cleared outside this
            # module): fall through to the deterministic rule table.
            label = auto_label(trace)
            trace_store.label_quality(trace_id, label)
        return label
    label = auto_label(trace)
    trace_store.label_quality(trace_id, label)
    _write_source(trace_store, trace_id, _AUTO_SOURCE)
    return label


def manual_override(trace_store: TraceStore, trace_id: str, label: str) -> str:
    """Apply a human label (``good | bad | needs-review``); returns it.

    Validates ``label`` against :data:`MANUAL_LABELS` (``ValueError``
    otherwise — ``acceptable`` is not a manual verdict) and applies it via
    :meth:`~hermes.traces.store.TraceStore.label_quality`, then records
    ``"manual"`` in the label-source sidecar so later :func:`label_trace`
    runs never clobber the override. Raises ``TypeError`` for non-str
    arguments and ``KeyError`` for an unknown ``trace_id``.
    """
    if not isinstance(trace_store, TraceStore):
        raise TypeError(
            f"trace_store must be a TraceStore, got {type(trace_store).__name__}"
        )
    if not isinstance(trace_id, str):
        raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
    if not isinstance(label, str):
        raise TypeError(f"label must be str, got {type(label).__name__}")
    if label not in MANUAL_LABELS:
        raise ValueError(
            f"manual label must be one of {sorted(MANUAL_LABELS)}, got {label!r}"
        )
    if trace_store.get(trace_id) is None:
        raise KeyError(f"unknown trace_id: {trace_id}")
    trace_store.label_quality(trace_id, label)
    _write_source(trace_store, trace_id, _MANUAL_SOURCE)
    return label


def label_source(trace_store: TraceStore, trace_id: str) -> str | None:
    """How ``trace_id`` was labeled: ``"auto"``, ``"manual"``, or ``None``.

    ``None`` means unlabeled by this module (never labeled, or labeled
    directly through the store). Reads the sidecar fresh on every call, so
    a fresh :class:`TraceStore` instance on the same root sees the same
    provenance (a reload keeps both the label and its source).
    """
    if not isinstance(trace_store, TraceStore):
        raise TypeError(
            f"trace_store must be a TraceStore, got {type(trace_store).__name__}"
        )
    if not isinstance(trace_id, str):
        raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
    return _read_sources(trace_store).get(trace_id)


# ── rule-table helpers (all defensive: payload shapes degrade, never raise) ─


def _stage_items(review: dict[str, Any] | None, key: str) -> list[Mapping[str, Any]]:
    """The ``review[key]`` list restricted to mapping items (``[]`` otherwise)."""
    if not isinstance(review, Mapping):
        return []
    raw = review.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _rationale(action: Mapping[str, Any]) -> str:
    """The action's rationale text (``""`` when absent or not a string)."""
    value = action.get("rationale")
    return value if isinstance(value, str) else ""


def _has_degradation_marker(actions: list[Mapping[str, Any]]) -> bool:
    """True when a conservative-degradation marker rides on a skip rationale.

    The pipeline only ever attaches these markers to ``skip`` actions (the
    plan 5.5 degradation channel), so the check is scoped to skips — a
    deliberate reviewer decision sharing a word with a marker must never
    condemn the trace.
    """
    for action in actions:
        if action.get("action") != "skip":
            continue
        rationale = _rationale(action)
        if (
            rationale.startswith(_REVIEWER_FAILED_PREFIX)
            or rationale.startswith(_CONFLICT_REREVIEW_FAILED_PREFIX)
            or rationale == _UNPARSEABLE_REVIEWER_RATIONALE
        ):
            return True
    return False


def _decided_writes_never_landed(
    actions: list[Mapping[str, Any]], applied: list[Mapping[str, Any]]
) -> bool:
    """Store-error signature: more add/update decisions than applied writes.

    ``apply_actions`` performs one store write per ``add``/``update`` action
    (a malformed ``update`` without ``merged_text``/``target_id`` is logged
    and skipped — nothing landed for it either). A store failure aborts the
    run with fewer applied writes than decisions; either way the trace
    records a turn whose glue failed.
    """
    decided = sum(1 for action in actions if action.get("action") in _WRITE_ACTIONS)
    landed = sum(1 for write in applied if write.get("action") in _WRITE_ACTIONS)
    return decided > landed


def _sources_path(trace_store: TraceStore) -> Path:
    """Path of the label-source sidecar for ``trace_store``."""
    return trace_store.root / _SOURCES_FILENAME


def _read_sources(trace_store: TraceStore) -> dict[str, str]:
    """Read the sidecar; a missing or corrupted file degrades to ``{}``."""
    try:
        raw: Any = json.loads(_sources_path(trace_store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(key, str)
        and value in _SOURCE_VALUES
    }


def _write_source(trace_store: TraceStore, trace_id: str, source: str) -> None:
    """Record ``source`` for ``trace_id`` in the sidecar (atomic replace)."""
    with _SOURCES_LOCK:
        sources = _read_sources(trace_store)
        sources[trace_id] = source
        path = _sources_path(trace_store)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(sources, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
