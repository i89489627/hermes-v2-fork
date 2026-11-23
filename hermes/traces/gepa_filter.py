"""GepaTraceFilter: GEPA-ready trace curation + eval-set export (plan P11.T4).

Turns the trace store into GEPA training/eval material (plan P11: "50-100
labeled examples per component"). Two concerns, one filter:

Selection
---------
:meth:`GepaTraceFilter.select` returns the traces that may enter a GEPA set:
``quality_label == "good"`` (plan P11.T3's auto-labeler writes exactly this
label for turns where a review decision actually applied and nothing
degraded) AND NOT PII-flagged. The PII gate is two layers, because the
stored ``pii_flagged`` field only reflects scans that ran upstream of the
store (the P17.T4 memory-write pre-hook) — turn prompts themselves are
never store-scanned:

1. the stored flag (cheap, SQLite-backed);
2. :meth:`scan_pii` — a fresh :class:`~hermes.memory.pii_detector.
   PIIDetector` pass over ``user_prompt`` plus the JSON-serialized stage
   payloads (extraction / review / gating / injection), so PII the store
   never saw is still caught before it reaches a training set. When no
   detector is injected the filter builds ``PIIDetector(action="warn")``
   internally: ``warn`` only logs — the filter decides exclusions itself
   from :func:`PIIDetector.scan` findings and never lets the detector's
   configured action silently drop text.

Export
------
:meth:`GepaTraceFilter.export` writes JSONL records matching the GEPA
eval-set schema of ``tests/fixtures/eval_sets/*.json`` (``{input,
expected_output, context}`` — the ``trace_id`` rides along as provenance
and doubles as the stable record id):

- ``input`` — the turn's ``user_prompt``;
- ``expected_output`` — the recorded pipeline outcome, paired per draft:
  each extraction draft as ``{"text", "type", "action"}`` where ``action``
  is that draft's final review decision (matched by ``draft_id``); when a
  trace recorded no drafts, the applied store writes stand in;
- ``context`` — the ``session_id`` (the conversation situation the example
  came from);
- ``trace_id`` — provenance / stable id.

PII handling at export time (``export_mode``):

- ``"skip"`` (default) — a trace whose scan finds anything is excluded
  from the export (and from :meth:`select`) entirely. Strictest and
  default: a GEPA example that cannot be proven clean never ships.
- ``"redact"`` — detected spans are replaced with ``[TYPE]`` placeholders
  (the detector's own :func:`PIIDetector.redact`) in the exported ``input``
  and in every draft ``text`` before the record is written; the ASSEMBLED
  record is then re-scanned and still-flagged records are skipped anyway
  (redaction cannot clean PII hiding in payload structures outside the
  exported text fields — those records are never written unsanitized).

Both modes count their outcome: ``written`` lines landed in the file,
``skipped`` records were withheld (PII in skip mode; PII redaction could
not fully clean in redact mode), ``redacted`` records were exported after
in-place placeholder substitution.

The export is ATOMIC: lines are staged in a temp file next to the target
and ``os.replace``d into place, so a crashed export can never leave a
truncated eval set behind (mirrors the store's own JSONL discipline).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from hermes.memory.pii_detector import PIIDetector
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

__all__ = ["EXPORT_MODES", "ExportCounts", "GepaTraceFilter"]

#: PII handling modes for :meth:`GepaTraceFilter.export` (see module docstring).
EXPORT_MODES: Final[tuple[str, ...]] = ("skip", "redact")

#: Export mode type (constructor / per-call override).
GepaExportMode = Literal["skip", "redact"]

#: Serialized format of :meth:`GepaTraceFilter.export` (the only one so far).
_EXPORT_FORMAT: Final[str] = "jsonl"

#: The quality label a trace must carry to enter a GEPA set (plan P11).
_GOOD_LABEL: Final[str] = "good"


@dataclass(frozen=True)
class ExportCounts:
    """Outcome of one :meth:`GepaTraceFilter.export` run (plan P11.T4).

    ``written`` records landed in the JSONL file, ``skipped`` were withheld
    (PII handling) and ``redacted`` is the subset of ``written`` that had
    detected spans replaced before export (``0`` in skip mode).
    """

    written: int
    skipped: int
    redacted: int


class GepaTraceFilter:
    """Curates the GEPA-ready subset of a trace store and exports eval sets.

    ``pii_detector`` defaults to an internally built
    ``PIIDetector(action="warn")`` (see the module docstring — the filter
    acts on scan findings itself regardless of the configured action);
    ``export_mode`` selects the default PII handling for :meth:`export`
    (``"skip"`` by default, ``"redact"`` as the redact-and-include variant;
    each call may override it).
    """

    def __init__(
        self,
        trace_store: TraceStore,
        pii_detector: PIIDetector | None = None,
        export_mode: GepaExportMode = "skip",
    ) -> None:
        if not isinstance(trace_store, TraceStore):
            raise TypeError(
                f"trace_store must be a TraceStore, got {type(trace_store).__name__}"
            )
        if export_mode not in EXPORT_MODES:
            raise ValueError(
                f"export_mode must be one of {EXPORT_MODES}, got {export_mode!r}"
            )
        self._store = trace_store
        self._pii = pii_detector if pii_detector is not None else PIIDetector(action="warn")
        self._export_mode: str = export_mode

    # ── PII scanning ────────────────────────────────────────────────────────
    def scan_pii(self, trace: Trace) -> bool:
        """True when the prompt or any serialized stage payload holds PII.

        Runs the detector over ``user_prompt`` and, for every recorded stage
        (``extraction`` / ``review`` / ``gating`` / ``injection``), over its
        JSON serialization — payload values are model/store text that a
        prompt-level scan alone would miss. A non-:class:`Trace` argument
        raises ``TypeError``.
        """
        if not isinstance(trace, Trace):
            raise TypeError(f"trace must be a Trace, got {type(trace).__name__}")
        return any(self._pii.scan(text).detected for text in _scannable_texts(trace))

    # ── selection ───────────────────────────────────────────────────────────
    def select(self) -> list[Trace]:
        """The GEPA-ready traces: ``good`` label, no stored flag, clean scan.

        Ordered oldest first (the store's deterministic iteration order), so
        a curated set is reproducible across runs. This is exactly the set
        skip-mode :meth:`export` writes.
        """
        return [
            trace
            for trace in self._good_candidates()
            if not self.scan_pii(trace)
        ]

    # ── export ──────────────────────────────────────────────────────────────
    def export(
        self,
        path: Path | str,
        format: str = _EXPORT_FORMAT,
        export_mode: GepaExportMode | None = None,
    ) -> ExportCounts:
        """Write the eval set to ``path`` (JSONL), atomically; return counts.

        Walks the ``good``-labeled, unflagged pool (the same candidates
        :meth:`select` screens) and applies the PII handling: ``"skip"``
        withholds scan-flagged traces; ``"redact"`` placeholder-substitutes
        the exported input and draft texts and ships the record only when
        the assembled line re-scans clean (module docstring). ``format``
        must be ``"jsonl"`` (the GEPA eval-set interchange format);
        ``export_mode`` defaults to the constructor's mode. The target is
        replaced atomically (temp file + ``os.replace``).
        """
        if format != _EXPORT_FORMAT:
            raise ValueError(
                f"format must be {_EXPORT_FORMAT!r} (GEPA eval-set interchange), "
                f"got {format!r}"
            )
        mode = self._export_mode if export_mode is None else export_mode
        if mode not in EXPORT_MODES:
            raise ValueError(f"export_mode must be one of {EXPORT_MODES}, got {mode!r}")

        lines: list[str] = []
        written = skipped = redacted = 0
        for trace in self._good_candidates():
            flagged = self.scan_pii(trace)
            if flagged and mode == "skip":
                skipped += 1
                continue
            record = _export_record(trace, redact=mode == "redact", detector=self._pii)
            if flagged and self._pii.scan(json.dumps(record, ensure_ascii=False)).detected:
                # redact mode could not fully clean the record (PII outside
                # the exported text fields) — never ship it unsanitized.
                skipped += 1
                continue
            lines.append(json.dumps(record, ensure_ascii=False))
            written += 1
            if flagged:
                redacted += 1

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write("".join(line + "\n" for line in lines))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return ExportCounts(written=written, skipped=skipped, redacted=redacted)

    # ── internals ───────────────────────────────────────────────────────────
    def _good_candidates(self) -> list[Trace]:
        """``good``-labeled traces without a stored PII flag, oldest first.

        The dynamic :meth:`scan_pii` pass is deliberately NOT applied here:
        :meth:`select` screens with it, while :meth:`export` needs the
        pre-screen pool to count skip-vs-redact handling per record.
        """
        return [
            trace
            for trace in self._store.get_all()
            if trace.quality_label == _GOOD_LABEL and not trace.pii_flagged
        ]


# ── module-level helpers (pure functions over a trace) ──────────────────────


def _scannable_texts(trace: Trace) -> list[str]:
    """The text surfaces one PII pass must cover: prompt + stage payloads.

    Stage payloads are serialized with ``ensure_ascii=False`` so the scanner
    sees the same characters the stored JSONL carries; a ``None`` stage
    contributes nothing.
    """
    texts = [trace.user_prompt]
    for stage_name in ("extraction", "review", "gating", "injection"):
        stage: Any = getattr(trace, stage_name)
        if isinstance(stage, Mapping):
            texts.append(json.dumps(stage, ensure_ascii=False))
    return texts


def _draft_items(trace: Trace) -> list[dict[str, Any]]:
    """Per-draft expected-output items paired with their final review action.

    Each extraction draft becomes ``{"text", "type", "action"}`` (``action``
    omitted when no review decision was recorded for the draft). Drafts are
    the extractor-facing supervision; the draft text is a plain string, so
    redact mode can clean it safely.
    """
    extraction = trace.extraction if isinstance(trace.extraction, Mapping) else {}
    review = trace.review if isinstance(trace.review, Mapping) else {}
    raw_drafts = extraction.get("drafts")
    actions_by_draft = {
        action["draft_id"]: action
        for action in review.get("actions", [])
        if isinstance(action, Mapping) and "draft_id" in action
    }
    items: list[dict[str, Any]] = []
    if isinstance(raw_drafts, list):
        for draft in raw_drafts:
            if not isinstance(draft, Mapping):
                continue
            item: dict[str, Any] = {
                "text": str(draft.get("text", "")),
                "type": str(draft.get("type", "")),
            }
            action = actions_by_draft.get(draft.get("draft_id"))
            if isinstance(action, Mapping) and "action" in action:
                item["action"] = str(action["action"])
            items.append(item)
    if not items:
        # No drafts recorded (e.g. a marker trace): the applied store writes
        # are the recorded outcome. ``good`` traces normally never land here.
        applied = review.get("applied")
        if isinstance(applied, list):
            items = [
                {
                    "action": str(write["action"]),
                    "entry_id": str(write.get("entry_id", "")),
                }
                for write in applied
                if isinstance(write, Mapping) and "action" in write
            ]
    return items


def _export_record(
    trace: Trace, redact: bool, detector: PIIDetector
) -> dict[str, Any]:
    """The GEPA eval-set record for ``trace`` (see the module docstring).

    With ``redact`` set, the exported input and every draft text have their
    detected spans replaced with ``[TYPE]`` placeholders first (identity for
    a clean text — :func:`PIIDetector.redact` finds nothing to replace).
    """
    prompt = detector.redact(trace.user_prompt) if redact else trace.user_prompt
    items = _draft_items(trace)
    if redact:
        items = [
            {**item, "text": detector.redact(item["text"])} if "text" in item else item
            for item in items
        ]
    return {
        "input": prompt,
        "expected_output": items,
        "context": trace.session_id,
        "trace_id": trace.trace_id,
    }
