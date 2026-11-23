"""MemoryPipeline orchestrator: extract -> review -> apply (plan P3.T4).

Runs async after the Stop hook (plan Phase 3): the
:class:`~hermes.memory.extractor.ExtractorAgent` proposes draft facts for the
finished turn, the :class:`~hermes.memory.reviewer.ReviewerAgent` decides
``add`` | ``update`` | ``skip`` | ``conflict`` per draft, and the decisions
are applied to the :class:`~hermes.memory.store.MemoryStore` — one git commit
per write, inside the store.

Conflict resolution (plan P3.T4 acceptance: "conflict actions trigger
ReviewerAgent second pass with conflict context")
------------------------------------------------------------------------------
A first-pass ``conflict`` action triggers exactly ONE re-review: the
conflicting entry is loaded from the store and re-injected into the reviewer
as the explicit ``existing`` candidate list (the reviewer's prompt carries
each candidate's id/type/text, so the conflicting entry text reaches the
model), and a wrapper :class:`~hermes.memory.draft.DraftFact` is built whose
``rationale`` field carries the conflict rationale naming the target entry
and quoting its text — the audit trail shows why a second pass ran. The
wrapper's ``text`` stays the original draft text: merging the entry text INTO
the draft text would corrupt the fact if the resolution pass lands on
``add``.

**Termination rule (documented):** the resolution pass is FINAL. A draft
that still conflicts after the second pass is dropped — never written, no
third pass is ever spawned, and the drop is logged with both rationales.
This keeps the review loop finite (1 pass + at most 1 resolution pass) and
memory write-safe: an unresolvable contradiction must never enter the store.

Conflict-resolution protocol (plan P8.T2, additive)
------------------------------------------------------------------------------
There are TWO conflict origins with two resolution mechanisms:

1. **Detector-flagged contradictions** (the reviewer's rules-over-model
   rule 1 — recognized via
   :func:`hermes.memory.reviewer.is_detector_flagged_conflict`): the
   contradiction verdict is FINAL and the pipeline runs the explicit
   protocol — :meth:`hermes.memory.reviewer.ReviewerAgent.resolve_conflict`
   asks the model once HOW to resolve and returns a
   ``ResolutionDecision``:

   - ``merge``   -> ``store.update(target, text=merged_text)``
   - ``replace`` -> ``store.update(target, text=draft.text)``
   - ``skip``    -> drop (the draft is never written)

   Each resolution is logged with its rationale. A merge/replace surfaces
   as an ``update`` action (so ``apply_actions`` performs the write); a
   skip surfaces as ``conflict`` (log-only drop).

2. **Model-declared conflicts** (rule 3): the P3.T4 second review pass above
   stays unchanged — the model may reverse its own conflict verdict
   (``add``/``update``/``skip``); a still-conflicting outcome drops the
   draft (termination rule). When that pass resolves the conflict as an
   ``update`` (the draft body was merged into the target — a merge
   resolution), its rationale is marked so the audit event below fires too.

Audit trail in the trust card (plan P8.T2 acceptance)
------------------------------------------------------------------------------
After a merge/replace resolution actually wrote the target entry,
``apply_actions`` appends ONE verification event to the target's trust card
via :func:`hermes.memory.verification.add_verification_event`:

- source ``llm`` with weight ``0.3`` — the honest low-weight provenance for
  a machine-decided resolution (per the plan Section 2.4 weight table;
  deliberately NOT ``code-review-human``, which would over-claim human
  review), ``verifier`` ``conflict-resolver``, ``value`` 1.0;
- evidence ``conflict resolution <verb>: <rationale>`` — the resolution
  verb and the model's rationale (plan P8.T2: "resolution ... with
  rationale"; "audit trail in trust card").

The resolution verb rides in the final action's rationale as the structured
prefix ``conflict resolution <verb>: ...`` (``ReviewAction`` in draft.py is
frozen this wave; the design-sanctioned rationale channel). Audit failures
are logged and never mask the completed write (same failure-safety as the
metrics/trace side effects). Skip resolutions and drops write nothing and
leave no event.

Resilience contract (plan Section 5.5, graceful degradation)
------------------------------------------------------------------------------
- Extractor failure (any exception from ``extract``, including transport
  errors from the injected model callable): logged, treated as zero drafts —
  "Memory extraction -> skip, retry next turn". The run returns an empty
  :class:`PipelineResult` instead of raising.
- Reviewer failure for one draft: logged, that draft degrades to a
  ``skip`` action (never auto-add unaudited content); the rest of the run
  continues.
- Store failures propagate: ``add``/``update`` are transactional on the
  store side (rollback + commit), so a store error aborts the run and
  surfaces to the caller (on the returned future in async mode).

Observability: every action is logged at INFO level with its rationale; an
optional ``metrics`` callable receives a per-run summary dict (drafts count,
actions by type, applied count, elapsed); an optional ``trace_store`` (plan
P11.T2) receives ONE :class:`~hermes.traces.schema.Trace` per turn with ALL
stage components — ``turn_id``/``session_id``/``user_prompt``/``models_used``
/``extraction``/``review``/``gating``/``injection``/``elapsed_ms`` (inside the
review payload; the Trace schema has no dedicated field)/``timestamp`` — trace
capture failures are logged, never fatal (plan Section 5.5).

Trace capture across the hooks (plan P11.T2 one-trace-per-turn contract)
------------------------------------------------------------------------------
A full turn runs the pre-step injection hook BEFORE the Stop-hook pipeline, so
the per-turn trace is assembled cooperatively and appended exactly once:

- the pre-step hook (:mod:`hermes.memory.hooks.user_prompt_submit_hook`)
  records its gating + injection stage summaries via
  :func:`record_turn_injection` (keyed by trace-store instance + session id);
- the pipeline consumes that pending record in ``_run_flow`` and merges it
  into the single trace it appends — ``gating``/``injection`` stay ``None``
  when no injection ran for the turn (plan P11.T2 acceptance: "gating summary
  (if injection ran — else None)").
- the Stop hook (:mod:`hermes.memory.hooks.stop_hook`) appends its own
  pipeline-run MARKER trace only when its injected trace store is NOT the
  pipeline's (a pipeline without a trace store must still leave a turn
  marker on a store the operator wired into the hook); when both share one
  store the pipeline's full trace covers the turn and no marker is written.

``models_used`` derives from the documented ``model_name`` convention: a
model callable may expose ``model_name`` as a string attribute or a
zero-argument callable returning one; anything else (including plain
functions) is recorded as ``"anonymous"`` — deterministic, never a repr.

Compaction summaries as primary extraction input (plan P7.T5, additive)
------------------------------------------------------------------------------
When the store holds a compaction summary for the run's session
(:func:`hermes.memory.compaction.find_compaction_summary` — tagged
``compaction`` + ``session:<id>``), the extraction context is rebuilt per the
DSH pattern: a SYNTHETIC context turn carrying the summary's entry text is
PREPENDED (turn id ``compaction-<session_id>``), and every raw context turn
is marked secondary by prefixing its context TEXT with ``"[secondary] "`` —
a prompt-only decoration (context material is never stored); the turn IDS
stay untouched so ``source_turn_ids`` provenance is never polluted. With no
raw context turns the summary still reaches the extractor alone. A summary
lookup failure degrades to the plain context (plan Section 5.5 — extraction
must never break on observability metadata).

Async mode: ``sync=False`` runs the whole flow on a daemon thread and
returns a :class:`concurrent.futures.Future[PipelineResult]` immediately —
the calling (turn) thread is never blocked (plan acceptance: "pipeline
completes within 10s for typical turn (async)").
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from hermes.memory.compaction import find_compaction_summary
from hermes.memory.draft import DraftFact, ReviewAction
from hermes.memory.types import VerificationSource

if TYPE_CHECKING:
    from hermes.memory.context_window import Turn
    from hermes.memory.extractor import ExtractorAgent
    from hermes.memory.reviewer import ReviewerAgent
    from hermes.memory.store import MemoryStore
    from hermes.traces.store import TraceStore

logger = logging.getLogger(__name__)

__all__ = [
    "AppliedAction",
    "MemoryPipeline",
    "PipelineResult",
    "model_label",
    "record_turn_injection",
]

#: Wrapper-draft id suffix marking a conflict-resolution re-review pass.
_CONFLICT_PASS_SUFFIX = "-conflict-pass"

# ── conflict-resolution protocol (plan P8.T2, additive) ──────────────────────

#: Structured rationale prefix marking a conflict-resolution decision.
#: ``ReviewAction`` (draft.py) is frozen this wave, so the resolution verb
#: rides in the rationale (the design-sanctioned rationale channel) in the
#: exact format the audit event's evidence requires:
#: ``conflict resolution <verb>: <rationale>``.
_RESOLUTION_RATIONALE_PREFIX: str = "conflict resolution"

#: Resolution verbs that produce a target-entry write (and thus an audit event).
_RESOLUTION_WRITE_VERBS: frozenset[str] = frozenset({"merge", "replace"})

#: Provenance of the conflict-resolution audit event (plan P8.T2): source
#: ``llm`` / weight 0.3 — the honest low-weight provenance for a
#: machine-decided resolution per the plan Section 2.4 weight table
#: (deliberately NOT ``code-review-human``, which would over-claim human
#: review), ``value`` 1.0 (the resolution was performed and recorded).
_RESOLUTION_EVENT_SOURCE: VerificationSource = "llm"
_RESOLUTION_EVENT_WEIGHT: float = 0.3
_RESOLUTION_EVENT_VERIFIER: str = "conflict-resolver"


# ── per-turn trace capture (plan P11.T2, additive) ───────────────────────────

#: Label recorded for a model callable that exposes no documented
#: ``model_name`` (string attribute or zero-argument callable) — deterministic
#: by rule, never a repr of the callable.
_ANONYMOUS_MODEL: Final[str] = "anonymous"

# ── compaction-summary context (plan P7.T5, additive) ────────────────────────

#: Turn id of the synthetic context turn carrying the compaction summary.
_COMPACTION_TURN_ID_PREFIX: Final[str] = "compaction-"

#: Text prefix marking a raw context turn as the SECONDARY extraction input
#: (after the compaction summary) — prompt-only decoration, never stored.
_SECONDARY_TURN_MARKER: Final[str] = "[secondary]"

#: Pending pre-step injection records (plan P11.T2 hook -> pipeline handoff),
#: keyed by trace-store instance (weakly — a closed store must not be kept
#: alive by its own registry) and session id; the LATEST record per session
#: wins (a turn's injection is recorded before the Stop hook dispatches the
#: pipeline that consumes it).
_pending_injection_records: weakref.WeakKeyDictionary[Any, dict[str, dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)
_PENDING_LOCK: Final[threading.Lock] = threading.Lock()


def model_label(model: Any) -> str:
    """Deterministic model id for ``model`` per the documented convention.

    A model callable may expose ``model_name`` as a plain string attribute or
    as a zero-argument callable returning one (fakes and provider adapters use
    this to identify themselves in traces); anything else — including plain
    functions and agents without one — is ``"anonymous"``. Never raises.
    """
    name = getattr(model, "model_name", None)
    if isinstance(name, str) and name.strip():
        return name
    if callable(name):
        try:
            value = name()
        except Exception:
            return _ANONYMOUS_MODEL
        if isinstance(value, str) and value.strip():
            return value
    return _ANONYMOUS_MODEL


def _agent_model(agent: Any) -> Any:
    """The single-model architecture's injected model callable of ``agent``.

    :class:`~hermes.memory.extractor.ExtractorAgent` and
    :class:`~hermes.memory.reviewer.ReviewerAgent` hold their model as the
    documented ``_model`` attribute (same package, P3.T2/P3.T3); reading it
    here only LABELS the trace — the pipeline never calls the model through
    this accessor. ``None`` (lazy default wiring not yet triggered) labels as
    :data:`_ANONYMOUS_MODEL`.
    """
    return getattr(agent, "_model", None)


def _models_used(
    extractor_model: Any, reviewer_model: Any, gate_model_label: str = ""
) -> list[str]:
    """Ordered, de-duplicated model ids for the trace (P11.T2 ``models_used``).

    Extraction, review and (when the pre-step injection ran) gating all serve
    from the single main model (plan r3), so the same id may appear once;
    order follows the turn's stage order: extractor, reviewer, gate.
    """
    labels = [model_label(extractor_model), model_label(reviewer_model)]
    if gate_model_label:
        labels.append(gate_model_label)
    ordered: list[str] = []
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def record_turn_injection(
    trace_store: TraceStore, session_id: str, record: Mapping[str, Any]
) -> None:
    """Record the pre-step injection stage payloads for the pending turn.

    The UserPromptSubmit hook (plan P4.T4) calls this after a completed
    injection chain so the pipeline can merge the record's ``gating`` /
    ``injection`` payloads into the turn's SINGLE trace (plan P11.T2
    one-trace-per-turn contract; see the module docstring). Keyed by the
    trace-store instance + session id; the latest record per session wins.
    Never raises: a trace-recording failure must not break the injection
    (plan Section 5.5 failure-safety, same contract as the append itself).
    """
    try:
        with _PENDING_LOCK:
            pending = _pending_injection_records.get(trace_store)
            if pending is None:
                pending = {}
                _pending_injection_records[trace_store] = pending
            pending[session_id] = dict(record)
    except Exception:
        logger.warning("pipeline: injection trace record failed", exc_info=True)


def _consume_turn_injection(trace_store: TraceStore, session_id: str) -> dict[str, Any] | None:
    """Pop the pending injection record for ``session_id`` (pipeline side)."""
    with _PENDING_LOCK:
        pending = _pending_injection_records.get(trace_store)
        if pending is None:
            return None
        return pending.pop(session_id, None)


def _resolution_rationale(verb: str, rationale: str) -> str:
    """Render the structured resolution rationale (the audit evidence format)."""
    return f"{_RESOLUTION_RATIONALE_PREFIX} {verb}: {rationale}"


def _resolution_verb_from(rationale: str) -> str | None:
    """Extract the audit-bearing resolution verb from a structured rationale.

    Returns ``"merge"``/``"replace"`` for rationales rendered by
    :func:`_resolution_rationale`; ``None`` for anything else (plain review
    rationales must never trigger the audit event).
    """
    if not rationale.startswith(_RESOLUTION_RATIONALE_PREFIX + " "):
        return None
    verb = rationale[len(_RESOLUTION_RATIONALE_PREFIX) + 1 :].split(":", 1)[0].strip()
    return verb if verb in _RESOLUTION_WRITE_VERBS else None


@dataclass(frozen=True)
class AppliedAction:
    """One completed store write (plan P3.T4 ``apply_actions`` result).

    ``entry_id`` is the created entry id for ``add`` and the updated entry id
    for ``update``; ``rationale`` is the reviewer rationale behind the write.
    ``skip``/``conflict`` decisions never produce an :class:`AppliedAction`.
    """

    action: str
    draft_id: str
    entry_id: str | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (trace payload friendly)."""
        return {
            "action": self.action,
            "draft_id": self.draft_id,
            "entry_id": self.entry_id,
            "rationale": self.rationale,
        }


@dataclass
class PipelineResult:
    """Outcome of one ``run_after_turn`` execution (plan Phase 3 files table).

    ``drafts`` and ``actions`` are index-aligned (``actions[i]`` is the FINAL
    decision for ``drafts[i]`` — for a conflict that the resolution pass
    resolved, the final action replaces the first-pass conflict); ``applied``
    lists the store writes that actually happened.
    """

    turn_id: str
    drafts: list[DraftFact] = field(default_factory=list)
    actions: list[ReviewAction] = field(default_factory=list)
    applied: list[AppliedAction] = field(default_factory=list)
    elapsed_ms: float = 0.0


class MemoryPipeline:
    """Two-stage memory pipeline orchestrator (plan P3.T4).

    All collaborators are injected: ``extractor`` proposes drafts, ``reviewer``
    decides actions, ``store`` persists them. ``trace_store`` optionally
    receives the per-turn trace (P11.T2 wires this later), ``metrics`` an
    optional per-run summary callback, and ``max_drafts_per_turn`` caps the
    drafts reviewed per turn (plan Section 2.5: ``max_drafts_per_turn: 5``).
    """

    def __init__(
        self,
        extractor: ExtractorAgent,
        reviewer: ReviewerAgent,
        store: MemoryStore,
        *,
        trace_store: TraceStore | None = None,
        metrics: Callable[[dict[str, Any]], None] | None = None,
        max_drafts_per_turn: int = 5,
    ) -> None:
        self._extractor = extractor
        self._reviewer = reviewer
        self._store = store
        self._trace_store = trace_store
        self._metrics = metrics
        self._max_drafts_per_turn = max(0, int(max_drafts_per_turn))

    @property
    def trace_store(self) -> TraceStore | None:
        """The wired trace store, or ``None`` (plan P11.T2).

        Public read for the Stop hook: it appends its own pipeline-run MARKER
        trace only on a trace store this pipeline does NOT itself cover (see
        the module docstring's one-trace-per-turn contract).
        """
        return self._trace_store

    # ── stage 1: extraction ────────────────────────────────────────────────
    def extract_drafts(
        self,
        turn_text: str,
        *,
        context: str = "global",
        context_turns: Sequence[Turn | Mapping[str, Any]] | None = None,
    ) -> list[DraftFact]:
        """Stage 1: draft facts for the turn (plan P3.T4 ``extract_drafts``).

        Delegates to :meth:`ExtractorAgent.extract` with the caller's
        ``context_turns`` (P3.T6 multi-turn context). Never raises: an
        extractor failure is logged and yields ``[]`` (plan 5.5 graceful
        degradation — extraction is skipped and retried next turn). The
        result is capped at ``max_drafts_per_turn``; the over-budget drafts
        are dropped with a logged notice.
        """
        try:
            drafts = self._extractor.extract(
                turn_text, context_hint=context, context_turns=context_turns
            )
        except Exception:
            logger.warning(
                "pipeline: extraction failed; skipping memory extraction for this turn "
                "(plan 5.5 graceful degradation, retry next turn)",
                exc_info=True,
            )
            return []
        if len(drafts) > self._max_drafts_per_turn:
            dropped = len(drafts) - self._max_drafts_per_turn
            logger.info(
                "pipeline: capping drafts at %d per turn (%d dropped over budget)",
                self._max_drafts_per_turn,
                dropped,
            )
            drafts = drafts[: self._max_drafts_per_turn]
        return list(drafts)

    # ── stage 2: review ────────────────────────────────────────────────────
    def review_drafts(self, drafts: Sequence[DraftFact]) -> list[ReviewAction]:
        """Stage 2: one reviewer decision per draft (plan P3.T4 ``review_drafts``).

        Index-aligned with ``drafts``. A reviewer failure degrades that draft
        to a conservative ``skip`` (plan 5.5) — never an auto-add. A
        first-pass ``conflict`` is resolved exactly once, by origin: a
        detector-flagged contradiction runs the P8.T2 explicit resolution
        protocol (:meth:`_resolve_detector_conflict`), a model-declared
        conflict the P3.T4 re-review with conflict context
        (:meth:`_resolve_conflict` — see the module docstring for both
        mechanisms and the termination rule).
        """
        actions: list[ReviewAction] = []
        for draft in drafts:
            actions.append(self._review_one(draft))
        return actions

    def _review_one(self, draft: DraftFact) -> ReviewAction:
        """Review one draft, resolving a first-pass conflict via a second pass."""
        try:
            action = self._reviewer.review(draft)
        except Exception as exc:
            logger.warning(
                "pipeline: reviewer failed for draft %s; degrading to skip (plan 5.5)",
                draft.draft_id,
                exc_info=True,
            )
            return ReviewAction(
                action="skip", target_id=None, rationale=f"reviewer failed: {exc}"
            )
        if action.action != "conflict":
            return action
        from hermes.memory.reviewer import is_detector_flagged_conflict

        if is_detector_flagged_conflict(action):
            return self._resolve_detector_conflict(draft, action)
        logger.info(
            "pipeline: conflict on draft %s against entry %s -> second reviewer pass "
            "with conflict context (rationale: %s)",
            draft.draft_id,
            action.target_id,
            action.rationale,
        )
        return self._resolve_conflict(draft, action)

    def _resolve_detector_conflict(
        self, draft: DraftFact, action: ReviewAction
    ) -> ReviewAction:
        """P8.T2 explicit resolution protocol for a detector-flagged contradiction.

        The ConflictDetector's contradiction verdict is final (rules > model):
        the model is never re-asked WHETHER there is a conflict, only HOW to
        resolve it (:meth:`ReviewerAgent.resolve_conflict`). ``merge`` ->
        ``store.update(target, text=merged_text)``; ``replace`` ->
        ``store.update(target, text=draft.text)``; ``skip`` -> drop. Each
        resolution is logged with its rationale; merge/replace surface as
        ``update`` actions whose rationale carries the structured
        ``conflict resolution <verb>:`` prefix so ``apply_actions`` appends
        the audit event after the write.
        """
        conflicting = self._store.get(action.target_id) if action.target_id else None
        if conflicting is None:
            logger.warning(
                "pipeline: detector-flagged conflict on draft %s names entry %s, "
                "which cannot be loaded; dropping the draft (conservative)",
                draft.draft_id,
                action.target_id,
            )
            return ReviewAction(
                action="conflict",
                target_id=action.target_id,
                rationale=(
                    f"detector-flagged conflict against entry {action.target_id} "
                    "that cannot be loaded; draft dropped"
                ),
            )
        logger.info(
            "pipeline: detector-flagged conflict on draft %s against entry %s -> "
            "resolution protocol (plan P8.T2)",
            draft.draft_id,
            action.target_id,
        )
        try:
            decision = self._reviewer.resolve_conflict(draft, conflicting)
        except Exception as exc:
            # defensive: resolve_conflict is failure-safe internally; this guard
            # keeps any unexpected resolution failure from aborting the run
            logger.warning(
                "pipeline: conflict resolution failed for draft %s; dropping the "
                "draft (plan 5.5)",
                draft.draft_id,
                exc_info=True,
            )
            return ReviewAction(
                action="conflict",
                target_id=action.target_id,
                rationale=f"conflict resolution failed: {exc}",
            )
        logger.info(
            "pipeline: conflict on draft %s resolved as %s (rationale: %s)",
            draft.draft_id,
            decision.verb,
            decision.rationale,
        )
        if (
            decision.verb == "merge"
            and decision.merged_text is not None
            and action.target_id is not None
        ):
            return ReviewAction(
                action="update",
                target_id=action.target_id,
                rationale=_resolution_rationale("merge", decision.rationale),
                merged_text=decision.merged_text,
            )
        if decision.verb == "replace" and action.target_id is not None:
            return ReviewAction(
                action="update",
                target_id=action.target_id,
                rationale=_resolution_rationale("replace", decision.rationale),
                merged_text=draft.text,
            )
        # skip (and any merge/replace that lost its target or merged text) ->
        # the draft is dropped, the target entry stays untouched
        return ReviewAction(
            action="conflict",
            target_id=action.target_id,
            rationale=_resolution_rationale("skip", decision.rationale),
        )

    def _resolve_conflict(self, draft: DraftFact, action: ReviewAction) -> ReviewAction:
        """Second (and FINAL) reviewer pass with the conflicting entry injected.

        The conflicting entry is re-injected as the explicit ``existing``
        candidate and a wrapper draft carries the conflict rationale (module
        docstring). If the resolution pass still says ``conflict``, the draft
        is dropped (never written) and the drop is logged — the loop
        terminates by rule, never by chance.
        """
        conflicting = self._store.get(action.target_id) if action.target_id else None
        if conflicting is not None:
            conflict_note = (
                f"conflicts with existing entry {action.target_id}: {conflicting.text}"
            )
        else:
            conflict_note = (
                f"conflicts with existing entry {action.target_id} "
                "(entry text unavailable for injection)"
            )
        wrapper = DraftFact(
            draft_id=f"{draft.draft_id}{_CONFLICT_PASS_SUFFIX}",
            text=draft.text,
            type=draft.type,
            proposed_source=draft.proposed_source,
            confidence=draft.confidence,
            context_hint=draft.context_hint,
            rationale=conflict_note,
        )
        existing = [conflicting] if conflicting is not None else None
        try:
            second = self._reviewer.review(wrapper, existing=existing)
        except Exception as exc:
            logger.warning(
                "pipeline: conflict re-review failed for draft %s; degrading to skip "
                "(plan 5.5)",
                draft.draft_id,
                exc_info=True,
            )
            return ReviewAction(
                action="skip",
                target_id=None,
                rationale=f"conflict re-review failed: {exc}",
            )
        if second.action == "conflict":
            # Termination rule: the resolution pass is final — drop the draft.
            logger.info(
                "pipeline: draft %s still conflicts after the resolution pass; dropping "
                "it (termination rule: at most one resolution pass, never a third). "
                "First pass rationale: %s | Resolution pass rationale: %s",
                draft.draft_id,
                action.rationale,
                second.rationale,
            )
            return ReviewAction(
                action="conflict",
                target_id=action.target_id,
                rationale=(
                    f"still conflicting after resolution pass; draft dropped. "
                    f"First pass: {action.rationale} | Resolution pass: {second.rationale}"
                ),
            )
        logger.info(
            "pipeline: conflict on draft %s resolved by the second pass -> %s "
            "(rationale: %s)",
            draft.draft_id,
            second.action,
            second.rationale,
        )
        if (
            second.action == "update"
            and second.target_id is not None
            and second.merged_text is not None
        ):
            # A second-pass update IS a merge resolution (the draft body was
            # merged into the target): mark the rationale so apply_actions
            # appends the conflict-resolution audit event (plan P8.T2).
            return ReviewAction(
                action="update",
                target_id=second.target_id,
                rationale=_resolution_rationale("merge", second.rationale),
                merged_text=second.merged_text,
            )
        return second

    # ── stage 3: apply ─────────────────────────────────────────────────────
    def apply_actions(
        self,
        *,
        drafts: Sequence[DraftFact],
        actions: Sequence[ReviewAction],
        context: str,
        session_id: str,
    ) -> list[AppliedAction]:
        """Stage 3: apply decisions to the store (plan P3.T4 ``apply_actions``).

        ``add`` -> :meth:`MemoryStore.add` with the draft's provenance and
        ``evidence="extractor draft <id>"``; ``update`` ->
        :meth:`MemoryStore.update` with the reviewer's merged text (an update
        without merged text is logged and skipped — nothing to merge);
        ``skip``/``conflict`` are log-only. Every action is logged with its
        rationale. Store failures propagate (transactional on the store).
        """
        applied: list[AppliedAction] = []
        for draft, action in zip(drafts, actions, strict=True):
            if action.action == "add":
                entry = self._store.add(
                    text=draft.text,
                    type=draft.type,
                    context=context,
                    source=draft.proposed_source,
                    evidence=f"extractor draft {draft.draft_id}",
                    session_id=session_id,
                )
                logger.info(
                    "pipeline: applied add for draft %s -> entry %s (rationale: %s)",
                    draft.draft_id,
                    entry.id,
                    action.rationale,
                )
                applied.append(
                    AppliedAction(
                        action="add",
                        draft_id=draft.draft_id,
                        entry_id=entry.id,
                        rationale=action.rationale,
                    )
                )
            elif action.action == "update":
                if action.merged_text is None or action.target_id is None:
                    logger.warning(
                        "pipeline: update for draft %s carries no merged_text or no "
                        "target_id; skipping the write (rationale: %s)",
                        draft.draft_id,
                        action.rationale,
                    )
                    continue
                self._store.update(
                    action.target_id,
                    text=action.merged_text,
                    session_id=session_id,
                )
                verb = _resolution_verb_from(action.rationale)
                if verb is not None:
                    # P8.T2 audit trail: the write was a conflict-resolution
                    # merge/replace -> record it on the target's trust card.
                    self._append_conflict_resolution_audit(
                        action.target_id, action.rationale
                    )
                logger.info(
                    "pipeline: applied update for draft %s -> entry %s (rationale: %s)",
                    draft.draft_id,
                    action.target_id,
                    action.rationale,
                )
                applied.append(
                    AppliedAction(
                        action="update",
                        draft_id=draft.draft_id,
                        entry_id=action.target_id,
                        rationale=action.rationale,
                    )
                )
            elif action.action == "skip":
                logger.info(
                    "pipeline: skipped draft %s (rationale: %s)",
                    draft.draft_id,
                    action.rationale,
                )
            else:  # conflict — final decision, draft dropped (module docstring)
                logger.info(
                    "pipeline: dropped draft %s without writing (rationale: %s)",
                    draft.draft_id,
                    action.rationale,
                )
        return applied

    def _append_conflict_resolution_audit(self, entry_id: str, evidence: str) -> None:
        """Audit trail (plan P8.T2): record the resolution on the target card.

        Appends ONE verification event to ``entry_id``'s trust card via
        :func:`hermes.memory.verification.add_verification_event` — source
        ``llm`` / weight 0.3 (the honest low-weight provenance for a
        machine-decided resolution, documented in the module docstring) with
        evidence ``conflict resolution <verb>: <rationale>``. Called only
        AFTER the resolution write actually committed, so the event always
        describes a performed resolution. Failures are logged and never mask
        the completed write (same failure-safety as the metrics/trace side
        effects); skip resolutions and drops never reach this method.
        """
        try:
            from hermes.memory.verification import add_verification_event

            add_verification_event(
                self._store,
                entry_id,
                source=_RESOLUTION_EVENT_SOURCE,
                verifier=_RESOLUTION_EVENT_VERIFIER,
                evidence=evidence,
                value=1.0,
                weight=_RESOLUTION_EVENT_WEIGHT,
            )
        except Exception:
            logger.warning(
                "pipeline: conflict-resolution audit event failed for entry %s",
                entry_id,
                exc_info=True,
            )

    # ── orchestration ──────────────────────────────────────────────────────
    def run_after_turn(
        self,
        turn_id: str,
        turn_text: str,
        session_id: str = "default",
        context: str = "global",
        context_turns: Sequence[Turn | Mapping[str, Any]] | None = None,
        sync: bool = True,
    ) -> PipelineResult | Future[PipelineResult]:
        """Run the full extract -> review -> apply flow for a finished turn.

        ``sync=True`` (default) runs inline and returns the
        :class:`PipelineResult`. ``sync=False`` submits the flow to a daemon
        thread (the Stop-hook async path — never blocks the turn) and returns
        a :class:`concurrent.futures.Future[PipelineResult]` immediately; a
        store failure surfaces from ``future.result()``.

        Extractor/reviewer failures degrade gracefully (plan 5.5 — logged,
        empty/skip results); store failures propagate. The ``metrics``
        callback (if wired) receives the per-run summary and the
        ``trace_store`` (if wired) one trace record — both are failure-safe.
        When the store holds a compaction summary for ``session_id`` (plan
        P7.T5), the extraction context leads with the summary and the raw
        ``context_turns`` are marked secondary (see
        :meth:`_context_turns_with_summary`).
        """
        if sync:
            return self._run_flow(
                turn_id=turn_id,
                turn_text=turn_text,
                session_id=session_id,
                context=context,
                context_turns=context_turns,
                async_run=False,
            )
        future: Future[PipelineResult] = Future()

        def _worker() -> None:
            try:
                future.set_result(
                    self._run_flow(
                        turn_id=turn_id,
                        turn_text=turn_text,
                        session_id=session_id,
                        context=context,
                        context_turns=context_turns,
                        async_run=True,
                    )
                )
            except Exception as exc:  # the future is the error channel
                future.set_exception(exc)

        thread = threading.Thread(
            target=_worker, name=f"memory-pipeline-{turn_id}", daemon=True
        )
        thread.start()
        return future

    # ── virtual-session scoped extraction (plan P7.T4, additive) ───────────
    def extract_session_scope(
        self,
        session_id: str,
        turns: Sequence[tuple[str, str]],
        *,
        context: str = "global",
    ) -> PipelineResult:
        """Run the full extract -> review -> apply flow over an EXPLICIT turn window.

        P7.T4 seam for :class:`hermes.gateway.session_manager.VirtualSessionManager`:
        memory extraction scoped to a VIRTUAL session. Extraction sees exactly
        the provided ``turns`` (``(turn_id, turn_text)`` pairs, oldest first)
        and nothing else — each pair becomes a context turn
        (``{"turn_id", "text"}``) and the scope's final turn anchors the run,
        so a closed virtual session yields entries derived ONLY from its own
        turns (clean virtual-session boundary; plan Phase 7: "Memory
        extraction runs per virtual session, not per gateway chat").

        Reuses the standard stages (``extract_drafts`` -> ``review_drafts``
        -> ``apply_actions``) and the failure-safety/observability of
        :meth:`_run_flow` (metrics + trace included, ``session_id`` carries
        the virtual session id for provenance). Synchronous: closing a
        virtual session is itself an async, post-turn-time event owned by the
        session manager. Extractor/reviewer failures degrade gracefully
        (plan 5.5); store failures propagate. An empty ``turns`` window
        returns an empty result without touching the store. When a compaction
        summary for this session exists in the store (plan P7.T5 — the
        session manager compacts BEFORE calling this method), the extraction
        context leads with the summary and the raw turns are marked secondary
        (DSH pattern; see :meth:`_context_turns_with_summary`).
        """
        if not turns:
            return PipelineResult(turn_id=f"session-scope-{session_id}")
        anchor_id, anchor_text = turns[-1]
        context_turns: list[dict[str, str]] = [
            {"turn_id": turn_id, "text": turn_text} for turn_id, turn_text in turns
        ]
        return self._run_flow(
            turn_id=anchor_id,
            turn_text=anchor_text,
            session_id=session_id,
            context=context,
            context_turns=context_turns,
            async_run=False,
        )

    def _run_flow(
        self,
        *,
        turn_id: str,
        turn_text: str,
        session_id: str,
        context: str,
        context_turns: Sequence[Turn | Mapping[str, Any]] | None,
        async_run: bool,
    ) -> PipelineResult:
        """Full synchronous flow; metrics/trace are failure-safe side effects."""
        start = time.perf_counter()
        drafts: list[DraftFact] = []
        actions: list[ReviewAction] = []
        applied: list[AppliedAction] = []
        try:
            drafts = self.extract_drafts(
                turn_text,
                context=context,
                context_turns=self._context_turns_with_summary(session_id, context_turns),
            )
            actions = self.review_drafts(drafts)
            applied = self.apply_actions(
                drafts=drafts, actions=actions, context=context, session_id=session_id
            )
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Failure-safe side effects: neither may mask a propagating store error.
            self._record_metrics(
                turn_id=turn_id,
                session_id=session_id,
                context=context,
                drafts=drafts,
                actions=actions,
                applied=applied,
                elapsed_ms=elapsed_ms,
                async_run=async_run,
            )
            self._write_trace(
                turn_id=turn_id,
                session_id=session_id,
                turn_text=turn_text,
                drafts=drafts,
                actions=actions,
                applied=applied,
                elapsed_ms=elapsed_ms,
            )
        return PipelineResult(
            turn_id=turn_id,
            drafts=drafts,
            actions=actions,
            applied=applied,
            elapsed_ms=elapsed_ms,
        )

    # ── compaction-summary context (plan P7.T5, additive) ──────────────────
    def _context_turns_with_summary(
        self,
        session_id: str,
        context_turns: Sequence[Turn | Mapping[str, Any]] | None,
    ) -> Sequence[Turn | Mapping[str, Any]] | None:
        """Summary FIRST, raw turns marked secondary (plan P7.T5 DSH pattern).

        When the store holds a compaction summary for ``session_id`` (tagged
        ``compaction`` + ``session:<id>``; see
        :func:`hermes.memory.compaction.find_compaction_summary`), the
        extraction context is rebuilt: a synthetic context turn carrying the
        summary's entry text (turn id ``compaction-<session_id>``) is
        PREPENDED and every raw turn's context TEXT is prefixed with
        ``"[secondary] "`` — the module docstring documents why the marker
        rides in the text (turn ids stay clean for ``source_turn_ids``
        provenance; context material is prompt-only, never stored). Without a
        summary the caller's context passes through unchanged. A summary
        lookup failure degrades to the unchanged context (plan Section 5.5).
        """
        try:
            summary = find_compaction_summary(self._store, session_id)
        except Exception:
            logger.warning(
                "pipeline: compaction-summary lookup failed for session %s; "
                "extracting without it (plan 5.5)",
                session_id,
                exc_info=True,
            )
            return context_turns
        if summary is None:
            return context_turns
        from hermes.memory.context_window import Turn

        items: list[Turn | Mapping[str, Any]] = [
            {
                "turn_id": f"{_COMPACTION_TURN_ID_PREFIX}{session_id}",
                "text": summary.text,
            }
        ]
        for turn in context_turns or []:
            if isinstance(turn, Mapping):
                marked = dict(turn)
                text = marked.get("text")
                if isinstance(text, str) and text.strip():
                    marked["text"] = f"{_SECONDARY_TURN_MARKER} {text}"
                items.append(marked)
            elif isinstance(turn, Turn):
                items.append(
                    Turn(
                        turn_id=turn.turn_id,
                        session_id=turn.session_id,
                        index=turn.index,
                        text=f"{_SECONDARY_TURN_MARKER} {turn.text}",
                        timestamp=turn.timestamp,
                    )
                )
            else:  # unusable items are passed through: the extractor's
                # context coercion drops them with a logged warning (P3.T6)
                items.append(turn)
        return items

    # ── observability ──────────────────────────────────────────────────────
    def _record_metrics(
        self,
        *,
        turn_id: str,
        session_id: str,
        context: str,
        drafts: Sequence[DraftFact],
        actions: Sequence[ReviewAction],
        applied: Sequence[AppliedAction],
        elapsed_ms: float,
        async_run: bool,
    ) -> None:
        """Emit the per-run summary to the injected metrics callback."""
        if self._metrics is None:
            return
        by_type = {"add": 0, "update": 0, "skip": 0, "conflict": 0}
        for action in actions:
            by_type[action.action] += 1
        summary: dict[str, Any] = {
            "turn_id": turn_id,
            "session_id": session_id,
            "context": context,
            "async": async_run,
            "drafts": len(drafts),
            "actions_by_type": by_type,
            "applied": len(applied),
            "elapsed_ms": round(elapsed_ms, 3),
        }
        try:
            self._metrics(summary)
        except Exception:
            logger.warning("pipeline: metrics callback failed", exc_info=True)

    def _write_trace(
        self,
        *,
        turn_id: str,
        session_id: str,
        turn_text: str,
        drafts: Sequence[DraftFact],
        actions: Sequence[ReviewAction],
        applied: Sequence[AppliedAction],
        elapsed_ms: float,
    ) -> None:
        """Append the ONE per-turn trace when a trace store is wired (P11.T2).

        The trace carries every stage component (plan P11.T2 acceptance):
        ``turn_id``/``session_id``/``user_prompt``/``timestamp`` (Trace
        fields), ``models_used`` (documented ``model_name`` labels of the
        extractor/reviewer model callables plus the gate's when the pre-step
        injection recorded one for this session on THIS store — see
        :func:`record_turn_injection`), ``extraction`` (draft count + ids +
        full drafts), ``review`` (actions by type + rationales + applied
        writes + the run's ``elapsed_ms`` — the Trace schema has no dedicated
        elapsed field), and ``gating``/``injection`` summaries when an
        injection ran for the turn (else ``None``). Trace failures are logged
        and never mask a completed or failing run (module docstring).
        """
        if self._trace_store is None:
            return
        try:
            from hermes.traces.schema import Trace

            injection_record = _consume_turn_injection(self._trace_store, session_id)
            gate_model_label = ""
            gating: dict[str, Any] | None = None
            injection: dict[str, Any] | None = None
            if injection_record is not None:
                raw_gate = injection_record.get("gate_model")
                gate_model_label = raw_gate if isinstance(raw_gate, str) else ""
                raw_gating = injection_record.get("gating")
                gating = raw_gating if isinstance(raw_gating, dict) else None
                raw_injection = injection_record.get("injection")
                injection = raw_injection if isinstance(raw_injection, dict) else None
            actions_by_type = {"add": 0, "update": 0, "skip": 0, "conflict": 0}
            for action in actions:
                actions_by_type[action.action] += 1
            trace = Trace(
                turn_id=turn_id,
                session_id=session_id,
                user_prompt=turn_text,
                models_used=_models_used(
                    _agent_model(self._extractor),
                    _agent_model(self._reviewer),
                    gate_model_label,
                ),
                extraction={
                    "drafts_count": len(drafts),
                    "draft_ids": [draft.draft_id for draft in drafts],
                    "drafts": [draft.to_dict() for draft in drafts],
                },
                review={
                    "actions": [action.to_dict() for action in actions],
                    "actions_by_type": actions_by_type,
                    "rationales": [action.rationale for action in actions],
                    "applied": [record.to_dict() for record in applied],
                    "elapsed_ms": round(elapsed_ms, 3),
                },
                gating=gating,
                injection=injection,
            )
            self._trace_store.append(trace)
        except Exception:
            logger.warning("pipeline: trace capture failed", exc_info=True)
