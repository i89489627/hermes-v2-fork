"""ReviewerAgent — dedup / update / conflict decisions on draft facts (plan P3.T3).

Two-stage pipeline (plan Phase 3): the extractor proposes
:class:`~hermes.memory.draft.DraftFact` candidates; the reviewer decides
``add`` | ``update`` | ``skip`` | ``conflict`` per draft against the top-5
semantically-matching existing entries (plan P3.T3 "semantic search top-5").

Single-model architecture (plan r3, same pattern as
:class:`hermes.memory.extractor.ExtractorAgent`): the LLM is injected as a
``Callable[[str], str]`` — the callable receives the full prompt and returns
the raw completion. ``model=None`` wires the fork-base auxiliary LLM lazily
at CALL time (:func:`hermes.memory.extractor.build_default_model`); importing
this module never imports the agent stack and never touches the network.

Precedence: RULES > MODEL. The reviewer consults the
:class:`hermes.memory.conflict_detector.ConflictDetector` FIRST and lets the
model only refine within what the rules allow:

1. Detector says ``contradiction`` (similarity > 0.85 plus a negation /
   antonym cue) -> action ``conflict`` with the detector's conflicting entry
   as ``target_id`` — the model output is not trusted for the decision.
2. Detector says ``duplicate`` but the model says ``add`` -> action ``skip``
   with the duplicated entry as ``target_id`` (the model's ``update`` and
   ``skip`` still pass through — merging and dropping are refinements only
   the model can judge).
3. The model may also declare ``conflict`` itself (it can catch semantic
   contradictions the lexical heuristic misses); its ``target_id`` must name
   one of the candidate entries.

Model-output validation (plan P3.T3 schema
``{"action", "target_id", "rationale", "merged_text"?}``):

- ``action`` must be one of the :data:`hermes.memory.types.ReviewAction`
  literals — anything else downgrades to ``skip`` with a rationale noting
  the invalid action (never coerced).
- ``target_id`` is required for ``update``/``skip``/``conflict`` and must
  name one of the candidate entries — a missing or unknown target downgrades
  the action to ``skip`` with a rationale (NOT to ``add``: never store a
  draft whose dedup check the model failed to anchor).
- ``merged_text`` is kept only for ``update`` and dropped for every other
  action.
- Malformed output (not a JSON object) triggers exactly one corrective
  retry; output that is still malformed yields ``skip`` with rationale
  ``"unparseable reviewer output"`` — conservative: a parse failure never
  auto-adds a draft to memory.

The ``search_fn`` dependency fetches the existing candidates; the default
lazily builds a :class:`hermes.memory.store.MemoryStore` over
``hermes_home() / "memory-repo"`` (paths derive from
:func:`hermes.config.hermes_home`) and uses its READ-ONLY
``search(query, top_k=5)`` spanning all contexts — dedup is safest when it
sees every stored entry; the pipeline (P3.T4) can inject a context-scoped
``search_fn`` instead. ``review`` itself never writes to the store.

Conflict-resolution protocol (plan P8.T2, additive)
------------------------------------------------------------------------------
When rule 1 fires (a detector-flagged contradiction), the conflict verdict is
FINAL — the model is never re-asked *whether* there is a conflict. The memory
pipeline instead calls :meth:`ReviewerAgent.resolve_conflict`, which asks the
injected model ONCE *how* to resolve it and returns a
:class:`ResolutionDecision` with an explicit verb:

- ``merge`` — both texts carry durable details; ``merged_text`` holds the
  merged entry body that resolves to the more current fact. The pipeline
  writes it with ``store.update(target, text=merged_text)``.
- ``replace`` — the draft is the newer truth and supersedes the entry; the
  pipeline writes ``store.update(target, text=draft.text)``.
- ``skip`` — the draft is dropped; the entry stays untouched.

Resolution validation (rules > model applies to the resolution too): a
``merge`` without a usable ``merged_text`` degrades to ``skip``; a model
attempt to overrule the contradiction (``add``) or re-declare it
(``conflict``) also degrades to ``skip`` — never stores anything; any model
failure (malformed output, transport error) yields the conservative
``skip``. Every resolution carries the model's ``rationale``, which becomes
part of the audit event's evidence (``conflict resolution <verb>:
<rationale>``) appended to the target's trust card by the pipeline.

Detector-flagged conflicts ride to the pipeline through a structured-suffix
marker in the rule-1 rationale (:data:`DETECTOR_CONFLICT_MARKER`, checked by
:func:`is_detector_flagged_conflict`) — :class:`~hermes.memory.draft.
ReviewAction` lives in ``draft.py`` which is frozen this wave, so the
design-sanctioned rationale channel carries the flag. Model-declared
conflicts (rule 3) keep the P3.T4 second review pass unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from hermes.memory.conflict_detector import ConflictDetector
from hermes.memory.draft import REVIEW_ACTIONS, DraftFact, ReviewAction
from hermes.memory.entry import MemoryEntry
from hermes.memory.extractor import build_default_model
from hermes.memory.types import ReviewAction as ReviewActionLiteral

if TYPE_CHECKING:
    from hermes.memory.store import MemoryStore

logger = logging.getLogger(__name__)

#: System prompt: dedup rules, conflict detection and merge rules (plan P3.T3).
REVIEWER_SYSTEM_PROMPT = """\
You are the hermes-v2 memory reviewer (plan Phase 3). One candidate memory \
draft is proposed for long-term storage; decide what to do with it against \
the candidate EXISTING entries (semantic top-5).

Decisions (use exactly one):
- "add": the draft is a new durable fact; no existing entry covers it.
- "update": an existing entry is related and should absorb the draft - \
merge both into one improved entry body and return it as "merged_text".
- "skip": the draft is an exact or near duplicate of an existing entry (no \
new information) - name that entry as "target_id". Also use "skip" when \
the draft is transient or not worth remembering.
- "conflict": the draft contradicts an existing entry - name that entry as \
"target_id". Never merge contradictory facts.

Merge rules: keep every durable detail from BOTH texts, resolve to the more \
current one, never invent facts, output one self-contained sentence.

Output format - return ONLY a JSON object (no prose, no markdown fences):
{"action": "<add|update|skip|conflict>",
 "target_id": "<id of the affected existing entry, or null>",
 "rationale": "<one sentence why>",
 "merged_text": "<merged entry body, only for update, else null>"}

"target_id" MUST be one of the provided existing entry ids for \
update/skip/conflict, and null for "add".\
"""

#: Corrective suffix appended on the single malformed-output retry.
REVIEWER_RETRY_SUFFIX = (
    "Your previous response was not a valid JSON object. "
    "Return ONLY a JSON object - no prose, no markdown fences, no code blocks."
)

#: Number of existing entries considered per review (plan P3.T3: top-5).
TOP_K_EXISTING: int = 5

#: Rationale emitted when the model output stays malformed after the retry.
_UNPARSEABLE_RATIONALE = "unparseable reviewer output"

#: Prompt framing around the review-request JSON object.
_REVIEW_HEADER = "Review request (JSON object):"
_REVIEW_FOOTER = "Respond with ONLY the JSON decision object."

# ── conflict-resolution protocol (plan P8.T2, additive) ──────────────────────

#: Structured-suffix marker appended to rule-1 (ConflictDetector) conflict
#: rationales. :class:`~hermes.memory.draft.ReviewAction` lives in
#: ``draft.py`` which is frozen this wave, so the detector-flagging rides in
#: the rationale (the design-sanctioned structured-suffix channel): the
#: memory pipeline checks it with :func:`is_detector_flagged_conflict` to
#: route detector-flagged contradictions into the explicit P8.T2 resolution
#: protocol, while model-declared conflicts keep the P3.T4 second review pass.
DETECTOR_CONFLICT_MARKER: str = "[conflict-source: conflict-detector]"

#: Verb of an explicit conflict resolution (plan P8.T2 acceptance:
#: "Resolution: merge, replace, or skip (with rationale)").
ResolutionVerb = Literal["merge", "replace", "skip"]

#: Runtime set of valid resolution verbs, derived from the literal.
RESOLUTION_VERBS: frozenset[str] = frozenset(("merge", "replace", "skip"))

#: Wrapper-draft id suffix marking the conflict-resolution request in the
#: prompt payload (same provenance marker as the pipeline's P3.T4 pass).
_RESOLUTION_DRAFT_SUFFIX = "-conflict-pass"

#: System prompt for the explicit resolution pass (plan P8.T2): the detector's
#: contradiction verdict is final; the model only chooses HOW to resolve it.
RESOLUTION_SYSTEM_PROMPT = """\
You are the hermes-v2 memory conflict resolver (plan Phase 8). A draft fact \
was flagged by the ConflictDetector as a CONTRADICTION of an existing entry. \
The contradiction verdict is final - do not re-litigate it. Decide only HOW \
to resolve it:

- "merge": both texts carry durable details; merge them into one improved, \
self-contained entry body that resolves to the more current fact. Return it \
as "merged_text". Never store contradictory claims side by side - the merge \
must resolve the contradiction (keep the current fact, retain still-valid \
details from the old one).
- "replace": the draft is the newer truth and fully supersedes the existing \
entry; the entry body becomes the draft text as-is.
- "skip": the draft is transient, redundant or not worth storing - the \
existing entry stays untouched and the draft is dropped.

Output format - return ONLY a JSON object (no prose, no markdown fences):
{"resolution": "<merge|replace|skip>",
 "merged_text": "<merged entry body, only for merge, else null>",
 "rationale": "<one sentence why>"}
"""

#: Prompt framing around the conflict-resolution request JSON object.
_RESOLUTION_HEADER = "Conflict resolution request (JSON object):"
_RESOLUTION_FOOTER = "Respond with ONLY the JSON resolution object."

#: Rationale emitted when the resolution model output is unusable or the
#: model tries to overrule the detector's verdict (conservative skip).
_UNRESOLVABLE_RATIONALE = (
    "model resolution not actionable; draft dropped "
    "(the ConflictDetector's contradiction verdict is final)"
)


@dataclass(frozen=True)
class ResolutionDecision:
    """Explicit conflict resolution chosen in the P8.T2 resolution pass.

    ``verb`` is one of :data:`RESOLUTION_VERBS`; ``merged_text`` carries the
    model-merged entry body for ``merge`` (always ``None`` for
    ``replace``/``skip`` — replace writes the draft text, skip writes
    nothing); ``rationale`` is the model's one-sentence justification and
    becomes part of the audit event's evidence
    (``conflict resolution <verb>: <rationale>``) that the pipeline appends
    to the target entry's trust card.
    """

    verb: ResolutionVerb
    rationale: str
    merged_text: str | None = None

    def __post_init__(self) -> None:
        if self.verb not in RESOLUTION_VERBS:
            allowed = sorted(RESOLUTION_VERBS)
            raise ValueError(
                f"invalid resolution verb: {self.verb!r}; expected one of {allowed}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (log/trace friendly)."""
        return {
            "verb": self.verb,
            "rationale": self.rationale,
            "merged_text": self.merged_text,
        }


def is_detector_flagged_conflict(action: ReviewAction) -> bool:
    """True when ``action`` is a conflict decided by the ConflictDetector rule.

    Rules-over-model precedence (module docstring rule 1): the detector's
    contradiction verdict fired and the model was never consulted. The
    memory pipeline (P8.T2) routes these into
    :meth:`ReviewerAgent.resolve_conflict`; model-declared conflicts (rule 3)
    return ``False`` and keep the P3.T4 second review pass.
    """
    return action.action == "conflict" and DETECTOR_CONFLICT_MARKER in action.rationale


def _validated_resolution(raw: dict[str, Any]) -> ResolutionDecision:
    """Schema-validate the parsed resolution output (see ``resolve_conflict``).

    ``{"resolution", "merged_text"?, "rationale"}`` is the native schema;
    legacy review-schema synonyms are tolerated (``"update"`` -> ``merge``,
    ``"skip"`` -> ``skip``). Everything else — including a model attempt to
    overrule the detector (``add``) or re-declare the conflict — degrades to
    the conservative ``skip``: a resolution never stores anything.
    """
    verb_raw = raw.get("resolution", raw.get("action"))
    rationale = str(raw.get("rationale", "") or "").strip()
    merged_raw = raw.get("merged_text")
    merged_text = str(merged_raw).strip() if isinstance(merged_raw, str) else None
    merged_text = merged_text or None

    verb = verb_raw if isinstance(verb_raw, str) else None
    if verb == "update":  # legacy review-schema synonym for merge
        verb = "merge"
    if verb not in RESOLUTION_VERBS:
        logger.warning(
            "reviewer: resolution verb %r is not actionable; downgrading to skip",
            verb_raw,
        )
        return ResolutionDecision(verb="skip", rationale=_UNRESOLVABLE_RATIONALE)
    if verb == "merge" and merged_text is None:
        logger.warning(
            "reviewer: merge resolution without merged_text; downgrading to skip"
        )
        return ResolutionDecision(
            verb="skip",
            rationale="merge resolution carried no merged_text; draft dropped",
        )
    if verb != "merge":
        merged_text = None
    return ResolutionDecision(
        verb=cast(ResolutionVerb, verb), rationale=rationale, merged_text=merged_text
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Strictly parse ``raw`` as a JSON object (no lenient parsing).

    Raises :class:`json.JSONDecodeError` for malformed JSON and
    :class:`ValueError` for JSON that is not an object.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("model output is JSON but not an object")
    return parsed


class ReviewerAgent:
    """Decides the :class:`~hermes.memory.draft.ReviewAction` for one draft.

    All dependencies are injectable (tests use deterministic fakes):

    - ``model(prompt) -> completion`` — the reviewer LLM; ``None`` wires the
      fork-base default lazily at first use (see module docstring).
    - ``conflict_detector`` — the pre-decision rule engine; defaults to a
      :class:`ConflictDetector` with the plan's 0.85 threshold.
    - ``search_fn(query) -> list[MemoryEntry]`` — fetches the existing
      candidates when :meth:`review` is called without ``existing``; the
      default uses :meth:`MemoryStore.search` READ-ONLY (see module
      docstring).

    :meth:`resolve_conflict` is the P8.T2 explicit resolution pass for
    detector-flagged contradictions (same injected model).
    """

    def __init__(
        self,
        model: Callable[[str], str] | None = None,
        conflict_detector: ConflictDetector | None = None,
        search_fn: Callable[[str], list[MemoryEntry]] | None = None,
    ) -> None:
        self._model = model
        self._conflict_detector = (
            conflict_detector if conflict_detector is not None else ConflictDetector()
        )
        self._search_fn = search_fn
        #: Lazily-built default store backing the default ``search_fn``.
        self._default_store: MemoryStore | None = None

    # ── dependency resolution (lazy defaults, extractor pattern) ──────────
    def _model_callable(self) -> Callable[[str], str]:
        """Resolve the injected model, wiring the fork-base default on first use."""
        if self._model is None:
            self._model = build_default_model()
        return self._model

    def _resolve_search_fn(self) -> Callable[[str], list[MemoryEntry]]:
        """Resolve the injected search, building the MemoryStore default lazily.

        The store import and construction happen at CALL time (like the
        extractor's default model wiring): importing this module never pulls
        the store stack, and the store is built once over
        ``hermes_home() / "memory-repo"`` and only ever searched (read-only).
        """
        if self._search_fn is None:
            if self._default_store is None:
                from hermes.memory.store import MemoryStore

                self._default_store = MemoryStore()
            store = self._default_store

            def _default_search(query: str) -> list[MemoryEntry]:
                results: list[MemoryEntry] = store.search(query, top_k=TOP_K_EXISTING)
                return results

            self._search_fn = _default_search
        return self._search_fn

    # ── prompt ────────────────────────────────────────────────────────────
    def _build_prompt(
        self, draft: DraftFact, candidates: Sequence[MemoryEntry], corrective: bool
    ) -> str:
        """Full prompt: review rules + draft and candidates as JSON (+ retry)."""
        payload = json.dumps(
            {
                "draft": {
                    "draft_id": draft.draft_id,
                    "text": draft.text,
                    "type": draft.type,
                    "confidence": draft.confidence,
                },
                "existing": [
                    {"id": entry.id, "type": entry.type, "text": entry.text}
                    for entry in candidates
                ],
            },
            ensure_ascii=True,
        )
        prompt = (
            f"{REVIEWER_SYSTEM_PROMPT}\n\n{_REVIEW_HEADER}\n{payload}\n\n{_REVIEW_FOOTER}"
        )
        if corrective:
            prompt += f"\n\n{REVIEWER_RETRY_SUFFIX}"
        return prompt

    # ── model call + validation ───────────────────────────────────────────
    def _call_model(
        self, draft: DraftFact, candidates: Sequence[MemoryEntry]
    ) -> dict[str, Any] | None:
        """Model decision with one corrective retry; ``None`` when unparseable."""
        model = self._model_callable()
        prompt = self._build_prompt(draft, candidates, corrective=False)
        try:
            return _parse_json_object(model(prompt))
        except (json.JSONDecodeError, ValueError):
            logger.warning("reviewer: model output was not a JSON object; retrying once")
        try:
            retry_prompt = self._build_prompt(draft, candidates, corrective=True)
            return _parse_json_object(model(retry_prompt))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "reviewer: model output malformed after corrective retry; "
                "falling back to the conservative skip action"
            )
            return None

    def _validated_decision(
        self, raw: dict[str, Any], candidates: Sequence[MemoryEntry]
    ) -> ReviewAction:
        """Schema-validate the parsed model output (rules constrain the model).

        See the module docstring for the exact validation contract: invalid
        actions and missing/unknown targets downgrade to ``skip``; ``merged_text``
        survives only on ``update``; ``add`` never carries a ``target_id``.
        """
        action = raw.get("action")
        # isinstance first: an unhashable JSON value (list/dict) must downgrade
        # to skip, never raise on the frozenset membership test.
        if not isinstance(action, str) or action not in REVIEW_ACTIONS:
            logger.warning("reviewer: downgrading invalid model action %r to skip", action)
            return ReviewAction(
                action="skip",
                target_id=None,
                rationale=f"model returned invalid action {action!r}; downgraded to skip",
            )
        rationale = str(raw.get("rationale", "") or "").strip()
        target_raw = raw.get("target_id")
        target_id = str(target_raw).strip() if target_raw is not None else None
        target_id = target_id or None
        merged_raw = raw.get("merged_text")
        merged_text = str(merged_raw).strip() if isinstance(merged_raw, str) else None
        merged_text = merged_text or None

        if action in {"update", "skip", "conflict"}:
            known_ids = {entry.id for entry in candidates}
            if target_id is None:
                logger.warning(
                    "reviewer: downgrading model action %r to skip (no target_id)", action
                )
                return ReviewAction(
                    action="skip",
                    target_id=None,
                    rationale=(
                        f"model returned {action!r} without a target_id; downgraded to skip"
                    ),
                )
            if target_id not in known_ids:
                logger.warning(
                    "reviewer: downgrading model action %r to skip (unknown target %r)",
                    action,
                    target_id,
                )
                return ReviewAction(
                    action="skip",
                    target_id=None,
                    rationale=(
                        f"model returned {action!r} for unknown target_id {target_id!r}; "
                        "downgraded to skip"
                    ),
                )
        if action != "update":
            merged_text = None
        if action == "add":
            target_id = None
        return ReviewAction(
            action=cast(ReviewActionLiteral, action),
            target_id=target_id,
            rationale=rationale,
            merged_text=merged_text,
        )

    # ── conflict resolution (plan P8.T2) ──────────────────────────────────
    def resolve_conflict(
        self, draft: DraftFact, conflicting_entry: MemoryEntry
    ) -> ResolutionDecision:
        """Explicit conflict-resolution pass (plan P8.T2).

        Called by the pipeline when the ConflictDetector flagged a
        contradiction (rules > model — see :func:`is_detector_flagged_conflict`).
        The conflicting entry is injected with its id/type/text and the model
        is asked ONCE how to resolve the conflict: ``merge`` (returns the
        merged body in ``merged_text``), ``replace`` (the draft supersedes
        the entry) or ``skip`` (drop the draft). Returns a validated
        :class:`ResolutionDecision` — never raises for unusable model output:
        malformed JSON, an unactionable verb, a merge without ``merged_text``,
        or a transport error from the injected model callable all yield the
        conservative ``skip`` (a failed resolution never writes anything).
        """
        payload = json.dumps(
            {
                "draft": {
                    "draft_id": f"{draft.draft_id}{_RESOLUTION_DRAFT_SUFFIX}",
                    "text": draft.text,
                    "type": draft.type,
                    "confidence": draft.confidence,
                },
                "conflicting_entry": {
                    "id": conflicting_entry.id,
                    "type": conflicting_entry.type,
                    "text": conflicting_entry.text,
                },
            },
            ensure_ascii=True,
        )
        prompt = (
            f"{RESOLUTION_SYSTEM_PROMPT}\n\n"
            f"{_RESOLUTION_HEADER}\n{payload}\n\n{_RESOLUTION_FOOTER}"
        )
        model = self._model_callable()
        try:
            raw = _parse_json_object(model(prompt))
        except Exception:
            logger.warning(
                "reviewer: conflict-resolution model output unusable; "
                "conservative skip (draft dropped)",
                exc_info=True,
            )
            return ResolutionDecision(
                verb="skip",
                rationale="conflict-resolution model output unusable; draft dropped",
            )
        decision = _validated_resolution(raw)
        logger.info(
            "reviewer: conflict resolution for draft %s against entry %s -> %s "
            "(rationale: %s)",
            draft.draft_id,
            conflicting_entry.id,
            decision.verb,
            decision.rationale,
        )
        return decision

    # ── public API ────────────────────────────────────────────────────────
    def review(
        self,
        draft: DraftFact,
        existing: Sequence[MemoryEntry] | None = None,
    ) -> ReviewAction:
        """Decide what happens to ``draft`` (plan P3.T3).

        ``existing`` carries the semantic top-5 matches for the draft text;
        when omitted the injected (or default) ``search_fn`` fetches them.
        At most :data:`TOP_K_EXISTING` candidates are considered. Returns a
        fully-validated :class:`ReviewAction` — never raises for malformed
        model output (conservative ``skip`` fallback); transport errors from
        the injected model callable propagate.
        """
        if existing is None:
            candidates = self._resolve_search_fn()(draft.text)[:TOP_K_EXISTING]
        else:
            candidates = list(existing)[:TOP_K_EXISTING]

        # Rule 1 (rules > model): the ConflictDetector decides conflicts first.
        conflict = self._conflict_detector.check_conflict(
            draft.text, [(entry.id, entry.text) for entry in candidates]
        )
        if conflict.is_conflict and conflict.kind == "contradiction":
            logger.info(
                "reviewer: conflict rule fired for draft %s against entry %s "
                "(similarity %.2f)",
                draft.draft_id,
                conflict.conflicting_entry_id,
                conflict.similarity,
            )
            return ReviewAction(
                action="conflict",
                target_id=conflict.conflicting_entry_id,
                rationale=(
                    f"contradicts existing entry {conflict.conflicting_entry_id} "
                    f"(similarity {conflict.similarity:.2f}); ConflictDetector rule "
                    f"precedes the model decision {DETECTOR_CONFLICT_MARKER}"
                ),
            )

        decision = self._call_model(draft, candidates)
        if decision is None:
            return ReviewAction(
                action="skip", target_id=None, rationale=_UNPARSEABLE_RATIONALE
            )
        action = self._validated_decision(decision, candidates)

        # Rule 2 (rules > model): the detector's duplicate verdict never lets
        # the model store the draft as new.
        if conflict.is_conflict and conflict.kind == "duplicate" and action.action == "add":
            logger.info(
                "reviewer: duplicate rule overrode model 'add' for draft %s "
                "against entry %s (similarity %.2f)",
                draft.draft_id,
                conflict.conflicting_entry_id,
                conflict.similarity,
            )
            return ReviewAction(
                action="skip",
                target_id=conflict.conflicting_entry_id,
                rationale=(
                    f"duplicate of existing entry {conflict.conflicting_entry_id} "
                    f"(similarity {conflict.similarity:.2f}); duplicate rule overrides "
                    "the model's add decision"
                ),
            )
        return action
