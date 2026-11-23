"""CBDCGate: one batched use/verify/ignore decision per turn (plan P4.T2).

Phase 4 pre-step: after BM25 retrieval (P4.T1) the gate classifies the TOP-3
candidates (plan: "only top-3 candidates are gated, not all retrieved") into
``use`` | ``verify`` | ``ignore`` — the Claim -> Belief -> Decision ->
Consumption chain — so the injector (P4.T3) can add only relevant facts to
the turn's context.

Single batched model call (check 3)
------------------------------------------------------------------------------
ALL top-3 candidates are classified in ONE prompt that asks for a JSON
decision LIST (``[{"candidate_id", "decision", "reason"}]``) — never one
call per candidate. The former "3 sequential calls" design (1-3s each) would
exceed the 5s hook budget on a slow main model; batching keeps the gate at
at most one model round-trip per turn (plan acceptance: "Gating uses <=1
batched main model call per turn"). Consequently there is NO corrective
retry: unparseable/failed model output is treated as a model failure and
degrades to the trust-only fallback below.

Budget enforcement is DETERMINISTIC post-processing — never the model's job
(mutation-tested Tier-1 logic per plan Section 4.7). Model decisions are
re-ordered by candidate retrieval score (trust-order: BM25 score descending,
id as deterministic tie-break) and then enforced:

- ``use`` consumes one claim; injection stops when ``budget_claims`` (3) is
  exhausted. Any further ``use`` decision is downgraded to ``ignore`` with
  reason ``"budget: claims budget exhausted"``.
- Every INJECTED text (``use`` AND ``verify``) counts against
  ``budget_chars`` (1200); overflow is downgraded to ``ignore`` with reason
  ``"budget: character budget exhausted"``. The chars cap protects the
  prompt, so every injected byte counts.

Verify semantics — injected with a needs-verification marker, plan-strict
budget (documented prominently)
------------------------------------------------------------------------------
A ``verify`` decision (e.g. medium-trust candidate the model finds
plausible but stale) IS returned in the decision list AND included in the
injection payload, flagged ``needs_verification=True``. Per the plan's
"Inject only 'use' items within budget", verify entries do NOT consume the
claims budget — only ``use`` items do. They DO count against the character
budget (their text is injected too). With the gate's top-3 pre-filter this
keeps every invariant checkable: injected count <= 3, use-count <=
``budget_claims``, injected chars <= ``budget_chars``.

Timeout / graceful degradation (plan Section 5.5)
------------------------------------------------------------------------------
The model call runs on a daemon worker thread bounded by
``timeout_seconds`` (default 3.5 — the plan's hard gate timeout). On
timeout, a model exception, or output that never parses to a decision list,
the model result is discarded and the gate FALLS BACK to trust-score-only
gating with ``fallback_used=True``: among the gated top-3, candidates with
``trust_score > 0.7`` are marked ``use`` (up to the same budgets) and the
rest are ignored — the pre-step hook never blocks a turn on a slow model
(plan 5.5: "fallback to trust-score-only gating, no LLM").

Output shape (plan P4.T2 acceptance): each :class:`GatedEntry` carries
``{decision, reason, candidate_id}`` plus the candidate text and the
needs-verification marker; :class:`GateResult.injected` is the exact
payload the injector should use (no ``ignore`` entries, budgets already
enforced).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from hermes.memory.extractor import build_default_model
from hermes.memory.retrieval import RetrievedEntry

logger = logging.getLogger(__name__)

__all__ = ["CBDCGate", "GatedEntry", "GateResult"]

#: Gate decision literal (plan Phase 4: use/verify/ignore).
GateDecision = Literal["use", "verify", "ignore"]

#: Runtime set of valid gate decisions, derived from the literal.
DECISIONS: frozenset[str] = frozenset(get_args(GateDecision))

#: Plan Section 2.5: "retrieve 20, gate only top 3" — the gate's own cap.
GATE_TOP_K: int = 3

#: Plan Section 5.5 fallback trust bar: inject only entries above it.
FALLBACK_TRUST_THRESHOLD: float = 0.7

#: System prompt: one batched decision LIST for all candidates (check 3).
GATE_SYSTEM_PROMPT = """\
You are the hermes-v2 CBDC gate (plan Phase 4). You decide which stored \
memory candidates may be injected as context for the CURRENT turn.

For EACH candidate decide exactly one:
- "use": relevant to the current turn; safe to inject as context.
- "verify": plausibly relevant but stale or uncertain - include flagged for \
verification (e.g. a medium-trust fact that needs re-confirmation).
- "ignore": not relevant to the current turn, contradicted by newer \
knowledge, or not worth the injection budget.

Return ONLY a JSON array (no prose, no markdown fences), one object per \
candidate, covering EVERY candidate id:
[{"candidate_id": "<id>", "decision": "<use|verify|ignore>", "reason": \
"<one short sentence>"}]\
"""

#: Prompt framing around the gating-request JSON object.
_REQUEST_HEADER = "Gating request (JSON object):"
_REQUEST_FOOTER = "Respond with ONLY the JSON decision array."

#: Markdown code-fence stripper (models often wrap JSON in ``` fences).
_LEADING_FENCE_RE = re.compile(r"^```[^\n]*\n?")

#: Fallback injected when the model gives no reason for a candidate.
_NO_REASON = "no reason given by model"


@dataclass(frozen=True)
class GatedEntry:
    """One gated candidate (plan P4.T2 output shape ``{decision, reason,
    candidate_id}`` plus the payload fields the injector needs).

    ``text`` is the candidate's entry text (injection payload);
    ``needs_verification`` is the injection marker for ``verify`` entries.
    """

    candidate_id: str
    decision: GateDecision
    reason: str
    text: str
    needs_verification: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (trace payload friendly)."""
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "reason": self.reason,
            "text": self.text,
            "needs_verification": self.needs_verification,
        }


@dataclass
class GateResult:
    """Outcome of one :meth:`CBDCGate.gate` call (plan P4.T2).

    ``decisions`` covers exactly the gated top-3 candidates (one entry per
    candidate); ``fallback_used`` reports the plan 5.5 degradation;
    ``injected`` is the derived injection payload — every decision that is
    not ``ignore`` (``use`` + ``verify``-flagged), budgets already enforced.
    """

    decisions: list[GatedEntry]
    fallback_used: bool
    elapsed_ms: float

    @property
    def injected(self) -> list[GatedEntry]:
        """The injection payload: all non-``ignore`` decisions, in gate order."""
        return [entry for entry in self.decisions if entry.decision != "ignore"]


class _GateOutputError(ValueError):
    """Internal: the model output never yielded a valid decision list."""


def _strip_markdown_fence(raw: str) -> str:
    """Strip a wrapping ``` fence (with optional language tag) from ``raw``."""
    text = raw.strip()
    if text.startswith("```"):
        text = _LEADING_FENCE_RE.sub("", text, count=1)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_decision_list(
    raw: str, gated_ids: frozenset[str]
) -> dict[str, tuple[GateDecision, str]]:
    """Strictly parse the model output into ``{candidate_id: (decision, reason)}``.

    Only decisions for the gated candidate ids survive; unknown ids,
    duplicate candidates (first wins), invalid decision literals and
    non-object items are dropped with a logged warning. Anything that is
    not a JSON array raises :class:`_GateOutputError` (-> trust-only
    fallback; the gate never retries — one batched call per turn).
    """
    try:
        parsed = json.loads(_strip_markdown_fence(raw))
    except json.JSONDecodeError as exc:
        raise _GateOutputError(f"model output is not JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise _GateOutputError("model output is JSON but not a decision array")
    model_decisions: dict[str, tuple[GateDecision, str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            logger.warning("cbdc gate: dropping non-object decision item %r", item)
            continue
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in gated_ids:
            logger.warning("cbdc gate: dropping decision for unknown candidate %r", candidate_id)
            continue
        if candidate_id in model_decisions:
            continue  # first decision per candidate wins
        decision = item.get("decision")
        if not isinstance(decision, str) or decision not in DECISIONS:
            logger.warning(
                "cbdc gate: dropping invalid decision %r for candidate %s", decision, candidate_id
            )
            continue
        reason = item.get("reason")
        reason_text = reason.strip() if isinstance(reason, str) else ""
        model_decisions[candidate_id] = (cast_gate_decision(decision), reason_text or _NO_REASON)
    return model_decisions


def cast_gate_decision(decision: str) -> GateDecision:
    """Narrow a ``DECISIONS``-validated string to the ``GateDecision`` literal."""
    return decision  # type: ignore[return-value]


def _trust_score_of(candidate: RetrievedEntry) -> float:
    """The candidate entry's trust score (fallback signal, plan Section 2.4)."""
    return candidate.entry.trust_score


class CBDCGate:
    """Batched CBDC gating over the top-3 retrieved candidates (plan P4.T2).

    ``gate_model`` is the injected main-model callable (``prompt -> raw
    completion``); ``None`` wires the fork-base default lazily at first use
    (same pattern as :class:`hermes.memory.extractor.ExtractorAgent` — tests
    always inject fakes). ``budget_claims`` / ``budget_chars`` /
    ``timeout_seconds`` mirror the plan Section 2.5 ``gating`` defaults.
    """

    def __init__(
        self,
        gate_model: Callable[[str], str] | None = None,
        budget_claims: int = 3,
        budget_chars: int = 1200,
        timeout_seconds: float = 3.5,
    ) -> None:
        if budget_claims < 0:
            raise ValueError(f"budget_claims must be >= 0, got {budget_claims}")
        if budget_chars < 0:
            raise ValueError(f"budget_chars must be >= 0, got {budget_chars}")
        if timeout_seconds < 0:
            raise ValueError(f"timeout_seconds must be >= 0, got {timeout_seconds}")
        self._gate_model = gate_model
        self._budget_claims = budget_claims
        self._budget_chars = budget_chars
        self._timeout_seconds = timeout_seconds

    # ── model resolution (lazy default, extractor pattern) ────────────────
    def _model_callable(self) -> Callable[[str], str]:
        """Resolve the injected model, wiring the fork-base default on first use."""
        if self._gate_model is None:
            self._gate_model = build_default_model()
        return self._gate_model

    # ── top-3 pre-filter ───────────────────────────────────────────────────
    def _gated_candidates(self, candidates: Sequence[RetrievedEntry]) -> list[RetrievedEntry]:
        """Top-3 candidates by retrieval score (BM25 pre-filter, plan Phase 4).

        Non-finite (NaN) scores degrade to 0.0 so a poisoned score can never
        corrupt the ranking; ties break deterministically by entry id.
        """
        def _sort_key(candidate: RetrievedEntry) -> tuple[float, str]:
            score = candidate.score if candidate.score == candidate.score else 0.0
            return (-score, candidate.entry.id)

        return sorted(candidates, key=_sort_key)[:GATE_TOP_K]

    # ── prompt ─────────────────────────────────────────────────────────────
    def _build_prompt(self, context: str, turn_text: str, gated: list[RetrievedEntry]) -> str:
        """One batched prompt: turn + ALL candidates (id/trust/text) as JSON."""
        payload = json.dumps(
            {
                "context": context,
                "turn_text": turn_text,
                "candidates": [
                    {
                        "candidate_id": candidate.entry.id,
                        "trust_score": round(_trust_score_of(candidate), 4),
                        "text": candidate.entry.text,
                    }
                    for candidate in gated
                ],
            },
            ensure_ascii=True,
        )
        return (
            f"{GATE_SYSTEM_PROMPT}\n\n{_REQUEST_HEADER}\n{payload}\n\n{_REQUEST_FOOTER}"
        )

    # ── batched model call with hard timeout (plan Section 5.5) ───────────
    def _call_model(self, prompt: str) -> str | None:
        """Run the model on a bounded daemon thread; ``None`` on any failure.

        The whole model round-trip (ONE call — no retries, see the module
        docstring) must finish within ``timeout_seconds``; on timeout the
        worker is abandoned (daemon thread, result discarded) and ``None``
        signals the trust-only fallback.
        """
        model = self._model_callable()
        outcome: dict[str, Any] = {}

        def _worker() -> None:
            try:
                outcome["raw"] = model(prompt)
            except Exception as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=_worker, name="cbdc-gate-model", daemon=True)
        worker.start()
        worker.join(self._timeout_seconds)
        if worker.is_alive():
            logger.warning(
                "cbdc gate: model call exceeded the %.2fs timeout; discarding the result "
                "and falling back to trust-score-only gating (plan 5.5)",
                self._timeout_seconds,
            )
            return None
        if "error" in outcome:
            logger.warning(
                "cbdc gate: model call failed (%r); falling back to trust-score-only "
                "gating (plan 5.5)",
                outcome["error"],
            )
            return None
        raw = outcome.get("raw")
        if not isinstance(raw, str):
            logger.warning(
                "cbdc gate: model returned no string output; falling back to "
                "trust-score-only gating (plan 5.5)"
            )
            return None
        return raw

    # ── deterministic budget enforcement (mutation-tested Tier-1 logic) ───
    def _enforce_budget(
        self,
        gated: list[RetrievedEntry],
        preliminary: list[tuple[GateDecision, str]],
    ) -> list[GatedEntry]:
        """Apply claims/chars budgets to per-candidate ``(decision, reason)``.

        Iterates in trust-order (the candidate ranking order, NOT the
        model's output order): ``use`` consumes a claim, every injected text
        (use or verify) consumes characters; overflow downgrades to
        ``ignore`` with a ``budget:`` reason. See the module docstring for
        the verify/claims rule.
        """
        enforced: list[GatedEntry] = []
        claims_used = 0
        chars_used = 0
        for candidate, (decision, reason) in zip(gated, preliminary, strict=True):
            candidate_id = candidate.entry.id
            text = candidate.entry.text
            if decision == "use" and claims_used >= self._budget_claims:
                logger.info(
                    "cbdc gate: candidate %s downgraded use->ignore (budget: claims "
                    "budget exhausted)",
                    candidate_id,
                )
                decision, reason = "ignore", "budget: claims budget exhausted"
            if decision != "ignore" and chars_used + len(text) > self._budget_chars:
                logger.info(
                    "cbdc gate: candidate %s downgraded %s->ignore (budget: character "
                    "budget exhausted)",
                    candidate_id,
                    decision,
                )
                decision, reason = "ignore", "budget: character budget exhausted"
            if decision == "use":
                claims_used += 1
            if decision != "ignore":
                chars_used += len(text)
            enforced.append(
                GatedEntry(
                    candidate_id=candidate_id,
                    decision=decision,
                    reason=reason,
                    text=text,
                    needs_verification=(decision == "verify"),
                )
            )
        return enforced

    # ── plan 5.5 trust-only fallback ──────────────────────────────────────
    def _trust_only_preliminary(
        self, gated: list[RetrievedEntry]
    ) -> list[tuple[GateDecision, str]]:
        """Trust-score-only decisions (no LLM): trust > 0.7 -> use, else ignore."""
        preliminary: list[tuple[GateDecision, str]] = []
        for candidate in gated:
            trust = _trust_score_of(candidate)
            if trust > FALLBACK_TRUST_THRESHOLD:
                preliminary.append((
                    "use",
                    f"trust-only fallback: trust {trust:.2f} > {FALLBACK_TRUST_THRESHOLD:.2f} "
                    "(model unavailable)",
                ))
            else:
                preliminary.append((
                    "ignore",
                    f"trust-only fallback: trust {trust:.2f} <= "
                    f"{FALLBACK_TRUST_THRESHOLD:.2f} (model unavailable)",
                ))
        return preliminary

    # ── public API (plan Phase 4) ─────────────────────────────────────────
    def gate(
        self,
        candidates: Sequence[RetrievedEntry],
        context: str = "global",
        turn_text: str = "",
    ) -> GateResult:
        """Gate the top-3 ``candidates`` for the current turn (plan P4.T2).

        One batched model call classifies all gated candidates (JSON
        decision list); the deterministic budget pass then enforces
        ``budget_claims`` / ``budget_chars``. On timeout, model failure or
        unparseable output the trust-only fallback runs instead
        (``fallback_used=True``). ``context`` and ``turn_text`` feed the
        model prompt. With no candidates the gate short-circuits without a
        model call (0 calls, nothing injected).
        """
        start = time.perf_counter()
        gated = self._gated_candidates(candidates)
        if not gated:
            return GateResult(decisions=[], fallback_used=False, elapsed_ms=0.0)

        fallback_used = False
        model_decisions: dict[str, tuple[GateDecision, str]] = {}
        raw = self._call_model(self._build_prompt(context, turn_text, gated))
        if raw is None:
            fallback_used = True
        else:
            try:
                model_decisions = _parse_decision_list(raw, frozenset(
                    candidate.entry.id for candidate in gated
                ))
            except _GateOutputError as exc:
                logger.warning(
                    "cbdc gate: unparseable model output (%s); falling back to "
                    "trust-score-only gating (plan 5.5)",
                    exc,
                )
                fallback_used = True

        if fallback_used:
            preliminary = self._trust_only_preliminary(gated)
        else:
            preliminary = [
                model_decisions.get(
                    candidate.entry.id,
                    ("ignore", "no model decision for this candidate (conservative default)"),
                )
                for candidate in gated
            ]
        decisions = self._enforce_budget(gated, preliminary)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GateResult(
            decisions=decisions,
            fallback_used=fallback_used,
            elapsed_ms=elapsed_ms,
        )
