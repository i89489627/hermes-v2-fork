"""ContextInjector: stable/dynamic context split for proactive injection (plan P4.T3).

Phase 4 pre-step composition: after BM25 retrieval (P4.T1) and CBDC gating
(P4.T2), the injector composes the ``additionalContext`` payload the
UserPromptSubmit hook (P4.T4) returns. The split follows the plan's prefix
cache preservation contract (plan Phase 4 acceptance: "stable context changes
<=1 time per session", Phase 7 trade-off note):

- STABLE — the global and active-context ``MEMORY.md`` indexes, loaded ONCE
  per session by :meth:`ContextInjector.load_stable_context` (via
  :class:`hermes.memory.context.ContextManager` detection + snapshot). The
  payload is then handed back into every per-turn call unchanged: the
  dynamic flow NEVER re-reads the stable documents, so the stable block is
  byte-identical across all turns of a session and the model prompt prefix
  stays cacheable. Reloading happens only at the next session boundary
  (e.g. after a ``/context switch``).

- DYNAMIC — per-turn retrieved facts, gated use/verify/ignore with budgets
  already enforced by the gate: ``use`` -> ``claims``, ``verify`` ->
  ``needs_verification`` (flagged), ``ignore`` -> dropped. Budget accounting
  mirrors the gate's deterministic enforcement: claims count the ``use``
  entries, chars count every injected text (use + verify).

- HOT MEMORY (plan P21.T2, issue #26568) — entries added DURING the current
  session (:func:`collect_hot_entries`), passed to
  :meth:`ContextInjector.build_dynamic_context` as ``hot_entries``. This is
  the injection half of the #26568 fix contract ("inject hot memory before
  each tool call"): hot entries used to be ignored because they had to
  survive the BM25 retrieval pre-filter mid-session. They now BYPASS
  retrieval (they are hot by definition — no query needed) but still pass
  through the SAME single batched CBDC gate call (the P4.T2 "one batched
  call per turn" budget semantics), where they take the gate's top-3 slots
  FIRST (priority order: newest entry first) and the shared claims/chars
  budgets apply. They are deduplicated out of the retrieval results so one
  entry never enters the gate twice. Injected hot entries appear in the
  dynamic block under ``"hot_memory"`` (rendered by
  :func:`format_additional_context` ahead of the retrieved claims), so the
  payload the pre-step hook hands the model before each tool call carries
  the session's fresh facts.

- SEGMENT AWARENESS (plan P7.T1) — when a :class:`~hermes.memory.hot_reload.
  MemoryHotReloader` is wired (:func:`wire_hot_reload` or the
  ``hot_reloader`` argument), the payload carries a top-level
  ``"invalidated_segments"`` list: the segment ids the reloader has pending.
  The agent harness consumes it to RE-INJECT ONLY those segments of the
  segmented system prompt (plan Phase 7: one segment per memory entry) —
  every other segment keeps its rendered bytes, preserving the prefix cache
  while mid-session memory updates stay visible.

Hard budget: full injection (retrieve + gate + compose) targets <2s typical,
5s hard (the hook timeout, plan Phase 4 acceptance) — the gate itself is
bounded at 3.5s with the trust-only fallback (plan Section 5.5), so the
injector can only add string assembly on top.

Output shape (:data:`InjectedContext`, JSON-safe for the hook):
``{"stable": <block> | None, "dynamic": {...}, "turn": "<sha256 of turn>"}``.
:func:`format_additional_context` renders it as the markdown-ish prompt
block and returns "" when there is nothing to inject, so the hook never
pollutes the prompt with an empty wrapper (plan P4.T4 acceptance).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeAlias

from hermes.memory.context import ContextManager
from hermes.memory.entry import MemoryEntry
from hermes.memory.gating import CBDCGate
from hermes.memory.hot_reload import MemoryHotReloader
from hermes.memory.retrieval import RetrievalEngine, RetrievedEntry
from hermes.memory.store import MemoryStore
from hermes.memory.types import to_utc

logger = logging.getLogger(__name__)

__all__ = [
    "BM25_PREFILTER_TOP_K",
    "ContextInjector",
    "HOT_PRIORITY_SCORE_BASE",
    "InjectedContext",
    "StableContextPayload",
    "collect_hot_entries",
    "wire_hot_reload",
]

#: Plan Section 2.5 ``gating.bm25_prefilter_top_k`` — the BM25 pre-filter
#: width feeding the gate ("retrieve 20, gate only top 3").
BM25_PREFILTER_TOP_K: int = 20

#: Retrieval score assigned to hot entries (plan P21.T2, issue #26568): the
#: gate ranks candidates by score, so hot entries get scores near this base
#: (newest first, one point apart) that dominate any realistic BM25 score —
#: they take the gate's top-3 slots first, per the #26568 fix contract
#: ("inject hot memory before each tool call"). The model still classifies
#: them: an irrelevant hot entry is gated ``ignore`` and consumes no budget.
HOT_PRIORITY_SCORE_BASE: float = 1_000_000.0

#: additionalContext payload for the UserPromptSubmit hook (plan P4.T3/P4.T4):
#: ``{"stable": dict | None, "dynamic": dict, "turn": str}``.
InjectedContext: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class StableContextPayload:
    """The session-stable context block (plan P4.T3).

    ``global_index`` / ``context_index`` carry the global and active-context
    ``MEMORY.md`` documents (``None`` when the file does not exist yet);
    ``loaded_at`` stamps the session start. The payload is frozen and handed
    back unchanged into every :meth:`ContextInjector.build_dynamic_context`
    call of the session — the stable block bytes must be identical across
    turns (prefix cache preservation, plan Phase 4 / Phase 7).
    """

    context_name: str
    global_index: str | None
    context_index: str | None
    loaded_at: datetime

    def to_block(self) -> dict[str, str]:
        """JSON-safe summary block; deterministic for a given payload.

        ``None`` indexes degrade to ``""`` so the block stays JSON-safe, and
        every value derives from the payload itself (never re-read from
        disk) — that is what makes the block byte-identical across the
        session's turns.
        """
        return {
            "context": self.context_name,
            "global_index": self.global_index or "",
            "context_index": self.context_index or "",
            "loaded_at": self.loaded_at.isoformat(),
        }


class ContextInjector:
    """Stable/dynamic context injector over retrieval + gating (plan P4.T3).

    All collaborators are injected. ``context_manager`` defaults to a
    :class:`~hermes.memory.context.ContextManager` rooted at the
    config-derived default path (``hermes_home() / "memory-repo"``).
    ``top_k`` is the BM25 pre-filter width (plan Section 2.5
    ``gating.bm25_prefilter_top_k``, default 20).
    """

    def __init__(
        self,
        retrieval: RetrievalEngine,
        gate: CBDCGate,
        context_manager: ContextManager | None = None,
        *,
        top_k: int = BM25_PREFILTER_TOP_K,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._retrieval = retrieval
        self._gate = gate
        self._context_manager = (
            context_manager if context_manager is not None else ContextManager()
        )
        self._top_k = top_k
        #: Wired by :func:`wire_hot_reload` (plan P7.T1); used as the default
        #: ``hot_reloader`` of :meth:`build_dynamic_context` when the caller
        #: does not pass one explicitly.
        self._default_hot_reloader: MemoryHotReloader | None = None

    # ── stable block (session start, loaded ONCE — plan P4.T3) ────────────
    def load_stable_context(self, session_metadata: dict[str, Any]) -> StableContextPayload:
        """Load the stable block ONCE at session start (plan P4.T3).

        Context detection delegates to
        :meth:`ContextManager.detect_context` (precedence: ``explicit`` >
        ``gateway_chat_id`` > ``cwd`` > ``"global"``); the snapshot reads the
        global and context ``MEMORY.md`` documents from disk at this moment.

        SESSION CONTRACT (prefix cache preservation): call this exactly once
        per session and pass the returned payload to every
        :meth:`build_dynamic_context` call of that session. The injector
        never caches internally and the dynamic flow never re-reads the
        stable documents — reloading happens only at the next session
        boundary. Detection/snapshot errors propagate: an ``explicit``
        context that does not exist must stay visible (the ContextManager
        contract), while rule-based lookups already degrade to global there.
        """
        context_name = self._context_manager.detect_context(session_metadata)
        snapshot = self._context_manager.stable_context_snapshot(context_name)
        return StableContextPayload(
            context_name=snapshot.context,
            global_index=snapshot.global_memory_md,
            context_index=snapshot.context_memory_md,
            loaded_at=snapshot.loaded_at,
        )

    # ── hot memory (plan P21.T2, issue #26568) ─────────────────────────────
    def _hot_candidates(self, hot_entries: list[MemoryEntry] | None) -> list[RetrievedEntry]:
        """Wrap session-added entries as priority gate candidates.

        Hot entries bypass the BM25 retrieve (they are hot by definition) and
        enter the ONE batched gate call with dominant scores so they take the
        gate's top-3 slots first, newest entry first (deterministic id
        tie-break). Duplicate ids keep their first occurrence.
        """
        if not hot_entries:
            return []
        deduped: dict[str, MemoryEntry] = {}
        for entry in hot_entries:
            deduped.setdefault(entry.id, entry)
        ordered = sorted(
            deduped.values(), key=lambda entry: (entry.created_at, entry.id), reverse=True
        )
        return [
            RetrievedEntry(entry=entry, score=HOT_PRIORITY_SCORE_BASE - position)
            for position, entry in enumerate(ordered)
        ]

    # ── dynamic block (per turn — plan P4.T3; hot memory P21.T2) ──────────
    def build_dynamic_context(
        self,
        turn_text: str,
        context_name: str,
        stable: StableContextPayload | None = None,
        *,
        hot_entries: list[MemoryEntry] | None = None,
        hot_reloader: MemoryHotReloader | None = None,
    ) -> InjectedContext:
        """Retrieve + gate + compose the per-turn injection (plan P4.T3).

        BM25 pre-filter over ``context_name`` (+ global) with ``top_k``
        candidates, then ONE batched CBDC gate call (use/verify/ignore with
        budgets enforced inside the gate), then composition of the
        additionalContext dict. ``stable`` (the session-start payload from
        :meth:`load_stable_context`) is passed through UNCHANGED — the
        returned ``"stable"`` block is a summary derived from the payload,
        so the same payload yields byte-identical stable blocks on every
        turn (prefix cache contract); ``None`` yields a ``None`` stable
        block (session start not wired).

        ``hot_entries`` (plan P21.T2, issue #26568) are the entries added
        during the current session (:func:`collect_hot_entries`). They skip
        the BM25 retrieve but join the SAME single batched gate call with
        top-priority scores (see :data:`HOT_PRIORITY_SCORE_BASE`), so the
        shared claims/chars budgets apply and the per-turn model-call count
        stays at one. Injected hot entries land in the dynamic block under
        ``"hot_memory"`` — the payload the pre-step hook returns is what the
        harness injects before each tool call, which is the #26568 fix
        contract. A hot entry decided ``verify`` stays inside
        ``"hot_memory"`` flagged ``needs_verification=True`` (it does not
        consume the claims budget); ``use``-decided hot entries count toward
        ``budget_used.claims`` together with the retrieved claims.

        ``hot_reloader`` (plan P7.T1) is the segment-aware hot reloader; when
        given (or wired via :func:`wire_hot_reload`) the payload gains a
        top-level ``"invalidated_segments"`` list with the reloader's pending
        segment ids. The agent harness consumes it to re-inject ONLY those
        segments of the segmented system prompt — every other segment keeps
        its rendered bytes (prefix cache preserved, plan Phase 7). The
        pending set is observed, not consumed; the harness clears it via
        :meth:`MemoryHotReloader.clear_invalidations` after re-injecting.

        With an empty/whitespace turn the retrieval short-circuits to no
        candidates; hot entries (when given) are still gated — they need no
        query. With neither retrieved nor hot candidates the gate
        short-circuits without a model call and the output degrades to an
        empty dynamic block. An unsafe ``context_name`` raises
        :class:`ValueError` from the retrieval engine (the pre-step hook owns
        graceful handling, P4.T4).
        """
        retrieved = self._retrieval.retrieve(
            turn_text, top_k=self._top_k, context=context_name
        )
        hot_candidates = self._hot_candidates(hot_entries)
        hot_ids = frozenset(candidate.entry.id for candidate in hot_candidates)
        # dedupe: an entry collected as hot must never enter the gate twice
        candidates = [
            candidate
            for candidate in retrieved
            if candidate.entry.id not in hot_ids
        ]
        candidates = [*hot_candidates, *candidates]
        gate_result = self._gate.gate(candidates, context=context_name, turn_text=turn_text)
        trust_by_id = {candidate.entry.id: candidate.entry.trust_score for candidate in candidates}
        claims: list[dict[str, Any]] = []
        needs_verification: list[dict[str, Any]] = []
        hot_memory: list[dict[str, Any]] = []
        chars_used = 0
        for entry in gate_result.injected:
            item: dict[str, Any] = {
                "candidate_id": entry.candidate_id,
                "text": entry.text,
                "trust": trust_by_id.get(entry.candidate_id, 0.0),
                "reason": entry.reason,
            }
            if entry.candidate_id in hot_ids:
                # hot: injected before tool calls (#26568); verify stays here, flagged
                hot_memory.append({**item, "needs_verification": entry.decision == "verify"})
            elif entry.decision == "verify":
                needs_verification.append(item)
            else:  # "use" — the gate result never carries "ignore" entries
                claims.append(item)
            chars_used += len(entry.text)
        hot_use_count = sum(1 for item in hot_memory if not item["needs_verification"])
        logger.info(
            "injector: dynamic context composed for context %r "
            "(claims=%d verify=%d hot=%d chars=%d fallback=%s)",
            context_name,
            len(claims),
            len(needs_verification),
            len(hot_memory),
            chars_used,
            gate_result.fallback_used,
        )
        injected: InjectedContext = {
            "stable": stable.to_block() if stable is not None else None,
            "dynamic": {
                # hot first: injected before the other facts / tool calls (#26568)
                "hot_memory": hot_memory,
                "claims": claims,
                "needs_verification": needs_verification,
                "budget_used": {
                    "claims": len(claims) + hot_use_count,
                    "chars": chars_used,
                },
                "fallback_used": gate_result.fallback_used,
            },
            "turn": hashlib.sha256(turn_text.encode("utf-8")).hexdigest(),
        }
        reloader = hot_reloader if hot_reloader is not None else self._default_hot_reloader
        if reloader is not None:
            # plan P7.T1: the harness re-injects ONLY these prompt segments
            injected["invalidated_segments"] = reloader.pending_invalidations()
        return injected


def collect_hot_entries(store: MemoryStore, session_started_at: datetime) -> list[MemoryEntry]:
    """Entries added during the current session (plan P21.T2, issue #26568).

    Reads :meth:`MemoryStore.list` (read-only; spans all contexts, ordered by
    creation time then id) and keeps the entries whose ``created_at`` is at
    or after ``session_started_at`` — the "hot memory" the injector must
    inject before each tool call. ``session_started_at`` must be
    timezone-aware (a naive datetime raises :class:`ValueError`); it is
    normalized to UTC before comparing, so session clocks in other
    timezones compare correctly against the store's UTC stamps.
    """
    started = to_utc(session_started_at, "session_started_at")
    return [entry for entry in store.list() if entry.created_at >= started]


def wire_hot_reload(injector: ContextInjector, store: MemoryStore) -> MemoryHotReloader:
    """Wire a default :class:`~hermes.memory.hot_reload.MemoryHotReloader` (P7.T1).

    Builds the reloader over ``store`` with the default fixed segments
    (stable profile + memory index; one segment per memory entry) and
    registers it on ``injector`` as the default: subsequent
    :meth:`ContextInjector.build_dynamic_context` calls without an explicit
    ``hot_reloader`` surface the reloader's ``pending_invalidations`` as the
    payload's top-level ``"invalidated_segments"`` for the agent harness.
    Returns the reloader so callers can drive
    :meth:`~hermes.memory.hot_reload.MemoryHotReloader.invalidate` /
    ``render_prompt`` directly. The store is only read (invalidation is
    caller-driven; the memory-tool wiring is plan P7.T2).
    """
    reloader = MemoryHotReloader(store)
    injector._default_hot_reloader = reloader
    return reloader


def format_additional_context(injected: InjectedContext) -> str:
    """Render the injected context as the markdown-ish prompt block (plan P4.T3).

    Deterministic given ``injected``: stable section (active context + the
    two index documents), then the hot-memory entries (plan P21.T2 —
    injected before the other facts, issue #26568), then the claims, then
    the verification entries — all in gate (trust) order. Returns ``""``
    when there is nothing to inject (no stable block, no hot entries, no
    claims, no verification entries) so the prompt is not polluted with an
    empty wrapper (plan P4.T4: "no injection when no relevant facts").
    """
    sections: list[str] = []

    stable = injected.get("stable")
    if isinstance(stable, dict) and stable:
        lines = ["## Stable session context"]
        context_name = stable.get("context")
        if context_name:
            lines.append(f"active context: {context_name}")
        global_index = stable.get("global_index")
        if global_index:
            lines.append("### Global memory index")
            lines.append(str(global_index).strip())
        context_index = stable.get("context_index")
        if context_index:
            lines.append("### Context memory index")
            lines.append(str(context_index).strip())
        sections.append("\n".join(lines))

    claims: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    hot_items: list[dict[str, Any]] = []
    dynamic = injected.get("dynamic")
    if isinstance(dynamic, dict):
        raw_claims = dynamic.get("claims")
        if isinstance(raw_claims, list):
            claims = [item for item in raw_claims if isinstance(item, dict)]
        raw_verification = dynamic.get("needs_verification")
        if isinstance(raw_verification, list):
            verification = [item for item in raw_verification if isinstance(item, dict)]
        raw_hot = dynamic.get("hot_memory")
        if isinstance(raw_hot, list):
            hot_items = [item for item in raw_hot if isinstance(item, dict)]

    if hot_items:
        # hot memory first among the dynamic facts: injected before tool
        # calls (plan P21.T2, issue #26568 fix contract)
        lines = ["## Hot memory (this session)"]
        for item in hot_items:
            marker = " [needs verification]" if item.get("needs_verification") else ""
            lines.append(
                f"- {item.get('text', '')} (trust {item.get('trust', 0.0):.2f}; "
                f"{item.get('reason', '')}){marker}"
            )
        sections.append("\n".join(lines))

    if claims:
        lines = ["## Relevant memory facts"]
        for item in claims:
            lines.append(
                f"- {item.get('text', '')} (trust {item.get('trust', 0.0):.2f}; "
                f"{item.get('reason', '')})"
            )
        sections.append("\n".join(lines))

    if verification:
        lines = ["## Facts needing verification"]
        for item in verification:
            lines.append(
                f"- {item.get('text', '')} (trust {item.get('trust', 0.0):.2f}; "
                f"{item.get('reason', '')})"
            )
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return "\n".join(["[hermes-v2 injected memory context]", *sections])
