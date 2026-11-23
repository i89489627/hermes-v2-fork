"""ExtractorAgent — turn to draft facts (plan P3.T2).

Single-model architecture (plan r3): the LLM is injected as a
``Callable[[str], str]`` — the callable receives the full prompt and returns
the raw completion — so tests always run against fakes and the extractor is
never called live. The default wiring (:func:`build_default_model`) reaches
the fork-base LLM API lazily at CALL time; importing this module never
imports the agent stack and never touches the network.

Resilience contract (plan acceptance: "produces valid JSON for 95%+ of
turns"): malformed model output triggers exactly one corrective retry
("return ONLY a JSON array"); output that is still malformed is dropped with
a logged warning and ``extract`` returns ``[]`` — a malformed turn must never
raise or break the Stop-hook pipeline (P3.T4 owns the retry/timeout policy
for transport errors, which propagate from the injected model callable).

Per-item validation keeps only schema-valid drafts (plan P3.T2 schema
``[{type, text, proposed_source, confidence, context_hint}]``): items with an
unknown entry type or empty text are dropped with a warning (never silently
coerced to another type), ``confidence`` is clamped into [0, 1] (non-finite
values degrade to 0.5), an unparsable ``proposed_source`` degrades to
``agent-inferred`` — provenance is never over-claimed — and the caller's
``context_hint`` always wins over the model's echo of it.

Multi-turn context (plan P3.T6): ``extract`` accepts ``context_turns`` —
recent turns, oldest first, as :class:`~hermes.memory.context_window.Turn`
objects or plain dicts (``{"text": ..., "turn_id": ..., "confirmed_facts":
[...]}``). When provided, the prompt gains a "RECENT TURNS (oldest first)"
block with reference-resolution rules and a "CONFIRMED FACTS" block; drafts
whose normalized text restates a confirmed fact are dropped post-hoc
(rule-based and deterministic — the model is instructed not to re-extract
them, but the rule gets the final word), and each draft's
``source_turn_ids`` is populated from the model's per-item proposal
(validated strings; ``[]`` when absent). Malformed context items are
dropped with a warning — context assembly must never break extraction.
Without ``context_turns`` the prompt is byte-identical to the P3.T2 format.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from hermes.memory.context_window import Turn
from hermes.memory.draft import DraftFact, normalize_fact_text
from hermes.memory.types import ENTRY_TYPES, SOURCE_WEIGHTS, EntryType, VerificationSource

logger = logging.getLogger(__name__)

#: System prompt: extraction rules per plan Phase 3 — the four entry types,
#: what must NOT be extracted, and the strict JSON output contract.
EXTRACTOR_SYSTEM_PROMPT = """\
You are the hermes-v2 memory extractor (plan Phase 3). Read ONE conversation \
turn and propose durable memory entries ("draft facts") worth keeping \
long-term.

Entry types (use exactly one per draft):
- "User": durable facts about the user — preferences, habits, environment, \
identity (e.g. "prefers concise answers", "runs Arch Linux").
- "Feedback": corrections and coaching about the agent's behavior \
(e.g. "stop restating the question", "always show the diff first").
- "Project": durable facts about the active project — structure, decisions, \
conventions, constraints (e.g. "deploys via GitHub Actions only").
- "Reference": pointers worth remembering — docs, URLs, file paths, contact \
points (e.g. "API reference lives at docs/api.md").

Do NOT extract:
- transient tool state: command output, temporary errors, running PIDs, \
anything stale within minutes;
- secrets or PII: API keys, tokens, passwords, credentials, phone numbers, \
addresses or other personal identifiers;
- already-known facts: routine conversation, small talk, restatements of \
information already present in memory.

Output format — return ONLY a JSON array (no prose, no markdown fences):
[{"type": "<User|Feedback|Project|Reference>",
  "text": "<one-sentence durable fact>",
  "proposed_source": "<user-explicit|agent-inferred|file-read|tool-output|\
test-green|code-review-human|llm>",
  "confidence": <number in [0, 1]>,
  "context_hint": "<the context_hint value from the turn object>"}]

"proposed_source" is your best provenance guess: "user-explicit" ONLY when \
the user stated the fact themselves in this turn; "file-read" or \
"tool-output" when the fact came from a file or tool result in this turn; \
otherwise "agent-inferred". If nothing in the turn is worth remembering, \
return an empty array: []\
"""

#: Corrective suffix appended on the single malformed-output retry.
RETRY_SUFFIX = (
    "Your previous response was not a valid JSON array. "
    "Return ONLY a JSON array — no prose, no markdown fences, no code blocks."
)

#: Prompt framing around the turn JSON object.
_TURN_HEADER = "Turn to analyze (JSON object):"
_TURN_FOOTER = "Respond with ONLY the JSON array of draft facts."

# ── multi-turn context (plan P3.T6) ────────────────────────────────────────

#: Rules prepended to the prompt when ``extract`` receives ``context_turns``.
_CONTEXT_RULES = """\
Multi-turn context rules (plan P3.T6):
- RECENT TURNS are ordered oldest-first; resolve pronouns and references \
("that approach", "it", "the one above") against them so every extracted \
fact stands alone without the context.
- NEVER re-extract anything listed under CONFIRMED FACTS - those facts are \
already stored in memory.
- For every draft you output, add a "source_turn_ids" field: the list of \
turn ids (the [<turn_id>] markers in RECENT TURNS, or the current turn) \
whose content the fact is derived from."""

#: Section headers inside the rendered context block.
_RECENT_HEADER = "RECENT TURNS (oldest first):"
_CONFIRMED_HEADER = "CONFIRMED FACTS (already stored - do NOT re-extract these):"

#: Upper bound for the slug component of a draft id.
_SLUG_MAX_CHARS = 48

#: Fallback provenance when the model's proposal is not a valid source.
_FALLBACK_SOURCE: VerificationSource = "agent-inferred"


class ExtractionError(RuntimeError):
    """Raised when the default model wiring cannot reach the fork-base LLM API.

    Never raised for malformed model *output* (that path returns ``[]`` with a
    logged warning — see the module docstring); transport/API errors from an
    injected model callable propagate unchanged (the pipeline owns retry and
    timeout policy).
    """


def build_default_model() -> Callable[[str], str]:
    """Build the default model callable against the fork-base LLM API.

    Single-model architecture (plan r3): the extractor reuses the main model
    via the fork-base auxiliary LLM client
    (:func:`agent.auxiliary_client.call_llm`). The import happens inside the
    returned closure at CALL time — on any failure (module missing, agent
    stack broken) an informative :class:`ExtractionError` is raised then, not
    at import time. Tests always inject fakes and never call this.
    """

    def _default_model(prompt: str) -> str:
        try:
            from agent.auxiliary_client import call_llm
        except Exception as exc:  # pragma: no cover - exercised only without fork-base stack
            raise ExtractionError(
                "fork-base LLM API unavailable (import of agent.auxiliary_client failed); "
                "inject a model callable into ExtractorAgent instead"
            ) from exc
        response = call_llm(
            "memory-extraction",
            messages=[{"role": "user", "content": prompt}],
        )
        choices = getattr(response, "choices", None)
        if not choices:
            logger.warning("default extractor model returned no choices; treating as malformed")
            return ""
        raw = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(raw, str):
            logger.warning("default extractor model returned non-string content; malformed")
            return ""
        return raw

    return _default_model


def _slugify(text: str) -> str:
    """ASCII slug of ``text`` for the draft id: [a-z0-9-]+ ("" -> "fact")."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:_SLUG_MAX_CHARS].rstrip("-") or "fact"


def _draft_id(text: str, now: datetime) -> str:
    """``<date>-<slug>-<8hex>`` draft id (plan P3.T2)."""
    return f"{now.strftime('%Y-%m-%d')}-{_slugify(text)}-{uuid.uuid4().hex[:8]}"


def _parse_json_array(raw: str) -> list[Any]:
    """Strictly parse ``raw`` as a JSON array (plan P3.T2: no lenient parsing).

    Raises :class:`json.JSONDecodeError` for malformed JSON and
    :class:`ValueError` for JSON that is not an array.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("model output is JSON but not an array")
    return parsed


def _coerce_confidence(value: Any) -> float:
    """Numeric coercion + clamp into [0, 1]; non-finite degrades to 0.5."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(number):
        return 0.5
    return min(1.0, max(0.0, number))


def _coerce_source(value: Any) -> VerificationSource:
    """Pass a valid provenance proposal through; degrade everything else.

    An unparsable proposal falls back to ``agent-inferred`` (weight 0.5) —
    the extractor's own inference — rather than over-claiming stronger
    evidence (e.g. ``user-explicit``) it cannot substantiate.
    """
    if isinstance(value, str) and value in SOURCE_WEIGHTS:
        return value
    return _FALLBACK_SOURCE


# ── multi-turn context helpers (plan P3.T6) ────────────────────────────────


def _one_line(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def _coerce_turn_ids(value: Any) -> list[str]:
    """Validate the model's per-item ``source_turn_ids`` proposal (P3.T6).

    Only non-empty strings survive; anything else is dropped — provenance
    metadata must never crash extraction nor be fabricated.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _coerce_context_items(
    context_turns: Sequence[Turn | Mapping[str, Any]] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse context items into ``(turn_id, one-line text)`` pairs + facts.

    Accepts :class:`~hermes.memory.context_window.Turn` objects and dicts
    with optional ``text`` / ``turn_id`` / ``confirmed_facts`` keys (plan
    P3.T6). Unusable items are dropped with a warning — context must never
    break extraction. A dict without usable text still donates its
    ``confirmed_facts``; a dict turn without a usable id gets the synthetic
    id ``context-turn-<position>`` (1-based, in list order).
    """
    turns: list[tuple[str, str]] = []
    confirmed: list[str] = []
    if not context_turns:
        return turns, confirmed
    for position, item in enumerate(context_turns, start=1):
        if isinstance(item, Turn):
            if item.text.strip():
                turns.append((item.turn_id, _one_line(item.text)))
            continue
        if isinstance(item, Mapping):
            raw_facts = item.get("confirmed_facts")
            if isinstance(raw_facts, (list, tuple)):
                confirmed.extend(
                    fact for fact in (_one_line(str(raw)) for raw in raw_facts) if fact
                )
            raw_text = item.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                raw_id = item.get("turn_id")
                turn_id = (
                    raw_id.strip()
                    if isinstance(raw_id, str) and raw_id.strip()
                    else f"context-turn-{position}"
                )
                turns.append((turn_id, _one_line(raw_text)))
            elif raw_text is not None:
                logger.warning(
                    "extractor: dropping context item with non-string text %r", raw_text
                )
            continue
        logger.warning("extractor: dropping unusable context item %r", item)
    return turns, confirmed


def _build_context_block(turns: list[tuple[str, str]], confirmed_facts: list[str]) -> str:
    """Render the P3.T6 context block (rules + RECENT TURNS + CONFIRMED FACTS).

    Empty when there is nothing to add, so the context-less prompt stays
    byte-identical to the P3.T2 format.
    """
    facts = list(dict.fromkeys(fact for fact in confirmed_facts if fact))
    if not turns and not facts:
        return ""
    sections = [_CONTEXT_RULES]
    if turns:
        sections.append(
            "\n".join([_RECENT_HEADER, *(f"- [{turn_id}] {text}" for turn_id, text in turns)])
        )
    if facts:
        sections.append("\n".join([_CONFIRMED_HEADER, *(f"- {fact}" for fact in facts)]))
    return "\n\n".join(sections)


def _suppress_confirmed(drafts: list[DraftFact], confirmed_facts: list[str]) -> list[DraftFact]:
    """Post-filter: drop drafts that restate a confirmed fact (plan P3.T6).

    Rule-based and deterministic: the model is instructed not to re-extract
    confirmed facts, but the rule gets the final word. Match is exact on
    :func:`normalize_fact_text` (lexical normalization); semantic
    near-duplicates remain the reviewer's (P3.T3) job.
    """
    confirmed = {normalize_fact_text(fact) for fact in confirmed_facts}
    confirmed.discard("")
    if not confirmed:
        return drafts
    kept: list[DraftFact] = []
    for draft in drafts:
        if normalize_fact_text(draft.text) in confirmed:
            logger.info(
                "extractor: dropping draft %s - restates a confirmed fact (P3.T6)",
                draft.draft_id,
            )
            continue
        kept.append(draft)
    return kept


class ExtractorAgent:
    """Extracts :class:`~hermes.memory.draft.DraftFact` candidates from a turn.

    The model is injected (``model(prompt) -> completion``); ``model=None``
    wires :func:`build_default_model` lazily at the first :meth:`extract`
    call. See the module docstring for the resilience and validation
    contract.
    """

    def __init__(self, model: Callable[[str], str] | None = None) -> None:
        self._model = model

    def _model_callable(self) -> Callable[[str], str]:
        """Resolve the injected model, wiring the fork-base default on first use."""
        if self._model is None:
            self._model = build_default_model()
        return self._model

    def _build_prompt(
        self, turn_text: str, context_hint: str, corrective: bool, context_block: str = ""
    ) -> str:
        """Full prompt: system rules + optional context block + turn (+ retry)."""
        turn_payload = json.dumps(
            {"turn_text": turn_text, "context_hint": context_hint},
            ensure_ascii=True,
        )
        parts = [EXTRACTOR_SYSTEM_PROMPT]
        if context_block:
            parts.append(context_block)
        parts.append(f"{_TURN_HEADER}\n{turn_payload}")
        parts.append(_TURN_FOOTER)
        prompt = "\n\n".join(parts)
        if corrective:
            prompt += f"\n\n{RETRY_SUFFIX}"
        return prompt

    def _validate_items(
        self, items: list[Any], context_hint: str, now: datetime
    ) -> list[DraftFact]:
        """Keep only schema-valid items; drop the rest with a logged warning."""
        drafts: list[DraftFact] = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("extractor: dropping non-object draft item %r", item)
                continue
            entry_type = item.get("type")
            if entry_type not in ENTRY_TYPES:
                logger.warning("extractor: dropping draft item with unknown type %r", entry_type)
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                logger.warning("extractor: dropping draft item with empty text")
                continue
            drafts.append(
                DraftFact(
                    draft_id=_draft_id(text, now),
                    text=text,
                    type=cast(EntryType, entry_type),
                    proposed_source=_coerce_source(item.get("proposed_source")),
                    confidence=_coerce_confidence(item.get("confidence")),
                    context_hint=context_hint,
                    source_turn_ids=_coerce_turn_ids(item.get("source_turn_ids")),
                )
            )
        return drafts

    def extract(
        self,
        turn_text: str,
        context_hint: str = "global",
        context_turns: Sequence[Turn | Mapping[str, Any]] | None = None,
    ) -> list[DraftFact]:
        """Extract draft facts from one turn (plan P3.T2; P3.T6 context).

        An empty or whitespace-only turn returns ``[]`` without calling the
        model. Otherwise the model is called with the system rules plus the
        turn JSON; malformed output triggers exactly one corrective retry,
        and output that is still malformed yields ``[]`` with a logged
        warning (never raises on malformed output — see the module
        docstring). Transport errors from the model callable propagate.

        With ``context_turns`` (P3.T6 — recent turns oldest first, as
        :class:`Turn` objects or plain dicts, optionally carrying
        ``confirmed_facts``), the prompt gains the multi-turn context block
        and the result is post-filtered against the confirmed facts;
        ``source_turn_ids`` on each surviving draft carries the model's
        validated per-item provenance proposal (``[]`` when absent).
        """
        if not turn_text.strip():
            return []
        turn_pairs, confirmed_facts = _coerce_context_items(context_turns)
        context_block = _build_context_block(turn_pairs, confirmed_facts)
        model = self._model_callable()
        prompt = self._build_prompt(
            turn_text, context_hint, corrective=False, context_block=context_block
        )
        try:
            items = _parse_json_array(model(prompt))
        except (json.JSONDecodeError, ValueError):
            logger.warning("extractor: model output was not a JSON array; retrying once")
            try:
                retry_prompt = self._build_prompt(
                    turn_text, context_hint, corrective=True, context_block=context_block
                )
                items = _parse_json_array(model(retry_prompt))
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "extractor: model output malformed after corrective retry; "
                    "returning no drafts for this turn"
                )
                return []
        drafts = self._validate_items(items, context_hint, datetime.now(UTC))
        return _suppress_confirmed(drafts, confirmed_facts)
