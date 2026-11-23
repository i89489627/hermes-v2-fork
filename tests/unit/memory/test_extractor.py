"""P3.T2 acceptance: ExtractorAgent extracts draft facts from turns.

The LLM is injected as a fake callable (single-model architecture — never
called live in tests). Covered here: the three positive extraction cases
from the plan (user preference, project fact, feedback correction), the
empty-turn short-circuit (no model call), and the strict structured-output
schema validation (unknown types dropped, confidence clamped, provenance
degraded to agent-inferred, caller context_hint wins, retry on malformed
JSON, resilient ``[]`` on persistent malformed output).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from hermes.memory.context_window import Turn
from hermes.memory.draft import DraftFact
from hermes.memory.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    ExtractorAgent,
)

_DRAFT_ID_PATTERN = r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+-[0-9a-f]{8}"


class _ScriptedModel:
    """Fake model: records prompts, replays scripted completions in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("unexpected extra model call")
        return self._responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _json_array(*items: dict[str, Any]) -> str:
    return json.dumps(list(items))


@pytest.mark.unit
def test_extracts_user_preference() -> None:
    """A stated preference becomes one User draft with provenance passthrough."""
    model = _ScriptedModel([
        _json_array({
            "type": "User",
            "text": "User prefers concise answers",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        })
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("I prefer concise answers from now on.")

    assert len(drafts) == 1
    fact = drafts[0]
    assert isinstance(fact, DraftFact)
    assert fact.type == "User"
    assert fact.text == "User prefers concise answers"
    assert fact.proposed_source == "user-explicit"
    assert fact.confidence == 0.9
    assert fact.context_hint == "global"  # default hint per plan schema
    assert re.fullmatch(_DRAFT_ID_PATTERN, fact.draft_id)
    assert "user-prefers-concise-answers" in fact.draft_id  # slug from the fact text

    # the prompt carried the system rules and the turn JSON
    assert model.calls == 1
    assert EXTRACTOR_SYSTEM_PROMPT in model.prompts[0]
    turn_payload = json.dumps({"turn_text": "I prefer concise answers from now on.",
                               "context_hint": "global"})
    assert turn_payload in model.prompts[0]

    # the draft serializes to the plan schema fields
    payload = fact.to_dict()
    assert payload["type"] == "User"
    assert payload["text"] == "User prefers concise answers"
    assert payload["proposed_source"] == "user-explicit"
    assert payload["confidence"] == 0.9
    assert payload["context_hint"] == "global"


@pytest.mark.unit
def test_extracts_project_fact() -> None:
    """A project convention becomes a Project draft; caller context_hint wins."""
    model = _ScriptedModel([
        _json_array({
            "type": "Project",
            "text": "The acme repo runs tests via pytest only",
            "proposed_source": "file-read",
            "confidence": 0.7,
        })
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("Remember: our acme repo runs its tests via pytest.", context_hint="work")

    assert len(drafts) == 1
    fact = drafts[0]
    assert fact.type == "Project"
    assert fact.text == "The acme repo runs tests via pytest only"
    assert fact.proposed_source == "file-read"  # valid proposal passes through
    assert fact.confidence == 0.7
    assert fact.context_hint == "work"  # caller-provided, not the model's echo
    assert re.fullmatch(_DRAFT_ID_PATTERN, fact.draft_id)


@pytest.mark.unit
def test_extracts_feedback_correction() -> None:
    """A behavioral correction becomes a Feedback draft."""
    model = _ScriptedModel([
        _json_array({
            "type": "Feedback",
            "text": "User corrected the agent: use spaces, never tabs, in code",
            "proposed_source": "user-explicit",
            "confidence": 0.95,
        })
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("Stop using tabs in my code — spaces only, please.")

    assert len(drafts) == 1
    fact = drafts[0]
    assert fact.type == "Feedback"
    assert fact.text == "User corrected the agent: use spaces, never tabs, in code"
    assert fact.proposed_source == "user-explicit"
    assert fact.confidence == 0.95
    assert fact.context_hint == "global"
    assert re.fullmatch(_DRAFT_ID_PATTERN, fact.draft_id)


@pytest.mark.unit
def test_empty_turn_no_drafts() -> None:
    """Empty / whitespace-only turns return [] without calling the model at all."""
    model = _ScriptedModel([])  # any call would raise AssertionError
    agent = ExtractorAgent(model=model)

    assert agent.extract("") == []
    assert agent.extract("   \n\t  ") == []
    assert model.prompts == []


@pytest.mark.unit
def test_structured_output_schema_valid() -> None:
    """Strict per-item validation: schema-valid drafts kept, invalid dropped."""
    model = _ScriptedModel([
        _json_array(
            {  # valid item
                "type": "User",
                "text": "User prefers concise answers",
                "proposed_source": "user-explicit",
                "confidence": 0.9,
                "context_hint": "model-echo-hint",
            },
            {  # unknown entry type -> dropped, never coerced
                "type": "Snapshot",
                "text": "invalid type item",
                "proposed_source": "llm",
                "confidence": 0.5,
            },
            "just-a-string",  # non-object item -> dropped
            {  # empty text -> dropped
                "type": "Project",
                "text": "   ",
                "proposed_source": "llm",
                "confidence": 0.5,
            },
            {  # invalid provenance + out-of-range confidence
                "type": "Feedback",
                "text": "User corrected the agent: use spaces",
                "proposed_source": "gossip",
                "confidence": 7,
            },
            {  # missing provenance + string confidence
                "type": "Reference",
                "text": "API docs live at docs/api.md",
                "confidence": "0.3",
            },
        )
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("Mixed turn with several candidates.", context_hint="work")

    assert len(drafts) == 3  # the three schema-valid items
    assert len({fact.draft_id for fact in drafts}) == 3  # draft ids unique

    first, feedback, reference = drafts
    assert first.type == "User"
    assert first.context_hint == "work"  # caller hint wins over the model's echo
    assert first.proposed_source == "user-explicit"

    assert feedback.type == "Feedback"
    assert feedback.proposed_source == "agent-inferred"  # "gossip" -> degraded, not over-claimed
    assert feedback.confidence == 1.0  # 7 clamped into [0, 1]

    assert reference.type == "Reference"
    assert reference.proposed_source == "agent-inferred"  # missing -> conservative default
    assert reference.confidence == 0.3  # "0.3" parsed and clamped

    for fact in drafts:
        payload = fact.to_dict()
        assert set(payload) >= {"type", "text", "proposed_source", "confidence", "context_hint"}
        assert DraftFact.from_dict(json.loads(json.dumps(payload))) == fact

    # malformed JSON: one corrective retry, then a valid array is accepted
    retry_model = _ScriptedModel(["I could not produce JSON, sorry.", _json_array({
        "type": "User",
        "text": "Recovered after corrective retry",
        "proposed_source": "agent-inferred",
        "confidence": 0.4,
    })])
    recovered = ExtractorAgent(model=retry_model).extract("Some turn.")
    assert len(recovered) == 1
    assert recovered[0].text == "Recovered after corrective retry"
    assert retry_model.calls == 2
    assert retry_model.prompts[1].endswith(
        "Return ONLY a JSON array — no prose, no markdown fences, no code blocks."
    )

    # persistent malformed output: resilient [] (never raises) after the retry
    broken_model = _ScriptedModel(["still not json", "```json\n{not an array}\n```"])
    assert ExtractorAgent(model=broken_model).extract("Some turn.") == []
    assert broken_model.calls == 2


# ── P3.T6: multi-turn context for the extractor ────────────────────────────


@pytest.mark.unit
def test_extracts_with_multi_turn_context() -> None:
    """context_turns add the RECENT TURNS block and populate source_turn_ids."""
    context = [
        Turn(
            turn_id="t1",
            session_id="s1",
            index=0,
            text="We chose alembic for database migrations",
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        ),
        Turn(
            turn_id="t2",
            session_id="s1",
            index=1,
            text="The migration config lives in alembic.ini",
            timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
        ),
        42,  # unusable context item -> dropped with a warning, never fatal
        None,
        {"text": 7},  # non-string text -> dropped
    ]
    model = _ScriptedModel([
        _json_array({
            "type": "Project",
            "text": "The project manages schema migrations with alembic",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
            "source_turn_ids": ["t1", "t2"],
        }),
        _json_array({
            "type": "Project",
            "text": "The project manages schema migrations with alembic",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }),
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract(
        "Keep that approach for the next service too.", context_turns=context
    )

    assert len(drafts) == 1
    fact = drafts[0]
    assert fact.text == "The project manages schema migrations with alembic"
    assert fact.source_turn_ids == ["t1", "t2"]  # provenance populated (P3.T6)

    # the context block: rules, turns oldest-first (t1 before t2); unusable
    # items contribute nothing.
    prompt = model.prompts[0]
    assert "Multi-turn context rules (plan P3.T6):" in prompt
    assert '"source_turn_ids"' in prompt
    assert "RECENT TURNS (oldest first):" in prompt
    assert prompt.index("- [t1] We chose alembic for database migrations") < prompt.index(
        "- [t2] The migration config lives in alembic.ini"
    )
    assert "- [42]" not in prompt and "- [context-turn" not in prompt  # junk dropped

    # without context_turns the prompt stays byte-identical to the P3.T2 form
    drafts = agent.extract("Keep that approach for the next service too.")
    assert len(drafts) == 1
    assert drafts[0].source_turn_ids == []  # no context -> no provenance
    assert "RECENT TURNS (oldest first):" not in model.prompts[1]
    assert "Multi-turn context rules" not in model.prompts[1]
    assert '"source_turn_ids"' not in model.prompts[1]


@pytest.mark.unit
def test_does_not_reextract_confirmed_facts() -> None:
    """Drafts restating a confirmed fact are dropped by the deterministic rule."""
    context = [{
        "text": "We picked alembic for the migrations",
        "turn_id": "t1",
        "confirmed_facts": ["The project uses Alembic for migrations!"],
    }]
    model = _ScriptedModel([
        _json_array(
            {  # naive re-extraction: normalized match of the confirmed fact
                "type": "Project",
                "text": "the project uses alembic for migrations",
                "proposed_source": "user-explicit",
                "confidence": 0.95,
                "source_turn_ids": ["t1"],
            },
            {  # genuinely new fact -> survives
                "type": "Reference",
                "text": "Migration history lives in the alembic/versions folder",
                "proposed_source": "file-read",
                "confidence": 0.7,
                "source_turn_ids": ["t1"],
            },
        )
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("Right, alembic it is.", context_turns=context)

    assert len(drafts) == 1  # the confirmed re-extraction is suppressed
    assert drafts[0].text == "Migration history lives in the alembic/versions folder"
    assert drafts[0].source_turn_ids == ["t1"]
    assert model.calls == 1  # suppression is a post-filter, not a model round

    # the prompt instructed the model about the confirmed fact
    prompt = model.prompts[0]
    assert "CONFIRMED FACTS (already stored - do NOT re-extract these):" in prompt
    assert "- The project uses Alembic for migrations!" in prompt


@pytest.mark.unit
def test_resolves_references_across_turns() -> None:
    """The context carries prior turns so 'that approach' resolves to a fact."""
    context = [
        Turn(
            turn_id="t1",
            session_id="s1",
            index=0,
            text="We should use the repository pattern for the data layer",
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        ),
        Turn(
            turn_id="t2",
            session_id="s1",
            index=1,
            text="The repository pattern keeps persistence isolated from the domain",
            timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
        ),
    ]
    model = _ScriptedModel([
        _json_array({
            "type": "Project",
            "text": "The project uses the repository pattern for its data layer",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
            "source_turn_ids": ["t1", "t2"],
        })
    ])
    agent = ExtractorAgent(model=model)

    drafts = agent.extract("yes, use that approach", context_turns=context)

    assert len(drafts) == 1
    fact = drafts[0]
    # the extracted fact is self-contained: the reference is resolved against
    # the context turns, not echoed as "that approach".
    assert fact.text == "The project uses the repository pattern for its data layer"
    assert "that approach" not in fact.text
    assert fact.source_turn_ids == ["t1", "t2"]  # both contributing turns

    prompt = model.prompts[0]
    assert "- [t1] We should use the repository pattern for the data layer" in prompt
    assert "- [t2] The repository pattern keeps persistence isolated from the domain" in prompt
    assert '"turn_text": "yes, use that approach"' in prompt
