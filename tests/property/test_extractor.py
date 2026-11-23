"""P3.T2 property: extractor output is always JSON-safe draft facts.

For arbitrary turn texts — empty, whitespace, unicode, JSON-lookalikes,
prompt-marker-like strings, huge (pinned example) — extraction never raises
and the result is always a list of
:class:`~hermes.memory.draft.DraftFact` objects that round-trip through JSON
(``to_dict`` -> ``json.dumps`` -> ``from_dict``). The fake model is
deterministic per prompt: it echoes the turn as a valid draft-fact array,
except when the turn carries a marker that makes it emit malformed output —
exercising both the accept path and the resilient retry-then-``[]`` path
inside the same property.

P3.T6 property (appended): the same validity contract holds for ANY
multi-turn context size and shape — :class:`Turn` objects, plain dicts
(with/without ``confirmed_facts``), malformed dict shapes and junk items —
and a confirmed fact never leaks through the deterministic suppression rule,
even when the fake model naively re-extracts it verbatim.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.context_window import Turn
from hermes.memory.draft import DraftFact, normalize_fact_text
from hermes.memory.extractor import _CONFIRMED_HEADER, ExtractorAgent

#: Turn marker that flips the deterministic fake to malformed output.
_MALFORMED_MARK = "HERMES-MAKE-IT-MALFORMED"

_TURN_HEADER = "Turn to analyze (JSON object):"

_TURN_STRATEGY = st.one_of(
    st.text(max_size=120),
    st.text(alphabet='{}[]()",: \\0123456789abcXYZ\n\t', max_size=120),
    st.just(""),
    st.just("   "),
    st.just('{"turn_text": "[not really json", "type": "User"}'),
    st.just(_MALFORMED_MARK),
    st.just("turn with 🌍 unicode and 控 character"),
)


def _deterministic_model(prompt: str) -> str:
    """Fake model, deterministic in the prompt: valid echo array or garbage.

    The echo text is deliberately non-empty ("" turns never reach the model:
    ``extract`` short-circuits them), and the malformed branch exercises the
    corrective-retry path deterministically (both calls return garbage).
    """
    if _MALFORMED_MARK in prompt:
        return "sorry, I cannot produce JSON for this one"
    payload = prompt.split(_TURN_HEADER + "\n", 1)[1].split("\n\nRespond with ONLY", 1)[0]
    turn = json.loads(payload)["turn_text"]
    item: dict[str, Any] = {
        "type": "User",
        "text": f"observed: {turn[:120]}",
        "proposed_source": "agent-inferred",
        "confidence": 0.6,
        "context_hint": "prop-ctx",
    }
    return json.dumps([item], ensure_ascii=True)


def _assert_draft_facts(result: object) -> None:
    """The P3.T2 output invariant: JSON-safe, validated DraftFact objects."""
    assert isinstance(result, list)
    for fact in result:
        assert isinstance(fact, DraftFact)
        payload = fact.to_dict()
        json.dumps(payload)  # must be JSON-serializable by construction
        assert DraftFact.from_dict(json.loads(json.dumps(payload))) == fact
        assert payload["type"] in {"User", "Feedback", "Project", "Reference"}
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["text"]
        assert payload["context_hint"] == "prop-ctx"


@pytest.mark.property
@given(turn=_TURN_STRATEGY)
@example("")  # empty turn -> [] without a model call
@example("   ")
@example(_MALFORMED_MARK)  # deterministic malformed output -> resilient []
@example("x" * 100_000)  # huge turn text follows the same contract
@settings(max_examples=50, deadline=None)
def test_output_always_valid_json(turn: str) -> None:
    """Extraction never raises and always yields JSON-safe DraftFact objects."""
    agent = ExtractorAgent(model=_deterministic_model)

    _assert_draft_facts(agent.extract(turn, context_hint="prop-ctx"))


# ── P3.T6: output validity for any context size ────────────────────────────

_TURN_FOOTER = "Respond with ONLY the JSON array of draft facts."

#: Turn-id markers rendered in the RECENT TURNS block ("- [<turn_id>] ...").
_CONTEXT_TURN_ID_RE = re.compile(r"^- \[([^\]]+)\]", re.MULTILINE)

_TURN_TIMESTAMP = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _confirmed_from_prompt(prompt: str) -> list[str]:
    """Parse the rendered CONFIRMED FACTS section (the fake model's view)."""
    if _CONFIRMED_HEADER not in prompt:
        return []
    section = prompt.split(_CONFIRMED_HEADER, 1)[1]
    return [
        line[2:]
        for line in section.splitlines()
        if line.startswith("- ") and not line.startswith("- [")
    ]


def _context_aware_model(prompt: str) -> str:
    """Fake model, deterministic in the prompt: naive re-extractor + echoer.

    When the (normalized) turn text restates a rendered confirmed fact, the
    model "re-extracts" that fact verbatim — exactly the mistake the P3.T6
    suppression rule must catch. Otherwise it echoes the turn as a draft
    carrying the rendered turn ids as ``source_turn_ids``. The malformed
    marker keeps exercising the retry-then-``[]`` path, now with context.
    """
    if _MALFORMED_MARK in prompt:
        return "sorry, I cannot produce JSON for this one"
    header_pos = prompt.rfind(_TURN_HEADER + "\n")
    tail = prompt[header_pos + len(_TURN_HEADER) + 1:]
    payload = tail.split("\n\n" + _TURN_FOOTER, 1)[0]
    turn = json.loads(payload)["turn_text"]
    fact_map = {normalize_fact_text(fact): fact for fact in _confirmed_from_prompt(prompt)}
    key = normalize_fact_text(turn)
    if key and key in fact_map:
        return json.dumps(
            [{
                "type": "User",
                "text": fact_map[key],  # naive re-extraction of a confirmed fact
                "proposed_source": "user-explicit",
                "confidence": 0.95,
                "context_hint": "prop-ctx",
            }],
            ensure_ascii=True,
        )
    item: dict[str, Any] = {
        "type": "User",
        "text": f"observed: {turn[:120]}",
        "proposed_source": "agent-inferred",
        "confidence": 0.6,
        "context_hint": "prop-ctx",
        "source_turn_ids": _CONTEXT_TURN_ID_RE.findall(prompt)[:2],
    }
    return json.dumps([item], ensure_ascii=True)


_CONTEXT_ITEM_STRATEGY = st.one_of(
    # well-formed dict turn, optionally carrying a turn_id / confirmed facts
    st.fixed_dictionaries(
        {"text": st.text(max_size=60)},
        optional={
            "turn_id": st.one_of(st.none(), st.text(max_size=12)),
            "confirmed_facts": st.lists(st.text(max_size=60), max_size=3),
        },
    ),
    # fact-carrier without text (donates only its confirmed facts)
    st.fixed_dictionaries(
        {"confirmed_facts": st.lists(st.text(max_size=60), max_size=3)}
    ),
    # malformed shapes the extractor must drop gracefully
    st.fixed_dictionaries({"text": st.sampled_from([7, None, "", "   ", ["x"]])}),
    st.just({"confirmed_facts": "not-a-list"}),
    st.just({}),
    st.none(),
    st.integers(min_value=0, max_value=99),
    st.text(max_size=10),
    # genuine Turn objects (turn_id validated non-empty by Turn itself)
    st.builds(
        Turn,
        turn_id=st.text(min_size=1, max_size=12),
        session_id=st.just("prop"),
        index=st.integers(min_value=0, max_value=20),
        text=st.text(max_size=60),
        timestamp=st.just(_TURN_TIMESTAMP),
    ),
)


@pytest.mark.property
@given(turn=_TURN_STRATEGY, items=st.lists(_CONTEXT_ITEM_STRATEGY, max_size=8))
@example(turn="", items=[])  # empty turn: short-circuit regardless of context
@example(turn=_MALFORMED_MARK, items=[])  # malformed output with context set up
@example(
    turn="not vim",
    items=[{"text": "we chose vim", "turn_id": "t1", "confirmed_facts": ["not vim"]}],
)  # direct suppression: the model re-extracts a confirmed fact verbatim
@example(turn="", items=[42, None, {"confirmed_facts": "junk"}])  # junk context
@settings(max_examples=50, deadline=None)
def test_output_valid_with_any_context_size(turn: str, items: list[Any]) -> None:
    """Any context size/shape: extraction never raises, output stays valid,
    and confirmed facts never leak through the suppression rule."""
    agent = ExtractorAgent(model=_context_aware_model)

    result = agent.extract(turn, context_hint="prop-ctx", context_turns=items)

    _assert_draft_facts(result)
    for fact in result:
        assert isinstance(fact.source_turn_ids, list)
        assert all(isinstance(turn_id, str) for turn_id in fact.source_turn_ids)

    # P3.T6 suppression invariant: no draft restates a confirmed fact the
    # context items declared — deterministic rule, the model cannot bypass it.
    confirmed: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            raw_facts = item.get("confirmed_facts")
            if isinstance(raw_facts, list | tuple):
                for raw in raw_facts:
                    if isinstance(raw, str) and normalize_fact_text(raw):
                        confirmed.add(normalize_fact_text(raw))
    for fact in result:
        assert normalize_fact_text(fact.text) not in confirmed
