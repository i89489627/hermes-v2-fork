"""P20.T2 property: the memory pipeline produces valid output for any turn.

For arbitrary turn texts — empty, whitespace, unicode, JSON-lookalikes,
prompt-marker-like strings, huge (pinned example) —
:meth:`MemoryPipeline.run_after_turn` (sync) never raises and returns a
structurally valid :class:`~hermes.memory.pipeline.PipelineResult`:

- ``drafts`` are validated :class:`~hermes.memory.draft.DraftFact` objects;
- ``drafts`` and ``actions`` are index-aligned and every action carries a
  valid :data:`~hermes.memory.types.ReviewAction` literal;
- every applied record points at an entry that actually exists in the store;
- an empty (or whitespace-only) turn produces empty drafts, actions and
  applied records (the extractor short-circuits without a model call).

The fakes are deterministic and no LLM is involved: the fake extractor model
echoes the turn (parsed back out of the prompt payload) as one valid draft;
the fake reviewer model draws from a fixed behavior set (add / skip /
unknown-target update / malformed output / model-declared conflict with an
unknown target). The reviewer's ``search_fn`` returns no candidates, so
store writes only ever come from ``add`` decisions — the shared module-scope
store stays small and the 1000-example run stays fast.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hermes.memory.draft import REVIEW_ACTIONS, DraftFact
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.pipeline import MemoryPipeline, PipelineResult
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_TURN_HEADER = "Turn to analyze (JSON object):"
_TURN_FOOTER = "Respond with ONLY the JSON array of draft facts."

_TURN_STRATEGY = st.one_of(
    st.text(max_size=200),
    st.text(alphabet='{}[]()",: \\0123456789abcXYZ\n\t', max_size=200),
    st.just(""),
    st.just("   "),
    st.just('{"turn_text": "[not really json", "type": "User"}'),
    st.just("turn with 🌍 unicode and 控 character"),
    st.just("x" * 100_000),
)

#: Reviewer behaviors: the malformed/unresolvable ones exercise the
#: conservative skip fallbacks inside the same property.
_BEHAVIORS = (
    "add",
    "skip",
    "update-unknown-target",
    "model-conflict-unknown-target",
    "malformed",
    "empty",
)


def _echo_extractor_model(prompt: str) -> str:
    """Deterministic fake extractor: one draft echoing the turn text.

    The turn is parsed back out of the prompt payload (byte-exact JSON), so
    the echo is deterministic in the drawn turn — including unicode and
    JSON-lookalike texts.
    """
    payload = prompt.split(_TURN_HEADER + "\n", 1)[1].split("\n\n" + _TURN_FOOTER, 1)[0]
    turn = json.loads(payload)["turn_text"]
    return json.dumps(
        [
            {
                "type": "User",
                "text": f"fact: {turn[:60]}",
                "proposed_source": "agent-inferred",
                "confidence": 0.6,
            }
        ],
        ensure_ascii=True,
    )


def _reviewer_model_for(behavior: str) -> Any:
    """Deterministic fake reviewer model for the drawn behavior."""

    def _model(_prompt: str) -> str:
        if behavior == "malformed":
            return "sorry, no JSON here"
        if behavior == "empty":
            return ""
        if behavior == "add":
            return json.dumps({"action": "add", "target_id": None, "rationale": "new"})
        if behavior == "skip":
            return json.dumps({"action": "skip", "target_id": None, "rationale": "transient"})
        if behavior == "update-unknown-target":
            # unknown target -> the reviewer downgrades to skip (never stored)
            return json.dumps({
                "action": "update",
                "target_id": "no-such-entry",
                "rationale": "merge",
                "merged_text": "merged body",
            })
        if behavior == "model-conflict-unknown-target":
            # model-declared conflict with an unknown target -> downgraded
            return json.dumps({
                "action": "conflict",
                "target_id": "no-such-entry",
                "rationale": "contradicts",
            })
        raise AssertionError(f"unknown behavior {behavior!r}")

    return _model


@pytest.fixture(scope="module")
def prop_store(tmp_path_factory: pytest.TempPathFactory) -> MemoryStore:
    """One shared git-backed store for the whole module (fast 1000-example runs).

    The reviewer ``search_fn`` injects no candidates, so the only store
    writes are ``add`` decisions — roughly one in six examples commits one
    small entry; the invariant under test is per-run and never depends on
    cross-example isolation.
    """
    root: Path = tmp_path_factory.mktemp("pipeline-property")
    return MemoryStore(root=root / "memory-repo")


def _assert_valid_result(result: object, turn_id: str, store: MemoryStore) -> None:
    """The P20.T2 output contract, regardless of what the fakes returned."""
    assert isinstance(result, PipelineResult)
    assert result.turn_id == turn_id
    assert result.elapsed_ms >= 0.0
    assert len(result.drafts) == len(result.actions)
    for draft, action in zip(result.drafts, result.actions, strict=True):
        assert isinstance(draft, DraftFact)
        assert draft.text  # validated DraftFacts always carry text
        assert action.action in REVIEW_ACTIONS
    for applied in result.applied:
        assert applied.action in {"add", "update"}
        assert applied.entry_id is not None
        # every applied write points at an entry that actually exists
        assert store.get(applied.entry_id) is not None


@pytest.mark.property
@given(turn=_TURN_STRATEGY, behavior=st.sampled_from(_BEHAVIORS))
@example(turn="", behavior="add")
@example(turn="   ", behavior="add")
@example(turn="turn with 🌍 unicode and 控 character", behavior="add")
@example(turn="x" * 100_000, behavior="add")
@example(turn='{"turn_text": "[not really json", "type": "User"}', behavior="malformed")
@settings(max_examples=1000, deadline=None)
def test_any_turn_produces_valid_output(turn: str, behavior: str, prop_store: MemoryStore) -> None:
    """run_after_turn(sync=True) never raises and always yields valid output."""
    extractor = ExtractorAgent(model=_echo_extractor_model)
    reviewer = ReviewerAgent(
        model=_reviewer_model_for(behavior), search_fn=lambda _query: []
    )
    pipeline = MemoryPipeline(extractor, reviewer, prop_store)

    result = pipeline.run_after_turn("t-prop", turn, session_id="s1", context="global")

    _assert_valid_result(result, "t-prop", prop_store)


@pytest.mark.property
@given(turn=st.one_of(st.just(""), st.text(alphabet=" \t\n\r", max_size=20)))
@settings(max_examples=1000, deadline=None)
def test_empty_turn_produces_empty_list(turn: str, prop_store: MemoryStore) -> None:
    """An empty (or whitespace-only) turn produces no drafts, actions or writes.

    The extractor short-circuits empty turns without a model call, so the
    whole run is a no-op: empty drafts, empty actions, empty applied records.
    """
    calls: list[str] = []

    def _counting_model(_prompt: str) -> str:
        calls.append(_prompt)
        return "[]"

    extractor = ExtractorAgent(model=_counting_model)
    reviewer = ReviewerAgent(
        model=lambda _prompt: json.dumps({"action": "add", "rationale": "new"}),
        search_fn=lambda _query: [],
    )
    pipeline = MemoryPipeline(extractor, reviewer, prop_store)

    result = pipeline.run_after_turn("t-empty", turn, session_id="s1", context="global")

    assert isinstance(result, PipelineResult)
    assert result.drafts == []
    assert result.actions == []
    assert result.applied == []
    assert calls == []  # the extractor never called the model
