"""P20.T3: subagent <-> main agent coordination through the shared stack.

- ``test_subagent_extracts_main_reviews`` — the SUBAGENT thread runs the
  extraction stage (:class:`~hermes.memory.extractor.ExtractorAgent` over a
  fake model) for its finished turn; the MAIN thread's
  :class:`~hermes.memory.reviewer.ReviewerAgent` (fake model) reviews the
  subagent's draft, and the add lands through the shared
  :class:`~hermes.memory.pipeline.MemoryPipeline` with full provenance (the
  trust card's verification event cites ``extractor draft <id>``).
- ``test_subagent_proposes_skill_main_approves`` — the subagent proposes a
  skill (:meth:`SkillLifecycle.propose`, plan P13.T1); the main agent
  approves it through the ``/skills`` command handler
  (:func:`hermes.core.agent.handle_skills_command`, plan P13.T4) and the
  skill ends ACTIVE in the persisted registry (a fresh lifecycle instance
  agrees).
"""

from __future__ import annotations

import json
import threading

import pytest

from hermes.core.agent import handle_skills_command
from hermes.memory.draft import DraftFact
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore
from hermes.skills.lifecycle import SkillLifecycle, SkillState
from hermes.skills.registry import SkillRegistry

#: Valid extractor completion: one draft proposed by the subagent's turn.
_EXTRACT_JSON = json.dumps(
    [
        {
            "text": "subagent coordination fact from its turn",
            "type": "Feedback",
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ]
)

#: Valid reviewer completion: the main agent's review approves the draft.
_REVIEW_JSON = json.dumps(
    {"action": "add", "rationale": "new fact from the subagent turn; no conflict"}
)

_SKILL_YAML = "name: greeter\ndescription: Greets the user.\n"


@pytest.mark.integration
def test_subagent_extracts_main_reviews() -> None:
    """Subagent extracts, main reviews: one add lands with provenance."""
    store = MemoryStore()  # the shared memory both agents write through
    pipeline = MemoryPipeline(
        extractor=ExtractorAgent(model=lambda prompt: _EXTRACT_JSON),
        reviewer=ReviewerAgent(model=lambda prompt: _REVIEW_JSON, search_fn=lambda query: []),
        store=store,
    )

    drafts_box: list[DraftFact] = []
    errors: list[str] = []

    def subagent_turn() -> None:
        """The subagent's stage-1 work on its own thread."""
        try:
            drafts_box.extend(
                pipeline.extract_drafts("subagent finished researching the topic")
            )
        except Exception as exc:  # collected, never swallowed
            errors.append(f"extract: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=subagent_turn, name="subagent")
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert errors == []
    assert len(drafts_box) == 1
    draft = drafts_box[0]

    # Main thread: review the subagent's draft and apply the decision.
    actions = pipeline.review_drafts(drafts_box)
    assert [action.action for action in actions] == ["add"]
    applied = pipeline.apply_actions(
        drafts=drafts_box, actions=actions, context="global", session_id="main"
    )
    assert len(applied) == 1
    assert applied[0].action == "add"
    assert applied[0].draft_id == draft.draft_id

    entry = store.get(applied[0].entry_id)
    assert entry is not None
    assert entry.text == "subagent coordination fact from its turn"

    # Provenance: the trust card cites the extractor draft that produced it.
    event = entry.verification_events[0]
    assert event.evidence == f"extractor draft {draft.draft_id}"
    assert event.source == "user-explicit"  # the draft's proposed provenance
    assert event.verifier == "main-model"
    assert entry.trust_score > 0.0
    assert store.git.is_clean()
    assert store.git.log(path=f"global/facts/{entry.id}.md", limit=5)


@pytest.mark.integration
def test_subagent_proposes_skill_main_approves(tmp_path) -> None:
    """Subagent proposes a skill; the main agent approves it -> active."""
    store_path = tmp_path / "memory-repo" / "skills" / "_skill_registry.json"
    lifecycle = SkillLifecycle(store_path=store_path)
    registry = SkillRegistry(store_path=store_path)

    proposed_box: list[str] = []
    errors: list[str] = []

    def subagent_proposes() -> None:
        """The subagent proposes its skill on its own thread (plan P13.T1)."""
        try:
            skill = lifecycle.propose(
                "Greeter Skill", _SKILL_YAML, "greets users on request"
            )
            proposed_box.append(skill.skill_id)
        except Exception as exc:  # collected, never swallowed
            errors.append(f"propose: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=subagent_proposes, name="subagent")
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []
    assert len(proposed_box) == 1
    skill_id = proposed_box[0]
    assert lifecycle.get(skill_id).state is SkillState.PROPOSED

    # Main agent: approve through the /skills command handler (plan P13.T4).
    out = handle_skills_command(None, ["approve", skill_id], lifecycle, registry)
    assert "active" in out
    assert lifecycle.get(skill_id).state is SkillState.ACTIVE

    # Consistent state everywhere: a freshly constructed registry re-reads
    # the shared persisted file (the first registry instance was built before
    # the proposal and never re-loads), and a fresh lifecycle instance reads
    # the approved state from disk.
    reloaded_registry = SkillRegistry(store_path=store_path)
    assert reloaded_registry.list_skills() == [skill_id]
    reloaded = SkillLifecycle(store_path=store_path)
    approved = reloaded.get(skill_id)
    assert approved is not None
    assert approved.state is SkillState.ACTIVE
    assert approved.rationale == "greets users on request"
