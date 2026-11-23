"""P20.T3: concurrent lifecycle actions race on one skill (plan Phase 20).

``test_concurrent_approve_deprecate`` — two agent threads race ``approve``
vs ``deprecate`` on the SAME skill. The transition table (plan P13.T1) is
the arbiter: ``approve`` is legal from {proposed, approved}, ``deprecate``
ONLY from active — there is no state from which both actions are legal, so
across the race EXACTLY ONE succeeds and the other raises
:class:`~hermes.skills.lifecycle.InvalidTransitionError`, with the registry
ending consistent (the state matches the successful action, the loser left
no trace, and a freshly reloaded registry agrees).

Race construction (honest): the skill starts ACTIVE (the state both agents
act on — e.g. the main agent approved it earlier while a stale agent still
tries to approve and another one retires it). From that start state the
outcome is deterministic under ANY interleaving: approve can never succeed
and deprecate always can, so exactly one transition lands no matter which
thread checks first. ``SkillLifecycle`` is deliberately not thread-safe
(documented on the class); the guarantee under test is the PLAN-level one —
the transition table, not thread scheduling, decides the outcome, and the
atomic registry persistence (tmp file + ``os.replace``) keeps the file
consistent. Several barrier-synchronized rounds sample different
interleavings of the two threads.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from hermes.skills.lifecycle import InvalidTransitionError, SkillLifecycle, SkillState

#: Barrier-synchronized race rounds per start state (interleaving samples).
RACE_ROUNDS = 4

_SKILL_YAML = "name: racer\ndescription: Races lifecycle transitions.\n"


def _race_once(store_path: Path, name: str, start_active: bool) -> tuple[dict[str, str], str, SkillState]:
    """One approve-vs-deprecate race; returns outcomes, skill id, final state."""
    lifecycle = SkillLifecycle(store_path=store_path)
    skill = lifecycle.propose(name, _SKILL_YAML, "racing skill")
    skill_id = skill.skill_id
    if start_active:
        lifecycle.approve(skill_id)  # now ACTIVE: deprecate legal, approve not

    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def do_approve() -> None:
        barrier.wait()
        try:
            lifecycle.approve(skill.skill_id)
            results["approve"] = "ok"
        except InvalidTransitionError:
            results["approve"] = "invalid"

    def do_deprecate() -> None:
        barrier.wait()
        try:
            lifecycle.deprecate(skill.skill_id, "no longer helpful")
            results["deprecate"] = "ok"
        except InvalidTransitionError:
            results["deprecate"] = "invalid"

    threads = [
        threading.Thread(target=do_approve, name="approver"),
        threading.Thread(target=do_deprecate, name="deprecator"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    # exactly one action succeeded; the other was refused as illegal
    assert sorted(results.values()) == ["invalid", "ok"], results
    final = lifecycle.get(skill_id)
    assert final is not None
    if results["deprecate"] == "ok":
        assert final.state is SkillState.DEPRECATED
        assert final.deprecation_reason == "no longer helpful"
    else:
        assert results["approve"] == "ok"
        assert final.state is SkillState.ACTIVE
        assert final.deprecation_reason is None
    return results, skill_id, final.state


@pytest.mark.integration
def test_concurrent_approve_deprecate(tmp_path: Path) -> None:
    """The approve/deprecate race always lands exactly one legal transition."""
    for round_index in range(RACE_ROUNDS):
        store_path = tmp_path / f"registry-race-{round_index}.json"
        results, skill_id, state = _race_once(
            store_path, f"Racer skill {round_index}", start_active=True
        )
        # the winner is the only legal action from ACTIVE: deprecate
        assert results["deprecate"] == "ok"
        assert results["approve"] == "invalid"
        assert state is SkillState.DEPRECATED

        # registry consistent: a fresh reload reads exactly the final state
        reloaded = SkillLifecycle(store_path=store_path)
        skill = reloaded.get(skill_id)
        assert skill is not None
        assert skill.state is SkillState.DEPRECATED
        assert skill.deprecation_reason == "no longer helpful"
