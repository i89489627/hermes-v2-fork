"""P1.T3 acceptance: MemoryStore — atomic trust-card files + git commits.

Covers the spec behaviours: add (atomic file + commit + transactional
rollback), get, update (trust score + tool-output event), remove (staged
deletion, committed), list filtering (context/type/tag) and strict context
isolation. Stores live under tmp_path so HERMES_HOME sandboxing from
conftest is not even needed for these tests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


@pytest.mark.unit
def test_add_creates_file_and_commits(tmp_path: Path) -> None:
    """add writes <root>/global/facts/<id>.md atomically and commits it."""
    store = _make_store(tmp_path)
    entry = store.add(
        "always run ruff before committing", type="Feedback", tags=["linting"]
    )

    # id = <date>-<slug> with an 8-hex uniqueness suffix
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+-[0-9a-f]{8}", entry.id)
    assert entry.id.startswith(entry.created_at.strftime("%Y-%m-%d"))

    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    assert path.is_file()
    card, body = parse(path.read_text(encoding="utf-8"))
    assert card.id == entry.id
    assert card.type == "Feedback"
    assert card.context == "global"
    assert card.tags == ["linting"]
    assert body == "always run ruff before committing"

    # one VerificationEvent created from the add() defaults
    assert entry.verification_events[0].source == "agent-inferred"
    assert entry.verification_events[0].verifier == "main-model"
    assert entry.trust_score == 1.0  # weighted mean of the single v=1.0 event

    history = store.git.log(path=path)
    assert [commit.message for commit in history] == [f"[memory] add {entry.id} global"]
    assert store.git.is_clean()  # entry + index committed, nothing dangling

    index_file = tmp_path / "memory-repo" / "MEMORY.md"
    assert index_file.is_file()
    assert entry.id in index_file.read_text(encoding="utf-8")

    # transactional add: a failing commit rolls the file write back
    original_commit = store.git.commit

    def _boom(message: str, files: Any) -> str:
        raise RuntimeError("git backend down")

    store.git.commit = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="git backend down"):
            store.add("half written fact", type="Project")
    finally:
        store.git.commit = original_commit

    assert sorted(p.name for p in path.parent.glob("*.md")) == [f"{entry.id}.md"]
    assert "half written fact" not in index_file.read_text(encoding="utf-8")
    assert store.git.is_clean()

    # the store remains usable after the rollback
    second = store.add("half written fact", type="Project")
    assert second.id != entry.id
    assert {e.id for e in store.list()} == {entry.id, second.id}


@pytest.mark.unit
def test_get_returns_entry(tmp_path: Path) -> None:
    """get round-trips entries through the trust card; unknown ids are None."""
    store = _make_store(tmp_path)
    a = store.add("user prefers dark mode", type="User")
    b = store.add("deploys happen on fridays", type="Project", context="ops")

    got_a = store.get(a.id)
    assert isinstance(got_a, MemoryEntry)
    assert got_a == a  # full dataclass equality across the file roundtrip

    got_b = store.get(b.id)
    assert got_b is not None
    assert got_b == b
    assert got_b.context == "ops"

    assert store.get("2026-01-01-does-not-exist-00000000") is None


@pytest.mark.unit
def test_update_modifies_trust_score(tmp_path: Path) -> None:
    """update with trust_score appends a tool-output event and persists it."""
    store = _make_store(tmp_path)
    entry = store.add("initial claim", type="Project")
    assert entry.trust_score == 1.0  # single confirming agent-inferred event

    updated = store.update(entry.id, trust_score=0.4)
    assert updated.trust_score == 0.4
    assert [event.source for event in updated.verification_events] == [
        "agent-inferred",
        "tool-output",
    ]
    assert updated.verification_events[-1].value == 0.4
    assert updated.last_verified_at is not None

    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    card, _body = parse(path.read_text(encoding="utf-8"))
    assert card.trust_score == 0.4  # change persisted to the trust card

    assert [commit.message for commit in store.git.log(path=path)] == [
        f"[memory] update {entry.id}",
        f"[memory] add {entry.id} global",
    ]

    # text-only update: score and event count unchanged, still one commit
    before = len(updated.verification_events)
    revised = store.update(entry.id, text="revised claim")
    assert revised.text == "revised claim"
    assert revised.trust_score == 0.4
    assert len(revised.verification_events) == before
    assert store.git.log(path=path)[0].message == f"[memory] update {entry.id}"
    assert store.git.is_clean()

    with pytest.raises(KeyError):
        store.update("2026-01-01-missing-entry-ffffffff")


@pytest.mark.unit
def test_remove_deletes_file(tmp_path: Path) -> None:
    """remove deletes + commits the deletion; removing again is False."""
    store = _make_store(tmp_path)
    entry = store.add("temporary note", type="Reference")
    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    assert path.is_file()

    assert store.remove(entry.id) is True
    assert not path.exists()
    assert store.get(entry.id) is None

    messages = [commit.message for commit in store.git.log(path=path)]
    assert messages[0] == f"[memory] remove {entry.id}"  # deletion was committed
    assert store.git.is_clean()  # removal fully staged + committed

    index_text = (tmp_path / "memory-repo" / "MEMORY.md").read_text(encoding="utf-8")
    assert entry.id not in index_text

    assert store.remove(entry.id) is False  # already gone: nothing to remove


@pytest.mark.unit
def test_list_filters_by_context_type_tag(tmp_path: Path) -> None:
    """list honours context (None = all), type and tag filters."""
    store = _make_store(tmp_path)
    feedback = store.add("prefer terse replies", type="Feedback", tags=["style"])
    project = store.add("uses pygit2 for storage", type="Project", tags=["storage", "git"])
    work = store.add(
        "client meets on mondays", type="Project", context="work", tags=["calendar"]
    )

    assert {e.id for e in store.list()} == {feedback.id, project.id, work.id}
    assert {e.id for e in store.list(context="global")} == {feedback.id, project.id}
    assert all(e.context == "global" for e in store.list(context="global"))
    assert {e.id for e in store.list(context="work")} == {work.id}

    assert {e.id for e in store.list(type="Project")} == {project.id, work.id}
    assert {e.id for e in store.list(type="Feedback")} == {feedback.id}

    assert {e.id for e in store.list(tag="git")} == {project.id}
    assert {e.id for e in store.list(tag="style")} == {feedback.id}
    assert {e.id for e in store.list(tag="nonexistent")} == set()

    assert {e.id for e in store.list(context="work", type="Project")} == {work.id}
    assert {e.id for e in store.list(context="work", type="Feedback")} == set()
    assert {e.id for e in store.list(context="work", tag="git")} == set()


@pytest.mark.unit
def test_context_isolation(tmp_path: Path) -> None:
    """A global fact must NOT appear when listing a specific context."""
    store = _make_store(tmp_path)
    global_fact = store.add("user is allergic to peanuts", type="User")
    work_fact = store.add("team standup at 9am", type="Project", context="team-alpha")

    work_listing = store.list(context="team-alpha")
    assert [e.id for e in work_listing] == [work_fact.id]  # exactly the context's facts
    assert global_fact.id not in {e.id for e in work_listing}
    assert work_listing[0] == work_fact

    global_listing = store.list(context="global")
    assert [e.id for e in global_listing] == [global_fact.id]
    assert work_fact.id not in {e.id for e in global_listing}

    # the two contexts live in separate layout roots on disk
    root = tmp_path / "memory-repo"
    assert (root / "global" / "facts" / f"{global_fact.id}.md").is_file()
    assert (root / "contexts" / "team-alpha" / "facts" / f"{work_fact.id}.md").is_file()
    assert not (root / "global" / "facts" / f"{work_fact.id}.md").exists()

    # the index groups entries per context
    index_text = (root / "MEMORY.md").read_text(encoding="utf-8")
    assert "## global" in index_text
    assert "## team-alpha" in index_text

    # get() still resolves ids across contexts (global searched first)
    assert store.get(work_fact.id) == work_fact
    assert store.get(global_fact.id) == global_fact
