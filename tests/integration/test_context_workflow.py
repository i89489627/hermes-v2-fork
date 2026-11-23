"""P5.T2 acceptance: /context slash command workflow (plan Phase 5).

Drives the subcommands through ``hermes.core.agent.handle_context_command``
over ``hermes.memory.context.ContextManager`` (create -> switch -> inspect
-> list -> delete), then exercises the plan's fact-isolation and
global-inheritance acceptance criteria with a MemoryStore sharing the same
repo root, using ``hermes.core.agent.list_effective_entries`` as the merged
view (context's own facts + inherited global facts, nearest copy wins).

The ``agent`` argument of the handler is the reserved harness context and is
unused by the dispatch, so the tests pass ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.core.agent import handle_context_command, list_effective_entries
from hermes.memory.context import ContextManager
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse, serialize


def _repo(tmp_path: Path) -> tuple[ContextManager, MemoryStore]:
    """ContextManager + MemoryStore over the same per-test repo root."""
    root = tmp_path / "memory-repo"
    return ContextManager(repo_root=root), MemoryStore(root=root)


@pytest.mark.integration
def test_user_creates_switches_contexts(tmp_path: Path) -> None:
    """create -> switch -> inspect -> list -> delete, all via the command."""
    context_manager, _store = _repo(tmp_path)

    # list on a fresh repo: only the implicit global root.
    out = handle_context_command(None, ["list"], context_manager)
    assert "global" in out

    # create <name>: the named context exists on disk afterwards.
    out = handle_context_command(None, ["create", "work"], context_manager)
    assert "work" in out and "created" in out
    assert context_manager.context_exists("work")

    # create rejects duplicates and bad slugs with an error line.
    out = handle_context_command(None, ["create", "work"], context_manager)
    assert out.startswith("error:")
    out = handle_context_command(None, ["create", "Bad Slug"], context_manager)
    assert out.startswith("error:")

    # switch <name>: explicit detection wins; unknown names error out.
    out = handle_context_command(None, ["switch", "work"], context_manager)
    assert "switched" in out and "work" in out
    out = handle_context_command(None, ["switch", "nope"], context_manager)
    assert out.startswith("error:") and "nope" in out

    # inspect [name]: details (default global).
    out = handle_context_command(None, ["inspect", "work"], context_manager)
    assert "work" in out and "parent" in out and "facts" in out
    out = handle_context_command(None, ["inspect"], context_manager)
    assert "global" in out and "facts" in out

    # list shows both contexts with the parent link.
    out = handle_context_command(None, ["list"], context_manager)
    assert "global" in out and "work" in out and "parent=global" in out

    # delete guard: a context with active sessions is refused without --force.
    context_manager.record_session("work", "sess-1")
    out = handle_context_command(None, ["delete", "work"], context_manager)
    assert out.startswith("error:") and "--force" in out
    assert context_manager.context_exists("work")

    # --force removes the directory AND the metadata entry.
    out = handle_context_command(None, ["delete", "work", "--force"], context_manager)
    assert "deleted" in out
    assert not context_manager.context_exists("work")

    # The implicit global root can never be deleted.
    out = handle_context_command(None, ["delete", "global"], context_manager)
    assert out.startswith("error:")


@pytest.mark.integration
def test_facts_isolated_per_context(tmp_path: Path) -> None:
    """A work fact never appears in the personal context (plan Phase 5)."""
    context_manager, store = _repo(tmp_path)
    assert handle_context_command(None, ["create", "work"], context_manager)
    assert handle_context_command(None, ["create", "personal"], context_manager)

    work_fact = store.add("ship the v2 CBDC gate", type="Project", context="work")
    personal_fact = store.add("book club meets on thursdays", type="User", context="personal")

    # Raw per-context listing: no bleeding in either direction.
    work_entries = {entry.id for entry in store.list(context="work")}
    personal_entries = {entry.id for entry in store.list(context="personal")}
    assert work_fact.id in work_entries and personal_fact.id not in work_entries
    assert personal_fact.id in personal_entries and work_fact.id not in personal_entries

    # The merged (effective) view respects the isolation too.
    effective_personal = {
        entry.id for entry in list_effective_entries(store, "personal", context_manager)
    }
    assert work_fact.id not in effective_personal
    assert personal_fact.id in effective_personal

    # inspect reports the per-context fact counts (1 fact file each).
    out = handle_context_command(None, ["inspect", "work"], context_manager)
    assert "facts: 1" in out


@pytest.mark.integration
def test_context_inherits_global_facts(tmp_path: Path) -> None:
    """A context's effective view includes global facts unless overridden."""
    context_manager, store = _repo(tmp_path)
    handle_context_command(None, ["create", "work"], context_manager)

    global_fact = store.add("deploy target is the eu cluster", type="Project")
    work_fact = store.add("work sprint ends friday", type="Project", context="work")

    effective = list_effective_entries(store, "work", context_manager)
    ids = [entry.id for entry in effective]
    assert work_fact.id in ids and global_fact.id in ids
    # Own facts first, inherited global facts appended (nearest-first order).
    assert ids.index(work_fact.id) < ids.index(global_fact.id)

    # Inheritance walks the parent chain: child of work still sees global.
    handle_context_command(None, ["create", "project-x", "--parent", "work"], context_manager)
    child_fact = store.add("project-x uses the staging cluster", type="Project", context="project-x")
    child_effective_ids = [
        entry.id for entry in list_effective_entries(store, "project-x", context_manager)
    ]
    assert child_fact.id in child_effective_ids
    assert work_fact.id in child_effective_ids  # inherited from the parent
    assert global_fact.id in child_effective_ids  # chain reaches global

    # "unless overridden": a same-id copy under the context wins over global.
    global_file = store.root / "global" / "facts" / f"{global_fact.id}.md"
    card, _body = parse(global_file.read_text(encoding="utf-8"))
    override_dir = store.root / "contexts" / "work" / "facts"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / f"{global_fact.id}.md").write_text(
        serialize(card, "deploy target is the us cluster"), encoding="utf-8"
    )
    overridden = [entry for entry in list_effective_entries(store, "work", context_manager) if entry.id == global_fact.id]
    assert len(overridden) == 1
    assert overridden[0].text == "deploy target is the us cluster"
