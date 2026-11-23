"""P15.T3 acceptance: /memory slash command (plan Phase 15).

Drives the subcommands through ``hermes.core.agent.handle_memory_command``
over a real git-backed ``MemoryStore``: list (per-context table), search
(BM25, plan P1.T4), stats (counts + trust distribution), export/import
round-trip (full trust cards preserved — ids, trust scores, tags, the whole
verification history) and conflict handling on import (skip keeps the
original, overwrite replaces).

The ``agent`` argument of the handler is the reserved harness context and is
unused by the dispatch, so the tests pass ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.core.agent import handle_memory_command
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse, serialize
from hermes.memory.verification import add_verification_event


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


@pytest.mark.integration
def test_list_returns_entries(tmp_path: Path) -> None:
    """list shows every entry (id/type/context/trust); [context] filters."""
    store = _make_store(tmp_path)
    alpha = store.add("alpha fact about the build", type="Project")
    beta = store.add("beta fact about the user", type="User", context="work")

    out = handle_memory_command(None, ["list"], store)
    assert alpha.id in out and beta.id in out
    assert "all contexts" in out
    assert "ctx=global" in out and "ctx=work" in out

    out = handle_memory_command(None, ["list", "work"], store)
    assert beta.id in out
    assert alpha.id not in out
    assert "work" in out

    # Empty context renders the (no entries) placeholder, not an exception.
    out = handle_memory_command(None, ["list", "personal"], store)
    assert "(no entries)" in out


@pytest.mark.integration
def test_search_returns_relevant(tmp_path: Path) -> None:
    """search ranks the entry sharing query tokens first; misses are clean.

    A filler third entry keeps the corpus above BM25Okapi's degenerate
    two-document regime (with N=2 a term present in exactly one document
    scores IDF 0 and can never match).
    """
    store = _make_store(tmp_path)
    pnpm = store.add("the team uses pnpm for workspace installs", type="Project")
    unrelated = store.add("the cat sleeps on the keyboard", type="User")
    store.add("weekly sync notes live in the calendar", type="Project")

    out = handle_memory_command(None, ["search", "pnpm"], store)
    assert pnpm.id in out
    assert unrelated.id not in out
    assert "1 match" in out

    out = handle_memory_command(None, ["search", "keyboard"], store)
    assert unrelated.id in out

    out = handle_memory_command(None, ["search", "zzz-no-hit"], store)
    assert "no entries match" in out


@pytest.mark.integration
def test_stats_shows_distribution(tmp_path: Path) -> None:
    """stats reports totals, per-type/context counts and trust distribution."""
    store = _make_store(tmp_path)
    low = store.add("stale claim awaiting re-verification", type="Project")
    store.add("user prefers short replies", type="User", context="work")

    # Drive one entry's trust below 0.3: two negative tool-output events.
    add_verification_event(store, low.id, source="tool-output", value=0.0, verifier="test")
    add_verification_event(store, low.id, source="tool-output", value=0.0, verifier="test")
    # Score now: (0.5*1.0) / (0.5 + 0.85 + 0.85) = 0.227.
    refreshed = store.get(low.id)
    assert refreshed is not None and refreshed.trust_score < 0.3

    out = handle_memory_command(None, ["stats"], store)
    assert "total: 2" in out
    assert "Project=1" in out and "User=1" in out
    assert "global=1" in out and "work=1" in out
    assert "below_0.3=1" in out
    assert f"min={refreshed.trust_score:.2f}" in out


@pytest.mark.integration
def test_export_import_roundtrip(tmp_path: Path) -> None:
    """export -> wipe -> import restores ids, texts and full trust cards."""
    store = _make_store(tmp_path)
    a = store.add("alpha fact about the build", type="Project", tags=["build"])
    b = store.add("beta fact about the user", type="User", context="work")
    add_verification_event(store, a.id, source="test-green", verifier="test-hook")
    originals = {entry.id: entry for entry in store.list()}

    export_path = tmp_path / "memory-export.json"
    out = handle_memory_command(None, ["export", str(export_path)], store)
    assert "exported 2 entries" in out
    assert export_path.is_file()
    doc = json.loads(export_path.read_text(encoding="utf-8"))
    assert doc["entry_count"] == 2
    assert len(doc["entries"]) == 2

    # Wipe the store, then import the export back.
    for entry_id in (a.id, b.id):
        assert store.remove(entry_id)
    assert store.list() == []

    out = handle_memory_command(None, ["import", str(export_path)], store)
    assert "added=2" in out and "skipped=0" in out and "overwritten=0" in out

    restored = {entry.id: entry for entry in store.list()}
    assert set(restored) == set(originals)
    for entry_id, original in originals.items():
        clone = restored[entry_id]
        assert clone.text == original.text
        assert clone.type == original.type
        assert clone.context == original.context
        assert clone.tags == original.tags
        assert clone.trust_score == original.trust_score
        assert [(event.source, event.value, event.evidence) for event in clone.verification_events] == [
            (event.source, event.value, event.evidence)
            for event in original.verification_events
        ]


@pytest.mark.integration
def test_import_handles_conflicts(tmp_path: Path) -> None:
    """Re-importing the same id: default skip keeps the original, --overwrite replaces."""
    store = _make_store(tmp_path)
    original = store.add("original text of the fact", type="Project")

    export_path = tmp_path / "conflict-export.json"
    handle_memory_command(None, ["export", str(export_path)], store)

    # Rewrite the export so the SAME id carries a different text.
    doc = json.loads(export_path.read_text(encoding="utf-8"))
    card, _body = parse(doc["entries"][0]["markdown"])
    assert card.id == original.id
    doc["entries"][0]["markdown"] = serialize(card, "rewritten text of the fact")
    export_path.write_text(json.dumps(doc), encoding="utf-8")

    # Default policy: skip — the original text survives untouched.
    out = handle_memory_command(None, ["import", str(export_path)], store)
    assert "skipped=1" in out and "added=0" in out and "overwritten=0" in out
    kept = store.get(original.id)
    assert kept is not None
    assert kept.text == "original text of the fact"
    assert len(kept.verification_events) == len(original.verification_events)

    # --overwrite: the entry is replaced with the imported text.
    out = handle_memory_command(None, ["import", str(export_path), "--overwrite"], store)
    assert "overwritten=1" in out and "added=0" in out and "skipped=0" in out
    replaced = store.get(original.id)
    assert replaced is not None
    assert replaced.text == "rewritten text of the fact"
    # The id is still unique in the store (no duplicate copy appeared).
    assert [entry.id for entry in store.list()].count(original.id) == 1
