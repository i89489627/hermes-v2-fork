"""P7.T1 acceptance: MemoryHotReloader segmented system prompt (plan Phase 7).

Covered: selective invalidation (only the affected entry's segment is marked
dirty, with old/new sha256 receipts), the prefix-cache contract (non-
invalidated segments byte-identical across renders — even when the store
changed behind the reloader's back), and fresh reads (reload_segment goes
through ``store.get`` / the renderer, never the reference cache). One custom
fixed segment (``session_header``) rides alongside the per-entry segments.
The additive injector integration (:func:`hermes.memory.injector.
wire_hot_reload` -> ``invalidated_segments`` payload key) is exercised at the
end of the reload test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes.memory.entry import MemoryEntry
from hermes.memory.gating import CBDCGate
from hermes.memory.hot_reload import MemoryHotReloader
from hermes.memory.injector import ContextInjector, wire_hot_reload
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore

_ENTRY_A_TEXT = "project alpha uses PostgreSQL 16"
_ENTRY_A_UPDATED = "project alpha uses PostgreSQL 17"
_ENTRY_B_TEXT = "weekly sync notes live in notion"
_HEADER_TEXT = "SESSION: acme"


def _make_parts(tmp_path: Path) -> tuple[MemoryStore, MemoryHotReloader, MemoryEntry, MemoryEntry]:
    """Store + reloader with two entries and one fixed segment."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    entry_a = store.add(_ENTRY_A_TEXT, type="Project")
    entry_b = store.add(_ENTRY_B_TEXT, type="Reference")
    reloader = MemoryHotReloader(
        store, segments={"session_header": lambda _store: _HEADER_TEXT}
    )
    return store, reloader, entry_a, entry_b


def _wrapped(entry_id: str, text: str) -> str:
    """The segment bytes for one entry (stable markers, plan Phase 7)."""
    return f'<mem-segment id="{entry_id}">{text}</mem-segment>'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.unit
def test_invalidate_specific_segment(tmp_path: Path) -> None:
    """invalidate() marks ONLY the affected entry's segment dirty."""
    store, reloader, entry_a, entry_b = _make_parts(tmp_path)

    layout = reloader.segment_layout()
    assert set(layout) == {entry_a.id, entry_b.id, "session_header"}
    assert layout[entry_a.id] == _wrapped(entry_a.id, _ENTRY_A_TEXT)
    assert layout[entry_b.id] == _wrapped(entry_b.id, _ENTRY_B_TEXT)
    assert layout["session_header"] == _HEADER_TEXT

    store.update(entry_a.id, text=_ENTRY_A_UPDATED)
    receipt = reloader.invalidate(entry_a.id)

    assert receipt.segment_id == entry_a.id
    assert receipt.old_hash == _sha(layout[entry_a.id])  # pre-update rendered bytes
    assert receipt.new_hash == _sha(_wrapped(entry_a.id, _ENTRY_A_UPDATED))
    assert receipt.changed is True

    # only the affected segment is pending — entry B and the fixed segment
    # were never invalidated (prefix cache preserved for them)
    assert reloader.pending_invalidations() == [entry_a.id]

    # the re-render honors it: A fresh, everything else byte-identical
    rendered = reloader.render_prompt(invalidated={entry_a.id})
    assert rendered[entry_a.id] == _wrapped(entry_a.id, _ENTRY_A_UPDATED)
    assert rendered[entry_b.id] == layout[entry_b.id]
    assert rendered["session_header"] == layout["session_header"]

    # unknown segment ids are refused
    with pytest.raises(KeyError):
        reloader.invalidate("no-such-entry")

    # a segment with NO previous render reports old_hash="" (never rendered)
    fresh_reloader = MemoryHotReloader(
        store, segments={"session_header": lambda _store: _HEADER_TEXT}
    )
    unrendered = fresh_reloader.invalidate(entry_b.id)
    assert unrendered.old_hash == ""
    assert unrendered.new_hash == _sha(_wrapped(entry_b.id, _ENTRY_B_TEXT))


@pytest.mark.unit
def test_other_segments_unchanged(tmp_path: Path) -> None:
    """Non-invalidated segments are byte-identical across renders (P7.T1)."""
    store, reloader, entry_a, entry_b = _make_parts(tmp_path)
    layout = reloader.segment_layout()

    store.update(entry_a.id, text=_ENTRY_A_UPDATED)
    receipt = reloader.invalidate(entry_a.id)  # only A is marked dirty
    assert receipt.changed is True
    render_1 = reloader.render_prompt(invalidated={entry_a.id})
    assert render_1[entry_a.id] != layout[entry_a.id]  # the invalidated one moved

    # even while the store changes BEHIND the reloader's back, segments that
    # were not invalidated keep their exact rendered bytes (the caller re-
    # injects only invalidated segments — the prefix-cache contract)
    store.update(entry_b.id, text="weekly sync notes moved to obsidian")
    render_2 = reloader.render_prompt(invalidated={entry_a.id})

    for segment_id in (entry_b.id, "session_header"):
        assert render_1[segment_id] == render_2[segment_id] == layout[segment_id]
        assert json.dumps(render_1[segment_id]) == json.dumps(render_2[segment_id])
    assert render_2[entry_a.id] == _wrapped(entry_a.id, _ENTRY_A_UPDATED)
    assert reloader.pending_invalidations() == [entry_a.id]  # B was NOT invalidated

    # a full render (no invalidated set) picks up the store's current state
    full = reloader.render_prompt(None)
    assert full[entry_b.id] == _wrapped(entry_b.id, "weekly sync notes moved to obsidian")
    assert full[entry_a.id] == _wrapped(entry_a.id, _ENTRY_A_UPDATED)


@pytest.mark.unit
def test_reload_returns_updated_content(tmp_path: Path) -> None:
    """reload_segment reads FRESH store content, not the reference cache."""
    store, reloader, entry_a, _entry_b = _make_parts(tmp_path)
    layout = reloader.segment_layout()  # reference now holds the old bytes

    store.update(entry_a.id, text=_ENTRY_A_UPDATED)
    # fresh read straight from the store — the cache still holds the old text
    reloaded = reloader.reload_segment(entry_a.id)
    assert reloaded == _wrapped(entry_a.id, _ENTRY_A_UPDATED)

    # reload without a further update returns the same content
    assert reloader.reload_segment(entry_a.id) == reloaded

    # the pure fresh read left the reference untouched: invalidate still
    # compares against the PRE-update rendered bytes
    receipt = reloader.invalidate(entry_a.id)
    assert receipt.old_hash == _sha(layout[entry_a.id])
    assert receipt.new_hash == _sha(reloaded)

    # fixed segments reload through their renderer; unknown ids are refused
    assert reloader.reload_segment("session_header") == _HEADER_TEXT
    with pytest.raises(KeyError):
        reloader.reload_segment("no-such-entry")

    # additive injector integration (plan P7.T1): wire_hot_reload surfaces
    # pending invalidations to the payload for the agent harness
    turn = "an unrelated turn matching nothing stored at all"
    plain = ContextInjector(retrieval=RetrievalEngine(store), gate=CBDCGate(gate_model=None))
    payload = plain.build_dynamic_context(turn, "global")
    assert "invalidated_segments" not in payload  # additive: absent when unwired

    injector = ContextInjector(retrieval=RetrievalEngine(store), gate=CBDCGate(gate_model=None))
    reloader_wired = wire_hot_reload(injector, store)
    assert isinstance(reloader_wired, MemoryHotReloader)
    injector_payload = injector.build_dynamic_context(turn, "global")
    assert injector_payload["invalidated_segments"] == []  # nothing pending yet

    reloader_wired.invalidate(entry_a.id)
    injector_payload = injector.build_dynamic_context(turn, "global")
    assert injector_payload["invalidated_segments"] == [entry_a.id]

    reloader_wired.clear_invalidations()
    injector_payload = injector.build_dynamic_context(turn, "global")
    assert injector_payload["invalidated_segments"] == []
