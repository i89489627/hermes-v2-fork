"""P4.T3 acceptance: ContextInjector stable/dynamic split (plan Phase 4).

Covered: stable context loaded ONCE at session start via ContextManager
detection + snapshot (cwd rules -> context name, global/context MEMORY.md
documents), per-turn dynamic context (retrieve -> gate -> compose claims /
needs_verification / budget accounting / turn hash), and the prefix-cache
preservation contract — the stable payload is never re-read by the dynamic
flow, so the stable block bytes are identical across turns while the dynamic
part changes per turn. Retrieval corpora are seeded with filler entries so
BM25 idf stays positive for the target terms (a term hitting the only
document of a 1-2 entry corpus scores 0 and is dropped by the P4.T1 engine).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hermes.memory.context import (
    METADATA_FILENAME,
    ContextManager,
    StableContext,
)
from hermes.memory.gating import CBDCGate
from hermes.memory.injector import ContextInjector, StableContextPayload
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.store import MemoryStore

_DEPLOY_FACT = "acme deploy pipeline runs via GitHub Actions"
_THEME_FACT = "editor theme dark mode high contrast"
#: Filler corpus so the target terms keep positive BM25 idf (see module docstring).
_FILLER_FACTS = (
    "quarterly budget review spreadsheet numbers",
    "gardening log tomatoes basil balcony",
    "weekly team sync offsite notes",
    "grandma apple pie recipe cinnamon nutmeg",
)


class _ScriptedGateModel:
    """Fake gate model: scripted decision per candidate id found in the prompt."""

    def __init__(self, decisions: dict[str, str]) -> None:
        self._decisions = decisions
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        # the gate prompt embeds its own instruction example ("<id>") — skip it
        found = [
            candidate_id
            for candidate_id in re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if candidate_id != "<id>"
        ]
        return json.dumps([
            {
                "candidate_id": candidate_id,
                "decision": self._decisions.get(candidate_id, "ignore"),
                "reason": "scripted decision",
            }
            for candidate_id in found
        ])


def _make_parts(
    tmp_path: Path,
) -> tuple[MemoryStore, ContextManager]:
    """Store + context manager sharing the tmp-rooted memory repo."""
    root = tmp_path / "memory-repo"
    return MemoryStore(root=root), ContextManager(repo_root=root)


def _make_injector(
    store: MemoryStore,
    manager: ContextManager,
    decisions: dict[str, str],
) -> tuple[ContextInjector, _ScriptedGateModel]:
    """Injector over the given store/manager with a scripted gate model."""
    gate_model = _ScriptedGateModel(decisions)
    injector = ContextInjector(
        retrieval=RetrievalEngine(store),
        gate=CBDCGate(gate_model=gate_model),
        context_manager=manager,
    )
    return injector, gate_model


@pytest.mark.unit
def test_stable_context_loaded_at_session_start(tmp_path: Path) -> None:
    """Session start: detect (cwd rules) + snapshot global/context MEMORY.md."""
    _store, manager = _make_parts(tmp_path)
    injector, _gate = _make_injector(_store, manager, {})
    manager.create_context("acme")
    (tmp_path / "memory-repo" / METADATA_FILENAME).write_text(
        json.dumps({"cwd_rules": [["/work/acme", "acme"]]}), encoding="utf-8"
    )
    (tmp_path / "memory-repo" / "MEMORY.md").write_text(
        "global index: user profile facts", encoding="utf-8"
    )
    (tmp_path / "memory-repo" / "contexts" / "acme" / "MEMORY.md").write_text(
        "acme index: deploy pipeline facts", encoding="utf-8"
    )

    stable = injector.load_stable_context({"cwd": "/home/user/work/acme"})

    assert stable.context_name == "acme"
    assert stable.global_index == "global index: user profile facts"
    assert stable.context_index == "acme index: deploy pipeline facts"
    assert stable.loaded_at.tzinfo is UTC
    assert (datetime.now(UTC) - stable.loaded_at).total_seconds() < 60

    # the block derivation is deterministic and JSON-safe
    block = stable.to_block()
    assert block == {
        "context": "acme",
        "global_index": "global index: user profile facts",
        "context_index": "acme index: deploy pipeline facts",
        "loaded_at": stable.loaded_at.isoformat(),
    }

    # no metadata match -> global fallback: context index missing, global read
    fallback = injector.load_stable_context({})
    assert fallback.context_name == "global"
    assert fallback.global_index == "global index: user profile facts"
    assert fallback.context_index is None


@pytest.mark.unit
def test_dynamic_context_changes_per_turn(tmp_path: Path) -> None:
    """Per-turn retrieval+gating: claims/verification/budget follow the turn."""
    store, manager = _make_parts(tmp_path)
    deploy_id = store.add(_DEPLOY_FACT, type="Project").id
    theme_id = store.add(_THEME_FACT, type="User").id
    for filler in _FILLER_FACTS:
        store.add(filler, type="Reference")

    # disjoint vocabularies -> each turn retrieves exactly its own entry
    injector, gate_model = _make_injector(
        store, manager, {deploy_id: "use", theme_id: "verify"}
    )

    turn_a = "acme deploy pipeline"
    turn_b = "editor theme dark mode"
    injected_a = injector.build_dynamic_context(turn_a, "global")
    injected_b = injector.build_dynamic_context(turn_b, "global")

    # turn hash identifies the turn deterministically (plan output shape)
    assert injected_a["turn"] == hashlib.sha256(turn_a.encode("utf-8")).hexdigest()
    assert injected_b["turn"] == hashlib.sha256(turn_b.encode("utf-8")).hexdigest()
    assert injected_a["turn"] != injected_b["turn"]

    dynamic_a: dict[str, Any] = injected_a["dynamic"]
    assert [claim["candidate_id"] for claim in dynamic_a["claims"]] == [deploy_id]
    assert dynamic_a["needs_verification"] == []
    assert dynamic_a["budget_used"] == {"claims": 1, "chars": len(_DEPLOY_FACT)}
    assert dynamic_a["fallback_used"] is False
    assert 0.0 <= dynamic_a["claims"][0]["trust"] <= 1.0

    # verify entries are injected flagged, consume no claim, but count chars
    dynamic_b: dict[str, Any] = injected_b["dynamic"]
    assert dynamic_b["claims"] == []
    verification = dynamic_b["needs_verification"]
    assert [item["candidate_id"] for item in verification] == [theme_id]
    assert dynamic_b["budget_used"] == {"claims": 0, "chars": len(_THEME_FACT)}
    assert dynamic_b["fallback_used"] is False

    # gate made one batched call per turn
    assert gate_model.calls == 2


@pytest.mark.unit
def test_stable_context_preserves_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different turns, one session: stable bytes identical, dynamic differs."""
    store, manager = _make_parts(tmp_path)
    deploy_entry = store.add(_DEPLOY_FACT, type="Project")  # corpus for both turns
    for filler in _FILLER_FACTS:
        store.add(filler, type="Reference")
    manager.create_context("acme")
    (tmp_path / "memory-repo" / "MEMORY.md").write_text(
        "global index: user profile facts", encoding="utf-8"
    )
    (tmp_path / "memory-repo" / "contexts" / "acme" / "MEMORY.md").write_text(
        "acme index: deploy pipeline facts", encoding="utf-8"
    )
    injector, _gate = _make_injector(store, manager, {deploy_entry.id: "use"})

    # spy: the stable snapshot must be read EXACTLY once (at session start)
    snapshot_calls: list[str] = []
    original_snapshot = manager.stable_context_snapshot

    def _spy(name: str) -> StableContext:
        snapshot_calls.append(name)
        return original_snapshot(name)

    monkeypatch.setattr(manager, "stable_context_snapshot", _spy)

    stable = injector.load_stable_context({"explicit": "acme"})
    assert snapshot_calls == ["acme"]

    injected_1 = injector.build_dynamic_context("acme deploy pipeline", "acme", stable)
    injected_2 = injector.build_dynamic_context("editor theme dark mode", "acme", stable)

    # stable block identical (object content AND serialized bytes) across turns
    assert injected_1["stable"] == injected_2["stable"] == stable.to_block()
    assert json.dumps(injected_1["stable"], sort_keys=True) == json.dumps(
        injected_2["stable"], sort_keys=True
    )
    # the block carries the session-start stamp, not a mid-session re-load
    assert injected_1["stable"]["loaded_at"] == stable.loaded_at.isoformat()
    assert snapshot_calls == ["acme"]  # the dynamic flow never re-read it

    # the payload object itself is untouched (frozen session-start snapshot)
    assert isinstance(stable, StableContextPayload)
    assert stable.context_name == "acme"
    assert stable.global_index == "global index: user profile facts"
    assert stable.context_index == "acme index: deploy pipeline facts"

    # ...while the dynamic part changes per turn
    assert injected_1["dynamic"] != injected_2["dynamic"]
    assert injected_1["turn"] != injected_2["turn"]


def _entry_id(store: MemoryStore, text: str) -> str:
    """The store entry id for ``text`` (single-entry lookup for the tests)."""
    matches = [entry.id for entry in store.list() if entry.text == text]
    assert len(matches) == 1, f"expected exactly one entry for {text!r}, got {matches}"
    return matches[0]
