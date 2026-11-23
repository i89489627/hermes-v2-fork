"""P5.T1 acceptance: ContextManager detects/switches/creates/lists contexts.

Covered: cwd-based detection via ``cwd_rules`` (segment-boundary suffix
matching, global fallback), gateway chat-id detection via ``chat_rules``,
switching with a fresh stable-context snapshot rebuilt from disk (session
boundary semantics), parented context creation with slug validation, and
``list_contexts`` always including ``global``. Layout aligns with
``hermes.memory.store.MemoryStore``: ``<root>/global``, ``<root>/contexts/<name>/``,
``<root>/MEMORY.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hermes.memory.context import (
    GLOBAL_CONTEXT,
    METADATA_FILENAME,
    ContextManager,
    StableContext,
    UnknownContextError,
)
from hermes.memory.types import Context


def _write_metadata(root: Path, doc: dict[str, Any]) -> None:
    """Test setup: seed contexts.json (rules tables live there by design)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / METADATA_FILENAME).write_text(json.dumps(doc), encoding="utf-8")


def _default_manager(tmp_path: Path) -> tuple[ContextManager, Path]:
    root = tmp_path / "memory-repo"
    return ContextManager(repo_root=root), root


@pytest.mark.unit
def test_detect_by_cwd(tmp_path: Path) -> None:
    """cwd detection maps through cwd_rules suffixes; non-matches fall back to global."""
    mgr, root = _default_manager(tmp_path)
    mgr.create_context("acme")
    _write_metadata(root, {"cwd_rules": [["/work/acme", "acme"]]})

    # POSIX-style cwd matching the rule suffix on a segment boundary
    assert mgr.detect_context({"cwd": "/home/user/work/acme"}) == "acme"
    # Windows-style separators normalize to the same match
    assert mgr.detect_context({"cwd": "C:\\Users\\dev\\work\\acme"}) == "acme"
    # partial-segment lookalikes must NOT match ("mywork" != "work")
    assert mgr.detect_context({"cwd": "/home/user/mywork/acme"}) == GLOBAL_CONTEXT
    # no matching rule -> global
    assert mgr.detect_context({"cwd": "/home/user/other"}) == GLOBAL_CONTEXT
    # no metadata at all -> global
    assert ContextManager(repo_root=tmp_path / "empty").detect_context(
        {"cwd": "/home/user/work/acme"}
    ) == GLOBAL_CONTEXT
    # empty metadata -> global
    assert mgr.detect_context({}) == GLOBAL_CONTEXT


@pytest.mark.unit
def test_detect_by_gateway_chat_id(tmp_path: Path) -> None:
    """gateway_chat_id maps through chat_rules (string keys, int ids ok); else global."""
    mgr, root = _default_manager(tmp_path)
    mgr.create_context("support")
    _write_metadata(root, {"chat_rules": {"12345": "support"}})

    assert mgr.detect_context({"gateway_chat_id": 12345}) == "support"
    assert mgr.detect_context({"gateway_chat_id": "12345"}) == "support"  # str key
    assert mgr.detect_context({"gateway_chat_id": "99999"}) == GLOBAL_CONTEXT  # unknown chat

    # a rule pointing at a deleted context degrades to global (detection never breaks)
    (root / "contexts" / "support").rmdir()
    assert mgr.detect_context({"gateway_chat_id": 12345}) == GLOBAL_CONTEXT


@pytest.mark.unit
def test_switch_reloads_stable_context(tmp_path: Path) -> None:
    """Switching is a session boundary: fresh snapshot objects, current disk state."""
    mgr, root = _default_manager(tmp_path)
    mgr.create_context("work")

    ctx1 = mgr.switch_context("work", session_id="session-1")
    assert isinstance(ctx1, Context)
    assert ctx1.name == "work"
    assert ctx1.parent is None
    assert ctx1.path == root / "contexts" / "work"
    assert ctx1.active_sessions == ["session-1"]

    snap1 = mgr.stable_context_snapshot("work")
    assert isinstance(snap1, StableContext)
    assert snap1.context == "work"
    assert snap1.global_memory_md is None  # nothing written yet
    assert snap1.context_memory_md is None
    assert snap1.global_memory_md_path == root / "MEMORY.md"
    assert snap1.context_memory_md_path == root / "contexts" / "work" / "MEMORY.md"

    # disk changes after the first snapshot...
    (root / "MEMORY.md").write_text("# global index", encoding="utf-8")
    (root / "contexts" / "work" / "MEMORY.md").write_text("# work facts v1", encoding="utf-8")

    # ...are only visible in a REBUILT snapshot: switching reloads stable context
    ctx2 = mgr.switch_context("work", session_id="session-2")
    assert ctx2 is not ctx1  # new snapshot object per call
    assert ctx2.active_sessions == ["session-1", "session-2"]  # appended, persisted
    snap2 = mgr.stable_context_snapshot("work")
    assert snap2 is not snap1
    assert snap2.context_memory_md == "# work facts v1"
    assert snap2.global_memory_md == "# global index"

    # the same switch again: still a fresh object reflecting current disk state
    (root / "contexts" / "work" / "MEMORY.md").write_text("# work facts v2", encoding="utf-8")
    snap3 = mgr.stable_context_snapshot("work")
    assert snap3 is not snap2
    assert snap3.context_memory_md == "# work facts v2"

    # session recording is idempotent; unknown contexts raise
    mgr.switch_context("work", session_id="session-2")
    assert len(mgr.switch_context("work").active_sessions) == 2
    with pytest.raises(UnknownContextError):
        mgr.switch_context("missing")


@pytest.mark.unit
def test_create_with_parent(tmp_path: Path) -> None:
    """create_context: slug validation, existing parent required, metadata recorded."""
    mgr, root = _default_manager(tmp_path)

    base = mgr.create_context("base")
    assert base.path.is_dir()
    assert base.parent is None

    child = mgr.create_context("child", parent="base")
    assert child.parent == "base"  # inheritance chain: child -> base -> global facts
    assert child.path == root / "contexts" / "child"
    assert child.active_sessions == []
    assert (root / "contexts" / "child").is_dir()

    # "global" is always an existing parent (contexts inherit global facts)
    assert mgr.create_context("proj", parent="global").parent == "global"

    with pytest.raises(UnknownContextError):
        mgr.create_context("orphan", parent="missing")
    with pytest.raises(ValueError, match="slug"):
        mgr.create_context("Bad_Name")
    with pytest.raises(ValueError, match="slug"):
        mgr.create_context("has space")
    with pytest.raises(ValueError, match="already exists"):
        mgr.create_context("base")
    with pytest.raises(ValueError, match="implicit root"):
        mgr.create_context("global")

    # duplicate detection also covers hand-created directories (dir presence = existence)
    (root / "contexts" / "manual").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        mgr.create_context("manual")


@pytest.mark.unit
def test_list_returns_all(tmp_path: Path) -> None:
    """list_contexts always includes global plus every context directory, sorted."""
    mgr, root = _default_manager(tmp_path)

    assert [ctx.name for ctx in mgr.list_contexts()] == [GLOBAL_CONTEXT]  # always present

    mgr.create_context("beta")
    mgr.create_context("alpha")
    mgr.create_context("child-ctx", parent="alpha")

    contexts = mgr.list_contexts()
    assert [ctx.name for ctx in contexts] == ["alpha", "beta", "child-ctx", "global"]

    by_name = {ctx.name: ctx for ctx in contexts}
    assert by_name["global"].parent is None
    assert by_name["global"].path == root / "global"  # store layout alignment
    assert by_name["child-ctx"].parent == "alpha"

    # hand-created directories (no metadata entry) are listed too
    (root / "contexts" / "manual").mkdir()
    assert "manual" in {ctx.name for ctx in mgr.list_contexts()}
    assert [ctx.name for ctx in mgr.list_contexts()] == sorted(
        ctx.name for ctx in mgr.list_contexts()
    )
