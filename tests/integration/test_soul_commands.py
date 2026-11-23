"""P15.T4 acceptance: /soul slash command (plan Phase 15).

Drives the subcommands through ``hermes.core.agent.handle_soul_command``
over a real ``SoulManager`` (plan P15.T1): inspect shows BOTH the immutable
core and the adaptive style with char counts + limit; edit-style performs
the manual override (2000-char enforcement, stdin ``-`` path); history comes
from the git log of ``soul/soul_style.md`` in a Git-backed repo (>= 2
entries after two updates) and falls back to ``.style_history.log`` without
Git.

The ``agent`` argument of the handler is the reserved harness context and is
unused by the dispatch, so the tests pass ``None``.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

from hermes.core.agent import handle_soul_command
from hermes.memory.git_backend import GitBackend
from hermes.memory.soul import SOUL_STYLE_LIMIT, SoulManager


_SHA_LINE_RE = re.compile(r"\b[0-9a-f]{12}\b")


@pytest.mark.integration
def test_inspect_shows_core_and_style(tmp_path: Path) -> None:
    """inspect shows both layers with char counts and the 2000-char limit."""
    soul_manager = SoulManager(repo_root=tmp_path / "memory-repo")
    soul_manager.update_style("Prefer concise replies.")

    out = handle_soul_command(None, ["inspect"], soul_manager)

    # Core layer: present (bundled template) and labeled immutable.
    assert "soul core" in out
    assert "immutable" in out
    assert "Soul — Core" in out  # template content actually shown
    # Style layer: content + counts + limit.
    assert "soul style" in out
    assert "Prefer concise replies." in out
    assert f"/{SOUL_STYLE_LIMIT} chars" in out
    assert f"({len('Prefer concise replies.')}/{SOUL_STYLE_LIMIT} chars)" in out


@pytest.mark.integration
def test_edit_style_updates(tmp_path: Path) -> None:
    """edit-style replaces the style; over-limit text is refused untouched."""
    soul_manager = SoulManager(repo_root=tmp_path / "memory-repo")

    out = handle_soul_command(None, ["edit-style", "Keep", "answers", "short."], soul_manager)
    assert "updated" in out
    assert soul_manager.get_style() == "Keep answers short."

    # The '-' form reads the replacement from stdin.
    monkey_old = sys.stdin
    sys.stdin = io.StringIO("style from stdin\n")
    try:
        out = handle_soul_command(None, ["edit-style", "-"], soul_manager)
    finally:
        sys.stdin = monkey_old
    assert "updated" in out
    assert soul_manager.get_style() == "style from stdin"

    # Over-limit: refused with an error, style left untouched.
    oversized = "x" * (SOUL_STYLE_LIMIT + 1)
    out = handle_soul_command(None, ["edit-style", oversized], soul_manager)
    assert out.startswith("error:")
    assert soul_manager.get_style() == "style from stdin"


@pytest.mark.integration
def test_history_shows_changes(tmp_path: Path) -> None:
    """Two style updates in a git-backed repo yield >= 2 history entries."""
    root = tmp_path / "memory-repo"
    GitBackend.init(root)
    soul_manager = SoulManager(repo_root=root)
    soul_manager.get_core()  # materializes soul + baseline commit (git-backed)

    soul_manager.update_style("Style draft one.")
    soul_manager.update_style("Style draft two, longer.")

    out = handle_soul_command(None, ["history"], soul_manager)
    assert "git log" in out
    shas = _SHA_LINE_RE.findall(out)
    assert len(shas) >= 2  # both updates are distinct commits
    assert "update soul style" in out  # SoulManager's commit message

    # Non-Git fallback: the .style_history.log content is shown instead.
    plain = SoulManager(repo_root=tmp_path / "plain-repo")
    plain.update_style("First style.")
    plain.update_style("Second style.")
    out = handle_soul_command(None, ["history"], plain)
    assert ".style_history.log" in out
    assert "chars" in out
    assert out.count("chars") >= 2  # one line per update
