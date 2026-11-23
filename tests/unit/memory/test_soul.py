"""P15.T1 acceptance: SoulManager — immutable core, bounded adaptive style.

Covered: the core never changes through style updates and has no write API
(immutability enforced by absence of ``update_core`` plus the Git-protected
note in the template; hooks use ``verify_core_integrity``), style updates
within the 2000-char bound (atomic write + history log + Git commit when the
repo is Git-backed), and over-limit updates raising ``StyleLimitError`` with
current vs limit while leaving file and history untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.memory.git_backend import GitBackend
from hermes.memory.soul import (
    SAFETY_CORE_TEMPLATE,
    SOUL_CORE_FILENAME,
    SOUL_STYLE_FILENAME,
    SOUL_STYLE_LIMIT,
    STYLE_HISTORY_FILENAME,
    SoulManager,
    StyleLimitError,
)


def _default_manager() -> tuple[SoulManager, Path]:
    """Manager on the default path (HERMES_HOME per-test tempdir, non-Git)."""
    mgr = SoulManager()
    return mgr, mgr.root / "soul"


@pytest.mark.unit
def test_core_immutable() -> None:
    """No core-write API exists; style updates leave core bytes untouched."""
    mgr, soul_dir = _default_manager()
    core_path = soul_dir / SOUL_CORE_FILENAME

    mgr.ensure_soul()
    assert mgr.get_core() == SAFETY_CORE_TEMPLATE  # bundled template materialized
    core_before = core_path.read_bytes()

    # the "git-protected note" is part of the immutability contract
    assert "IMMUTABLE" in SAFETY_CORE_TEMPLATE
    assert "Git-protected" in SAFETY_CORE_TEMPLATE

    mgr.update_style("plain, concise tone; no emoji")
    mgr.update_style("another style generation")
    assert core_path.read_bytes() == core_before  # core bytes unchanged by style writes

    # immutability enforced by ABSENCE of API: no update_core / write_core / set_core
    for forbidden in ("update_core", "write_core", "set_core"):
        assert not hasattr(SoulManager, forbidden)
        assert not hasattr(mgr, forbidden)

    # hook helper: integrity snapshot + tamper detection
    assert mgr.verify_core_integrity(hashlib.sha256(core_before).hexdigest())
    assert not mgr.verify_core_integrity("0" * 64)
    assert mgr.core_hash() == hashlib.sha256(core_path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_style_updatable_within_limit(tmp_path: Path) -> None:
    """Style updates at/within the bound succeed: write + history + Git commit."""
    mgr, soul_dir = _default_manager()
    style_path = soul_dir / SOUL_STYLE_FILENAME
    history_path = soul_dir / STYLE_HISTORY_FILENAME

    # exactly at the limit is allowed
    at_limit = "a" * SOUL_STYLE_LIMIT
    mgr.update_style(at_limit)
    assert mgr.get_style() == at_limit

    # a further update replaces the content and appends history
    mgr.update_style("short style")
    assert mgr.get_style() == "short style"
    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    timestamp, char_count = lines[-1].split("\t")
    assert char_count == str(len("short style"))
    assert timestamp.endswith("Z")  # ISO 8601 UTC

    # Git-backed root: updates become commits via GitBackend
    git_root = tmp_path / "git-soul-repo"
    GitBackend.init(git_root)
    git_mgr = SoulManager(repo_root=git_root)
    git_mgr.update_style("concise tone for git-backed repo")

    backend = GitBackend(git_root)
    commits = backend.log(path=f"soul/{SOUL_STYLE_FILENAME}")
    assert len(commits) >= 2  # baseline (created) + the update
    assert "[P15.T1]" in commits[0].message
    assert backend.is_clean()  # core, style and history are all tracked

    # the git-backed core was baselined too (manual-override workflow needs a baseline)
    assert git_mgr.get_core() == SAFETY_CORE_TEMPLATE
    assert len(GitBackend(git_root).log(path=f"soul/{SOUL_CORE_FILENAME}")) >= 1


@pytest.mark.unit
def test_style_over_limit_errors() -> None:
    """Over-limit updates raise StyleLimitError (current vs limit) and change nothing."""
    mgr, soul_dir = _default_manager()
    style_path = soul_dir / SOUL_STYLE_FILENAME
    history_path = soul_dir / STYLE_HISTORY_FILENAME

    mgr.update_style("kept content")
    history_before = history_path.read_text(encoding="utf-8")

    with pytest.raises(StyleLimitError) as excinfo:
        mgr.update_style("x" * (SOUL_STYLE_LIMIT + 1))

    message = str(excinfo.value)
    assert "2001" in message  # the attempted (current) length
    assert "2000" in message  # the limit
    assert message == (
        f"soul style update refused: {SOUL_STYLE_LIMIT + 1} chars exceeds the "
        f"{SOUL_STYLE_LIMIT}-char limit"
    )

    assert mgr.get_style() == "kept content"  # style file untouched
    assert history_path.read_text(encoding="utf-8") == history_before  # no history line
