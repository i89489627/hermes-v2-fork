"""P1.T2 acceptance: GitBackend — every write commits; branch/merge; clean state.

Repos are tmp_path-based (hermetic); pygit2 is used directly for both the
backend under test and the cross-checks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pygit2
import pytest

from hermes.memory.git_backend import GitBackend


def _write(root: Path, rel: str, content: str) -> Path:
    """Write ``content`` to ``root/rel`` (creating parent directories)."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


@pytest.mark.unit
def test_init_creates_repo(tmp_path: Path) -> None:
    """init creates the repository and stamps a local commit identity."""
    repo_path = tmp_path / "memory"
    backend = GitBackend.init(repo_path)
    assert (repo_path / ".git").is_dir()
    assert backend.repo_path == repo_path
    assert backend.current_branch is None  # unborn until the first commit
    assert backend.is_clean()  # empty repository is clean

    repo = pygit2.Repository(str(repo_path))
    assert repo.config["user.name"] == "hermes-v2"
    assert repo.config["user.email"] == "hermes-v2@local"


@pytest.mark.unit
def test_commit_returns_sha(tmp_path: Path) -> None:
    """commit stages the listed files and returns the hex sha of a new commit."""
    backend = GitBackend.init(tmp_path / "memory")
    first = _write(backend.repo_path, "entries/a.md", "v1")
    sha1 = backend.commit("add a", [first])  # absolute path form
    assert re.fullmatch(r"[0-9a-f]{40}", sha1)

    _write(backend.repo_path, "entries/a.md", "v2")
    sha2 = backend.commit("update a", ["entries/a.md"])  # repo-relative path form
    assert sha2 != sha1
    assert len(backend.log()) == 2  # every write created one Git commit

    repo = pygit2.Repository(str(backend.repo_path))
    head = repo[repo.head.target]
    assert head.message.strip() == "update a"
    assert backend.is_clean()


@pytest.mark.unit
def test_log_returns_history(tmp_path: Path) -> None:
    """log returns newest-first history, honours limit and path filters."""
    backend = GitBackend.init(tmp_path / "memory")
    _write(backend.repo_path, "a.md", "1")
    sha_a1 = backend.commit("a v1", ["a.md"])
    _write(backend.repo_path, "b.md", "1")
    sha_b1 = backend.commit("b v1", ["b.md"])
    _write(backend.repo_path, "a.md", "2")
    sha_a2 = backend.commit("a v2", ["a.md"])

    entries = backend.log()
    assert [e.sha for e in entries] == [sha_a2, sha_b1, sha_a1]
    assert [e.message for e in entries] == ["a v2", "b v1", "a v1"]
    assert all(e.timestamp.tzinfo is not None for e in entries)  # aware UTC

    assert [e.sha for e in backend.log(limit=2)] == [sha_a2, sha_b1]
    assert [e.sha for e in backend.log(path="a.md")] == [sha_a2, sha_a1]
    assert backend.log(path="b.md")[-1].sha == sha_b1

    # Nested paths (absolute on disk) are accepted and normalized.
    nested = _write(backend.repo_path, "sub/c.md", "1")
    sha_c = backend.commit("c v1", [nested])
    assert [e.sha for e in backend.log(path="sub/c.md")] == [sha_c]


@pytest.mark.unit
def test_branch_and_merge(tmp_path: Path) -> None:
    """branch creates+switches; merge fast-forwards cleanly."""
    backend = GitBackend.init(tmp_path / "memory")
    _write(backend.repo_path, "base.md", "base")
    backend.commit("initial", ["base.md"])
    default_branch = backend.current_branch
    assert default_branch is not None

    backend.branch("feature")
    assert backend.current_branch == "feature"
    _write(backend.repo_path, "feature.md", "from feature")
    feat_sha = backend.commit("feature work", ["feature.md"])

    backend.branch(default_branch)  # switch back to the default branch
    assert backend.current_branch == default_branch
    assert not (backend.repo_path / "feature.md").exists()  # workdir switched back

    result = backend.merge("feature")
    assert result.status == "fast-forward"
    assert backend.current_branch == default_branch  # HEAD never moved away
    assert (backend.repo_path / "feature.md").exists()  # merged content restored
    assert backend.is_clean()
    assert backend.log()[0].sha == feat_sha  # ff moved the branch to the tip


@pytest.mark.unit
def test_merge_conflict_reported(tmp_path: Path) -> None:
    """Divergent edits to one file produce a reported conflict, not a raise."""
    backend = GitBackend.init(tmp_path / "memory")
    _write(backend.repo_path, "x.md", "base")
    backend.commit("initial", ["x.md"])
    default_branch = backend.current_branch
    assert default_branch is not None

    backend.branch("feature")
    _write(backend.repo_path, "x.md", "feature version")
    backend.commit("feature edit", ["x.md"])

    backend.branch(default_branch)
    _write(backend.repo_path, "x.md", "default version")
    backend.commit("default edit", ["x.md"])

    result = backend.merge("feature")
    assert result.status == "conflict"
    assert result.conflicts == ("x.md",)
    assert not backend.is_clean()  # conflicted repository is dirty


@pytest.mark.unit
def test_is_clean_after_commit(tmp_path: Path) -> None:
    """is_clean flips dirty on untracked/modified files and back after commit."""
    backend = GitBackend.init(tmp_path / "memory")
    assert backend.is_clean()

    _write(backend.repo_path, "entry.md", "hello")
    assert not backend.is_clean()  # untracked file
    backend.commit("add entry", ["entry.md"])
    assert backend.is_clean()

    _write(backend.repo_path, "entry.md", "changed")
    assert not backend.is_clean()  # modified tracked file
    backend.commit("edit entry", ["entry.md"])
    assert backend.is_clean()
