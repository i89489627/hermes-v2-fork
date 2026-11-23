"""Git-backed storage backend for the v2 memory graph (plan P1.T2).

Every write to the memory store becomes a Git commit, giving the atomic,
append-only history the memory-first design needs (plan Section 2.4). Uses
``pygit2`` (libgit2) directly — no shelling out to the ``git`` CLI — and
stamps a repo-local identity so commits never fail on machines without a
global Git configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pygit2

#: Local identity written into every memory repository's config (plan P1.T2).
GIT_USER_NAME = "hermes-v2"
GIT_USER_EMAIL = "hermes-v2@local"

#: Outcome of :meth:`GitBackend.merge`.
MergeStatus = Literal["fast-forward", "merged", "up-to-date", "conflict"]


@dataclass(frozen=True)
class CommitEntry:
    """One commit from :meth:`GitBackend.log`."""

    sha: str
    message: str
    timestamp: datetime  # aware UTC


@dataclass(frozen=True)
class MergeResult:
    """Result of :meth:`GitBackend.merge`.

    Content conflicts are *reported* (``status="conflict"`` plus the
    conflicting paths), never raised; the repository is left in the merge
    state for the caller to resolve or abort, like ``git merge``.
    """

    status: MergeStatus
    conflicts: tuple[str, ...] = ()


class GitBackend:
    """Thin pygit2 wrapper around one on-disk memory repository.

    Paths handed to :meth:`commit` and :meth:`log` are interpreted relative
    to the repository root; absolute paths inside the repository are
    accepted and relativized, POSIX separators are used internally.
    """

    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path)
        try:
            self._repo: Any = pygit2.Repository(str(self.repo_path))
        except Exception as exc:
            raise ValueError(f"{self.repo_path} is not a git repository") from exc

    @classmethod
    def init(cls, repo_path: Path | str) -> GitBackend:
        """Create (or re-open) the repository at ``repo_path``.

        Sets repo-local ``user.name``/``user.email`` so commits never fail
        on machines without a global Git identity.
        """
        path = Path(repo_path)
        path.mkdir(parents=True, exist_ok=True)
        repo = pygit2.init_repository(str(path), bare=False)
        repo.config["user.name"] = GIT_USER_NAME
        repo.config["user.email"] = GIT_USER_EMAIL
        return cls(path)

    def _rel_path(self, path: str | Path) -> str:
        """Normalize ``path`` to a POSIX-style path relative to the repo root."""
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(self.repo_path.resolve())
            except ValueError:
                raise ValueError(
                    f"path {path!r} is outside repository {self.repo_path}"
                ) from None
        return candidate.as_posix()

    def commit(self, message: str, files: Sequence[str | Path]) -> str:
        """Stage ``files`` and commit them; returns the commit sha (hex).

        The commit is always created — every write leaves exactly one Git
        commit behind, even when the resulting tree is unchanged.
        """
        repo = self._repo
        index = repo.index
        for path in files:
            index.add(self._rel_path(path))
        index.write()
        tree = index.write_tree()
        signature = pygit2.Signature(GIT_USER_NAME, GIT_USER_EMAIL)
        parents: list[Any] = [] if repo.head_is_unborn else [repo.head.target]
        sha = repo.create_commit("HEAD", signature, signature, message, tree, parents)
        return str(sha)

    def branch(self, name: str) -> None:
        """Create ``name`` at the current commit (if missing) and check it out.

        Switching to an existing branch updates HEAD, the working tree and
        the index; creating a new branch only moves HEAD (the tree is
        already identical to the current one).
        """
        repo = self._repo
        if repo.head_is_unborn:
            raise ValueError("cannot branch before the initial commit")
        branches = repo.branches.local
        if name in branches:
            repo.checkout(branches[name])  # moves HEAD, updates workdir + index
            return
        created = branches.create(name, repo[repo.head.target])
        repo.set_head(created.name)  # symbolic HEAD move; tree unchanged

    def merge(self, branch: str) -> MergeResult:
        """Merge ``branch`` into the current branch.

        Fast-forwards when possible, otherwise three-way-merges and creates
        a merge commit. Content conflicts are reported, never raised.
        """
        repo = self._repo
        if repo.head_is_unborn:
            raise ValueError("cannot merge before the initial commit")
        branches = repo.branches.local
        if branch not in branches:
            raise ValueError(f"unknown branch: {branch!r}")
        target = branches[branch].target
        analysis, _preference = repo.merge_analysis(target)
        if analysis & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            return MergeResult(status="up-to-date")
        if analysis & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            target_commit = repo[target]
            # Checkout the target *tree* (HEAD stays on the current branch),
            # then move the current branch reference forward.
            repo.checkout_tree(target_commit.tree)
            repo.index.write()
            repo.lookup_reference(repo.head.name).set_target(target)
            return MergeResult(status="fast-forward")
        repo.merge(target)
        conflicts = repo.index.conflicts
        if conflicts is not None:
            paths: list[str] = []
            for ours, theirs, _base in conflicts:
                entry = ours if ours is not None else theirs
                if entry is not None and entry.path and entry.path not in paths:
                    paths.append(entry.path)
            return MergeResult(status="conflict", conflicts=tuple(paths))
        signature = pygit2.Signature(GIT_USER_NAME, GIT_USER_EMAIL)
        parents: list[Any] = [repo.head.target, target]
        repo.create_commit(
            "HEAD",
            signature,
            signature,
            f"Merge branch '{branch}'",
            repo.index.write_tree(),
            parents,
        )
        repo.state_cleanup()  # drop MERGE_HEAD now that the merge commit exists
        return MergeResult(status="merged")

    def log(self, path: str | Path | None = None, limit: int = 10) -> list[CommitEntry]:
        """Commits touching ``path`` (all files when ``None``), newest first."""
        if limit <= 0:
            return []
        repo = self._repo
        if repo.head_is_unborn:
            return []
        rel = self._rel_path(path) if path is not None else None
        entries: list[CommitEntry] = []
        # Topological + time sort: newest first with a deterministic order
        # for commits sharing a timestamp (children strictly before parents).
        sort_flags = pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_TOPOLOGICAL
        for commit in repo.walk(repo.head.target, sort_flags):
            if rel is not None and not self._touches(commit, rel):
                continue
            entries.append(
                CommitEntry(
                    sha=str(commit.id),
                    message=commit.message,
                    timestamp=datetime.fromtimestamp(commit.commit_time, tz=UTC),
                )
            )
            if len(entries) >= limit:
                break
        return entries

    def _touches(self, commit: Any, rel_path: str) -> bool:
        """True when ``commit`` changed ``rel_path`` relative to its parent."""
        parents = commit.parents
        # Root commits have no parent: diff against the empty tree.
        diff = (
            commit.tree.diff_to_tree(parents[0].tree)
            if parents
            else commit.tree.diff_to_tree()
        )
        return any(
            delta.old_file.path == rel_path or delta.new_file.path == rel_path
            for delta in diff.deltas
        )

    def is_clean(self) -> bool:
        """True when workdir and index match HEAD and nothing is untracked."""
        return not self._repo.status()

    @property
    def current_branch(self) -> str | None:
        """Short name of the checked-out branch (None if unborn or detached)."""
        repo = self._repo
        if repo.head_is_unborn:
            return None
        name = repo.head.name
        prefix = "refs/heads/"
        return name[len(prefix):] if name.startswith(prefix) else None
