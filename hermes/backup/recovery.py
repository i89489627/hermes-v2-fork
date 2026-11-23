"""Recovery: restore the memory repository from a backup (plan P18.T2).

The full restore workflow of :meth:`Recovery.restore`:

1. a **safety backup** of the CURRENT (possibly corrupted) repository is
   copied to a sibling ``<repo>-pre-restore`` directory and KEPT — a restore
   is destructive (the live tree is replaced wholesale), so even the state
   being recovered from stays on disk unless ``pre_restore_backup=False``;
2. the live repository directory is wiped with :func:`force_rmtree` (a plain
   ``rmtree`` fails on Windows because libgit2 stamps ``.git`` objects
   read-only — this helper is the shared, production-grade fix);
3. the archive is restored via :meth:`BackupManager.restore` (its safe
   ``data`` filter), bringing back both the entry files and the ``.git``
   history;
4. the result is verified: :class:`RecoveryResult` carries the entry count
   and the number of Git commits (callers check entry count > 0 and a
   non-empty log), and :meth:`Recovery.verify_recovered` lists concrete
   problems (missing ``MEMORY.md`` index, entries listed in the index whose
   fact files are gone, broken trust cards, missing/empty Git history).

Stopping and restarting Hermes processes (plan P18.T2 steps 1 and 4) is the
operator's responsibility — a library cannot signal live processes — so the
result carries a fixed ``stopped_note`` stating it.

All paths default to the :class:`~hermes.backup.manager.BackupManager`
layout (``<hermes_home>/memory-repo``, ``<hermes_home>/backups``) so tests
and scripts stay hermetic via ``HERMES_HOME``.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hermes.backup.manager import BackupManager
from hermes.memory.git_backend import GitBackend
from hermes.memory.store import INDEX_NAME
from hermes.memory.trust_card import parse

__all__ = ["Recovery", "RecoveryResult", "force_rmtree"]

#: Cap for the commit walk used to verify a recovered repository's history.
_COMMIT_LOG_LIMIT = 100_000

#: Fixed note in every :class:`RecoveryResult` (plan P18.T2 workflow steps 1
#: and 4 are operator actions, not library actions).
_STOPPED_NOTE = (
    "Hermes processes are NOT stopped or restarted automatically: stop Hermes "
    "before restoring and restart it afterwards (plan P18.T2 steps 1 and 4)."
)


@dataclass
class RecoveryResult:
    """Outcome of one :meth:`Recovery.restore` run."""

    #: The backup archive the repository was restored from.
    restored_from: Path
    #: Fact files restored under ``global/facts`` + ``contexts/*/facts``
    #: (verify: > 0 for a non-empty backup).
    entries_count: int
    #: Commits in the restored repository's history (verify: > 0).
    git_commits: int
    #: Note about the manual stop/restart of Hermes processes (see above).
    stopped_note: str


def force_rmtree(path: Path) -> None:
    """``shutil.rmtree`` that clears the read-only bit first.

    libgit2 stamps packed objects under ``.git/objects`` read-only on
    Windows, which makes a plain ``rmtree`` of a memory repository raise
    ``PermissionError``. Every file is made writable before the removal;
    removal of a missing path is a no-op. Shared by the recovery workflow
    and any other code that has to delete a Git-backed tree (e.g. the
    P1.T6 migration rollback keeps its own copy for independence).
    """
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            # best effort: rmtree reports anything that still fails
            with contextlib.suppress(OSError):
                child.chmod(0o644)
    shutil.rmtree(path)


def _safety_backup(repo_dir: Path) -> Path:
    """Copy ``repo_dir`` to a sibling ``<repo>-pre-restore`` dir; returns it.

    The copy is KEPT (never auto-deleted): it preserves the pre-restore
    state — including whatever corruption triggered the recovery — for
    forensics and for a second recovery attempt from a different archive.
    An existing safety backup from a previous restore is left untouched and
    a UTC-timestamped sibling is created instead.
    """
    target = repo_dir.with_name(repo_dir.name + "-pre-restore")
    if target.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        target = repo_dir.with_name(f"{repo_dir.name}-pre-restore-{stamp}")
    shutil.copytree(repo_dir, target)
    return target


def _count_entries(repo_dir: Path) -> int:
    """Fact files in the v2 layout: ``global/facts`` + ``contexts/*/facts``."""
    count = 0
    global_facts = repo_dir / "global" / "facts"
    if global_facts.is_dir():
        count += sum(1 for path in global_facts.glob("*.md") if path.is_file())
    contexts_root = repo_dir / "contexts"
    if contexts_root.is_dir():
        for context_dir in contexts_root.iterdir():
            facts_dir = context_dir / "facts"
            if facts_dir.is_dir():
                count += sum(1 for path in facts_dir.glob("*.md") if path.is_file())
    return count


def _fact_files(repo_dir: Path) -> list[Path]:
    """All fact file paths in the v2 layout (global first, contexts sorted)."""
    files: list[Path] = []
    global_facts = repo_dir / "global" / "facts"
    if global_facts.is_dir():
        files.extend(sorted(path for path in global_facts.glob("*.md") if path.is_file()))
    contexts_root = repo_dir / "contexts"
    if contexts_root.is_dir():
        for context_dir in sorted(path for path in contexts_root.iterdir() if path.is_dir()):
            facts_dir = context_dir / "facts"
            if facts_dir.is_dir():
                files.extend(
                    sorted(path for path in facts_dir.glob("*.md") if path.is_file())
                )
    return files


def _count_commits(repo_dir: Path) -> int:
    """Commits in the repository's history; 0 when missing/corrupt/unborn."""
    try:
        return len(GitBackend(repo_dir).log(limit=_COMMIT_LOG_LIMIT))
    except ValueError:
        return 0  # not a git repository (GitBackend's constructor signal)


class Recovery:
    """Restores the memory repository from a backup archive (plan P18.T2)."""

    def __init__(self, backup_manager: BackupManager | None = None) -> None:
        """Scope the recovery with ``backup_manager`` (default: hermes home)."""
        self.backup_manager: BackupManager = (
            backup_manager if backup_manager is not None else BackupManager()
        )

    @property
    def repo_dir(self) -> Path:
        """The memory repository directory this recovery operates on."""
        return self.backup_manager.repo_dir

    def restore(
        self, backup_path: Path | str, *, pre_restore_backup: bool = True
    ) -> RecoveryResult:
        """Replace the live repository with ``backup_path``'s contents.

        Workflow: safety backup of the current repo (``-pre-restore`` sibling,
        kept; skipped when ``pre_restore_backup=False`` or the repo directory
        is already gone), full wipe via :func:`force_rmtree`, then
        :meth:`BackupManager.restore` extracts the archive (``.git`` included,
        so the commit history comes back with the files). Raises
        :class:`FileNotFoundError` when the archive does not exist.
        """
        archive = Path(backup_path)
        if not archive.is_file():
            raise FileNotFoundError(f"backup archive does not exist: {archive}")
        repo_dir = self.repo_dir
        if pre_restore_backup and repo_dir.is_dir():
            _safety_backup(repo_dir)
        force_rmtree(repo_dir)
        self.backup_manager.restore(archive)
        return RecoveryResult(
            restored_from=archive,
            entries_count=_count_entries(repo_dir),
            git_commits=_count_commits(repo_dir),
            stopped_note=_STOPPED_NOTE,
        )

    def verify_recovered(self, repo_path: Path | None = None) -> list[str]:
        """Integrity problems of the (recovered) repository; empty = healthy.

        Checks: the repository directory exists, the ``MEMORY.md`` index is
        present, every entry listed in the index still has its fact file
        (partial corruption), every fact file parses as a valid trust card,
        at least one entry exists, and the Git history is present and
        non-empty.
        """
        repo = repo_path if repo_path is not None else self.repo_dir
        if not repo.is_dir():
            return [f"memory repository directory missing: {repo}"]
        problems: list[str] = []

        index_path = repo / INDEX_NAME
        if not index_path.is_file():
            problems.append(f"index file missing: {INDEX_NAME}")
        listed_ids: set[str] = set()
        if index_path.is_file():
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("- "):
                    listed_ids.add(line[2:].strip())

        fact_files = _fact_files(repo)
        on_disk = {path.stem for path in fact_files}
        for missing_id in sorted(listed_ids - on_disk):
            problems.append(
                f"entry {missing_id!r} is listed in {INDEX_NAME} "
                "but its fact file is missing"
            )
        if not fact_files:
            problems.append("no memory entries found (empty global/facts and contexts)")
        for path in fact_files:
            try:
                parse(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"broken trust card: {path.name} ({exc})")

        if not (repo / ".git").exists():
            problems.append("git history missing: no .git directory")
        elif _count_commits(repo) == 0:
            problems.append("git history empty: repository has no commits")
        return problems
