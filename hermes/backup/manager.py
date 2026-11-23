"""hermes-v2 automatic backups (plan P18.T1).

Daily ``tar.gz`` archives of the memory repository (default
``<hermes_home>/memory-repo``) under ``<hermes_home>/backups/YYYY-MM-DD.tar.gz``:

- :meth:`BackupManager.create_backup` verifies the repository first via
  ``git fsck --no-progress`` (pygit2 exposes no fsck, so the git CLI runs via
  ``subprocess``; a plain non-git tree is archived without the check) and
  then archives the whole tree — including ``.git``, so a restore brings back
  both the entry files and the commit history. Same-day backups are
  idempotent: the archive for the current UTC date is refreshed in place via
  a temp file + ``os.replace``. Afterwards, retention runs automatically so
  archives older than 30 days are auto-deleted.
- :meth:`BackupManager.enforce_retention` deletes ``*.tar.gz`` files older
  than 30 days. Age comes from the ``YYYY-MM-DD`` archive name when it is
  parseable (deterministic under injected clocks) and from the file mtime
  otherwise; exactly 30 days old is still kept.
- :meth:`BackupManager.restore` empties the target directory and extracts
  the archive into it — a restore returns the repository to the archived
  state, it does not overlay the current contents.
- :func:`due_for_backup` is the pure scheduling helper: the backup runs once
  per local day at 02:00 local time (configurable ``hour``), and a never-
  backed-up system is due immediately.

All paths live under the injected ``hermes_root`` so tests stay hermetic;
``now`` is injectable everywhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

from hermes.config import hermes_home

__all__ = [
    "BACKUP_HOUR",
    "RETENTION_DAYS",
    "BackupIntegrityError",
    "BackupManager",
    "due_for_backup",
]

#: Retention window for archived backups (plan P18.T1 acceptance: 30 days).
RETENTION_DAYS: Final[int] = 30

#: Local hour of the daily backup run (plan P18.T1 acceptance: 02:00 local).
BACKUP_HOUR: Final[int] = 2

_SUFFIX_LEN = len(".tar.gz")


class BackupIntegrityError(RuntimeError):
    """Raised when ``git fsck`` rejects the memory repository before archiving."""


def _as_local(value: datetime) -> datetime:
    """Normalize ``value`` to an aware local datetime.

    Naive input is taken to be local time (matching the "02:00 local" design),
    aware input is converted to the system's local timezone.
    """
    return value.astimezone()


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _archive_date(path: Path) -> date | None:
    """The ``YYYY-MM-DD`` date encoded in an archive name, or ``None``."""
    try:
        return datetime.strptime(path.name[:-_SUFFIX_LEN], "%Y-%m-%d").date()
    except ValueError:
        return None


def due_for_backup(
    last_backup: datetime | None, now: datetime, hour: int = BACKUP_HOUR
) -> bool:
    """Pure scheduling helper: is the daily backup due at ``now``?

    The design runs one backup per local day at ``hour``:00 local time.
    ``True`` when the system has never been backed up (``last_backup`` is
    ``None`` — the first backup happens immediately, not at the next 02:00),
    or when ``now`` has reached today's scheduled hour and ``last_backup``
    happened before it. Backing up at exactly the scheduled hour counts as
    done for that day.
    """
    if last_backup is None:
        return True
    now_local = _as_local(now)
    scheduled = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    return now_local >= scheduled and _as_local(last_backup) < scheduled


class BackupManager:
    """Creates, verifies, retains and restores daily memory-repo backups."""

    def __init__(
        self,
        hermes_root: Path | None = None,
        repo_subdir: str = "memory-repo",
        backups_subdir: str = "backups",
    ) -> None:
        """Scope the manager under ``hermes_root`` (default ``hermes_home()``)."""
        self.hermes_root: Path = hermes_root if hermes_root is not None else hermes_home()
        self.repo_subdir: str = repo_subdir
        self.backups_subdir: str = backups_subdir
        self.repo_dir: Path = self.hermes_root / repo_subdir
        self.backups_dir: Path = self.hermes_root / backups_subdir

    # ── public API ────────────────────────────────────────────────────────
    def create_backup(self, now: datetime | None = None) -> Path:
        """Archive the memory repo to ``backups/YYYY-MM-DD.tar.gz``; returns it.

        Integrity is verified with ``git fsck --no-progress`` before archiving
        (:class:`BackupIntegrityError` on failure; skipped cleanly for a
        non-git tree). The per-day archive is idempotent — an existing file is
        refreshed atomically. Retention (30 days) runs after the archive is
        written, so stale backups auto-delete.
        """
        moment = _as_utc(now) if now is not None else datetime.now(UTC)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self._verify_repo_integrity()
        target = self.backups_dir / f"{moment:%Y-%m-%d}.tar.gz"
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                self._add_tree(tar)
            os.replace(tmp, target)  # refresh in place when the day already has one
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self.enforce_retention(now=moment)
        return target

    def enforce_retention(self, now: datetime | None = None) -> list[Path]:
        """Delete ``*.tar.gz`` backups older than 30 days; returns deleted paths."""
        moment = _as_utc(now) if now is not None else datetime.now(UTC)
        if not self.backups_dir.is_dir():
            return []
        cutoff = moment.date() - timedelta(days=RETENTION_DAYS)
        deleted: list[Path] = []
        for path in sorted(self.backups_dir.glob("*.tar.gz")):
            archive_date = _archive_date(path)
            if archive_date is not None:
                is_old = archive_date < cutoff
            else:
                mtime_date = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
                is_old = mtime_date < cutoff
            if is_old:
                path.unlink()
                deleted.append(path)
        return deleted

    def restore(self, backup_path: Path | str, target_dir: Path | None = None) -> Path:
        """Restore an archive over the memory repo (or ``target_dir``); returns it.

        The destination directory is emptied first — a restore returns the
        tree to the archived state rather than overlaying it — then the
        archive is extracted with tarfile's safe ``data`` filter.
        """
        archive = Path(backup_path)
        if not archive.is_file():
            raise FileNotFoundError(f"backup archive does not exist: {archive}")
        destination = Path(target_dir) if target_dir is not None else self.repo_dir
        _clear_directory(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(destination, filter="data")
        return destination

    # ── internals ─────────────────────────────────────────────────────────
    def _verify_repo_integrity(self) -> None:
        """Run ``git fsck --no-progress`` in the repo; raise on corruption.

        pygit2 (used by the memory store) has no fsck binding, so the git CLI
        is invoked via subprocess with captured output. A tree that is not a
        git repository (missing ``.git``) is skipped cleanly — there is no
        object database to validate. A missing ``git`` binary propagates the
        ``FileNotFoundError``: that is an environment problem, not evidence of
        corruption.
        """
        if not self.repo_dir.is_dir() or not (self.repo_dir / ".git").exists():
            return
        result = subprocess.run(
            ["git", "fsck", "--no-progress"],
            cwd=self.repo_dir,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            output = result.stderr or result.stdout or b""
            detail = output.decode("utf-8", errors="replace").strip()
            raise BackupIntegrityError(
                f"git fsck failed for {self.repo_dir} "
                f"(exit {result.returncode}): {detail}"
            )

    def _add_tree(self, tar: tarfile.TarFile) -> None:
        """Add every entry of the repo tree under its repo-relative arcname.

        ``.git`` is included so restores recover the commit history too.
        Entries are added non-recursively (each file/directory exactly once).
        A missing repository yields an empty archive — the scheduled run on a
        fresh install must not fail.
        """
        if not self.repo_dir.is_dir():
            return
        for path in sorted(self.repo_dir.rglob("*")):
            arcname = path.relative_to(self.repo_dir).as_posix()
            tar.add(path, arcname=arcname, recursive=False)


def _clear_directory(path: Path) -> None:
    """Remove every child of ``path``; a missing directory is fine."""
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
