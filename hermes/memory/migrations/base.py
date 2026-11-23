"""Migration safety base: dry-run + automatic backup + rollback (plan P18.T4).

Every memory-format migration — the fork-base -> v2.0 conversion
(:mod:`hermes.memory.migrations.fork_to_v2_0`) and future schema changes —
must honor the P18.T4 safety contract:

1. **Automatic backup before any migration** (:meth:`Migration.prepare_backup`):
   a byte-exact copy of the source directory at
   ``<source>-premigration-backup`` is created BEFORE any write, kept after
   the run, and never clobbered by a re-run (a timestamped sibling is used
   instead).
2. **Dry run without state changes** (:meth:`Migration.dry_run`): the full
   report is produced by walking the SAME transform logic with a
   null-writer — :meth:`Migration.discover` is read-only and
   :meth:`Migration.write_units` (the ONLY mutating step) is never called —
   so the target receives zero filesystem writes.
3. **Rollback** (:meth:`Migration.rollback`): an applied (or failed) run is
   undone by removing the produced target and restoring the source
   byte-exact from the backup, so a partial migration is always recoverable.

Subclasses implement :meth:`Migration.discover` (read-only source walk) and
:meth:`Migration.write_units` (the guarded apply step); :meth:`Migration.run`
orchestrates backup -> discover -> write. The class is deliberately thin —
no hooks beyond the source-existence check — to stay honest about what is
shared: the safety mechanics, not the per-migration transform.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "MIGRATION_BACKUP_SUFFIX",
    "Migration",
    "MigrationPlan",
    "MigrationReport",
    "atomic_write",
    "backup_source",
    "rollback_migration",
    "rmtree_force",
    "source_from_backup",
]

#: Suffix of the pre-migration backup directory created next to the source.
#: :func:`source_from_backup` derives the original source path from it.
MIGRATION_BACKUP_SUFFIX = "-premigration-backup"

#: Fallback message of :meth:`Migration._require_source`; subclasses may
#: override the method to tailor the wording to their source format.
_SOURCE_MISSING_MESSAGE = "migration source directory does not exist: {path}"


@dataclass
class MigrationReport:
    """Outcome of one migration run (plan P1.T6; shared via the P18.T4 base).

    The same report is produced by a dry run and an applied run: a dry run
    fills ``entries_found``/``skipped``/``errors`` completely and reports
    ``entries_migrated == 0`` with ``dry_run=True``.
    """

    #: Candidate entries discovered in the successfully-read source files.
    entries_found: int
    #: Entries actually written to the target repository (always 0 on a dry run).
    entries_migrated: int
    #: Discovered-but-not-migrated units: bullets with no content after
    #: marker stripping, plus unreadable source files (one unit each, with a
    #: message in ``errors``). ``entries_migrated == entries_found - skipped``.
    skipped: int
    #: True when nothing was written to the target (dry run).
    dry_run: bool
    #: Pre-migration backup directory of the source (always created, kept).
    backup_path: Path | None
    #: Context every migrated entry landed in (plan P5.T4): the explicit
    #: ``"global"`` context — entries under ``global/facts/`` with
    #: ``context=global`` trust cards and a global-naming ``MEMORY.md`` index.
    context: str = "global"
    #: Per-failure messages (unreadable files). Empty list = clean run.
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MigrationPlan:
    """What one migration run WOULD write: the read-only discovery result.

    Produced by :meth:`Migration.discover` — the same walk an applied run
    performs, without any side effect. ``found`` counts every candidate unit
    discovered in readable source files (including ones later skipped);
    ``units`` carries only the units :meth:`Migration.write_units` will
    write; ``skipped`` counts discovered-but-not-migrated units (empty
    content, unreadable files — one unit each, message in ``errors``).
    """

    units: list[Any]
    found: int
    skipped: int
    errors: list[str] = field(default_factory=list)


def rmtree_force(path: Path) -> None:
    """``shutil.rmtree`` that clears the read-only bit first.

    libgit2 stamps packed objects under ``.git/objects`` read-only on
    Windows, which makes a plain ``rmtree`` of a v2 memory repository raise
    ``PermissionError``. Every file is made writable before the removal;
    removal of a missing path is a no-op.
    """
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            # best effort: rmtree reports anything that still fails
            with contextlib.suppress(OSError):
                child.chmod(stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(path)


def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def backup_source(source_dir: Path) -> Path:
    """Copy ``source_dir`` to ``<source>-premigration-backup``; returns it.

    Runs BEFORE anything else (plan P18.T4: automatic backup before any
    migration; created even for dry runs, matching the P21.T11 config
    migration's "backup before any change" rule). An existing backup from a
    previous attempt is left untouched and a UTC-timestamped sibling is
    created instead.
    """
    source_dir = Path(source_dir)
    backup = source_dir.with_name(source_dir.name + MIGRATION_BACKUP_SUFFIX)
    if backup.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        backup = source_dir.with_name(f"{source_dir.name}{MIGRATION_BACKUP_SUFFIX}-{stamp}")
    shutil.copytree(source_dir, backup)
    return backup


def source_from_backup(backup_path: Path) -> Path:
    """The source directory a pre-migration backup belongs to (name-derived)."""
    name = backup_path.name
    if not name.endswith(MIGRATION_BACKUP_SUFFIX) or name == MIGRATION_BACKUP_SUFFIX:
        raise ValueError(
            f"{backup_path} is not a pre-migration backup "
            f"(name must end with {MIGRATION_BACKUP_SUFFIX!r})"
        )
    return backup_path.with_name(name[: -len(MIGRATION_BACKUP_SUFFIX)])


def rollback_migration(target_repo: Path, backup_path: Path) -> None:
    """Shared rollback: remove the target, restore the source from the backup.

    ``backup_path`` is the ``<source>-premigration-backup`` directory that
    :func:`backup_source` created (the source path is derived from its
    name). The target repository — whatever the migration produced — is
    removed, and the source directory is replaced byte-exact with the backup
    contents, so a run that failed halfway (or damaged/consumed the source)
    ends in the pre-migration state. Raises :class:`FileNotFoundError` when
    the backup does not exist and :class:`ValueError` when the path is not a
    pre-migration backup.
    """
    backup_path = Path(backup_path)
    if not backup_path.is_dir():
        raise FileNotFoundError(f"pre-migration backup does not exist: {backup_path}")
    source_dir = source_from_backup(backup_path)
    target_repo = Path(target_repo)
    if target_repo.exists():
        rmtree_force(target_repo)
    if source_dir.exists():
        rmtree_force(source_dir)
    shutil.copytree(backup_path, source_dir)


class Migration(ABC):
    """Base class enforcing the P18.T4 migration safety contract.

    A migration reads ``source_dir`` and produces ``target_repo``. The base
    owns the safety mechanics — automatic pre-migration backup, the
    null-writer dry run, the guarded apply orchestration and rollback — and
    delegates the actual transform to two abstract methods:

    - :meth:`discover`: read-only source walk returning a
      :class:`MigrationPlan`;
    - :meth:`write_units`: the ONLY method permitted to mutate the target;
      :meth:`run` calls it exclusively on applied runs.
    """

    def __init__(self, source_dir: Path, target_repo: Path) -> None:
        self.source_dir = Path(source_dir)
        self.target_repo = Path(target_repo)

    # ── subclass contract ────────────────────────────────────────────────────

    @abstractmethod
    def discover(self) -> MigrationPlan:
        """Read-only walk of the source: the plan an applied run would write.

        MUST NOT mutate anything — it is the null-writer half of the dry
        run. Read errors are folded into the plan (``skipped``/``errors``),
        not raised, wherever the migration can meaningfully continue.
        """

    @abstractmethod
    def write_units(self, units: list[Any]) -> int:
        """Write the discovered units into the target; returns the count.

        The ONLY mutating step of a migration. :meth:`run` calls it only
        when ``apply=True``; :meth:`dry_run` never does, which is what makes
        the dry run write-free by construction.
        """

    def _require_source(self) -> None:
        """Raise :class:`FileNotFoundError` when the source is missing.

        Called before the backup on BOTH dry and applied runs (the P1.T6
        behavior: a missing source fails before anything is written).
        Subclasses may override to tailor the error message.
        """
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                _SOURCE_MISSING_MESSAGE.format(path=self.source_dir)
            )

    # ── safety mechanics (plan P18.T4) ──────────────────────────────────────

    def prepare_backup(self, source_dir: Path | None = None) -> Path:
        """Create the automatic pre-migration backup; returns its path.

        Copies ``source_dir`` (default: the migration's source) to
        ``<source>-premigration-backup`` — byte-exact, BEFORE any write,
        kept after the run, never clobbered by a re-run (a timestamped
        sibling is created instead, see :func:`backup_source`).
        """
        return backup_source(
            Path(source_dir) if source_dir is not None else self.source_dir
        )

    def dry_run(self) -> MigrationReport:
        """Full report with ZERO writes to the target (null-writer guarantee).

        Walks the same transform logic as an applied run — backup, then
        :meth:`discover` — but never calls :meth:`write_units`, so the
        target receives no filesystem writes at all. The pre-migration
        backup IS created (always-backup rule) next to the source.
        """
        self._require_source()
        backup = self.prepare_backup()
        plan = self.discover()
        return MigrationReport(
            entries_found=plan.found,
            entries_migrated=0,
            skipped=plan.skipped,
            dry_run=True,
            backup_path=backup,
            errors=list(plan.errors),
        )

    def run(self, apply: bool = True) -> MigrationReport:
        """Dry run when ``apply=False``; otherwise backup + discover + write.

        The applied path creates the automatic pre-migration backup FIRST
        (:meth:`prepare_backup`), then discovers (read-only), then — and only
        then — hands the units to :meth:`write_units`, the sole mutating
        step. The returned report carries the backup path for
        :meth:`rollback`.
        """
        if not apply:
            return self.dry_run()
        self._require_source()
        backup = self.prepare_backup()
        plan = self.discover()
        report = MigrationReport(
            entries_found=plan.found,
            entries_migrated=0,
            skipped=plan.skipped,
            dry_run=False,
            backup_path=backup,
            errors=list(plan.errors),
        )
        report.entries_migrated = self.write_units(plan.units)
        return report

    def rollback(self, backup_path: Path) -> None:
        """Undo this migration; restores the pre-migration state.

        Delegates to :func:`rollback_migration` with this migration's target
        repository: the target is removed and the source is restored
        byte-exact from ``backup_path``.
        """
        rollback_migration(self.target_repo, backup_path)
