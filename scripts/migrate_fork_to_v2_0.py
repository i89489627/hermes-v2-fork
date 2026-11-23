#!/usr/bin/env python3
"""CLI: migrate the fork-base memory directory to the v2.0 format (plan P1.T6).

Migrate (backup of the source is always created and kept, even on a dry run):

    python scripts/migrate_fork_to_v2_0.py SOURCE TARGET [--dry-run]

Undo an applied migration (removes the v2 repo, restores SOURCE byte-exact
from the pre-migration backup that ``migrate`` created and reported):

    python scripts/migrate_fork_to_v2_0.py TARGET --rollback BACKUP

Exit codes: 0 on success, 1 on migration errors (unreadable source files) or
any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running as a script puts scripts/ (not the repo root) on sys.path, so the
# vendored hermes-v2 package is made importable explicitly (repo convention,
# see scripts/contributor_audit.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes.memory.migrations.fork_to_v2_0 import MigrationReport, migrate, rollback  # noqa: E402


def _print_report(report: MigrationReport, source: Path, target: Path) -> None:
    """Human-readable summary of a :func:`migrate` run."""
    print("Fork-base memory migration to v2.0 (plan P1.T6)")
    print(f"  source:           {source}")
    print(f"  target:           {target}")
    print(f"  dry run:          {'yes' if report.dry_run else 'no'}")
    print(f"  entries found:    {report.entries_found}")
    print(f"  entries migrated: {report.entries_migrated}")
    print(f"  skipped:          {report.skipped}")
    print(f"  backup:           {report.backup_path}")
    if report.errors:
        print("errors:")
        for error in report.errors:
            print(f"  - {error}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the migration or the rollback, print the result."""
    parser = argparse.ArgumentParser(
        description="Migrate fork-base memory (~/.hermes/memories/*.md) to the v2.0 "
        "git-backed trust-card memory repository, with dry-run and rollback."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=None,
        help="fork-base memory directory (e.g. ~/.hermes/memories); "
        "not needed with --rollback",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="v2.0 memory repository to create (e.g. ~/.hermes/memory-repo); "
        "with --rollback, the v2 repo to remove",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the migration without writing the target "
        "(the source backup is still created)",
    )
    parser.add_argument(
        "--rollback",
        type=Path,
        metavar="BACKUP",
        default=None,
        help="undo an applied migration: remove TARGET and restore the source "
        "byte-exact from the given pre-migration backup directory",
    )
    args = parser.parse_args(argv)

    if args.rollback is not None:
        try:
            rollback(args.target, args.rollback)
        except (OSError, ValueError) as exc:
            print(f"rollback failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Rollback complete: {args.target} removed, "
            f"source restored from {args.rollback}"
        )
        return 0

    if args.source is None:
        parser.error("source is required unless --rollback is given")

    try:
        report = migrate(args.source, args.target, apply=not args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    _print_report(report, args.source, args.target)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
