#!/usr/bin/env python3
"""CLI: restore the memory repository from a backup archive (plan P18.T2).

    python scripts/hermes-restore.py BACKUP_PATH [--hermes-root ROOT] [--no-safety-backup]

Runs the full recovery workflow of :class:`hermes.backup.recovery.Recovery`:
a safety backup of the current (possibly corrupted) repository is kept as a
``-pre-restore`` sibling (unless ``--no-safety-backup``), the live repo is
wiped, the archive is extracted over it (entries + Git history), and the
result is verified. Stop Hermes before running this and restart it
afterwards — the library cannot signal live processes.

Exit codes: 0 when the restore completed with a clean verification, 1 on
failure or when the recovered repository still has problems.
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

from hermes.backup.manager import BackupManager  # noqa: E402
from hermes.backup.recovery import Recovery  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the recovery, print the result."""
    parser = argparse.ArgumentParser(
        description="Restore the hermes-v2 memory repository from a backup "
        "archive (plan P18.T2), with a kept safety backup of the current state."
    )
    parser.add_argument(
        "backup_path",
        type=Path,
        help="backup archive to restore from (e.g. ~/.hermes/backups/2026-11-20.tar.gz)",
    )
    parser.add_argument(
        "--hermes-root",
        type=Path,
        default=None,
        help="Hermes home directory holding memory-repo/ and backups/ "
        "(default: the HERMES_HOME environment override or ~/.hermes)",
    )
    parser.add_argument(
        "--no-safety-backup",
        action="store_true",
        help="skip the pre-restore safety backup of the current repository",
    )
    args = parser.parse_args(argv)

    manager = BackupManager(hermes_root=args.hermes_root) if args.hermes_root else None
    recovery = Recovery(manager)

    try:
        result = recovery.restore(
            args.backup_path, pre_restore_backup=not args.no_safety_backup
        )
    except (OSError, ValueError) as exc:
        print(f"restore failed: {exc}", file=sys.stderr)
        return 1

    print("Memory repository recovery (plan P18.T2)")
    print(f"  restored from:  {result.restored_from}")
    print(f"  entries:        {result.entries_count}")
    print(f"  git commits:    {result.git_commits}")
    print(f"  note:           {result.stopped_note}")
    problems = recovery.verify_recovered()
    if problems:
        print("verification problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  verification:   clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
