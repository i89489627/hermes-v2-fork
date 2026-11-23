"""hermes-v2 memory format migrations (plan P1.T6).

One-time conversions between memory storage formats. The first migration
ships here: :mod:`hermes.memory.migrations.fork_to_v2_0` converts the
fork-base ``~/.hermes/memories/*.md`` layout (plain markdown without
frontmatter) into the v2.0 git-backed trust-card memory repository. Plan
P18.T4 later factors the dry-run + backup + rollback contract into a shared
migration base; both tasks share this package.
"""

from hermes.memory.migrations.fork_to_v2_0 import (
    MIGRATION_BACKUP_SUFFIX,
    MigrationReport,
    migrate,
    rollback,
)

__all__ = [
    "MIGRATION_BACKUP_SUFFIX",
    "MigrationReport",
    "migrate",
    "rollback",
]
