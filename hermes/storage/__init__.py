"""hermes-v2 storage layer (plan P21.T6, issue #156160-family).

Public surface of the v2 SQLite storage fix for "state.db WAL errors /
session storage unavailable" under concurrent access:
:func:`hermes.storage.database.connect_database` (WAL + busy_timeout +
NORMAL sync), :func:`hermes.storage.database.write_transaction` (BEGIN
IMMEDIATE writes), :class:`hermes.storage.database.ConnectionPool` (the
thread-safe pooled handle, a.k.a. :data:`DatabaseHandle`), and
:func:`hermes.storage.database.ensure_schema`. Serves the v2 tree's
session/state databases; the fork-base ``hermes_state.py`` stays untouched.
"""

from hermes.storage.database import (
    DEFAULT_BUSY_TIMEOUT_MS,
    DEFAULT_POOL_SIZE,
    SESSIONS_TABLE_DDL,
    ConnectionPool,
    DatabaseHandle,
    connect_database,
    ensure_schema,
    runtime_journal_mode,
    write_transaction,
)

__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "DEFAULT_POOL_SIZE",
    "SESSIONS_TABLE_DDL",
    "ConnectionPool",
    "DatabaseHandle",
    "connect_database",
    "ensure_schema",
    "runtime_journal_mode",
    "write_transaction",
]
