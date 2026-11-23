"""P21.T6 acceptance: SQLite storage fix (plan Phase 21, #156160-family).

Covered: the v2 storage module (:mod:`hermes.storage.database`) sustains
concurrent write load with ZERO SQLITE_BUSY/WAL errors — 8 threads x 50
IMMEDIATE transactions through the pooled connections, all 400 rows
present — and every connection carries the configured ``busy_timeout``
(default 5000ms) with the runtime-policy journal mode and
``synchronous=NORMAL``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hermes.storage.database import (
    DEFAULT_BUSY_TIMEOUT_MS,
    DEFAULT_POOL_SIZE,
    SESSIONS_TABLE_DDL,
    ConnectionPool,
    connect_database,
    runtime_journal_mode,
    write_transaction,
)

_WRITERS = 8
_TXNS_PER_WRITER = 50
_NOW = "2026-01-01T00:00:00+00:00"


@pytest.mark.unit
def test_wal_no_errors_under_load(tmp_path: Path) -> None:
    """8 threads x 50 pooled IMMEDIATE transactions -> zero errors, all rows."""
    pool = ConnectionPool(tmp_path / "state-v2.db")  # maxsize 8 (default)
    try:
        pool.ensure_schema()

        errors: list[Exception] = []

        def _writer(writer: int) -> None:
            for txn in range(_TXNS_PER_WRITER):
                try:
                    with pool.transaction() as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO sessions "
                            "(session_id, state, updated_at) VALUES (?, ?, ?)",
                            (
                                f"writer-{writer}-txn-{txn}",
                                json.dumps({"writer": writer, "txn": txn}),
                                _NOW,
                            ),
                        )
                except Exception as exc:  # collected, never swallowed silently
                    errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(w,)) for w in range(_WRITERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
            assert not thread.is_alive(), "writer thread did not finish"

        # acceptance: zero SQLITE_BUSY / WAL errors under concurrent load
        assert errors == []

        with pool.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == _WRITERS * _TXNS_PER_WRITER  # all 8x50 present
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE session_id = ?", ("writer-3-txn-49",)
            ).fetchone()
        assert row is not None
        assert json.loads(str(row[0])) == {"writer": 3, "txn": 49}
    finally:
        pool.close()


@pytest.mark.unit
def test_busy_timeout_configured(tmp_path: Path) -> None:
    """busy_timeout pragma is set (default 5000ms) on every connection."""
    conn = connect_database(tmp_path / "db.sqlite3", busy_timeout_ms=1234)
    conn_default = connect_database(tmp_path / "db-default.sqlite3")
    try:
        # the configured timeout reaches the actual SQLite pragma
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
        assert (
            conn_default.execute("PRAGMA busy_timeout").fetchone()[0]
            == DEFAULT_BUSY_TIMEOUT_MS
            == 5000
        )

        # journal mode follows the runtime policy (wal on safe SQLite builds,
        # delete on builds carrying the upstream WAL-reset bug — see module)
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        assert mode == runtime_journal_mode()
        assert mode in ("wal", "delete")

        # synchronous=NORMAL (pragma value 1) per the plan fix
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1

        # manual transaction control: write_transaction's BEGIN IMMEDIATE
        # must never race Python's implicit transaction management
        assert conn.isolation_level is None
        with write_transaction(conn):
            conn.execute(SESSIONS_TABLE_DDL)
        assert conn.in_transaction is False  # committed cleanly
    finally:
        conn.close()
        conn_default.close()
    assert DEFAULT_POOL_SIZE == 8  # the documented pool width
