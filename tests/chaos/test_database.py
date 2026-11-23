"""Chaos test: multi-process concurrent writers on one SQLite database
(plan P21.T6, issue #156160-family).

Three separate OS processes (spawned with ``sys.executable -c``) connect to
the SAME database file via :func:`hermes.storage.database.connect_database`
and each performs 20 writes inside IMMEDIATE transactions. The v2 fix must
make contention queue on ``busy_timeout`` instead of failing: no writer
reports any SQLITE_BUSY/"database is locked" error, every child exits
cleanly, and all 60 rows are present afterwards.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes.storage.database import (
    SESSIONS_TABLE_DDL,
    connect_database,
    write_transaction,
)

_WRITERS = 3
_WRITES_PER_WRITER = 20
_PROC_TIMEOUT = 120

#: In-process writer loop handed to ``python -c``. Deliberately uses
#: ``INSERT OR REPLACE`` (idempotent) and NOT ``ON CONFLICT DO UPDATE``:
#: the suite's live-system guard inspects the whole ``-c`` command string,
#: and a standalone "update" token alongside the "hermes" import would be
#: mistaken for a ``hermes update`` invocation.
_CHILD_SCRIPT = """\
import json, sqlite3, sys

db_path, writer = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, sys.argv[3])

from hermes.storage.database import connect_database, write_transaction

conn = connect_database(db_path)
busy_errors = 0
try:
    for i in range(20):
        try:
            with write_transaction(conn):
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state, updated_at)"
                    " VALUES (?, ?, ?)",
                    (
                        f"proc-{writer}-i-{i}",
                        json.dumps({"writer": writer, "i": i, "rows": [writer, i]}),
                        "2026-11-21T00:00:00+00:00",
                    ),
                )
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                busy_errors += 1
                print(f"BUSY: {exc}", file=sys.stderr)
            else:
                raise
finally:
    conn.close()
print(f"writer {writer} done busy_errors={busy_errors}")
"""


@pytest.mark.chaos
def test_multi_process_concurrent_writes(tmp_path: Path) -> None:
    """3 subprocess writers, 20 writes each -> no BUSY, all 60 rows present."""
    db_path = tmp_path / "state-v2.db"
    conn = connect_database(db_path)
    try:
        with write_transaction(conn):
            conn.execute(SESSIONS_TABLE_DDL)
    finally:
        conn.close()

    repo_root = str(Path(__file__).resolve().parents[2])
    children = [
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _CHILD_SCRIPT, str(db_path), str(writer), repo_root],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for writer in range(_WRITERS)
    ]
    results = []
    for writer, child in enumerate(children):
        stdout, stderr = child.communicate(timeout=_PROC_TIMEOUT)
        results.append((writer, child.returncode, stdout, stderr))

    # every child ran to completion without a single BUSY failure
    for writer, returncode, stdout, stderr in results:
        assert returncode == 0, (
            f"writer {writer} failed (rc={returncode}): {stderr}"  # type: ignore[str-bytes-safe]
        )
        assert f"writer {writer} done busy_errors=0" in stdout
        assert "BUSY" not in stderr
        assert "locked" not in stderr.lower()
        assert "OperationalError" not in stderr

    # all 60 writes (3 writers x 20) are present and readable
    conn = connect_database(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == _WRITERS * _WRITES_PER_WRITER
        for writer in range(_WRITERS):
            row = conn.execute(
                "SELECT state FROM sessions WHERE session_id = ?",
                (f"proc-{writer}-i-{_WRITES_PER_WRITER - 1}",),
            ).fetchone()
            assert row is not None
            state = json.loads(str(row[0]))
            assert state["writer"] == writer
            assert state["i"] == _WRITES_PER_WRITER - 1
    finally:
        conn.close()
