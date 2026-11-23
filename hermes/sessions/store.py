"""Durable session state store for hermes-v2 ACP (plan P21.T12, issue #156173).

Upstream bug: an ACP turn kept running after the editor's ACP process exited,
and a later ``session/load`` of that session failed with a bare
``Internal error`` — the session state had never been persisted on the failure
path. The v2 :class:`SessionStore` is the persistence half of the fix: every
session lives in one JSON file under ``<hermes_home>/sessions`` and is written
with a temp-file + ``os.replace`` atomic swap, so a reader never observes a
torn write and the ACP server can checkpoint turn progress *before* surfacing
any failure (see :mod:`hermes.acp.server`).

Session ids must match ``^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`` (they become
file names), which covers the url-safe ids ACP clients generate; anything
else raises ``ValueError`` instead of silently escaping the store root.

SQLite-backed mode (plan P21.T6, issue #156160-family — ADDITIVE)
-------------------------------------------------------------------------------
``SessionStore(storage=<pool>)`` switches persistence to the v2 SQLite
storage layer (:mod:`hermes.storage.database`): state dicts are written via
pooled connections inside ``BEGIN IMMEDIATE`` transactions
(:func:`hermes.storage.database.write_transaction`) into the ``sessions``
table, with the JSON serialization PRESERVED — the stored payload is the
same ``json.dumps(state, indent=2, sort_keys=True)`` text the JSON mode
writes, so consumers never see a different state shape. Reads/list run on
pooled connections (WAL readers never block writers). This mode exists for
concurrent-access deployments where the JSON file-swap showed the upstream
"state.db WAL errors / session storage unavailable" symptoms; the DEFAULT
stays the JSON file mode, which is unchanged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from hermes.config import hermes_home
from hermes.storage.database import (
    SESSIONS_TABLE_DDL,
    ConnectionPool,
    write_transaction,
)

__all__ = ["SESSION_ID_PATTERN", "SessionStore"]

SESSION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# os.replace on Windows raises PermissionError while another thread holds the
# destination open (e.g. a concurrent load_session); retry briefly instead of
# failing a save that is only contended for microseconds.
_REPLACE_RETRIES: Final[int] = 25
_REPLACE_RETRY_DELAY: Final[float] = 0.01


class SessionStore:
    """Atomic JSON-file store of ACP session state (optional SQLite-backed mode).

    ``storage`` (plan P21.T6) is an optional
    :class:`hermes.storage.database.ConnectionPool` — when provided, all
    three operations persist into that pool's ``sessions`` table instead of
    the JSON files (see the module docstring); the default ``storage=None``
    keeps the JSON mode byte-for-byte unchanged.
    """

    def __init__(
        self,
        root: Path | None = None,
        storage: ConnectionPool | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else hermes_home() / "sessions"
        self._storage = storage
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @property
    def root(self) -> Path:
        """Directory holding the ``<session_id>.json`` state files (JSON mode)."""
        return self._root

    def save_session(self, session_id: str, state: dict[str, Any]) -> Path:
        """Atomically write ``state`` for ``session_id``; returns the state file path.

        The write is a temp-file + ``os.replace`` swap: concurrent readers see
        either the previous or the new state, never a partial file, so a crash
        mid-save cannot corrupt a session that a later ``session/load`` needs.

        Storage-backed mode: the same JSON-serialized state goes into the
        ``sessions`` table inside an IMMEDIATE transaction; the returned
        path is the pool's database file (no state file is written).
        """
        self._validate_session_id(session_id)
        payload = json.dumps(state, indent=2, sort_keys=True)
        if self._storage is None:
            path = self._path_for(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            try:
                tmp.write_text(payload, encoding="utf-8")
                _atomic_replace(tmp, path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return path
        self._ensure_schema()
        with self._storage.connection() as conn, write_transaction(conn):
            conn.execute(
                "INSERT INTO sessions (session_id, state, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "state = excluded.state, updated_at = excluded.updated_at",
                (session_id, payload, datetime.now(UTC).isoformat()),
            )
        return self._storage.db_path

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Return the saved state for ``session_id``, or ``None`` if never saved.

        A present-but-corrupt state file raises a descriptive ``ValueError``:
        silently returning ``None`` would turn a data-loss bug into a mystery.
        """
        self._validate_session_id(session_id)
        if self._storage is None:
            path = self._path_for(session_id)
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            where = f" at {path}"
        else:
            self._ensure_schema()
            with self._storage.connection() as conn:
                row = conn.execute(
                    "SELECT state FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
            if row is None:
                return None
            raw = str(row[0])
            where = ""
        try:
            state = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"session state for {session_id!r}{where} is not valid JSON "
                f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
            ) from exc
        if not isinstance(state, dict):
            raise ValueError(
                f"session state for {session_id!r}{where} must be a JSON object, "
                f"got {type(state).__name__}"
            )
        return state

    def list_sessions(self) -> list[str]:
        """Sorted ids of every saved session."""
        if self._storage is None:
            if not self._root.exists():
                return []
            return sorted(p.stem for p in self._root.glob("*.json") if p.is_file())
        self._ensure_schema()
        with self._storage.connection() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY session_id"
            ).fetchall()
        return sorted(str(row[0]) for row in rows)

    # ── storage-backed mode internals (plan P21.T6) ─────────────────────────
    def _ensure_schema(self) -> None:
        """Create the ``sessions`` table once per store instance (idempotent DDL).

        The CREATE runs inside an IMMEDIATE transaction so concurrent
        instances/processes serialise; ``IF NOT EXISTS`` makes the loser a
        no-op. Errors propagate — a storage-backed store that cannot create
        its table must fail loudly, not silently lose sessions.
        """
        if self._schema_ready or self._storage is None:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._storage.connection() as conn, write_transaction(conn):
                conn.execute(SESSIONS_TABLE_DDL)
            self._schema_ready = True

    def _validate_session_id(self, session_id: str) -> None:
        """Shared id validation (JSON mode turns it into a file name)."""
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.match(session_id):
            raise ValueError(
                f"invalid session id {session_id!r}: must match "
                f"{SESSION_ID_PATTERN.pattern}"
            )

    def _path_for(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        return self._root / f"{session_id}.json"


def _atomic_replace(tmp: Path, path: Path) -> None:
    """``os.replace`` with a bounded retry for Windows sharing violations."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY)
