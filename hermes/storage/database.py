"""hermes-v2 storage layer: SQLite session/state databases (plan P21.T6).

Upstream bug (plan Phase 21, #156160-family — "state.db Write-Ahead Logging
errors / session storage unavailable"): under concurrent access the state
database surfaced WAL errors and "session storage unavailable" failures.
The fix is disciplined SQLite configuration, not a new database:

- :func:`connect_database` — every connection carries
  ``PRAGMA busy_timeout`` (default 5000ms, the plan acceptance value),
  ``PRAGMA journal_mode=WAL`` (with the fork-base safety fallback below)
  and ``PRAGMA synchronous=NORMAL`` (WAL's documented durable-enough
  setting for session state). The default database lives under the config
  home (``hermes_home() / "state-v2.db"``) — this module serves the v2
  tree's session/state databases ONLY; the fork-base ``hermes_state.py``
  machinery (and its ``state.db``) stays untouched.
- :func:`write_transaction` — writes run inside ``BEGIN IMMEDIATE``
  transactions. IMMEDIATE takes the write lock UP FRONT instead of
  upgrading a deferred transaction at first write, which is the classic
  SQLITE_BUSY deadlock shape (two writers both hold the shared read lock,
  one fails its upgrade); combined with ``busy_timeout`` competing writers
  queue instead of failing.
- :class:`ConnectionPool` — a small thread-safe checkout/checkin pool
  (default maxsize 8, context-manager use) so threads share a bounded set
  of long-lived connections instead of reopen-per-write.

WAL safety policy (mirrors the fork-base single source of truth)
------------------------------------------------------------------------------
The fork-base (``hermes_cli.sqlite_runtime.is_sqlite_wal_reset_vulnerable``,
mirrored by ``tests/conftest.py::_wal_is_usable``) refuses ``journal_mode=WAL``
on SQLite builds carrying the upstream WAL-reset corruption bug
(3.7.0–3.51.2, excluding the 3.50.7 / 3.44.6 backports) and falls back to
``DELETE``. :func:`runtime_journal_mode` exposes the same decision so the
v2 tree never becomes the one WAL-enabled database on a vulnerable runtime;
on such builds ``connect_database`` logs and uses ``DELETE`` (concurrency
safety still comes from ``busy_timeout`` + IMMEDIATE transactions, so the
"no SQLITE_BUSY under concurrent load" acceptance holds in both modes).
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, TypeAlias

from hermes.config import hermes_home

logger = logging.getLogger(__name__)

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

#: Plan P21.T6 acceptance: "busy_timeout configured (default 5000ms)".
DEFAULT_BUSY_TIMEOUT_MS: Final[int] = 5000

#: Plan P21.T6: "connection pooling" — the default pool width.
DEFAULT_POOL_SIZE: Final[int] = 8

#: The v2 session-state table (plan P21.T6: sessions/store.py's optional
#: storage-backed mode persists the same session state dicts here; JSON
#: serialization is preserved).
SESSIONS_TABLE_DDL: Final[str] = (
    "CREATE TABLE IF NOT EXISTS sessions ("
    " session_id TEXT PRIMARY KEY,"
    " state TEXT NOT NULL,"
    " updated_at TEXT NOT NULL)"
)

#: Bound on waiting for a checked-out connection to come back (defensive —
#: callers always release through the :meth:`ConnectionPool.connection`
#: context manager).
_CHECKOUT_WAIT_SECONDS: Final[float] = 60.0


def runtime_journal_mode() -> str:
    """The journal mode this runtime may use: ``"wal"`` or ``"delete"``.

    ``"delete"`` exactly when the linked SQLite carries the upstream
    WAL-reset bug (fork-base policy,
    :func:`hermes_cli.sqlite_runtime.is_sqlite_wal_reset_vulnerable` — the
    stdlib-only, import-safe single source of truth). If the fork-base
    module is not importable (installed-package deployments), the decision
    fails open to ``"wal"``: the bug window is old and the alternative
    (silently degrading every v2 database) is worse than the risk.
    """
    try:
        from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable
    except Exception:  # pragma: no cover - fork-base missing (packaged installs)
        logger.debug("sqlite_runtime unavailable; assuming a WAL-capable runtime")
        return "wal"
    return (
        "delete"
        if is_sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info)
        else "wal"
    )


def _default_db_path() -> Path:
    """The v2 session/state database default: ``<hermes_home>/state-v2.db``.

    Config-derived (:func:`hermes.config.hermes_home`) and deliberately NOT
    the fork-base ``state.db`` — the two trees must never share a database
    file.
    """
    return hermes_home() / "state-v2.db"


def connect_database(
    db_path: Path | None = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open one configured SQLite connection (plan P21.T6 fix).

    ``db_path`` defaults to :func:`_default_db_path`. The connection carries
    the three pragmas the bug fix requires — ``busy_timeout``
    (``busy_timeout_ms``, default 5000), the runtime-safe journal mode
    (:func:`runtime_journal_mode`) and ``synchronous=NORMAL`` — plus
    ``check_same_thread=False`` (pooled connections move between threads,
    one at a time) and manual transaction control (``isolation_level=None``)
    so :func:`write_transaction`'s explicit ``BEGIN IMMEDIATE`` never races
    Python's implicit transaction management. The parent directory is
    created if needed.
    """
    if busy_timeout_ms < 0:
        raise ValueError(f"busy_timeout_ms must be >= 0, got {busy_timeout_ms}")
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000.0, check_same_thread=False)
    conn.isolation_level = None  # write_transaction owns transaction boundaries
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    journal_mode = runtime_journal_mode()
    if journal_mode == "wal":
        conn.execute("PRAGMA journal_mode = WAL")
    else:
        logger.warning(
            "SQLite %s has the upstream WAL-reset bug; the v2 database at %s "
            "uses journal_mode=DELETE (fork-base policy)",
            sqlite3.sqlite_version,
            path,
        )
        conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write inside a ``BEGIN IMMEDIATE`` transaction (plan P21.T6 fix).

    IMMEDIATE takes the write lock at BEGIN, so concurrent writers QUEUE on
    ``busy_timeout`` instead of failing the deferred-transaction upgrade
    with SQLITE_BUSY. Commits on success; on any failure rolls back
    (rollback errors are logged, never allowed to mask the original
    exception) and re-raises. Connections must use manual transaction
    control (:func:`connect_database` sets it).
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            logger.warning("write_transaction: rollback failed", exc_info=True)
        raise
    else:
        conn.execute("COMMIT")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the v2 session-state table (idempotent; caller wraps in a write tx)."""
    conn.execute(SESSIONS_TABLE_DDL)


class ConnectionPool:
    """Thread-safe SQLite connection pool (plan P21.T6: "connection pooling").

    ``connection()`` is the (only) use surface: a context manager that
    checks a configured connection out, yields it, and always checks it
    back in (residue transactions rolled back; a poisoned connection is
    discarded). Connections are created lazily up to ``maxsize`` (default
    8, the pool width this task ships) and reused; checkout beyond capacity
    waits for a checkin. For session/state databases under the v2 tree —
    the fork-base ``hermes_state.py`` stays untouched.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        maxsize: int = DEFAULT_POOL_SIZE,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if busy_timeout_ms < 0:
            raise ValueError(f"busy_timeout_ms must be >= 0, got {busy_timeout_ms}")
        self._db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._maxsize = maxsize
        self._busy_timeout_ms = busy_timeout_ms
        self._idle: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._opened = 0
        self._closed = False

    @property
    def db_path(self) -> Path:
        """The database file every pooled connection opens."""
        return self._db_path

    @property
    def maxsize(self) -> int:
        """The configured pool width."""
        return self._maxsize

    @property
    def busy_timeout_ms(self) -> int:
        """The busy timeout every pooled connection is configured with."""
        return self._busy_timeout_ms

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Check a connection out for the with-block; always checked back in."""
        conn = self._checkout()
        try:
            yield conn
        finally:
            self._checkin(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A :func:`write_transaction` over a pooled connection (convenience).

        ``with pool.transaction() as conn:`` = checkout + BEGIN IMMEDIATE +
        commit/rollback + checkin — the shape every session write uses.
        """
        with self.connection() as conn, write_transaction(conn) as tx_conn:
            yield tx_conn

    def ensure_schema(self) -> None:
        """Create the v2 session-state table through the pool (idempotent)."""
        with self.transaction() as conn:
            conn.execute(SESSIONS_TABLE_DDL)

    def close(self) -> None:
        """Close every idle pooled connection; checked-out ones close on checkin.

        Idempotent; a closed pool refuses further checkouts.
        """
        self._closed = True
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            conn.close()

    # ── internals ──────────────────────────────────────────────────────────
    def _checkout(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("connection pool is closed")
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._opened < self._maxsize:
                self._opened += 1
                return connect_database(self._db_path, self._busy_timeout_ms)
        try:
            return self._idle.get(timeout=_CHECKOUT_WAIT_SECONDS)
        except queue.Empty as exc:
            raise RuntimeError(
                f"connection pool exhausted ({self._maxsize} connection(s)); "
                f"no checkin within {_CHECKOUT_WAIT_SECONDS:.0f}s"
            ) from exc

    def _checkin(self, conn: sqlite3.Connection) -> None:
        if self._closed:
            conn.close()
            return
        try:
            conn.rollback()  # leave no residue transaction for the next user
        except sqlite3.Error:
            logger.warning(
                "connection pool: discarding a poisoned connection", exc_info=True
            )
            self._discard(conn)
            return
        try:
            self._idle.put_nowait(conn)
        except queue.Full:
            self._discard(conn)

    def _discard(self, conn: sqlite3.Connection) -> None:
        conn.close()
        with self._lock:
            self._opened -= 1


#: The pool handle injected into storage-backed consumers (plan P21.T6:
#: ``SessionStore(storage: DatabaseHandle | None=None)`` — the documented
#: spec name for the pooled-connection handle).
DatabaseHandle: TypeAlias = ConnectionPool
