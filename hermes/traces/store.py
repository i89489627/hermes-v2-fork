"""hermes-v2 trace store (plan P11.T1).

Every completed turn produces one :class:`~hermes.traces.schema.Trace`
record (plan Section 2.6, "memory-first" observability). The store keeps two
synchronized representations:

- **JSONL source of truth**: one append-only file per day under
  ``<root>/v1.0/YYYY-MM-DD.jsonl`` (``root`` defaults to
  ``hermes_home() / "traces"``). Appends are a single write, so the hot path
  stays fast. Quality relabeling rewrites that day's file atomically.
- **SQLite search index**: ``<root>/traces.db`` mirrors every trace as a row
  (``traces`` table) with an FTS5 virtual table (``traces_fts``) over
  ``user_prompt`` plus the serialized stage payloads (extraction / review /
  gating / injection). If the runtime SQLite lacks FTS5, the store degrades
  to a LIKE-based scan and :attr:`fts_available` reports which path is
  active; CPython on Windows ships FTS5.

Schema DDL lives in ``migrations.sql`` next to this module; it is idempotent
(``CREATE ... IF NOT EXISTS``) and its FTS5 statement is last so a missing
FTS5 module cannot prevent the base table from being created.

``trace_id`` values are uuid4 hex strings, generated on append when the
caller leaves them blank and assigned back onto the passed trace.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from hermes.config import hermes_home
from hermes.traces.schema import QUALITY_LABELS, Trace

__all__ = ["TraceStore"]

_JSONL_VERSION_DIR: Final[str] = "v1.0"
_DB_FILENAME: Final[str] = "traces.db"
_MIGRATIONS_FILENAME: Final[str] = "migrations.sql"

_SELECT_COLUMNS: Final[str] = (
    "trace_id, turn_id, session_id, timestamp, user_prompt, "
    "models_used, payload, quality_label, pii_flagged"
)

_UPSERT_SQL: Final[str] = (
    "INSERT INTO traces (trace_id, turn_id, session_id, timestamp, day, "
    "user_prompt, models_used, payload, quality_label, pii_flagged) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(trace_id) DO UPDATE SET "
    "turn_id=excluded.turn_id, session_id=excluded.session_id, "
    "timestamp=excluded.timestamp, day=excluded.day, "
    "user_prompt=excluded.user_prompt, models_used=excluded.models_used, "
    "payload=excluded.payload, quality_label=excluded.quality_label, "
    "pii_flagged=excluded.pii_flagged"
)

_FTS_SEARCH_SQL: Final[str] = (
    "SELECT t.* FROM traces_fts, traces AS t "
    "WHERE traces_fts MATCH ? AND t.trace_id = traces_fts.trace_id "
    "ORDER BY bm25(traces_fts) LIMIT ?"
)

_LIKE_SEARCH_SQL: Final[str] = (
    f"SELECT {_SELECT_COLUMNS} FROM traces "
    "WHERE user_prompt LIKE ? ESCAPE '\\' OR payload LIKE ? ESCAPE '\\' "
    "ORDER BY timestamp DESC LIMIT ?"
)


def _day_stamp(timestamp: float) -> str:
    """UTC calendar day (``YYYY-MM-DD``) of an epoch timestamp."""
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")


def _jsonl_path(root: Path, day: str) -> Path:
    """Path of the JSONL file for ``day`` (``<root>/v1.0/<day>.jsonl``)."""
    return root / _JSONL_VERSION_DIR / f"{day}.jsonl"


def _fts_match_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (ANDed quoted phrases).

    Every whitespace token is wrapped in double quotes (with embedded quotes
    stripped), so user input can never inject FTS5 operators or break the
    query syntax; consecutive phrases are implicitly ANDed.
    """
    tokens = [token.replace('"', "") for token in query.split()]
    return " ".join(f'"{token}"' for token in tokens if token)


def _like_pattern(query: str) -> str:
    """Escape ``query`` for a LIKE pattern with ``ESCAPE '\\'``."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _validate_trace(trace: Trace) -> None:
    """Eagerly validate a trace before it reaches JSONL or the index."""
    if not isinstance(trace, Trace):
        raise TypeError(f"trace must be a Trace, got {type(trace).__name__}")
    for name in ("trace_id", "turn_id", "session_id", "user_prompt"):
        value = getattr(trace, name)
        if not isinstance(value, str):
            raise TypeError(f"trace.{name} must be str, got {type(value).__name__}")
    if not isinstance(trace.models_used, list) or not all(
        isinstance(item, str) for item in trace.models_used
    ):
        raise TypeError("trace.models_used must be a list[str]")
    if isinstance(trace.timestamp, bool) or not isinstance(
        trace.timestamp, int | float
    ):
        raise TypeError(
            f"trace.timestamp must be float epoch seconds, got "
            f"{type(trace.timestamp).__name__}"
        )
    label = trace.quality_label
    if label is not None and label not in QUALITY_LABELS:
        raise ValueError(
            f"trace.quality_label must be one of {sorted(QUALITY_LABELS)} or "
            f"None, got {label!r}"
        )
    if not isinstance(trace.pii_flagged, bool):
        raise TypeError(
            f"trace.pii_flagged must be bool, got {type(trace.pii_flagged).__name__}"
        )
    for name in ("extraction", "review", "gating", "injection"):
        value = getattr(trace, name)
        if value is not None and not isinstance(value, dict):
            raise TypeError(
                f"trace.{name} must be a dict or None, got {type(value).__name__}"
            )


def _stage_from_payload(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Extract a stage payload from a deserialized db ``payload`` blob."""
    value = payload.get(key)
    return value if isinstance(value, dict) else None


class TraceStore:
    """Append-only turn traces (JSONL) with a searchable SQLite/FTS5 index."""

    def __init__(self, root: Path | None = None) -> None:
        """Open (and lazily create) the trace store rooted at ``root``."""
        self.root: Path = root if root is not None else hermes_home() / "traces"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._fts_available = False
        self.root.mkdir(parents=True, exist_ok=True)
        self._connection()

    @property
    def fts_available(self) -> bool:
        """True when the FTS5 index is active (``False`` -> LIKE fallback)."""
        return self._fts_available

    # ── public API ────────────────────────────────────────────────────────
    def append(self, trace: Trace) -> str:
        """Write ``trace`` to the day's JSONL file and the index; returns id.

        The trace id is generated (uuid4 hex) when the caller leaves it blank
        and is assigned back onto ``trace``. Appending a trace whose id
        already exists upserts it (JSONL gains a new line; the db row and the
        FTS entry are replaced).
        """
        _validate_trace(trace)
        with self._lock:
            if not trace.trace_id:
                trace.trace_id = uuid.uuid4().hex
            self._append_jsonl(trace)
            self._upsert_db(trace)
            return trace.trace_id

    def search(self, query: str, limit: int = 10) -> list[Trace]:
        """Full-text search over prompts + stage payloads, most relevant first.

        Runs via the FTS5 ``MATCH`` index (bm25 ranking) when available, via a
        LIKE scan otherwise. A whitespace-only query returns ``[]``. ``limit``
        must be a positive int.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"limit must be int, got {type(limit).__name__}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        trimmed = query.strip()
        if not trimmed:
            return []
        with self._lock:
            if self._fts_available:
                rows = self._search_fts(trimmed, limit)
            else:
                rows = self._search_like(trimmed, limit)
        return [self._trace_from_row(row) for row in rows]

    def label_quality(self, trace_id: str, label: str) -> None:
        """Set ``quality_label`` on the JSONL line and the db row.

        ``label`` must be one of :data:`QUALITY_LABELS`. The trace's day JSONL
        file is rewritten atomically; the db row is updated in place. Raises
        ``KeyError`` for an unknown ``trace_id`` and ``ValueError`` for an
        invalid ``label``.
        """
        if not isinstance(trace_id, str):
            raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
        if label not in QUALITY_LABELS:
            raise ValueError(
                f"quality label must be one of {sorted(QUALITY_LABELS)}, got {label!r}"
            )
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT day FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown trace_id: {trace_id}")
            self._rewrite_jsonl_label(str(row["day"]), trace_id, label)
            conn.execute(
                "UPDATE traces SET quality_label = ? WHERE trace_id = ?",
                (label, trace_id),
            )
            conn.commit()

    def get_gepa_ready(self) -> list[Trace]:
        """Traces labeled ``good`` with no PII flag — the GEPA training set.

        Ordered by timestamp ascending for a stable training order.
        """
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM traces "
                "WHERE quality_label = 'good' AND pii_flagged = 0 "
                "ORDER BY timestamp ASC"
            ).fetchall()
        return [self._trace_from_row(row) for row in rows]

    def get(self, trace_id: str) -> Trace | None:
        """Return one trace by id, or ``None`` when unknown (ops/test helper)."""
        if not isinstance(trace_id, str):
            raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        return self._trace_from_row(row) if row is not None else None

    def get_all(self, limit: int | None = None) -> list[Trace]:
        """Every stored trace, oldest first (additive helper for plan P11.T4).

        The GEPA-ready filter (``hermes/traces/gepa_filter.py``) walks the
        whole store to run its own PII scan over prompts + stage payloads —
        :meth:`get_gepa_ready` alone cannot serve that, because the stored
        ``pii_flagged`` flag only reflects scans that happened upstream of
        the store. ``limit`` (when given, a positive int) keeps only the
        NEWEST ``limit`` traces (the tail of the same timestamp-ascending
        order; ``trace_id`` breaks timestamp ties deterministically).
        """
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError(f"limit must be a positive int or None, got {limit!r}")
        with self._lock:
            conn = self._connection()
            if limit is None:
                rows = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM traces "
                    "ORDER BY timestamp ASC, trace_id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM traces "
                    "ORDER BY timestamp DESC, trace_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        traces = [self._trace_from_row(row) for row in rows]
        if limit is not None:
            traces.reverse()
        return traces

    def close(self) -> None:
        """Close the SQLite connection (the store reconnects lazily on reuse)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── internals ─────────────────────────────────────────────────────────
    def _connection(self) -> sqlite3.Connection:
        conn = self._conn
        if conn is None:
            conn = sqlite3.connect(self.root / _DB_FILENAME, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._migrate(conn)
            self._conn = conn
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply migrations.sql (idempotent); degrade gracefully without FTS5."""
        migrations_path = Path(__file__).resolve().with_name(_MIGRATIONS_FILENAME)
        script = migrations_path.read_text(encoding="utf-8")
        try:
            conn.executescript(script)
        except sqlite3.OperationalError:
            # The FTS5 statement is last in migrations.sql, so the base table
            # and indexes already exist when the fts5 module is unavailable.
            self._fts_available = False
            return
        self._fts_available = True

    def _append_jsonl(self, trace: Trace) -> None:
        path = _jsonl_path(self.root, _day_stamp(trace.timestamp))
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _upsert_db(self, trace: Trace) -> None:
        conn = self._connection()
        payload = json.dumps(
            {
                "extraction": trace.extraction,
                "review": trace.review,
                "gating": trace.gating,
                "injection": trace.injection,
            },
            ensure_ascii=False,
        )
        models = json.dumps(trace.models_used, ensure_ascii=False)
        day = _day_stamp(trace.timestamp)
        conn.execute(
            _UPSERT_SQL,
            (
                trace.trace_id,
                trace.turn_id,
                trace.session_id,
                float(trace.timestamp),
                day,
                trace.user_prompt,
                models,
                payload,
                trace.quality_label,
                int(trace.pii_flagged),
            ),
        )
        conn.execute("DELETE FROM traces_fts WHERE trace_id = ?", (trace.trace_id,))
        conn.execute(
            "INSERT INTO traces_fts (trace_id, user_prompt, payload) VALUES (?, ?, ?)",
            (trace.trace_id, trace.user_prompt, payload),
        )
        conn.commit()

    def _search_fts(self, query: str, limit: int) -> list[sqlite3.Row]:
        match_expr = _fts_match_query(query)
        if not match_expr:
            return []
        conn = self._connection()
        return list(conn.execute(_FTS_SEARCH_SQL, (match_expr, limit)).fetchall())

    def _search_like(self, query: str, limit: int) -> list[sqlite3.Row]:
        conn = self._connection()
        pattern = _like_pattern(query)
        return list(conn.execute(_LIKE_SEARCH_SQL, (pattern, pattern, limit)).fetchall())

    def _rewrite_jsonl_label(self, day: str, trace_id: str, label: str) -> None:
        """Rewrite the day's JSONL with ``quality_label`` updated (atomic)."""
        path = _jsonl_path(self.root, day)
        if not path.exists():
            raise KeyError(f"trace {trace_id} missing from JSONL file {path}")
        updated: list[str] = []
        found = False
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    doc: Any = json.loads(stripped)
                except json.JSONDecodeError:
                    updated.append(stripped)  # preserve unreadable lines as-is
                    continue
                if isinstance(doc, dict) and doc.get("trace_id") == trace_id:
                    doc["quality_label"] = label
                    found = True
                    updated.append(json.dumps(doc, ensure_ascii=False))
                else:
                    updated.append(stripped)
        if not found:
            raise KeyError(f"trace {trace_id} missing from JSONL file {path}")
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write("".join(entry + "\n" for entry in updated))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def _trace_from_row(self, row: sqlite3.Row) -> Trace:
        """Rebuild a :class:`Trace` from a ``traces`` table row."""
        payload: dict[str, Any] = json.loads(row["payload"]) if row["payload"] else {}
        models_raw: Any = json.loads(row["models_used"]) if row["models_used"] else []
        models = (
            [str(item) for item in models_raw] if isinstance(models_raw, list) else []
        )
        return Trace(
            trace_id=str(row["trace_id"]),
            turn_id=str(row["turn_id"]),
            session_id=str(row["session_id"]),
            timestamp=float(row["timestamp"]),
            user_prompt=str(row["user_prompt"]),
            models_used=models,
            extraction=_stage_from_payload(payload, "extraction"),
            review=_stage_from_payload(payload, "review"),
            gating=_stage_from_payload(payload, "gating"),
            injection=_stage_from_payload(payload, "injection"),
            quality_label=row["quality_label"],
            pii_flagged=bool(row["pii_flagged"]),
        )
