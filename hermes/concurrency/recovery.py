"""Writer crash recovery: client-side journal + replay to a fresh writer (plan P6.T3).

Plan Phase 6: "Crash recovery: if writer dies, agents detect and restart."
Agents detect writer death (the loopback connection dies) and must replay the
writes that never reached durable storage. This module is that recovery
layer, built entirely CLIENT-SIDE so ``hermes/concurrency/writer.py`` stays
untouched — the writer never learns about the journal (plan P6.T3 design: a
"durable sidecar the WRITER does NOT know about"):

- :class:`WriteJournal` — append-only JSONL sidecar (default
  ``<hermes_home()>/journal/writer_journal.jsonl``, derived from
  :func:`hermes.config.hermes_home`); every op is journaled and fsynced
  BEFORE it is sent, and marked acked when the
  :class:`~hermes.concurrency.protocol.WriteResponse` reports success.
- :class:`JournalingClient` — a writer connection with journal-before-send
  semantics; connection failures raise :class:`WriterConnectionError` and
  leave the op journaled-unacked, which is exactly the state the next
  startup replays. An op-level rejection (``ok=False``) is returned as-is
  and stays journaled-pending — the caller may correct and retry.
- :func:`recover` — after a crash: read the journal, skip acked ops, resend
  UNACKED ops to a fresh writer (built by the caller-supplied
  ``writer_factory`` — the "agents detect writer death and restart" step),
  mark each replay acked or failed, then make the surviving state durable
  in Git.

Idempotent replay — the dedupe rule (P6.T3 design note)
------------------------------------------------------------------------------
``add`` creates a NEW entry on every application and the store has no
request dedupe, so a blind replay would duplicate any op the dead writer
already applied (applied-but-unacked: the response was lost with the crash,
or the client died between response and journal ack). Recovery therefore
makes ``add`` idempotent AT THE RECOVERY LAYER with this rule:

    before replaying an ``add`` op, check whether an entry with IDENTICAL
    text AND context already exists (``store.list()`` text match); if so,
    the op is presumed applied by the dead writer — skip the wire replay
    (counted as ``deduped`` in the summary and marked acked in the journal).

``update`` ops carry their target ``entry_id`` (re-applying is naturally
idempotent) and ``remove`` answers ``removed=False`` for an already-removed
id, so neither needs the dedupe check. A replayed-but-rejected op
(``ok=False``) is terminal: it is marked failed in the journal so later
recovery passes do not retry it forever.

Durability sweep — the writer's batching window
------------------------------------------------------------------------------
The writer applies ops to the store (files on disk) immediately but
coalesces Git commits into flush windows (``_BatchingGitBackend``); a hard
writer crash inside a window leaves applied entries present as files but
UNCOMMITTED — and acked ops can be in that state too. After the replay
loop, recovery commits the repository's dirty state as ONE
``[memory] recovery`` commit (under the store's shared ``.lock`` file, via
:func:`hermes.concurrency.manager.get_shared_lock`), so "no write lost on
crash" holds at the Git level as well. When ``store`` is the fresh writer's
own store instance, that durability commit flows through the writer's
batching window and lands within ``batch_window_s`` — callers finish with
``WriterProcess.flush_pending()``.

IMPORTANT — single committing handle: pass the store instance the fresh
writer itself uses. libgit2 keeps one in-memory Git index per repository
handle and never reloads it from disk (see ``MemoryStore._stage_removal``);
two handles committing interleaved on one repository drop each other's
staged files from the tree. Recovery's durability commit reloads the
handle's index from disk before committing, but the replayed ops themselves
MUST flow through the writer's own handle.

Bounded replay (plan acceptance: "crash recovery within 5s")
------------------------------------------------------------------------------
``recover`` runs the whole replay under ``budget_s`` (default 5.0): the
fresh-writer connection gets the remaining budget as its socket timeout and
the loop stops when the budget is exhausted. Un-replayed ops simply stay
journaled-unacked (counted in ``left_pending``) for the next :func:`recover`
call. Nothing in the replay path retries: one pass, one connection, bounded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from hermes.concurrency.manager import get_shared_lock
from hermes.concurrency.protocol import (
    ProtocolError,
    WriteRequest,
    WriteResponse,
    read_response,
    write_request,
)
from hermes.config import hermes_home
from hermes.memory.entry import MemoryEntry
from hermes.memory.git_backend import GitBackend
from hermes.memory.locking import LockTimeoutError
from hermes.memory.store import MemoryStore

__all__ = [
    "DEFAULT_IO_TIMEOUT_S",
    "DEFAULT_REPLAY_BUDGET_S",
    "JournalRecord",
    "JournalingClient",
    "RecoverySummary",
    "WriteJournal",
    "WriterConnectionError",
    "default_journal_path",
    "recover",
]

logger = logging.getLogger(__name__)

#: Replay budget in seconds (plan acceptance: "crash recovery within 5s").
DEFAULT_REPLAY_BUDGET_S: Final[float] = 5.0

#: Socket I/O timeout for ordinary (non-recovery) client round trips.
DEFAULT_IO_TIMEOUT_S: Final[float] = 30.0

#: Minimum socket timeout handed to any single replay step.
_MIN_STEP_TIMEOUT_S: Final[float] = 0.05

#: Commit message of the post-replay durability sweep (git-history marker).
RECOVERY_COMMIT_MESSAGE: Final[str] = (
    "[memory] recovery: commit writes surviving a writer crash"
)


def default_journal_path() -> Path:
    """The writer journal sidecar: ``<hermes_home()>/journal/writer_journal.jsonl``."""
    return hermes_home() / "journal" / "writer_journal.jsonl"


class WriterConnectionError(ConnectionError):
    """The writer could not be reached (crashed, refused, reset, timed out).

    The failed op is already journaled-pending, so the recovery layer (or the
    next :meth:`JournalingClient.send_op` on a restarted writer) replays it.
    """


@dataclass
class JournalRecord:
    """One journaled write op with its recovery status.

    ``acked``/``failed`` are resolved from the journal's ``ack``/``failed``
    lines; a record with neither is UNACKED — the replay candidate.
    """

    request_id: int | str
    client_id: str
    op: dict[str, Any]
    ts: float
    acked: bool = False
    failed: bool = False
    error: str | None = None
    entry_id: str | None = None


class WriteJournal:
    """Append-only JSONL journal of write ops (the crash-recovery sidecar).

    Layout: one JSON object per line.

    - ``{"type": "pending", "request_id", "client_id", "op", "ts"}`` —
      appended BEFORE the op is sent (fsynced);
    - ``{"type": "ack", "request_id", "entry_id"?, "ts"}`` — appended when
      the writer answered ``ok=True`` (``entry_id`` from ``add`` results);
    - ``{"type": "failed", "request_id", "error", "ts"}`` — appended for a
      terminal ``ok=False`` rejection so recovery never retries it.

    A request_id is journal-unique (clients emit ``"<client_id>-<n>"``), so
    an ack/failed line resolves the pending record it finalizes. Torn lines
    (a crash mid-append) are skipped with a warning on read — the record
    before the tear stays pending and is replayed, never lost.
    """

    def __init__(self, journal_path: Path | None = None) -> None:
        self.path: Path = (
            Path(journal_path) if journal_path is not None else default_journal_path()
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    # ── append path ────────────────────────────────────────────────────────
    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())  # durable BEFORE the op goes on the wire

    def append_pending(self, request_id: int | str, client_id: str, op: dict[str, Any]) -> None:
        """Journal one op as pending — call BEFORE sending it to the writer."""
        self._append({
            "type": "pending",
            "request_id": request_id,
            "client_id": client_id,
            "op": op,
            "ts": time.time(),
        })

    def mark_acked(self, request_id: int | str, result: Any | None = None) -> None:
        """Record a successful application (``entry_id`` extracted from ``add`` results)."""
        record: dict[str, Any] = {"type": "ack", "request_id": request_id, "ts": time.time()}
        if isinstance(result, dict) and isinstance(result.get("entry_id"), str):
            record["entry_id"] = result["entry_id"]
        self._append(record)

    def mark_failed(self, request_id: int | str, error: str) -> None:
        """Record a terminal rejection so later recovery passes skip the op."""
        self._append({
            "type": "failed",
            "request_id": request_id,
            "error": error,
            "ts": time.time(),
        })

    # ── read path ──────────────────────────────────────────────────────────
    def records(self) -> list[JournalRecord]:
        """All journaled pending records, resolved with their ack/failure state."""
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        by_id: dict[int | str, JournalRecord] = {}
        order: list[JournalRecord] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frame = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("recovery journal %s: skipping torn line", self.path)
                continue
            if not isinstance(frame, dict):
                continue
            request_id = frame.get("request_id")
            if not isinstance(request_id, (int, str)) or isinstance(request_id, bool):
                continue
            kind = frame.get("type")
            if kind == "pending":
                op = frame.get("op")
                client_id = frame.get("client_id")
                ts = frame.get("ts")
                if not isinstance(op, dict) or not isinstance(client_id, str):
                    continue
                record = JournalRecord(
                    request_id=request_id,
                    client_id=client_id,
                    op=op,
                    ts=float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool)
                    else 0.0,
                )
                by_id[request_id] = record
                order.append(record)
            elif kind == "ack":
                target = by_id.get(request_id)
                if target is not None:
                    target.acked = True
                    entry_id = frame.get("entry_id")
                    if isinstance(entry_id, str):
                        target.entry_id = entry_id
            elif kind == "failed":
                target = by_id.get(request_id)
                if target is not None:
                    target.failed = True
                    error = frame.get("error")
                    target.error = error if isinstance(error, str) else "unknown"
        return order

    def unacked(self) -> list[JournalRecord]:
        """Pending records with no ack/failed line, in journal (send) order."""
        return [record for record in self.records() if not record.acked and not record.failed]

    def acked(self) -> list[JournalRecord]:
        """Pending records the journal proves were applied before the crash."""
        return [record for record in self.records() if record.acked]

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self) -> None:
        """Close the append handle; the file remains readable for recovery."""
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> WriteJournal:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class JournalingClient:
    """A writer connection that journals every op BEFORE sending it (plan P6.T3).

    ``send_op`` appends a ``pending`` record (fsynced), sends the op over the
    loopback protocol, waits for the :class:`WriteResponse` and marks the
    record acked on ``ok=True``. If the writer is dead (connection refused,
    reset, EOF, I/O timeout) the op STAYS journaled-unacked and
    :class:`WriterConnectionError` is raised — the "agents detect writer
    death" step; the journal is exactly the pending-writes list the next
    :func:`recover` replays. Connections are established lazily and re-
    established after a failure.
    """

    def __init__(
        self,
        writer_address: tuple[str, int],
        client_id: str,
        journal: WriteJournal | None = None,
        *,
        connect_timeout_s: float = 5.0,
        io_timeout_s: float = DEFAULT_IO_TIMEOUT_S,
    ) -> None:
        self._address = (str(writer_address[0]), int(writer_address[1]))
        self._client_id = client_id
        self._journal = journal if journal is not None else WriteJournal()
        self._owns_journal = journal is None
        self._connect_timeout_s = connect_timeout_s
        self._io_timeout_s = io_timeout_s
        self._send_lock = threading.Lock()
        self._counter = 0
        self._sock: socket.socket | None = None
        self._rfile: IO[bytes] | None = None
        self._wfile: IO[bytes] | None = None

    # ── public API ─────────────────────────────────────────────────────────
    def send_op(self, op: dict[str, Any]) -> WriteResponse:
        """Journal, send and await one op; returns the writer's response.

        Raises :class:`WriterConnectionError` when the writer is unreachable
        (the op remains journaled-unacked for :func:`recover`).
        """
        with self._send_lock:
            self._counter += 1
            request_id = f"{self._client_id}-{self._counter}"
            self._journal.append_pending(request_id, self._client_id, op)
            try:
                response = self._roundtrip(request_id, op)
            except (OSError, ProtocolError) as exc:
                self._drop_connection()
                raise WriterConnectionError(
                    f"writer at {self._address} unreachable "
                    f"({type(exc).__name__}: {exc}); op stays journaled-unacked"
                ) from exc
            if response.ok:
                self._journal.mark_acked(request_id, response.result)
            return response

    def close(self) -> None:
        """Drop the connection; the journal is closed only if this client owns it."""
        self._drop_connection()
        if self._owns_journal:
            self._journal.close()

    def __enter__(self) -> JournalingClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── connection mechanics ───────────────────────────────────────────────
    def _roundtrip(self, request_id: int | str, op: dict[str, Any]) -> WriteResponse:
        sock, rfile, wfile = self._ensure_connection()
        write_request(
            wfile,
            WriteRequest(request_id=request_id, client_id=self._client_id, op=op),
        )
        return read_response(rfile)

    def _ensure_connection(self) -> tuple[socket.socket, IO[bytes], IO[bytes]]:
        if self._sock is not None and self._rfile is not None and self._wfile is not None:
            return self._sock, self._rfile, self._wfile
        sock = socket.create_connection(self._address, timeout=self._connect_timeout_s)
        try:
            sock.settimeout(self._io_timeout_s)
            # Same Nagle+delayed-ACK avoidance as the writer side (see
            # WriterProcess._accept_loop): small frames must not cost ~40ms.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rfile = sock.makefile("rb")
            wfile = sock.makefile("wb")
        except BaseException:
            sock.close()
            raise
        self._sock, self._rfile, self._wfile = sock, rfile, wfile
        return sock, rfile, wfile

    def _drop_connection(self) -> None:
        sock, rfile, wfile = self._sock, self._rfile, self._wfile
        self._sock = self._rfile = self._wfile = None
        for stream in (rfile, wfile):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


@dataclass(frozen=True)
class RecoverySummary:
    """Outcome of one :func:`recover` run (plan P6.T3: replay pending writes).

    ``replayed`` — unacked ops the fresh writer applied (``ok=True``);
    ``acked_skipped`` — ops the journal already proved applied (no replay);
    ``failed`` — ops the fresh writer rejected (``ok=False``, marked failed);
    ``deduped`` — unacked adds skipped by the dedupe rule (already applied);
    ``left_pending`` — ops the budget/replay did not reach (still journaled);
    ``elapsed_s`` / ``writer_address`` — replay wall time and the fresh
    writer endpoint the ops were replayed to.
    """

    replayed: int
    acked_skipped: int
    failed: int
    deduped: int = 0
    left_pending: int = 0
    elapsed_s: float = 0.0
    writer_address: tuple[str, int] | None = None


class _ReplayConnection:
    """One throwaway connection to the fresh writer, budget-aware per op."""

    def __init__(self, address: tuple[str, int], connect_timeout: float) -> None:
        self._sock = socket.create_connection(address, timeout=connect_timeout)
        self._sock.settimeout(connect_timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rfile = self._sock.makefile("rb")
        self._wfile = self._sock.makefile("wb")

    def roundtrip(
        self, request_id: int | str, client_id: str, op: dict[str, Any], timeout: float
    ) -> WriteResponse:
        """Send one op, wait at most ``timeout`` seconds for its response."""
        self._sock.settimeout(timeout)
        write_request(
            self._wfile,
            WriteRequest(request_id=request_id, client_id=client_id, op=op),
        )
        return read_response(self._rfile)

    def close(self) -> None:
        for stream in (self._rfile, self._wfile):
            with contextlib.suppress(OSError):
                stream.close()
        with contextlib.suppress(OSError):
            self._sock.close()


def _find_applied_duplicate(store: MemoryStore, op: dict[str, Any]) -> MemoryEntry | None:
    """The dedupe rule: an existing entry with identical text AND context."""
    text = op.get("text")
    context = op.get("context", "global")
    if not isinstance(text, str) or not isinstance(context, str):
        return None
    for entry in store.list(context=context):
        if entry.text == text:
            return entry
    return None


def _commit_dirty_state(store: MemoryStore) -> str | None:
    """Commit the repository's dirty state as ONE recovery commit.

    A hard writer crash inside a batching window leaves applied entries on
    disk but uncommitted (acked ops included). This sweep makes the whole
    surviving state durable: every dirty path that still exists on disk is
    staged and committed (paths already staged as removals by
    ``MemoryStore._stage_removal`` are recorded through the shared index).
    Runs under the repo's shared ``.lock`` so it serializes against every
    store mutation and the writer's flush commits.

    Returns the commit sha, or None when the state was already clean, the
    backend is not Git-backed, or the shared lock could not be taken.
    """
    git = store.git
    if not isinstance(git, GitBackend):
        return None
    repo_root = store.root
    shared_lock = get_shared_lock(repo_root)
    try:
        shared_lock.acquire()
    except LockTimeoutError:
        logger.warning("recovery sweep: shared lock busy; durability commit skipped")
        return None
    try:
        # pygit2 status: {repo-relative path: flags}; CURRENT == 0 == clean.
        # Documented repo-internal access (same seam as MemoryStore._stage_removal).
        status = git._repo.status()
        dirty = sorted(
            str(path)
            for path, flags in status.items()
            if flags and (repo_root / str(path)).is_file()
        )
        if not dirty:
            return None
        # libgit2 caches one in-memory index per handle and never reloads it
        # from disk (see MemoryStore._stage_removal) — reload so this commit's
        # tree is built from the on-disk index, never a stale cache.
        git._repo.index.read()
        sha: str = git.commit(RECOVERY_COMMIT_MESSAGE, dirty)
        logger.info("recovery sweep: committed %d surviving path(s) as %s", len(dirty), sha)
        return sha
    finally:
        shared_lock.release()


def recover(
    writer_factory: Callable[[], tuple[str, int]],
    journal_path: Path | None = None,
    *,
    store: MemoryStore | None = None,
    budget_s: float = DEFAULT_REPLAY_BUDGET_S,
    journal: WriteJournal | None = None,
) -> RecoverySummary:
    """Replay journaled-unacked writes to a fresh writer (plan P6.T3).

    Reads the journal (``journal_path``, default :func:`default_journal_path`),
    skips the acked ops, and resends every UNACKED op — in journal order, so
    per-client write order is preserved — to a fresh writer whose address
    comes from calling ``writer_factory()`` (the "restart writer" step; the
    factory owns the restart policy and is only called when there is
    something to replay). Each replayed add is deduped against ``store``
    (pass the fresh writer's OWN store instance — see the module docstring);
    acked ops are skipped, ``ok=False`` replays are marked failed, and the
    post-replay durability sweep commits the surviving Git state.

    Bounded: the whole replay runs under ``budget_s`` (default 5.0 — the
    plan's recovery budget); ops the budget did not cover stay
    journaled-unacked (``left_pending``) for the next :func:`recover` call.
    ``writer_factory`` failures propagate to the caller.
    """
    if budget_s <= 0:
        raise ValueError(f"budget_s must be > 0, got {budget_s}")
    started = time.monotonic()
    owned_journal = journal if journal is not None else WriteJournal(journal_path)
    owns = journal is None
    try:
        unacked = owned_journal.unacked()
        acked_skipped = len(owned_journal.acked())
        if not unacked:
            return RecoverySummary(
                replayed=0,
                acked_skipped=acked_skipped,
                failed=0,
                elapsed_s=time.monotonic() - started,
            )

        deadline = started + budget_s
        address = writer_factory()
        replayed = failed = deduped = left_pending = 0
        remaining = deadline - time.monotonic()
        connection = _ReplayConnection(address, max(remaining, _MIN_STEP_TIMEOUT_S))
        try:
            for index, record in enumerate(unacked):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    left_pending = len(unacked) - index
                    break
                if store is not None and record.op.get("kind") == "add":
                    duplicate = _find_applied_duplicate(store, record.op)
                    if duplicate is not None:
                        deduped += 1
                        owned_journal.mark_acked(
                            record.request_id, {"entry_id": duplicate.id}
                        )
                        continue
                try:
                    response = connection.roundtrip(
                        record.request_id, record.client_id, record.op, remaining
                    )
                except (OSError, ProtocolError) as exc:
                    left_pending = len(unacked) - index
                    logger.warning(
                        "recovery: replay of op %r lost (%s); %d op(s) stay journaled",
                        record.request_id,
                        exc,
                        left_pending,
                    )
                    break
                if response.ok:
                    owned_journal.mark_acked(record.request_id, response.result)
                    replayed += 1
                else:
                    owned_journal.mark_failed(
                        record.request_id, response.error or "op rejected by writer"
                    )
                    failed += 1
        finally:
            connection.close()

        if store is not None:
            _commit_dirty_state(store)
        return RecoverySummary(
            replayed=replayed,
            acked_skipped=acked_skipped,
            failed=failed,
            deduped=deduped,
            left_pending=left_pending,
            elapsed_s=time.monotonic() - started,
            writer_address=address,
        )
    finally:
        if owns:
            owned_journal.close()
