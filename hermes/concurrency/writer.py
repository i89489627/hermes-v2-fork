"""WriterProcess: single-writer daemon applying queued memory ops (plan P6.T2).

Multiple agent processes (main + subagents, plan Phase 6) share one Hermes
home; instead of every process racing on the memory repository, they send
write operations to ONE writer process over the IPC protocol of
:mod:`hermes.concurrency.protocol` and the writer applies them serially.

**Transport.** TCP on the loopback interface (``127.0.0.1`` with
``port=0`` = OS-assigned) — Unix domain sockets do not exist on Windows
Python, so loopback TCP is the portable equivalent of the plan's "Unix
socket" (see the protocol module's platform note).

**Concurrency model.**

- One listener thread accepts connections; each connection gets its own
  handler thread. A connection's requests are processed strictly in arrival
  order and answered in the same order — per-client write order is
  preserved by construction.
- Cross-client serialization is inherited, not re-implemented: every op is
  applied through :class:`~hermes.memory.store.MemoryStore`, whose
  :class:`~hermes.memory.locking.FileLock` (plan P1.T5) makes each mutation
  atomic against any other process/thread touching the same store. The
  writer is the *queue* seam the plan describes — the store lock is the
  serializer.
- **Crash safety**: an op that raises (unknown entry id, rejected write,
  ...) is answered ``ok=False`` with the error text and the connection
  stays usable; a client that dies mid-conversation only closes its own
  connection — the listener and all other clients keep working.

**Throughput mechanism — commit batching.** The plan acceptance is
"writer handles 100 writes/sec", but the store commits one Git commit per
mutation, which is far slower than 100/sec on a typical disk. The writer
therefore installs a batching wrapper around the store's Git backend
(:class:`_BatchingGitBackend`): ops are applied to the store immediately
(files + index written, response sent — the response reflects the applied
state, which is what clients read back), while the underlying Git commits
are coalesced into ONE commit per *flush window* — when 50 ops have
accumulated (``batch_max_ops``) or the oldest buffered op is 200ms old
(``batch_window_s``), whichever comes first; :meth:`WriterProcess.stop`
flushes the remainder. This bounds the durability window to
``batch_window_s`` seconds / ``batch_max_ops`` ops and is the documented
trade-off behind the throughput number: an acknowledged-but-unflushed
write survives a writer restart (files are on disk and re-pended) but not
a machine crash — plan P6.T3's crash recovery (agents detect writer death
and replay pending writes) is the designed completion of that seam.

Flushes run under the store's own :class:`FileLock` — the same lock that
serializes store mutations and the index-touching ``_stage_removal``
helper — so the batcher's commit can never interleave with a mutation's
pygit2 index access.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from hermes.concurrency.protocol import (
    ProtocolError,
    WriteRequest,
    WriteResponse,
    read_request,
    write_response,
)
from hermes.memory.git_backend import GitBackend
from hermes.memory.locking import FileLock
from hermes.memory.store import MemoryStore
from hermes.memory.types import ENTRY_TYPES

__all__ = ["WriterProcess"]

_LOGGER = logging.getLogger("hermes.concurrency.writer")

#: Flush-window size in ops: the batcher commits once this many ops are
#: buffered (whichever of this or ``window_s`` comes first).
_DEFAULT_BATCH_MAX_OPS: Final[int] = 50

#: Flush-window duration in seconds: the batcher commits once its oldest
#: buffered op is this old.
_DEFAULT_BATCH_WINDOW_S: Final[float] = 0.2

#: Poll cadence of the accept loop (the listener socket's timeout).
_LISTENER_POLL_S: Final[float] = 0.5

#: Poll cadence of the flusher thread (it wakes to honor the window).
_FLUSHER_POLL_S: Final[float] = 0.02

#: ``listen`` backlog for the writer's loopback socket.
_LISTEN_BACKLOG: Final[int] = 16


class _BatchingGitBackend(GitBackend):
    """Git backend that coalesces a flush window of commits into ONE commit.

    Wraps the same repository path as the backend it replaces (one fresh
    pygit2 handle — the replaced instance is dropped, so ``MemoryStore``
    stages removals and commits through THIS handle only, keeping libgit2's
    per-handle in-memory index consistent).

    ``commit()`` (called by store mutations, always while the store's
    :class:`FileLock` is held) buffers the files+message instead of
    committing, and :meth:`flush` performs the coalesced commit:

    - inline, when ``max_ops`` ops have buffered (the triggering store
      mutation's thread already holds the flush lock — the store's own
      :class:`FileLock` — so the reentrant acquire cannot block), and
    - from the writer's flusher thread when ``window_s`` elapsed.

    Durability contract: buffered ops' files and the MEMORY.md index are
    already on disk, so a flush failure re-pends them for the next window
    (counted in ``flush_errors``) instead of losing them; only a machine
    crash inside the window can lose an acknowledged op — recovered by the
    P6.T3 replay seam. A buffered ``commit()`` call cannot return the real
    commit sha (it does not exist yet), so it returns ``""``; the
    triggering call of an inline flush returns the batch's sha.

    ``close()`` freezes batching: afterwards ``commit()`` commits directly
    again (per-op commits, original semantics), so the store stays usable
    standalone after :meth:`WriterProcess.stop` — through the same handle,
    never a stale second one.
    """

    def __init__(
        self,
        repo_path: Path | str,
        *,
        max_ops: int = _DEFAULT_BATCH_MAX_OPS,
        window_s: float = _DEFAULT_BATCH_WINDOW_S,
        flush_lock: FileLock | None = None,
    ) -> None:
        super().__init__(repo_path)
        if max_ops < 1:
            raise ValueError(f"max_ops must be >= 1, got {max_ops}")
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.max_ops = max_ops
        self.window_s = window_s
        #: Held across every flush; the store's own FileLock (its mutations
        #: hold it too), so flushes serialize against index mutations.
        self._flush_lock = flush_lock
        self._cond = threading.Condition()
        self._pending_files: set[str] = set()
        self._messages: list[str] = []
        self._pending_ops = 0
        self._window_start: float | None = None
        self._closed = False
        #: Batch commits performed / flush failures since construction.
        self.commits_flushed = 0
        self.flush_errors = 0
        #: Error text of the most recent flush failure (None when healthy).
        self.last_flush_error: str | None = None
        #: sha of the most recent coalesced commit (None until one happens).
        self.last_flush_sha: str | None = None

    # ── GitBackend seam ────────────────────────────────────────────────────
    def commit(self, message: str, files: Sequence[str | Path]) -> str:
        """Buffer one store mutation's commit (see the class docstring)."""
        if self._closed:
            return super().commit(message, files)
        with self._cond:
            self._pending_files.update(self._rel_path(f) for f in files)
            self._messages.append(message)
            self._pending_ops += 1
            if self._window_start is None:
                self._window_start = time.monotonic()
            flush_now = self._pending_ops >= self.max_ops
        if flush_now:
            sha = self.flush()
            if sha is not None:
                return sha
        return ""

    # ── flush mechanics ────────────────────────────────────────────────────
    @property
    def pending_ops(self) -> int:
        """Number of buffered (not yet committed) ops."""
        with self._cond:
            return self._pending_ops

    def flush_if_due(self) -> str | None:
        """Flush when the oldest buffered op is older than ``window_s``."""
        due = False
        with self._cond:
            if self._window_start is not None:
                due = (time.monotonic() - self._window_start) >= self.window_s
        return self.flush() if due else None

    def flush(self) -> str | None:
        """Commit the buffered window as one Git commit; returns its sha.

        Returns None when nothing was buffered or the commit failed (the
        failure is counted in ``flush_errors`` and the ops are re-pended).
        Serialized against store mutations via the store's FileLock.
        """
        if self._flush_lock is not None:
            self._flush_lock.acquire()  # reentrant for the inline path
        try:
            return self._flush_locked()
        finally:
            if self._flush_lock is not None:
                self._flush_lock.release()

    def _flush_locked(self) -> str | None:
        with self._cond:
            pending = sorted(self._pending_files)
            messages = self._messages
            ops = self._pending_ops
            if not pending:
                return None
            self._pending_files = set()
            self._messages = []
            self._pending_ops = 0
            self._window_start = None
        # Paths whose file no longer exists were removed by a later op inside
        # the same window (or their write was rolled back): Git must not
        # record them. A removal of a PREVIOUSLY COMMITTED file is recorded
        # anyway — MemoryStore._stage_removal already dropped it from the
        # index the flush's tree is built from. Dropping (not re-pending)
        # these is what keeps a flush from failing on its own history.
        files = [rel for rel in pending if (self.repo_path / rel).is_file()]
        if not files:
            return None
        message = (
            messages[0]
            if ops == 1
            else "[memory] writer flush: " + "; ".join(messages)
        )
        try:
            sha = super().commit(message, files)
        except Exception as exc:  # every failure must re-pend, never lose ops
            with self._cond:
                self._pending_files.update(files)
                self._messages = messages
                self._pending_ops = ops
                if self._window_start is None:
                    self._window_start = time.monotonic()
            self.flush_errors += 1
            self.last_flush_error = f"{type(exc).__name__}: {exc}"
            _LOGGER.warning(
                "writer flush commit failed (ops re-pended for the next window): %s",
                self.last_flush_error,
            )
            return None
        self.commits_flushed += 1
        self.last_flush_sha = sha
        self.last_flush_error = None
        return sha

    def close(self) -> None:
        """Freeze batching: subsequent ``commit()`` calls commit directly."""
        if self.pending_ops:
            _LOGGER.warning(
                "writer batcher closed with %d ops still pending (unflushed)",
                self.pending_ops,
            )
        self._closed = True


def _install_batching(
    store: MemoryStore, max_ops: int, window_s: float
) -> _BatchingGitBackend | None:
    """Swap ``store.git`` for the batching backend; returns it (or None).

    Returns None (no batching, per-op commits) when the store's backend is
    not a Git-backed :class:`GitBackend`. The flush lock is the store's own
    :class:`FileLock` — the same lock every store mutation holds, which is
    the only correct serialization point against the index-touching
    ``_stage_removal`` helper (repo-internal access mirroring how
    ``MemoryStore`` itself reaches into ``git._repo``).
    """
    current = store.git
    if isinstance(current, _BatchingGitBackend) and not current._closed:
        return current  # already batching (e.g. a restarted writer)
    if not isinstance(current, GitBackend):
        _LOGGER.info(
            "writer: store backend %s is not Git-backed; commits stay per-op",
            type(current).__name__,
        )
        return None
    batcher = _BatchingGitBackend(
        current.repo_path,
        max_ops=max_ops,
        window_s=window_s,
        flush_lock=store._lock,
    )
    store.git = batcher
    return batcher


class WriterProcess:
    """The Phase 6 single-writer daemon for one :class:`MemoryStore`.

    ``start()`` binds a loopback TCP listener (``host``/``port``, default
    ``127.0.0.1``/OS-assigned), installs commit batching on the store (see
    :func:`_install_batching`), starts the accept and flusher threads and
    returns the ``(host, port)`` clients connect to. ``stop()`` shuts down
    cleanly: no more accepts, clients disconnected, threads joined, the
    pending commit batch flushed, batching frozen — the store remains
    usable standalone (per-op commits) before ``start()`` and after
    ``stop()``, and a writer can be restarted.
    """

    def __init__(
        self,
        store: MemoryStore,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        batch_max_ops: int = _DEFAULT_BATCH_MAX_OPS,
        batch_window_s: float = _DEFAULT_BATCH_WINDOW_S,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self._batch_max_ops = batch_max_ops
        self._batch_window_s = batch_window_s
        self._batcher: _BatchingGitBackend | None = None
        self._listener: socket.socket | None = None
        self._address: tuple[str, int] | None = None
        self._stopping = threading.Event()
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._client_counter = 0
        self._started = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> tuple[str, int]:
        """Bind the listener, start the daemon threads, return ``(host, port)``."""
        if self._started:
            raise RuntimeError("WriterProcess is already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((self.host, self.port))
            listener.listen(_LISTEN_BACKLOG)
        except BaseException:
            listener.close()
            raise
        listener.settimeout(_LISTENER_POLL_S)
        name, port = listener.getsockname()[:2]
        self._listener = listener
        self._address = (str(name), int(port))
        self._stopping.clear()
        self._batcher = _install_batching(
            self.store, self._batch_max_ops, self._batch_window_s
        )
        self._started = True
        self._spawn(self._accept_loop, "hermes-writer-accept")
        if self._batcher is not None:
            self._spawn(self._flusher_loop, "hermes-writer-flusher")
        _LOGGER.info("writer listening on %s:%d", self._address[0], self._address[1])
        return self._address

    def stop(self, timeout: float = 5.0) -> None:
        """Clean shutdown; idempotent and a no-op before :meth:`start`.

        Stops accepting, disconnects clients (in-flight ops finish being
        applied; their responses may be lost with the socket — clients
        reconnect and replay per P6.T3), joins the threads, flushes the
        remaining commit batch and freezes batching so the store stays
        usable standalone.
        """
        if not self._started:
            return
        self._stopping.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            with contextlib.suppress(OSError):
                listener.close()
        with self._clients_lock:
            clients = list(self._clients)
        for conn in clients:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                conn.close()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self.flush_pending()
        if self._batcher is not None:
            self._batcher.close()
        self._threads.clear()
        self._started = False
        _LOGGER.info("writer stopped")

    @property
    def address(self) -> tuple[str, int]:
        """The ``(host, port)`` :meth:`start` bound (client connect target)."""
        if self._address is None:
            raise RuntimeError("WriterProcess is not started")
        return self._address

    @property
    def pending_ops(self) -> int:
        """Ops applied but not yet committed to Git (batching window)."""
        return 0 if self._batcher is None else self._batcher.pending_ops

    def flush_pending(self) -> int:
        """Commit the pending batch now; returns the number of ops committed."""
        if self._batcher is None:
            return 0
        before = self._batcher.pending_ops
        self._batcher.flush()
        return before - self._batcher.pending_ops

    # ── threads ────────────────────────────────────────────────────────────
    def _spawn(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _flusher_loop(self) -> None:
        """Wake periodically to honor the 200ms half of the flush window.

        The ``batch_max_ops`` half flushes inline in the triggering store
        mutation; this thread is what makes an idle writer's buffered ops
        reach Git within ``batch_window_s``. The final flush (including
        anything this loop leaves) happens in :meth:`stop`.
        """
        batcher = self._batcher
        if batcher is None:
            return
        while not self._stopping.wait(_FLUSHER_POLL_S):
            try:
                batcher.flush_if_due()
            except Exception as exc:  # never let the flusher thread die
                _LOGGER.warning("writer flusher error: %s", exc)

    def _accept_loop(self) -> None:
        """Accept connections until stopped; one handler thread per client."""
        listener = self._listener
        if listener is None:
            return
        while not self._stopping.is_set():
            try:
                conn, _addr = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stopping.is_set() or listener.fileno() < 0:
                    break
                continue  # transient accept error; keep serving
            # accept() inherits the listener's poll timeout — clients need
            # blocking reads/writes, so restore it before serving. TCP_NODELAY
            # is load-bearing for the 100 writes/sec acceptance: without it
            # the small request/response frames hit the Nagle + delayed-ACK
            # interaction and every round trip costs ~40ms on Windows.
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._clients_lock:
                self._clients.add(conn)
            self._client_counter += 1
            self._spawn(self._client_handler(conn), f"hermes-writer-client-{self._client_counter}")

    def _client_handler(self, conn: socket.socket) -> Callable[[], None]:
        def run() -> None:
            try:
                self._serve_connection(conn)
            finally:
                with self._clients_lock:
                    self._clients.discard(conn)
                with contextlib.suppress(OSError):
                    conn.close()

        return run

    def _serve_connection(self, conn: socket.socket) -> None:
        """Serve one client sequentially — per-connection order is the write order."""
        rfile = conn.makefile("rb")
        wfile = conn.makefile("wb")
        try:
            while not self._stopping.is_set():
                try:
                    request = read_request(rfile)
                except ProtocolError as exc:
                    if exc.eof:
                        break  # client closed the connection — clean end
                    # Malformed frame: exactly one line consumed, framing
                    # stays line-synced — answer the error and continue.
                    _LOGGER.debug("malformed request frame: %s", exc)
                    write_response(
                        wfile,
                        WriteResponse(
                            request_id=None, ok=False, error=f"malformed request: {exc}"
                        ),
                    )
                    continue
                write_response(wfile, self._process(request))
        except OSError:
            # Client vanished mid-conversation: its connection dies, the
            # writer (listener + all other clients) stays alive.
            _LOGGER.debug("client connection lost", exc_info=True)
        finally:
            with contextlib.suppress(OSError):
                rfile.close()
            with contextlib.suppress(OSError):
                wfile.close()

    # ── op application ─────────────────────────────────────────────────────
    def _process(self, request: WriteRequest) -> WriteResponse:
        """Apply one request; never raises (crash-safe handler contract)."""
        try:
            result = self._apply_op(request.op)
        except Exception as exc:  # answered with ok=False, never fatal
            _LOGGER.warning(
                "op %r from %s failed: %s", request.request_id, request.client_id, exc
            )
            return WriteResponse(
                request_id=request.request_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return WriteResponse(request_id=request.request_id, ok=True, result=result)

    def _apply_op(self, op: dict[str, Any]) -> dict[str, Any]:
        """Apply one op dict through the store; returns the response result."""
        kind = op.get("kind")
        if kind == "add":
            return self._apply_add(op)
        if kind == "update":
            return self._apply_update(op)
        if kind == "remove":
            return self._apply_remove(op)
        raise ValueError(f"unknown op kind: {kind!r}")

    def _apply_add(self, op: dict[str, Any]) -> dict[str, Any]:
        text = op.get("text")
        if not isinstance(text, str):
            raise ValueError("add op requires a string 'text'")
        entry_type = op.get("type", "Feedback")
        if entry_type not in ENTRY_TYPES:
            raise ValueError(
                f"add op 'type' must be one of {sorted(ENTRY_TYPES)}, got {entry_type!r}"
            )
        context = op.get("context", "global")
        if not isinstance(context, str):
            raise ValueError("add op 'context' must be a string")
        entry = self.store.add(text, type=entry_type, context=context)
        return {
            "entry_id": entry.id,
            "type": entry.type,
            "context": entry.context,
            "trust_score": entry.trust_score,
        }

    def _apply_update(self, op: dict[str, Any]) -> dict[str, Any]:
        entry_id = op.get("entry_id")
        if not isinstance(entry_id, str):
            raise ValueError("update op requires a string 'entry_id'")
        text = op.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError("update op 'text' must be a string")
        trust_score = op.get("trust_score")
        if trust_score is not None and (
            isinstance(trust_score, bool) or not isinstance(trust_score, (int, float))
        ):
            raise ValueError("update op 'trust_score' must be a number")
        entry = self.store.update(
            entry_id,
            text=text,
            trust_score=None if trust_score is None else float(trust_score),
        )
        return {
            "entry_id": entry.id,
            "type": entry.type,
            "context": entry.context,
            "trust_score": entry.trust_score,
        }

    def _apply_remove(self, op: dict[str, Any]) -> dict[str, Any]:
        entry_id = op.get("entry_id")
        if not isinstance(entry_id, str):
            raise ValueError("remove op requires a string 'entry_id'")
        removed = self.store.remove(entry_id)
        return {"entry_id": entry_id, "removed": removed}
