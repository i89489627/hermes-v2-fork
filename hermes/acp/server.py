"""Transport-agnostic ACP server that survives editor crashes (plan P21.T12, issue #156173).

Upstream bug: an ACP turn kept running after the editor's ACP process exited,
and a later ``session/load`` of that session failed with a bare
``Internal error``.

The v2 :class:`AcpServer` fixes the failure path end to end:

1. **Peer exit terminates the turn.** EOF from the transport reader is treated
   as the editor exiting: every active turn is marked cancelled via a
   cooperative :class:`threading.Event` (exposed to the turn runner through
   :attr:`active_turns` / :meth:`cancel_event_for`), and the worker thread is
   given a bounded grace period to stop.
2. **Session state is saved BEFORE the failure surfaces.** Each turn writes a
   ``turn_checkpoint`` into the session state (via
   :class:`hermes.sessions.store.SessionStore`) before running, and the peer
   disconnect handler re-saves the checkpoint as ``cancelled`` *before* any
   error reply is emitted. The audit trail is kept in :attr:`event_log`
   (``session_saved:<sid>`` always precedes ``error_replied:<sid>``).
3. **Descriptive errors, never a bare ``Internal error``.** Every error reply
   names the method, session, or turn involved. A peer disconnect during turn
   ``<id>`` answers with
   ``"ACP peer disconnected during turn <id>; session <sid> saved; turn
   cancelled"``.

Transport contract: ``reader.readline()`` returns one JSON-RPC 2.0 line per
call and ``""`` on EOF; ``writer.write_line(line)`` emits one JSON-RPC line.
Tests use in-memory fakes; no real editor process is required.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from hermes import __version__
from hermes.sessions.store import SessionStore

__all__ = ["AcpServer", "TurnHandle", "TurnRunner"]

#: Turn runner contract: called as ``turn_runner(prompt, emit)``; ``emit``
#: streams a JSON-able dict event to the client. Cooperative cancellation is
#: exposed to the runner via :attr:`AcpServer.active_turns` and
#: :meth:`AcpServer.cancel_event_for` — the runner polls
#: ``cancel_event.is_set()`` and returns promptly once it is set.
TurnRunner = Callable[[str, Callable[[dict[str, Any]], None]], None]

# JSON-RPC 2.0 error codes plus server-defined ones (-32000 .. -32099).
_JSONRPC_PARSE_ERROR: Final[int] = -32700
_JSONRPC_INVALID_REQUEST: Final[int] = -32600
_JSONRPC_METHOD_NOT_FOUND: Final[int] = -32601
_JSONRPC_INVALID_PARAMS: Final[int] = -32602
_JSONRPC_INTERNAL_ERROR: Final[int] = -32603
_PEER_DISCONNECTED: Final[int] = -32000
_SESSION_NOT_FOUND: Final[int] = -32001
_TURN_FAILED: Final[int] = -32002

#: Bounded grace period a cancelled worker gets when the peer disconnects.
_TURN_JOIN_TIMEOUT: Final[float] = 5.0

_SUPPORTED_METHODS: Final[str] = "initialize, session/new, session/prompt, session/load"


class _AcpError(Exception):
    """JSON-RPC error with its code; messages are always descriptive."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _LineReader(Protocol):
    """Transport read side: one JSON-RPC line per call, ``""`` on EOF."""

    def readline(self) -> str: ...


class _LineWriter(Protocol):
    """Transport write side: accepts JSON-RPC lines."""

    def write_line(self, line: str) -> None: ...


@dataclass
class TurnHandle:
    """One in-flight turn; ``cancel_event`` is the cooperative stop signal."""

    session_id: str
    turn_id: str
    prompt: str
    started_utc: str = field(default_factory=lambda: _utc_now())
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    cancelled: bool = False
    # Set BEFORE ``cancel_event`` so the worker records the true cause.
    cancel_reason: str | None = None


@dataclass(frozen=True)
class _PendingRequest:
    """A ``session/prompt`` request awaiting its turn's outcome."""

    request_id: int | str
    session_id: str
    turn_id: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AcpServer:
    """ACP server core: JSON-lines transport, turn lifecycle, crash safety."""

    def __init__(
        self,
        turn_runner: TurnRunner | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._turn_runner = turn_runner
        self.session_store = session_store if session_store is not None else SessionStore()
        #: session_id -> in-flight turn (the runner reads ``cancel_event`` here)
        self.active_turns: dict[str, TurnHandle] = {}
        #: ordered audit trail, e.g. ``session_saved:<sid>:<turn>`` then
        #: ``error_replied:<sid>:<message>`` — proves state was saved first.
        self.event_log: list[str] = []
        self._pending: dict[int | str, _PendingRequest] = {}
        self._reader: _LineReader | None = None
        self._writer: _LineWriter | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = threading.Event()

    # ── transport ───────────────────────────────────────────────────────────

    def attach_transport(self, reader: _LineReader, writer: _LineWriter) -> None:
        """Start serving JSON-RPC lines from ``reader`` and replying via ``writer``."""
        if self._reader is not None:
            raise RuntimeError("AcpServer already has a transport attached")
        self._reader = reader
        self._writer = writer
        threading.Thread(target=self._read_loop, name="acp-reader", daemon=True).start()

    def close(self) -> None:
        """Stop the reader loop (worker threads are daemon threads)."""
        self._closed.set()

    def _read_loop(self) -> None:
        reader = self._reader
        if reader is None:  # pragma: no cover - guarded by attach_transport
            return
        while not self._closed.is_set():
            try:
                line = reader.readline()
            except Exception as exc:  # transport blew up: treat as peer exit
                self._log(f"reader_error:{type(exc).__name__}: {exc}")
                break
            if not line:
                break  # EOF — the editor's ACP process exited
            stripped = line.strip()
            if stripped:
                self._handle_line(stripped)
        self._on_peer_disconnected()

    # ── request handling ────────────────────────────────────────────────────

    def _handle_line(self, line: str) -> None:
        try:
            message: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            self._reply_error(
                None,
                _JSONRPC_PARSE_ERROR,
                f"unable to parse JSON-RPC message ({exc.msg} at line {exc.lineno}, "
                f"column {exc.colno}); each message must be one JSON object per line",
            )
            return
        if not isinstance(message, dict):
            self._reply_error(
                None,
                _JSONRPC_INVALID_REQUEST,
                f"JSON-RPC message must be a JSON object, got {type(message).__name__}",
            )
            return
        self._dispatch(message)

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        has_id = "id" in message
        request_id: int | str | None = message.get("id")
        if not isinstance(method, str):
            if has_id:
                self._reply_error(
                    request_id,
                    _JSONRPC_INVALID_REQUEST,
                    f"JSON-RPC request is missing a string 'method' field (got {method!r})",
                )
            else:
                self._log("notification_missing_method")
            return
        params = message.get("params")
        if params is None:
            params = {}
        try:
            if method == "initialize":
                result = self._initialize()
            elif method == "session/new":
                result = self._session_new()
            elif method == "session/prompt":
                # Response is deferred: it is emitted when the turn completes,
                # or as a descriptive error if the peer disconnects first.
                self._session_prompt(params, request_id if has_id else None)
                return
            elif method == "session/load":
                result = self._session_load(params)
            else:
                raise _AcpError(
                    _JSONRPC_METHOD_NOT_FOUND,
                    f"ACP method '{method}' is not supported by this server "
                    f"(supported methods: {_SUPPORTED_METHODS})",
                )
        except _AcpError as exc:
            self._log(f"request_rejected:{method}:{exc.code}")
            if has_id:
                self._reply_error(request_id, exc.code, exc.message)
            return
        except Exception as exc:
            # Never the bare string "Internal error": name the method and cause.
            self._log(f"dispatch_error:{method}:{type(exc).__name__}: {exc}")
            if has_id:
                self._reply_error(
                    request_id,
                    _JSONRPC_INTERNAL_ERROR,
                    f"ACP request '{method}' failed: {type(exc).__name__}: {exc}",
                )
            return
        if has_id:
            self._reply_result(request_id, result)

    def _initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": 1,
            "serverInfo": {"name": "hermes-v2-acp", "version": __version__},
        }

    def _session_new(self) -> dict[str, Any]:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        self.session_store.save_session(
            session_id,
            {"session_id": session_id, "created_utc": _utc_now(), "turn_checkpoint": None},
        )
        return {"sessionId": session_id}

    def _session_prompt(self, params: dict[str, Any], request_id: int | str | None) -> None:
        session_id = params.get("sessionId")
        prompt = params.get("prompt")
        if not isinstance(session_id, str) or not session_id:
            raise _AcpError(
                _JSONRPC_INVALID_PARAMS,
                "ACP session/prompt requires a non-empty string 'sessionId' param, "
                f"got {session_id!r}",
            )
        if not isinstance(prompt, str) or not prompt:
            raise _AcpError(
                _JSONRPC_INVALID_PARAMS,
                "ACP session/prompt requires a non-empty string 'prompt' param, "
                f"got {prompt!r}",
            )
        handle = self.start_turn(session_id, prompt)
        if request_id is not None:
            with self._lock:
                self._pending[request_id] = _PendingRequest(
                    request_id=request_id, session_id=session_id, turn_id=handle.turn_id
                )

    def _session_load(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise _AcpError(
                _JSONRPC_INVALID_PARAMS,
                "ACP session/load requires a non-empty string 'sessionId' param, "
                f"got {session_id!r}",
            )
        state = self.session_store.load_session(session_id)
        if state is None:
            raise _AcpError(
                _SESSION_NOT_FOUND,
                f"ACP session/load failed: no saved state for session '{session_id}' under "
                f"{self.session_store.root}; the session was never created or its state was lost",
            )
        return {"sessionId": session_id, "state": state}

    # ── turn lifecycle ──────────────────────────────────────────────────────

    def start_turn(self, session_id: str, prompt: str) -> TurnHandle:
        """Checkpoint the session, then run the turn on a worker thread."""
        if self._turn_runner is None:
            raise _AcpError(
                _JSONRPC_INTERNAL_ERROR,
                f"ACP server has no turn runner configured; cannot start a turn for "
                f"session '{session_id}'",
            )
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        handle = TurnHandle(session_id=session_id, turn_id=turn_id, prompt=prompt)
        state = self.session_store.load_session(session_id)
        if state is None:
            state = {"session_id": session_id, "created_utc": _utc_now()}
        state["turn_checkpoint"] = {
            "turn_id": turn_id,
            "session_id": session_id,
            "prompt": prompt,
            "status": "running",
            "cancelled": False,
            "started_utc": handle.started_utc,
        }
        # The checkpoint is durable BEFORE the turn starts running.
        self.session_store.save_session(session_id, state)
        self._log(f"checkpoint_saved:{session_id}:{turn_id}")
        with self._lock:
            self.active_turns[session_id] = handle
        thread = threading.Thread(
            target=self._run_turn, args=(handle,), name=f"acp-turn-{turn_id}", daemon=True
        )
        handle.thread = thread
        thread.start()
        return handle

    def cancel_event_for(self, session_id: str) -> threading.Event | None:
        """The cooperative cancel event for the session's active turn, if any."""
        with self._lock:
            handle = self.active_turns.get(session_id)
        return handle.cancel_event if handle is not None else None

    def _run_turn(self, handle: TurnHandle) -> None:
        runner = self._turn_runner
        outcome = "completed"
        detail: str | None = None
        try:
            if runner is None:  # pragma: no cover - start_turn refuses first
                raise RuntimeError(f"no turn runner configured for turn {handle.turn_id}")

            def emit(event: dict[str, Any]) -> None:
                self._send_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": handle.session_id,
                            "turnId": handle.turn_id,
                            "update": event,
                        },
                    }
                )

            runner(handle.prompt, emit)
        except Exception as exc:
            if handle.cancel_event.is_set():
                outcome = "cancelled"
                detail = f"turn runner stopped after cancellation ({type(exc).__name__}: {exc})"
            else:
                outcome = "failed"
                detail = f"{type(exc).__name__}: {exc}"
        if handle.cancel_event.is_set():
            outcome = "cancelled"
        self._finalize_turn(handle, outcome, detail)

    def _finalize_turn(self, handle: TurnHandle, outcome: str, detail: str | None) -> None:
        self._checkpoint_state(
            handle,
            status=outcome,
            cancelled=outcome == "cancelled",
            cancel_reason=handle.cancel_reason,
            detail=detail,
        )
        self._log(f"checkpoint_finalized:{handle.session_id}:{handle.turn_id}:{outcome}")
        with self._lock:
            if self.active_turns.get(handle.session_id) is handle:
                del self.active_turns[handle.session_id]
            pending = self._pop_pending_for(handle.session_id)
        if pending is None:
            handle.done.set()
            return
        if outcome == "completed":
            self._reply_result(pending.request_id, {"stopReason": "end_turn"})
        elif outcome == "cancelled":
            self._reply_result(pending.request_id, {"stopReason": "cancelled"})
        else:
            self._reply_error(
                pending.request_id,
                _TURN_FAILED,
                f"ACP turn {handle.turn_id} for session '{handle.session_id}' failed: {detail}",
            )
        handle.done.set()

    def _on_peer_disconnected(self) -> None:
        """The editor's ACP process exited: cancel turns, save state, then fail."""
        with self._lock:
            handles = list(self.active_turns.values())
            pendings = list(self._pending.values())
        for handle in handles:
            # 1) Mark the turn cancelled (cooperative stop signal). The reason
            #    is recorded on the handle first so the worker's own finalize
            #    write carries the true cause even if it lands last on disk.
            handle.cancelled = True
            handle.cancel_reason = "ACP peer disconnected"
            handle.cancel_event.set()
            # 2) Save the session state BEFORE surfacing the failure.
            self._checkpoint_state(
                handle,
                status="cancelled",
                cancelled=True,
                cancel_reason="ACP peer disconnected",
                detail=None,
            )
            self._log(f"session_saved:{handle.session_id}:{handle.turn_id}")
        # 3) Surface the failure with a descriptive JSON-RPC error reply.
        for pending in pendings:
            message = (
                f"ACP peer disconnected during turn {pending.turn_id}; "
                f"session {pending.session_id} saved; turn cancelled"
            )
            self._reply_error(pending.request_id, _PEER_DISCONNECTED, message)
            self._log(f"error_replied:{pending.session_id}:{message}")
            with self._lock:
                self._pending.pop(pending.request_id, None)
        # Bounded wait so cancelled turns actually terminate.
        for handle in handles:
            if handle.thread is not None:
                handle.thread.join(timeout=_TURN_JOIN_TIMEOUT)

    def _checkpoint_state(
        self,
        handle: TurnHandle,
        status: str,
        cancelled: bool,
        cancel_reason: str | None,
        detail: str | None,
    ) -> None:
        state = self.session_store.load_session(handle.session_id)
        if state is None:
            state = {"session_id": handle.session_id, "created_utc": _utc_now()}
        checkpoint = state.get("turn_checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("turn_id") != handle.turn_id:
            checkpoint = {
                "turn_id": handle.turn_id,
                "session_id": handle.session_id,
                "prompt": handle.prompt,
                "started_utc": handle.started_utc,
            }
        checkpoint["status"] = status
        checkpoint["cancelled"] = cancelled
        if cancelled:
            if cancel_reason:
                checkpoint["cancel_reason"] = cancel_reason
            else:
                existing_reason = checkpoint.get("cancel_reason")
                checkpoint["cancel_reason"] = (
                    existing_reason
                    if isinstance(existing_reason, str) and existing_reason
                    else "cancelled"
                )
        if detail:
            checkpoint["detail"] = detail
        checkpoint["updated_utc"] = _utc_now()
        state["turn_checkpoint"] = checkpoint
        self.session_store.save_session(handle.session_id, state)

    # ── messaging ───────────────────────────────────────────────────────────

    def _pop_pending_for(self, session_id: str) -> _PendingRequest | None:
        for request_id, pending in self._pending.items():
            if pending.session_id == session_id:
                del self._pending[request_id]
                return pending
        return None

    def _reply_result(self, request_id: int | str | None, result: dict[str, Any]) -> None:
        self._send_message({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _reply_error(self, request_id: int | str | None, code: int, message: str) -> None:
        error = {"code": code, "message": message}
        self._send_message({"jsonrpc": "2.0", "id": request_id, "error": error})

    def _send_message(self, message: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            writer.write_line(json.dumps(message))
        except Exception as exc:  # transport may already be dead after a crash
            self._log(f"write_failed:{type(exc).__name__}: {exc}")

    def _log(self, entry: str) -> None:
        with self._lock:
            self.event_log.append(entry)
