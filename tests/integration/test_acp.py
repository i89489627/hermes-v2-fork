"""Integration tests: ACP server survives editor exit (plan P21.T12, issue #156173).

Reproduces the upstream bug: an ACP turn kept running after the editor's ACP
process exited, and a later ``session/load`` failed with a bare
``Internal error``. These tests drive the v2
:class:`hermes.acp.server.AcpServer` over in-memory JSON-lines transports
(threads, no real editor process) and verify:

* a peer disconnect terminates the associated turn within a bounded wait;
* the session state (with its ``turn_checkpoint``) is saved BEFORE the
  failure is surfaced, and stays reloadable via
  :class:`hermes.sessions.store.SessionStore`;
* every error reply is descriptive — never the bare string "Internal error".
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hermes.acp.server import AcpServer
from hermes.sessions.store import SessionStore

SESSION_ID = "sess-ace-face-0001"
TIMEOUT = 10.0


class FakeReader:
    """In-memory JSON-lines reader; ``close()`` simulates the editor exiting."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str | None] = queue.Queue()

    def send_line(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put(None)

    def readline(self) -> str:
        item = self._lines.get()
        return "" if item is None else item


class FakeWriter:
    """In-memory JSON-lines writer that records everything it is sent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def json_messages(self) -> list[dict[str, Any]]:
        with self._lock:
            snapshot = list(self.lines)
        return [json.loads(line) for line in snapshot if line.strip()]

    def error_messages(self) -> list[str]:
        return [
            str(message.get("error", {}).get("message", ""))
            for message in self.json_messages()
            if isinstance(message.get("error"), dict)
        ]


class OrderCheckingWriter(FakeWriter):
    """Fails the test if an error reply outruns the session checkpoint save."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        super().__init__()
        self._store = store
        self._session_id = session_id
        self.violations: list[str] = []

    def write_line(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            message = {}
        if isinstance(message, dict) and isinstance(message.get("error"), dict):
            text = str(message["error"].get("message", ""))
            if "peer disconnected" in text:
                state = self._store.load_session(self._session_id)
                checkpoint = (state or {}).get("turn_checkpoint")
                if state is None:
                    self.violations.append("error reply sent before any session state existed")
                elif not isinstance(checkpoint, dict) or checkpoint.get("status") != "cancelled":
                    self.violations.append(f"error reply sent before checkpoint saved: {checkpoint!r}")
        super().write_line(line)


def _wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _error_response_by_id(writer: FakeWriter, request_id: Any) -> dict[str, Any]:
    assert _wait_until(lambda: any(m.get("id") == request_id and "error" in m for m in writer.json_messages())), (
        f"no error response for request id {request_id!r}: {writer.json_messages()!r}"
    )
    matches = [m for m in writer.json_messages() if m.get("id") == request_id and "error" in m]
    return matches[-1]


def _prompt_message(request_id: int, session_id: str, prompt: str) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": prompt},
        }
    )


def _blocking_turn(
    server: AcpServer,
    session_id: str,
    started: threading.Event,
    recorded_prompts: list[str],
) -> Callable[[str, Callable[[dict[str, Any]], None]], None]:
    """A turn that keeps running until its cooperative cancel event is set."""

    def run(prompt: str, emit: Callable[[dict[str, Any]], None]) -> None:
        recorded_prompts.append(prompt)
        handle = server.active_turns[session_id]
        started.set()
        deadline = time.monotonic() + TIMEOUT * 2
        while not handle.cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        emit({"type": "turn_stopped", "cancelled": handle.cancel_event.is_set()})

    return run


@pytest.mark.integration
def test_turn_terminated_on_editor_exit(tmp_path: Path) -> None:
    """Closing the transport (editor exit) must terminate the running turn."""
    store = SessionStore(root=tmp_path / "sessions")
    started = threading.Event()
    recorded_prompts: list[str] = []
    server = AcpServer(
        turn_runner=lambda prompt, emit: _blocking_turn(server, SESSION_ID, started, recorded_prompts)(prompt, emit),
        session_store=store,
    )
    reader, writer = FakeReader(), FakeWriter()
    server.attach_transport(reader, writer)

    reader.send_line(_prompt_message(1, SESSION_ID, "long-running analysis"))
    assert started.wait(TIMEOUT), "the turn never started running"
    handle = server.active_turns[SESSION_ID]

    reader.close()  # the editor's ACP process exits mid-turn
    assert handle.thread is not None
    handle.thread.join(timeout=TIMEOUT)  # bounded wait for turn termination
    assert not handle.thread.is_alive(), "the turn kept running after the editor exited"
    assert handle.cancel_event.is_set(), "the turn was never marked cancelled"
    assert handle.cancelled

    # The peer-exit failure is surfaced with a descriptive error reply.
    message = _error_response_by_id(writer, 1)["error"]["message"]
    assert "peer disconnected" in message
    assert SESSION_ID in message
    assert recorded_prompts == ["long-running analysis"]

    # The cancelled session stays reloadable with its checkpoint.
    state = store.load_session(SESSION_ID)
    assert state is not None
    checkpoint = state["turn_checkpoint"]
    assert checkpoint["status"] == "cancelled"
    assert checkpoint["cancelled"] is True
    server.close()


@pytest.mark.integration
def test_session_saved_before_failure(tmp_path: Path) -> None:
    """The checkpoint (status cancelled) is on disk before the error reply goes out."""
    store = SessionStore(root=tmp_path / "sessions")
    store.save_session(SESSION_ID, {"session_id": SESSION_ID, "history": ["earlier exchange"]})
    started = threading.Event()
    recorded_prompts: list[str] = []
    server = AcpServer(
        turn_runner=lambda prompt, emit: _blocking_turn(server, SESSION_ID, started, recorded_prompts)(prompt, emit),
        session_store=store,
    )
    reader = FakeReader()
    writer = OrderCheckingWriter(store, SESSION_ID)
    server.attach_transport(reader, writer)

    reader.send_line(_prompt_message(7, SESSION_ID, "stateful turn"))
    assert started.wait(TIMEOUT), "the turn never started running"

    reader.close()  # editor exits mid-turn
    assert _wait_until(lambda: any("peer disconnected" in m for m in writer.error_messages())), (
        f"no descriptive disconnect error was sent: {writer.error_messages()!r}"
    )

    # The writer inspected the store at the instant the error was written:
    # any violation means the failure was surfaced before the save.
    assert writer.violations == [], writer.violations

    # The server-side audit log proves the ordering too.
    saved_at = [i for i, e in enumerate(server.event_log) if e.startswith(f"session_saved:{SESSION_ID}")]
    replied_at = [i for i, e in enumerate(server.event_log) if e.startswith(f"error_replied:{SESSION_ID}")]
    assert saved_at and replied_at, f"missing audit entries: {server.event_log!r}"
    assert saved_at[0] < replied_at[0], "session must be saved before the error reply"

    # The saved state survived: reloadable, checkpoint cancelled, prior
    # session content preserved (the checkpoint is merged, not clobbered).
    state = store.load_session(SESSION_ID)
    assert state is not None
    checkpoint = state["turn_checkpoint"]
    assert checkpoint["status"] == "cancelled"
    assert "peer disconnected" in checkpoint["cancel_reason"]
    assert checkpoint["turn_id"].startswith("turn-")
    assert state["history"] == ["earlier exchange"]
    server.close()


@pytest.mark.integration
def test_no_bare_internal_error(tmp_path: Path) -> None:
    """Every failure path answers with a descriptive message, never 'Internal error'."""
    store = SessionStore(root=tmp_path / "sessions")
    store.save_session(SESSION_ID, {"session_id": SESSION_ID, "history": []})
    started = threading.Event()
    recorded_prompts: list[str] = []
    server = AcpServer(
        turn_runner=lambda prompt, emit: _blocking_turn(server, SESSION_ID, started, recorded_prompts)(prompt, emit),
        session_store=store,
    )
    reader, writer = FakeReader(), FakeWriter()
    server.attach_transport(reader, writer)

    # 1) Unsupported method: the message names the offending method.
    reader.send_line(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "workspace/readFile", "params": {}}))
    unsupported = _error_response_by_id(writer, 1)["error"]["message"]
    assert "workspace/readFile" in unsupported
    assert "not supported" in unsupported

    # 2) Malformed JSON: the message explains the parse failure.
    reader.send_line('{"jsonrpc": "2.0", oops')
    assert _wait_until(lambda: any("parse" in m for m in writer.error_messages())), (
        f"no parse error reply: {writer.error_messages()!r}"
    )
    parse_errors = [m for m in writer.error_messages() if "parse" in m]
    assert "JSON" in parse_errors[0]

    # 3) session/load for an unknown session: descriptive, names the session
    #    (the upstream bug surfaced a bare 'Internal error' exactly here).
    reader.send_line(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "session/load", "params": {"sessionId": "sess-ghost"}}))
    load_error = _error_response_by_id(writer, 3)["error"]["message"]
    assert "sess-ghost" in load_error
    assert "no saved state" in load_error

    # 4) Editor exits mid-turn: the disconnect error names the turn and session.
    reader.send_line(_prompt_message(4, SESSION_ID, "turn that outlives its editor"))
    assert started.wait(TIMEOUT), "the turn never started running"
    reader.close()
    disconnect = _error_response_by_id(writer, 4)["error"]["message"]
    assert "peer disconnected" in disconnect
    assert SESSION_ID in disconnect
    assert "turn cancelled" in disconnect

    # No reply anywhere is the bare upstream string.
    errors = writer.error_messages()
    assert len(errors) >= 4
    for message in errors:
        assert message.strip() != "Internal error", f"bare internal error leaked: {errors!r}"
        assert len(message) >= 20, f"error lacks descriptive context: {message!r}"

    # The session remains recoverable after the crash.
    state = store.load_session(SESSION_ID)
    assert state is not None and state["turn_checkpoint"]["status"] == "cancelled"
    server.close()
