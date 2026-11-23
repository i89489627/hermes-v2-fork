"""Chaos test: the editor's ACP process crashes mid-turn (plan P21.T12, issue #156173).

Upstream bug: an ACP turn kept running after the editor's ACP process exited,
leaving an orphaned turn and a session that later failed to load with a bare
``Internal error``. This test crashes the peer (closes the transport reader)
while a turn is actively streaming progress, then asserts the v2
:class:`hermes.acp.server.AcpServer` failure path holds together:

* the turn terminates within a bounded wait (cooperative cancel event);
* the session state — with its ``turn_checkpoint`` — is reloadable via
  :class:`hermes.sessions.store.SessionStore` and records the cancellation
  reason *before* the failure was surfaced (audit-log ordering);
* the JSON-RPC error reply is descriptive ("peer disconnected" + session id),
  never a bare ``Internal error``.
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

SESSION_ID = "sess-chaos-crash-42"
TIMEOUT = 10.0


class FakeReader:
    """In-memory JSON-lines reader; ``close()`` crashes the peer mid-turn."""

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


def _wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.chaos
def test_editor_crash_during_turn(tmp_path: Path) -> None:
    """A peer crash mid-turn cancels the turn, saves the session, explains itself."""
    store = SessionStore(root=tmp_path / "sessions")
    store.save_session(SESSION_ID, {"session_id": SESSION_ID, "history": ["pre-crash exchange"]})
    started = threading.Event()
    emits: list[dict[str, Any]] = []
    emit_lock = threading.Lock()

    def run(prompt: str, emit: Callable[[dict[str, Any]], None]) -> None:
        handle = server.active_turns[SESSION_ID]
        started.set()
        tick = 0
        while not handle.cancel_event.is_set():
            with emit_lock:
                emits.append({"type": "progress", "tick": tick})
            emit({"type": "progress", "tick": tick})
            tick += 1
            time.sleep(0.01)

    server = AcpServer(turn_runner=run, session_store=store)
    reader, writer = FakeReader(), FakeWriter()
    server.attach_transport(reader, writer)

    reader.send_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/prompt",
                "params": {"sessionId": SESSION_ID, "prompt": "stream until the editor dies"},
            }
        )
    )
    assert started.wait(TIMEOUT), "the turn never started running"
    assert _wait_until(lambda: len(emits) >= 5), f"turn produced no progress: {len(emits)} emits"
    handle = server.active_turns[SESSION_ID]

    reader.close()  # the editor crashes mid-turn, mid-stream
    assert handle.thread is not None
    handle.thread.join(timeout=TIMEOUT)  # bounded wait: the turn must stop
    assert not handle.thread.is_alive(), "the turn kept running after the editor crashed"
    assert handle.cancel_event.is_set()
    assert handle.cancelled

    # Progress stopped flowing after cancellation (turn really terminated).
    with emit_lock:
        count_after_cancel = len(emits)
    time.sleep(0.3)
    with emit_lock:
        assert len(emits) == count_after_cancel, "the turn kept emitting after cancellation"

    # Session is reloadable with a cancelled checkpoint, prior content intact.
    state = store.load_session(SESSION_ID)
    assert state is not None, "session state was lost in the crash"
    checkpoint = state["turn_checkpoint"]
    assert checkpoint["status"] == "cancelled"
    assert "peer disconnected" in checkpoint["cancel_reason"]
    assert checkpoint["turn_id"].startswith("turn-")
    assert state["history"] == ["pre-crash exchange"]

    # The failure is surfaced descriptively, saved-first, never bare.
    assert _wait_until(lambda: any("peer disconnected" in m for m in writer.error_messages())), (
        f"no descriptive disconnect error: {writer.error_messages()!r}"
    )
    errors = writer.error_messages()
    assert any("peer disconnected" in m and SESSION_ID in m for m in errors), errors
    assert all(m.strip() != "Internal error" for m in errors), errors
    saved_at = [i for i, e in enumerate(server.event_log) if e.startswith(f"session_saved:{SESSION_ID}")]
    replied_at = [i for i, e in enumerate(server.event_log) if e.startswith(f"error_replied:{SESSION_ID}")]
    assert saved_at and replied_at, f"missing audit entries: {server.event_log!r}"
    assert saved_at[0] < replied_at[0], "session must be saved before the failure is surfaced"
    server.close()
