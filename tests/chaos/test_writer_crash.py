"""P20.T1 chaos: writer hard-crash mid-batch with 20 concurrent client ops.

Distinct angle vs. the P6.T3 integration tests
(:mod:`tests.integration.test_writer`): the crash happens DURING a batch.
The writer runs with a huge commit-batching window; 20 concurrent ops from
four journaling clients are in flight or already applied-but-uncommitted
when the writer is killed WITHOUT any graceful stop — no flush, the batching
window dies with the process exactly like a dead process would. Client-side
recovery via the :class:`~hermes.concurrency.recovery.WriteJournal` +
:func:`~hermes.concurrency.recovery.recover` seam (plan P6.T3: agents detect
writer death and replay) replays the journaled-unacked ops to a fresh
writer; the test goes through the chaos conftest fixtures and asserts NO
DATA LOSS AT THE STORE LEVEL: every one of the 20 ops is present EXACTLY
ONCE, the journal is fully resolved, and the repository ends git-clean
(plan P20.T1 acceptance).

Platform note: the transport is TCP loopback (127.0.0.1, OS-assigned port);
the crash closes the listener and every client socket abruptly — the same
hard-kill shape the P6.T3 integration tests use.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from hermes.concurrency.recovery import (
    JournalingClient,
    RecoverySummary,
    WriteJournal,
    WriterConnectionError,
    recover,
)
from hermes.concurrency.writer import WriterProcess
from hermes.memory.store import MemoryStore


def _add_op(text: str) -> dict[str, Any]:
    """One writer-protocol add op (Feedback entry, global context)."""
    return {"kind": "add", "text": text, "type": "Feedback", "context": "global"}


def _hard_kill(writer: WriterProcess) -> None:
    """Simulate a writer PROCESS crash: kill the server without any flush.

    ``stop()`` would flush the pending commit batch — a graceful shutdown,
    the opposite of a crash (same shape as the P6.T3 helper in
    ``tests/integration/test_writer.py``). The stop flag makes the accept
    loop and flusher thread exit WITHOUT flushing, then the listener and
    every client connection are closed. The WriterProcess object is
    abandoned afterwards, never stop()ed.
    """
    writer._stopping.set()
    listener, writer._listener = writer._listener, None
    if listener is not None:
        with contextlib.suppress(OSError):
            listener.close()
    with writer._clients_lock:
        clients = list(writer._clients)
    for conn in clients:
        with contextlib.suppress(OSError):
            conn.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            conn.close()
    for thread in writer._threads:
        thread.join(timeout=5.0)


@contextlib.contextmanager
def _fresh_writer(store: MemoryStore) -> Iterator[WriterProcess]:
    """A fresh (recovered) writer over ``store``; stopped on exit."""
    writer = WriterProcess(store)
    writer.start()
    try:
        yield writer
    finally:
        writer.stop()


def _drain(writer: WriterProcess, timeout_s: float = 5.0) -> None:
    """Wait for the writer's batching window to drain, then flush the rest."""
    deadline = time.monotonic() + timeout_s
    while writer.pending_ops and time.monotonic() < deadline:
        time.sleep(0.02)
    writer.flush_pending()


@pytest.mark.chaos
def test_writer_recovery_after_crash(chaos_store, tmp_path) -> None:
    """20 concurrent ops, crash mid-batch, recovery -> every op exactly once."""
    store = chaos_store
    writer = WriterProcess(
        store, batch_max_ops=10**9, batch_window_s=float(10**9)
    )  # huge window: NOTHING commits — the crash happens inside the batch
    writer.start()
    journal = WriteJournal(tmp_path / "journal" / "writer_journal.jsonl")
    agents = [f"agent-{index}" for index in range(4)]
    texts_by_agent = {
        agent: [f"{agent} chaos fact {index}" for index in range(5)] for agent in agents
    }
    clients = {
        agent: JournalingClient(writer.address, client_id=agent, journal=journal)
        for agent in agents
    }
    connection_errors: list[str] = []

    def run(agent: str) -> None:
        for text in texts_by_agent[agent]:
            try:
                clients[agent].send_op(_add_op(text))
            except WriterConnectionError as exc:
                # Writer died mid-op: the op stays journaled-unacked and is
                # replayed (or deduped) by the recovery below — never lost.
                connection_errors.append(f"{agent}: {exc}")

    threads = [
        threading.Thread(target=run, args=(agent,), name=agent) for agent in agents
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.05)  # mid-batch: ops applied (files on disk), nothing committed
    _hard_kill(writer)  # crash: sockets die abruptly, batching window never flushed
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    for client in clients.values():
        client.close()

    # Crash evidence: the huge window means the dead writer committed nothing.
    assert store.git.log(limit=10) == []

    fresh_store = MemoryStore(root=store.root)  # the fresh writer's own handle
    try:
        with _fresh_writer(fresh_store) as fresh_writer:
            def writer_factory() -> tuple[str, int]:
                return fresh_writer.address

            summary: RecoverySummary = recover(
                writer_factory, store=fresh_store, journal=journal
            )

            # Every journaled op is accounted for: acked skipped, unacked
            # replayed or deduped (applied-but-ack-lost), nothing failed,
            # nothing left pending, all within the plan's 5s budget.
            accounted = (
                summary.acked_skipped
                + summary.replayed
                + summary.deduped
                + summary.failed
                + summary.left_pending
            )
            assert accounted == 20
            assert summary.failed == 0
            assert summary.left_pending == 0
            assert summary.elapsed_s < 5.0, f"recovery took {summary.elapsed_s:.2f}s"
            assert summary.writer_address == fresh_writer.address

            # NO DATA LOSS at the store level: all 20 texts, each exactly once.
            expected = [
                text for agent in agents for text in texts_by_agent[agent]
            ]
            entries = fresh_store.list()
            assert len(entries) == 20
            assert sorted(entry.text for entry in entries) == sorted(expected)
            assert len({entry.id for entry in entries}) == 20

            _drain(fresh_writer)
            assert fresh_store.git.is_clean()
            # the journal is fully resolved: no op left unacked
            assert journal.unacked() == []
            # every surviving entry is durably committed
            for entry in entries:
                assert fresh_store.git.log(
                    path=f"global/facts/{entry.id}.md", limit=5
                )
    finally:
        journal.close()
