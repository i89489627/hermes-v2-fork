"""P20.T3: multi-agent memory sharing (plan Phase 20; P6.T3 finding applied).

Modeling note (honest, per the P6.T3 finding): multiple agents share one
Hermes home, and the plan's single-writer model is what keeps the shared
repository corruption-free. libgit2 keeps ONE in-memory git index per
repository HANDLE and never reloads it from disk (documented on
``MemoryStore._stage_removal`` and verified experimentally in
``tests/integration/test_writer.py::spawn_second_agent_writer``): two handles
committing interleaved on one repository drop each other's staged files from
the tree. Concurrent writers here therefore share ONE committing store
handle — the agents are THREADS over a single :class:`MemoryStore` instance
(the corruption-safe shape; the store's :class:`~hermes.memory.locking.FileLock`
serializes every mutation) — while a separate READ-ONLY verification store
instance (a fresh handle that only re-reads from disk, never commits) proves
cross-agent visibility of everything the writers persisted.

Scenarios (plan P20.T3):
- ``test_two_agents_concurrent_writes`` — 2 agents x 10 adds: 20 entries,
  git clean, the read view sees everything.
- ``test_main_plus_three_subagents`` — main 5 + three subagents 5 each:
  20 entries in a consistent state on both handles.
- ``test_write_visibility_immediate`` — agent A writes, the fresh
  cross-instance read view of agent B sees the entry within 1s.
- ``test_concurrent_update_same_entry`` — two agents update the same entry:
  last-writer-wins, exactly one of the two texts survives, the trust card
  parses, both updates are in the git history, git clean.
- ``test_writer_crash_during_multi_agent`` — writer + journaling clients
  crash mid-stream; the P6.T3 journal/recovery seam replays every op; no
  lost ops at the store level, repository ends git-clean.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
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
from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore


def _add_op(text: str) -> dict[str, Any]:
    """One writer-protocol add op (Feedback entry, global context)."""
    return {"kind": "add", "text": text, "type": "Feedback", "context": "global"}


def _hard_kill(writer: WriterProcess) -> None:
    """Simulate a writer PROCESS crash: kill the server without any flush.

    Same shape as the P6.T3 helper in ``tests/integration/test_writer.py``:
    ``stop()`` would flush the pending commit batch (graceful — the opposite
    of a crash), so the stop flag makes the accept loop and flusher thread
    exit WITHOUT flushing, then the listener and every client connection are
    closed and the WriterProcess object is abandoned.
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


def _run_agents(
    targets: list[tuple[str, int, str]],
    store: MemoryStore,
    errors: list[str],
) -> None:
    """Run ``count`` agent threads each performing ``per_agent`` adds."""
    start = threading.Barrier(len(targets))

    def agent(name: str, per_agent: int, prefix: str) -> None:
        start.wait()
        for index in range(per_agent):
            try:
                store.add(f"{prefix} fact {index}", type="Feedback", source="user-explicit")
            except Exception as exc:  # any failure is collected, never swallowed
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=agent, args=target, name=target[0]) for target in targets
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive()


def _assert_shared_state(store: MemoryStore, reader: MemoryStore, expected: list[str]) -> None:
    """Both handles agree on exactly ``expected``; git clean; commits durable."""
    entries = store.list()
    assert sorted(entry.text for entry in entries) == sorted(expected)
    assert len({entry.id for entry in entries}) == len(expected)
    # the cross-instance read view (fresh handle, re-reads from disk) agrees
    assert sorted(entry.text for entry in reader.list()) == sorted(expected)
    assert store.git.is_clean()
    # every entry is durably committed (one commit per add)
    for entry in entries:
        assert store.git.log(path=f"global/facts/{entry.id}.md", limit=5)


@pytest.mark.integration
def test_two_agents_concurrent_writes() -> None:
    """2 agents x 10 concurrent adds: 20 entries, no corruption, git clean."""
    store = MemoryStore()  # the ONE committing handle both agents share
    reader = MemoryStore(root=store.root)  # read-only cross-agent verification view
    errors: list[str] = []
    expected = sorted(
        f"agent-{agent} fact {index}" for agent in (1, 2) for index in range(10)
    )

    _run_agents(
        [("agent-1", 10, "agent-1"), ("agent-2", 10, "agent-2")], store, errors
    )

    assert errors == []
    _assert_shared_state(store, reader, expected)


@pytest.mark.integration
def test_main_plus_three_subagents() -> None:
    """Main agent 5 + three subagents 5 each: 20 entries, consistent state."""
    store = MemoryStore()
    reader = MemoryStore(root=store.root)
    for index in range(5):
        store.add(f"main fact {index}", type="Feedback", source="user-explicit")
    assert len(store.list()) == 5

    errors: list[str] = []
    _run_agents(
        [
            ("subagent-1", 5, "subagent-1"),
            ("subagent-2", 5, "subagent-2"),
            ("subagent-3", 5, "subagent-3"),
        ],
        store,
        errors,
    )
    assert errors == []

    expected = sorted(
        [f"main fact {index}" for index in range(5)]
        + [
            f"subagent-{agent} fact {index}"
            for agent in (1, 2, 3)
            for index in range(5)
        ]
    )
    _assert_shared_state(store, reader, expected)
    # the main agent's earlier writes are still visible to every subagent view
    assert {entry.id for entry in store.list(context="global")} >= {
        entry.id for entry in reader.list() if entry.text.startswith("main fact")
    }


@pytest.mark.integration
def test_write_visibility_immediate() -> None:
    """Agent A writes; agent B's own store instance sees it within 1s."""
    store = MemoryStore()  # agent A's committing handle
    reader = MemoryStore(root=store.root)  # agent B: fresh handle, reads from disk
    ready = threading.Event()
    published: dict[str, Any] = {}

    def agent_a() -> None:
        entry = store.add("visibility probe fact", type="Feedback", source="user-explicit")
        published["entry_id"] = entry.id
        published["written_at"] = time.perf_counter()
        ready.set()

    thread = threading.Thread(target=agent_a, name="agent-a")
    thread.start()
    assert ready.wait(timeout=10.0), "agent A did not finish its write"
    entry_id: str = published["entry_id"]

    # Agent B polls its OWN instance for A's write — visible within 1s.
    deadline = time.monotonic() + 1.0
    seen: MemoryEntry | None = None
    while time.monotonic() < deadline:
        seen = reader.get(entry_id)
        if seen is not None:
            break
        time.sleep(0.01)
    assert seen is not None, "agent B did not see agent A's write within 1s"
    assert seen.text == "visibility probe fact"
    visibility_latency = time.perf_counter() - published["written_at"]
    assert visibility_latency < 1.0, f"visibility took {visibility_latency:.3f}s"

    # the same entry is visible from the writing store and durably committed
    assert store.get(entry_id) is not None
    assert store.git.log(path=f"global/facts/{entry_id}.md", limit=5)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert store.git.is_clean()


@pytest.mark.integration
def test_concurrent_update_same_entry() -> None:
    """Same-entry race: both updates serialize, exactly one text survives."""
    store = MemoryStore()
    entry = store.add("original text before the race", type="Feedback", source="user-explicit")
    text_a = "agent A's competing update"
    text_b = "agent B's competing update"
    start = threading.Barrier(2)
    outcomes: dict[str, str] = {}

    def updater(name: str, new_text: str) -> None:
        start.wait()
        try:
            store.update(entry.id, text=new_text)
            outcomes[name] = "ok"
        except Exception as exc:
            outcomes[name] = f"{type(exc).__name__}: {exc}"

    threads = [
        threading.Thread(target=updater, args=("agent-a", text_a), name="agent-a"),
        threading.Thread(target=updater, args=("agent-b", text_b), name="agent-b"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()

    # Both writes completed (the shared FileLock serialized them); the entry
    # resolved to last-writer-wins: EXACTLY one of the two texts survives.
    assert outcomes == {"agent-a": "ok", "agent-b": "ok"}
    final = store.get(entry.id)
    assert final is not None
    assert final.id == entry.id
    assert final.text in {text_a, text_b}
    assert final.text != "original text before the race"
    # the trust card parses intact (history preserved through both rewrites)
    assert final.verification_events
    assert final.trust_score > 0.0

    # both updates are recorded in the entry's git history (no interleaving)
    history = store.git.log(path=f"global/facts/{entry.id}.md", limit=10)
    assert len(history) == 3  # add + update + update
    update_messages = [commit.message for commit in history[:2]]
    assert all("[memory] update" in message for message in update_messages)

    # a fresh read-only instance agrees — no corruption either way
    reader = MemoryStore(root=store.root)
    fresh = reader.get(entry.id)
    assert fresh is not None and fresh.text == final.text
    assert store.git.is_clean()


@pytest.mark.integration
def test_writer_crash_during_multi_agent(tmp_path: Path) -> None:
    """Writer + journaling clients crash mid-stream; recovery loses no ops."""
    store = MemoryStore()
    writer = WriterProcess(
        store, batch_max_ops=10**9, batch_window_s=float(10**9)
    )  # huge window: the crash lands inside the uncommitted batch
    writer.start()
    journal = WriteJournal(tmp_path / "journal" / "writer_journal.jsonl")
    main_texts = [f"main agent midstream fact {index}" for index in range(10)]
    sub_texts = [f"subagent midstream fact {index}" for index in range(10)]
    main_client = JournalingClient(writer.address, client_id="main", journal=journal)
    sub_client = JournalingClient(writer.address, client_id="sub", journal=journal)
    connection_errors: list[str] = []

    def run(client: JournalingClient, texts: list[str]) -> None:
        for text in texts:
            try:
                client.send_op(_add_op(text))
            except WriterConnectionError as exc:
                # Writer died mid-op: journaled-unacked -> replayed below.
                connection_errors.append(str(exc))

    threads = [
        threading.Thread(target=run, args=(main_client, main_texts), name="main"),
        threading.Thread(target=run, args=(sub_client, sub_texts), name="sub"),
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.05)  # mid-stream: ops applied (files), nothing committed yet
    _hard_kill(writer)
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    main_client.close()
    sub_client.close()
    assert store.git.log(limit=10) == []  # huge window -> nothing was committed

    fresh_store = MemoryStore(root=store.root)  # the fresh writer's own handle
    try:
        with _fresh_writer(fresh_store) as fresh_writer:
            def writer_factory() -> tuple[str, int]:
                return fresh_writer.address

            summary: RecoverySummary = recover(
                writer_factory, store=fresh_store, journal=journal
            )
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

            # No lost ops, no duplicates: every op exactly once, both agents.
            expected = sorted(main_texts + sub_texts)
            entries = fresh_store.list()
            assert len(entries) == 20
            assert sorted(entry.text for entry in entries) == expected
            assert len({entry.text for entry in entries}) == 20

            _drain(fresh_writer)
            assert fresh_store.git.is_clean()
            assert journal.unacked() == []
    finally:
        journal.close()
