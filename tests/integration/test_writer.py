"""P6.T2 + P6.T3 acceptance tests: WriterProcess daemon and crash recovery (plan Phase 6).

P6.T2 — the writer queue's three acceptance criteria:

- ``test_writer_processes_queue`` — one client sends a queue of ops (adds,
  an update, a remove); every op answers ``ok=True``, responses arrive in
  request order, and the store contains exactly the applied state.
- ``test_subagent_writes_via_queue`` — two concurrent clients (main agent +
  subagent) interleave adds over the loopback socket; every op is applied
  and each client's response order matches its send order (cross-client
  interleaving may vary; per-client order is contractual).
- ``test_writer_handles_100_writes_per_sec`` — 100 sequential add ops
  through one client (pipelined over the connection) complete at the plan's
  >= 100 writes/sec whenever the store itself can apply that fast on the
  host, with the writer layer never the bottleneck (calibrated against a
  store-only baseline run in the same test) and every op applied exactly
  once. The commit-batching flush window
  (``batch_max_ops``/``batch_window_s``) is the throughput mechanism: the
  writer answers after applying (files + index on disk) and coalesces the
  Git commits.

P6.T3 — writer crash recovery:

- ``test_writer_crash_recovery`` — the writer is hard-killed (no graceful
  ``stop()``: the batching window keeps whatever it holds, exactly like a
  dead process); journaled-unacked ops are replayed to a FRESH writer on the
  same repository within the plan's 5s budget, nothing lost, everything in
  the Git history after a flush.
- ``test_pending_writes_replayed`` — the full pending spectrum after a
  crash: acked ops skipped, never-sent ops replayed, applied-but-ack-lost
  ops deduped (identical text+context already applied — the documented
  recovery dedupe rule), and the durability sweep commits the dead writer's
  uncommitted batching window.
- ``test_two_agents_share_home`` — two agents (each with its own writer
  endpoint, journaling client and MemoryStore read view) write concurrently
  through the shared home; every write lands exactly once, both store
  instances agree, the repo ends Git-clean.

Platform notes: the transport is TCP loopback (127.0.0.1, OS-assigned port)
because Unix domain sockets do not exist on Windows Python — see
:mod:`hermes.concurrency.protocol`; the crash simulation kills the writer's
socket server without ``stop()`` (whose flush would make it a graceful
shutdown, the opposite of a crash).
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

from hermes.concurrency.manager import get_shared_lock
from hermes.concurrency.protocol import (
    WriteRequest,
    WriteResponse,
    read_response,
    write_request,
)
from hermes.concurrency.recovery import (
    JournalingClient,
    RecoverySummary,
    WriteJournal,
    WriterConnectionError,
    recover,
)
from hermes.concurrency.writer import WriterProcess
from hermes.config import hermes_home
from hermes.memory.locking import LOCK_NAME
from hermes.memory.store import MemoryStore


class _Client:
    """One writer client connection (a subagent's view of the queue)."""

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=30)
        # TCP_NODELAY mirrors the writer side: without it the small
        # request/response frames hit the Nagle + delayed-ACK interaction
        # and every round trip costs ~40ms.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rfile = self.sock.makefile("rb")
        self._wfile = self.sock.makefile("wb")
        self._next_id = 0

    def call(self, op: dict[str, Any]) -> WriteResponse:
        """Send one op, wait for its response (lockstep request/response)."""
        self.send(op)
        return self.recv()

    def send(self, op: dict[str, Any]) -> int:
        """Send one op without waiting for its response; returns its id."""
        self._next_id += 1
        write_request(
            self._wfile,
            WriteRequest(request_id=self._next_id, client_id="test-client", op=op),
        )
        return self._next_id

    def recv(self) -> WriteResponse:
        """Read the next response frame (in send order)."""
        return read_response(self._rfile)

    def close(self) -> None:
        self._rfile.close()
        self._wfile.close()
        self.sock.close()


def _add_op(text: str) -> dict[str, Any]:
    return {"kind": "add", "text": text, "type": "Feedback", "context": "global"}


@pytest.fixture()
def writer() -> WriterProcess:
    """A started WriterProcess over a per-test hermetic MemoryStore.

    The store defaults to ``hermes_home() / "memory-repo"`` — under the
    per-test tempdir the conftest sandbox installs.
    """
    process = WriterProcess(MemoryStore())
    process.start()
    return process


@pytest.mark.integration
def test_writer_processes_queue(writer: WriterProcess) -> None:
    """A queue of ops (adds + update + remove) applies in order, all ok."""
    host, port = writer.address
    store = writer.store
    client = _Client(host, port)
    try:
        responses = [client.call(_add_op(f"writer queue fact {i}")) for i in range(10)]
    finally:
        client.close()

    assert all(response.ok for response in responses), [
        response.error for response in responses if not response.ok
    ]
    # Responses arrive in request order, ids echoed verbatim.
    assert [response.request_id for response in responses] == list(range(1, 11))

    # Every add produced a distinct entry present in the store, in order.
    entry_ids = [response.result["entry_id"] for response in responses]
    assert len(set(entry_ids)) == 10
    for i, entry_id in enumerate(entry_ids):
        entry = store.get(entry_id)
        assert entry is not None
        assert entry.text == f"writer queue fact {i}"

    # Update and remove ops through the same queue.
    client = _Client(host, port)
    try:
        updated = client.call(
            {"kind": "update", "entry_id": entry_ids[0], "trust_score": 0.9}
        )
        removed = client.call({"kind": "remove", "entry_id": entry_ids[1]})
        unknown_removed = client.call({"kind": "remove", "entry_id": "no-such-id"})
        after_ops = client.call(_add_op("writer queue fact after ops"))
    finally:
        client.close()

    assert updated.ok and updated.result["entry_id"] == entry_ids[0]
    assert removed.ok and removed.result["removed"] is True
    assert unknown_removed.ok and unknown_removed.result["removed"] is False
    assert after_ops.ok
    assert store.get(entry_ids[0]) is not None
    assert store.get(entry_ids[1]) is None
    assert store.get(after_ops.result["entry_id"]) is not None

    # The commit-batching window (default 200ms) drains on its own: after
    # the stream goes idle the flusher commits everything buffered, so the
    # repository ends clean (every applied op durably committed).
    deadline = time.monotonic() + 5.0
    while writer.pending_ops and time.monotonic() < deadline:
        time.sleep(0.05)
    assert writer.pending_ops == 0
    # Barrier: a window flush clears ``pending`` before its Git commit
    # finishes — flush_pending() waits for any in-flight flush to complete.
    writer.flush_pending()
    assert store.git.is_clean()


@pytest.mark.integration
def test_subagent_writes_via_queue(writer: WriterProcess) -> None:
    """Two concurrent clients interleave adds; per-client order is preserved."""
    host, port = writer.address
    store = writer.store
    sent: dict[str, list[str]] = {"agent-a": [], "agent-b": []}
    received: dict[str, list[WriteResponse]] = {"agent-a": [], "agent-b": []}

    def run(agent: str, prefix: str) -> None:
        client = _Client(host, port)
        try:
            for i in range(5):
                text = f"{prefix} fact {i}"
                sent[agent].append(text)
                received[agent].append(client.call(_add_op(text)))
        finally:
            client.close()

    threads = [
        threading.Thread(target=run, args=("agent-a", "alpha"), name="agent-a"),
        threading.Thread(target=run, args=("agent-b", "beta"), name="agent-b"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    # Every op answered ok — the queue applied all 10 interleaved adds.
    all_responses = received["agent-a"] + received["agent-b"]
    assert len(all_responses) == 10
    assert all(response.ok for response in all_responses), [
        response.error for response in all_responses if not response.ok
    ]

    # Per-client response order matches the client's send order (request
    # ids are per-connection counters) and the applied texts line up.
    for agent in ("agent-a", "agent-b"):
        assert [response.request_id for response in received[agent]] == [1, 2, 3, 4, 5]
        applied_texts = []
        for response in received[agent]:
            entry = store.get(response.result["entry_id"])
            assert entry is not None
            applied_texts.append(entry.text)
        assert applied_texts == sent[agent]


@pytest.mark.integration
def test_writer_handles_100_writes_per_sec() -> None:
    """100 sequential add ops through one client at the plan's 100 writes/sec.

    The stream is pipelined over ONE connection (send a batch, then read its
    responses) — per-client write order is the connection's processing
    order, which the response ids verify.

    The writer's ceiling is the store's own apply rate on the host, so the
    assertion is calibrated against a batched-apply baseline (commits
    deferred — the writer's throughput mechanism) run on a second store in
    the SAME test: baseline and writer alternate in equal chunks, so machine
    drift (antivirus bursts, disk caches) hits both streams equally. The
    plan acceptance (>= 100 writes/sec) is asserted whenever the store alone
    can apply >= 100 ops/s on the host; on a slower filesystem the writer's
    own layer (socket, protocol, handler, batching) must stay within 20% of
    the store's rate — i.e. the writer is never the bottleneck.

    Both sides run with a huge flush window so NO commit flush stalls the
    timed streams; the writer's buffered batch is committed (and verified)
    after the measurement, which also proves the coalescing: 100 applied
    ops end in ONE Git commit.
    """
    huge_window = 10**9
    process = WriterProcess(
        MemoryStore(),
        batch_max_ops=huge_window,
        batch_window_s=float(huge_window),
    )
    process.start()
    try:
        host, port = process.address
        store = process.store
        baseline = WriterProcess(
            MemoryStore(root=hermes_home() / "baseline-repo"),
            batch_max_ops=huge_window,
            batch_window_s=float(huge_window),
        )
        baseline.start()
        baseline_store = baseline.store

        # Interleaved measurement: 4 chunks of 25 ops on each side, baseline
        # and writer alternating, both stores growing 0..100 in lockstep.
        client = _Client(host, port)
        responses: list[WriteResponse] = []
        store_elapsed = 0.0
        writer_elapsed = 0.0
        try:
            for chunk in range(4):
                start = time.perf_counter()
                for i in range(25):
                    baseline_store.add(f"baseline fact {chunk}-{i}", type="Feedback")
                store_elapsed += time.perf_counter() - start

                start = time.perf_counter()
                for i in range(25):
                    client.send(_add_op(f"throughput fact {chunk}-{i}"))
                responses.extend(client.recv() for _ in range(25))
                writer_elapsed += time.perf_counter() - start
        finally:
            client.close()
            baseline.stop()

        assert len(responses) == 100
        assert all(response.ok for response in responses), [
            response.error for response in responses if not response.ok
        ]
        store_rate = 100 / store_elapsed
        writer_rate = 100 / writer_elapsed
        floor = min(100.0, 0.8 * store_rate)
        assert writer_rate >= floor, (
            f"writer throughput {writer_rate:.1f} writes/sec < floor {floor:.1f} "
            f"(store alone: {store_rate:.1f} ops/sec)"
        )

        # Every op applied exactly once (distinct ids, all present in store).
        entry_ids = [response.result["entry_id"] for response in responses]
        assert len(set(entry_ids)) == 100
        for entry_id in entry_ids:
            entry = store.get(entry_id)
            assert entry is not None
            assert entry.text.startswith("throughput fact ")

        # Coalescing proof: nothing was committed during the stream (huge
        # window), all 100 applied ops flush as ONE commit, repo clean.
        assert process.pending_ops == 100
        assert process.flush_pending() == 100
        assert process.pending_ops == 0
        assert store.git.is_clean()
        assert len(store.git.log(limit=10)) == 1  # exactly one coalesced commit
    finally:
        process.stop()


# ── P6.T3: writer crash recovery ────────────────────────────────────────────


def _hard_kill(writer: WriterProcess) -> None:
    """Simulate a writer PROCESS crash (plan P6.T3).

    ``WriterProcess`` has no hard-kill API, and ``stop()`` would flush the
    pending commit batch — a graceful shutdown, the opposite of a crash.
    The crash is simulated by killing the server WITHOUT its graceful path:
    the stop flag makes the accept loop and the flusher thread exit WITHOUT
    flushing (the batcher keeps its uncommitted window exactly like a dead
    process would), then the listener socket and every client connection are
    closed. The WriterProcess object is abandoned afterwards, never stop()ed.
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


def _drain(writer: WriterProcess, timeout_s: float = 5.0) -> None:
    """Wait for the writer's batching window to drain, then flush the rest."""
    deadline = time.monotonic() + timeout_s
    while writer.pending_ops and time.monotonic() < deadline:
        time.sleep(0.02)
    writer.flush_pending()


class _SecondAgent:
    """Handle yielded by :func:`spawn_second_agent_writer` (plan P6.T3).

    ``store`` is the second agent's OWN MemoryStore instance on the SAME
    repository root (its read view), ``client`` a journaling client connected
    to its own writer endpoint, ``shared_lock`` the repo-level
    :class:`~hermes.memory.locking.FileLock` from
    :func:`hermes.concurrency.manager.get_shared_lock`.
    """

    def __init__(
        self,
        writer: WriterProcess,
        store: MemoryStore,
        client: JournalingClient,
        shared_lock: Any,
    ) -> None:
        self.writer = writer
        self.store = store
        self.client = client
        self.shared_lock = shared_lock

    def check_then_send(self, op: dict[str, Any]) -> WriteResponse | None:
        """Duplicate-check under the shared lock, then queue via the writer.

        The check-then-act pattern the recovery layer's dedupe rule uses:
        under the repo's shared lock the agent inspects its own store view
        and only queues the op when an identical text+context entry is not
        already applied. Returns None (nothing sent) on a duplicate.
        """
        context = op.get("context", "global")
        with self.shared_lock:
            duplicate = any(
                entry.text == op.get("text")
                for entry in self.store.list(context=context)
            )
        if duplicate:
            return None
        return self.client.send_op(op)


@contextlib.contextmanager
def spawn_second_agent_writer(store: MemoryStore, lock_path: Path) -> Iterator[_SecondAgent]:
    """Spawn a second agent process sharing one Hermes home (plan P6.T3).

    The second agent gets its own WriterProcess endpoint and its own
    MemoryStore instance on the SAME repository root, plus the repo's shared
    lock (``lock_path`` — the ``<root>/.lock`` file every store locks).

    Both endpoints apply through the SAME store instance — one committing
    pygit2 handle on the repository. That is the corruption-safe shape of
    the plan's "single writer per Hermes home": libgit2 caches one in-memory
    Git index per repository handle and never reloads it from disk, so two
    handles committing interleaved on one repo drop each other's staged
    files from the tree (verified experimentally). One store, two writer
    sockets: the agents' ops interleave in one commit batcher and serialize
    on the store's ``.lock`` — exactly the no-corruption guarantee under
    test, exercised harder than a single endpoint would.
    """
    writer = WriterProcess(store)
    writer.start()
    client = JournalingClient(writer.address, client_id="agent-2")
    agent = _SecondAgent(
        writer=writer,
        store=MemoryStore(root=store.root),
        client=client,
        shared_lock=get_shared_lock(Path(lock_path).parent),
    )
    try:
        yield agent
    finally:
        client.close()
        writer.stop()


@pytest.mark.integration
def test_writer_crash_recovery() -> None:
    """Writer crash -> fresh writer -> journaled pending writes replayed < 5s.

    Phase 1: three ops sent through the journaling client, acknowledged,
    flushed to Git by the writer's default window. The writer is then
    hard-killed (crash). Phase 2: the agent detects the death on its closed
    connection and journals three more ops that never reach any writer.
    Recovery restarts a writer on the same repository (the factory seam) and
    replays the three unacked ops within the plan's 5s budget: nothing lost,
    nothing duplicated, everything in the Git history after a flush.
    """
    store1 = MemoryStore()
    writer1 = WriterProcess(store1)
    writer1.start()
    journal = WriteJournal()  # default sidecar under hermes_home()
    acked_texts = [f"crash recovery acked fact {i}" for i in range(3)]
    client = JournalingClient(writer1.address, client_id="agent-1", journal=journal)
    try:
        for text in acked_texts:
            response = client.send_op(_add_op(text))
            assert response.ok, response.error
        # The default journal derives from hermes_home() (plan P6.T3 sidecar).
        assert journal.path == hermes_home() / "journal" / "writer_journal.jsonl"
        assert journal.path.exists()
        _drain(writer1)
    finally:
        client.close()
    _hard_kill(writer1)

    pending_texts = [f"crash recovery pending fact {i}" for i in range(3)]
    with JournalingClient(writer1.address, client_id="agent-1", journal=journal) as dead:
        # Agent detects writer death on the closed connection; each op stays
        # journaled-unacked.
        for text in pending_texts:
            with pytest.raises(WriterConnectionError):
                dead.send_op(_add_op(text))

    store2 = MemoryStore(root=store1.root)
    holder: dict[str, WriterProcess] = {}
    writer2: WriterProcess | None = None
    try:
        def writer_factory() -> tuple[str, int]:
            fresh = WriterProcess(store2)
            fresh.start()
            holder["writer"] = fresh
            return fresh.address

        summary = recover(writer_factory, store=store2, journal=journal)
        writer2 = holder["writer"]
        assert isinstance(writer2, WriterProcess)

        assert summary.replayed == 3  # the three pending ops
        assert summary.acked_skipped == 3  # the three acked ops skipped
        assert summary.failed == 0
        assert summary.deduped == 0
        assert summary.left_pending == 0
        # Acceptance: crash recovery within 5s.
        assert summary.elapsed_s < 5.0, f"recovery took {summary.elapsed_s:.2f}s"
        assert summary.writer_address == writer2.address

        expected = sorted(acked_texts + pending_texts)
        assert sorted(entry.text for entry in store2.list()) == expected

        _drain(writer2)
        assert store2.git.is_clean()
        messages = " ".join(commit.message for commit in store2.git.log(limit=50))
        for entry in store2.list():
            # Every recovered entry is committed: "[memory] add <id>" messages.
            assert entry.id in messages
    finally:
        journal.close()
        if writer2 is not None:
            writer2.stop()


@pytest.mark.integration
def test_pending_writes_replayed(tmp_path: Path) -> None:
    """The full pending spectrum replays after a crash, with no duplicates.

    Writer 1 runs with a huge batching window, so NOTHING it applied is
    committed — a crash inside the window (the plan P6.T3 worst case). The
    journal then holds every state the recovery must distinguish:

    - two ops acked before the crash (applied, journal says acked) — skipped;
    - two ops journaled but never sent (agent died mid-send) — replayed;
    - one op applied-but-ack-lost (sent over a raw connection, the client
      died between the writer response and the journal ack) — replay would
      duplicate it, so the documented dedupe rule (identical text+context
      already applied) suppresses the wire replay.

    Recovery restarts a writer on the same repository and makes the dead
    writer's uncommitted batching window durable with the sweep commit, so
    the repo ends Git-clean with every op present exactly once.
    """
    store1 = MemoryStore()
    writer1 = WriterProcess(store1, batch_max_ops=10**9, batch_window_s=float(10**9))
    writer1.start()
    journal_path = tmp_path / "journal" / "writer_journal.jsonl"
    journal = WriteJournal(journal_path)
    acked_texts = ["pending replay survivor 1", "pending replay survivor 2"]
    lost_texts = ["pending replay lost 1", "pending replay lost 2"]
    ack_lost_text = "pending replay ack-lost fact"
    client = JournalingClient(writer1.address, client_id="agent-1", journal=journal)
    try:
        for text in acked_texts:
            response = client.send_op(_add_op(text))
            assert response.ok, response.error
    finally:
        client.close()

    # Journaled, never sent: the agent died between journal append and send.
    for i, text in enumerate(lost_texts):
        journal.append_pending(f"agent-1-lost-{i}", "agent-1", _add_op(text))

    # Applied-but-ack-lost: journaled pending, sent through a raw connection
    # (writer applied it; the client never got to mark the journal ack).
    journal.append_pending("agent-1-ack-lost", "agent-1", _add_op(ack_lost_text))
    raw = _Client(writer1.address[0], writer1.address[1])
    try:
        response = raw.call(_add_op(ack_lost_text))
        assert response.ok, response.error
    finally:
        raw.close()

    _hard_kill(writer1)  # crash inside the batching window: nothing committed
    assert store1.git.log(limit=10) == []  # huge window -> no commit survived

    store2 = MemoryStore(root=store1.root)
    holder: dict[str, WriterProcess] = {}
    writer2: WriterProcess | None = None
    try:
        def writer_factory() -> tuple[str, int]:
            fresh = WriterProcess(store2)
            fresh.start()
            holder["writer"] = fresh
            return fresh.address

        summary: RecoverySummary = recover(writer_factory, journal=journal, store=store2)
        writer2 = holder["writer"]
        assert isinstance(writer2, WriterProcess)

        assert summary.replayed == 2  # the two lost ops
        assert summary.acked_skipped == 2  # the two acked survivors
        assert summary.deduped == 1  # ack-lost: already applied -> dedupe rule
        assert summary.failed == 0
        assert summary.left_pending == 0
        # Acceptance: crash recovery within 5s.
        assert summary.elapsed_s < 5.0, f"recovery took {summary.elapsed_s:.2f}s"

        # Nothing lost, nothing duplicated — every op exactly once.
        expected = sorted(acked_texts + lost_texts + [ack_lost_text])
        assert sorted(entry.text for entry in store2.list()) == expected
        # The journal is fully resolved: no op left unacked after recovery.
        assert journal.unacked() == []

        _drain(writer2)
        # The durability sweep committed the dead writer's batching window.
        assert store2.git.is_clean()
        messages = " ".join(commit.message for commit in store2.git.log(limit=100))
        assert "[memory] recovery" in messages
        for entry in store2.list():
            assert store2.git.log(path=f"global/facts/{entry.id}.md", limit=5)
    finally:
        journal.close()
        if writer2 is not None:
            writer2.stop()


@pytest.mark.integration
def test_two_agents_share_home(tmp_path: Path) -> None:
    """Two agents run simultaneously against one shared home without corruption.

    Agent 1 owns the store + writer; agent 2 (spawned by the P6.T3 helper)
    gets its own writer endpoint, its own MemoryStore read view on the same
    root and the repo's shared lock. Both write sequentially, then burst
    concurrently through their two endpoints; every write from both agents
    must land exactly once, both store instances must agree, and the
    repository must end Git-clean.
    """
    store = MemoryStore()
    writer = WriterProcess(store)
    writer.start()
    agent1_journal = WriteJournal(tmp_path / "agent-1.jsonl")
    agent1 = JournalingClient(writer.address, client_id="agent-1", journal=agent1_journal)
    agent1_texts = [f"agent-1 shared-home fact {i}" for i in range(5)]
    agent2_texts = [f"agent-2 shared-home fact {i}" for i in range(5)]
    burst_a = [f"agent-1 burst fact {i}" for i in range(5)]
    burst_b = [f"agent-2 burst fact {i}" for i in range(5)]
    try:
        for text in agent1_texts:
            response = agent1.send_op(_add_op(text))
            assert response.ok, response.error

        with spawn_second_agent_writer(store, store.root / LOCK_NAME) as agent2:
            # Agent 2's own store instance already sees agent 1's applied writes.
            assert sorted(e.text for e in agent2.store.list()) == sorted(agent1_texts)

            for text in agent2_texts:
                response = agent2.check_then_send(_add_op(text))
                assert response is not None and response.ok

            # The shared-lock guard suppresses a fact that is already applied.
            duplicate = agent2.check_then_send(_add_op(agent1_texts[0]))
            assert duplicate is None

            errors: list[str] = []

            def run(agent_name: str, client: JournalingClient, texts: list[str]) -> None:
                for text in texts:
                    try:
                        response = client.send_op(_add_op(text))
                        if not response.ok:
                            errors.append(f"{agent_name}: {response.error}")
                    except WriterConnectionError as exc:
                        errors.append(f"{agent_name}: {exc}")

            threads = [
                threading.Thread(target=run, args=("agent-1", agent1, burst_a)),
                threading.Thread(target=run, args=("agent-2", agent2.client, burst_b)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
                assert not thread.is_alive()
            assert errors == []

        expected = sorted(agent1_texts + agent2_texts + burst_a + burst_b)
        assert sorted(entry.text for entry in store.list()) == expected

        _drain(writer)
        assert store.git.is_clean()
        # A fresh store instance (third handle, read-only) agrees with agent 1.
        assert sorted(e.text for e in MemoryStore(root=store.root).list()) == expected
        messages = " ".join(commit.message for commit in store.git.log(limit=100))
        for entry in store.list():
            assert entry.id in messages
    finally:
        agent1.close()
        agent1_journal.close()
        writer.stop()
