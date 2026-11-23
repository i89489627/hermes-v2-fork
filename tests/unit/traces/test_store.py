"""P11.T1 acceptance: TraceStore JSONL + SQLite/FTS5 index (plan Phase 11).

Every append writes one JSONL line under ``<root>/v1.0/YYYY-MM-DD.jsonl`` and
upserts a row into ``traces.db`` with an FTS5 index synced from the payload;
search returns relevant traces in under 100 ms; quality relabeling updates
both representations; ``get_gepa_ready`` filters the GEPA training set.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

_TS = 1_700_000_000.0  # fixed epoch (2023-11-14 UTC) for deterministic day files


def _day_file(root: Path, timestamp: float) -> Path:
    day = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    return root / "v1.0" / f"{day}.jsonl"


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.unit
def test_append_creates_trace(tmp_path: Path) -> None:
    """append() writes JSONL per day, syncs the db row and assigns the id."""
    store = TraceStore(root=tmp_path / "traces")
    # Default root derives from hermes_home() so tests stay hermetic.
    assert TraceStore().root == hermes_home() / "traces"

    trace = Trace(
        turn_id="turn-1",
        session_id="sess-1",
        timestamp=_TS,
        user_prompt="hello world",
        models_used=["main-model"],
        gating={"decision": "use"},
    )
    trace_id = store.append(trace)
    assert trace_id
    assert trace.trace_id == trace_id  # id assigned back onto the caller's object

    jsonl = _day_file(tmp_path / "traces", _TS)
    assert jsonl.exists()
    records = _jsonl_records(jsonl)
    assert len(records) == 1
    assert records[0]["trace_id"] == trace_id
    assert records[0]["user_prompt"] == "hello world"
    assert records[0]["gating"] == {"decision": "use"}

    stored = store.get(trace_id)
    assert stored is not None
    assert stored.user_prompt == "hello world"
    assert stored.models_used == ["main-model"]
    assert stored.gating == {"decision": "use"}
    assert stored.quality_label is None
    assert stored.pii_flagged is False

    # A second trace on the same day appends a second line.
    second_id = store.append(
        Trace(
            turn_id="turn-2",
            session_id="sess-1",
            timestamp=_TS + 60.0,
            user_prompt="second",
        )
    )
    assert second_id != trace_id
    assert len(_jsonl_records(jsonl)) == 2
    assert store.get(second_id) is not None
    store.close()


@pytest.mark.unit
def test_search_returns_relevant(tmp_path: Path) -> None:
    """FTS5 search hits the right prompts (and payloads), ranked, with limit."""
    store = TraceStore(root=tmp_path / "traces")
    store.append(
        Trace(
            turn_id="a",
            session_id="s",
            timestamp=_TS,
            user_prompt="how do I reset the redis cluster password",
        )
    )
    store.append(
        Trace(
            turn_id="b",
            session_id="s",
            timestamp=_TS + 1.0,
            user_prompt="kubernetes pod stuck in terminating state",
        )
    )
    store.append(
        Trace(
            turn_id="c",
            session_id="s",
            timestamp=_TS + 2.0,
            user_prompt="redis replication lag alarm keeps firing",
        )
    )
    store.append(
        Trace(
            turn_id="d",
            session_id="s",
            timestamp=_TS + 3.0,
            user_prompt="completely unrelated question",
            extraction={"facts": ["sqlite wal checkpoint tuning helped"]},
        )
    )

    # CPython on Windows ships FTS5 — the store must be using it here.
    assert store.fts_available is True

    hits = store.search("redis", limit=10)
    assert {t.turn_id for t in hits} == {"a", "c"}
    assert all("redis" in t.user_prompt.lower() for t in hits)
    # Multi-word queries AND their terms.
    assert {t.turn_id for t in store.search("redis cluster")} == {"a"}
    # Stage payloads are indexed too.
    assert [t.turn_id for t in store.search("checkpoint")] == ["d"]
    # limit is honoured.
    assert len(store.search("redis", limit=1)) == 1
    # Whitespace-only query -> no hits.
    assert store.search("   ") == []
    # Returned records are full traces, not just ids.
    first = store.search("replication", limit=1)[0]
    assert first.session_id == "s"
    assert first.user_prompt.startswith("redis replication")
    # Invalid limits are rejected.
    with pytest.raises(ValueError):
        store.search("redis", limit=0)
    store.close()


@pytest.mark.unit
def test_label_quality_updates(tmp_path: Path) -> None:
    """label_quality rewrites the JSONL line, the db row and GEPA readiness."""
    store = TraceStore(root=tmp_path / "traces")
    trace_id = store.append(
        Trace(
            turn_id="t1",
            session_id="s",
            timestamp=_TS,
            user_prompt="fix the login redirect loop",
        )
    )
    jsonl = _day_file(tmp_path / "traces", _TS)
    stored = store.get(trace_id)
    assert stored is not None
    assert stored.quality_label is None

    store.label_quality(trace_id, "good")

    # JSONL (source of truth) was rewritten in place.
    records = {
        str(record["trace_id"]): record for record in _jsonl_records(jsonl)
    }
    assert records[trace_id]["quality_label"] == "good"
    # The db row was updated too.
    stored = store.get(trace_id)
    assert stored is not None
    assert stored.quality_label == "good"
    # GEPA-ready picks it up.
    assert [t.trace_id for t in store.get_gepa_ready()] == [trace_id]

    # Re-labeling to "bad" removes it from the GEPA-ready set.
    store.label_quality(trace_id, "bad")
    assert store.get_gepa_ready() == []
    stored = store.get(trace_id)
    assert stored is not None
    assert stored.quality_label == "bad"

    # Invalid labels and unknown ids are rejected.
    with pytest.raises(ValueError):
        store.label_quality(trace_id, "excellent")
    with pytest.raises(KeyError):
        store.label_quality("does-not-exist", "good")
    store.close()


@pytest.mark.unit
def test_get_gepa_ready_filters(tmp_path: Path) -> None:
    """Only non-PII traces labeled "good" enter the GEPA-ready set."""
    store = TraceStore(root=tmp_path / "traces")
    good_a = store.append(
        Trace(turn_id="ga", session_id="s", timestamp=_TS, user_prompt="a")
    )
    good_b = store.append(
        Trace(turn_id="gb", session_id="s", timestamp=_TS + 1.0, user_prompt="b")
    )
    bad = store.append(
        Trace(turn_id="bad", session_id="s", timestamp=_TS + 2.0, user_prompt="c")
    )
    unlabeled = store.append(
        Trace(turn_id="un", session_id="s", timestamp=_TS + 3.0, user_prompt="d")
    )
    pii_good = store.append(
        Trace(
            turn_id="pii",
            session_id="s",
            timestamp=_TS + 4.0,
            user_prompt="e",
            pii_flagged=True,
        )
    )
    store.label_quality(good_a, "good")
    store.label_quality(good_b, "good")
    store.label_quality(bad, "bad")
    store.label_quality(pii_good, "good")

    ready = store.get_gepa_ready()
    assert {t.turn_id for t in ready} == {"ga", "gb"}
    # Stable training order: ascending timestamp.
    assert [t.timestamp for t in ready] == sorted(t.timestamp for t in ready)

    # The unlabeled trace exists but is not ready; the PII "good" trace is
    # excluded despite its label.
    unlabeled_stored = store.get(unlabeled)
    assert unlabeled_stored is not None
    assert unlabeled_stored.quality_label is None
    pii_stored = store.get(pii_good)
    assert pii_stored is not None
    assert pii_stored.quality_label == "good"
    assert pii_stored.pii_flagged is True
    store.close()


@pytest.mark.unit
def test_search_under_100ms(tmp_path: Path) -> None:
    """Search latency budget: several queries over 300 traces, each < 100 ms."""
    store = TraceStore(root=tmp_path / "traces")
    topics = [
        "redis cache eviction policy",
        "kubernetes node drain stuck",
        "billing invoice reconciliation mismatch",
        "sqlite wal checkpoint starvation",
        "nginx rate limiting false positives",
    ]
    for i in range(300):
        topic = topics[i % len(topics)]
        store.append(
            Trace(
                turn_id=f"turn-{i}",
                session_id=f"sess-{i % 7}",
                timestamp=_TS + float(i),
                user_prompt=f"{topic} report {i}",
                models_used=["main-model"],
            )
        )
    queries = ["redis", "kubernetes drain", "invoice", "checkpoint", "rate limiting"]
    for query in queries:
        start = time.perf_counter()
        hits = store.search(query, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert hits, f"expected hits for {query!r}"
        assert elapsed_ms < 100.0, f"{query!r} took {elapsed_ms:.1f} ms (budget 100 ms)"
    store.close()
