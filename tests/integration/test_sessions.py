"""P21.T6 acceptance: session storage stays available under concurrency.

(plan Phase 21, #156160-family — "state.db WAL errors / session storage
unavailable")

Covered: the storage-backed :class:`~hermes.sessions.store.SessionStore`
(``SessionStore(storage=<pool>)``, plan P21.T6) persists the SAME session
state dicts with the JSON serialization preserved, never surfaces a
"session storage unavailable" failure (every save/load/list succeeds; the
error string never reaches the logs), and sustains concurrent access
through the pooled IMMEDIATE-transaction writes with zero WAL/locked
errors. The default JSON file mode is untouched: without ``storage`` the
store still writes its atomic ``<session_id>.json`` files.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from hermes.sessions.store import SessionStore
from hermes.storage.database import ConnectionPool

_WRITERS = 8
_SAVES_PER_WRITER = 25
_NOW = "2026-11-21T00:00:00+00:00"


def _state(writer: int, seq: int) -> dict[str, object]:
    """A representative nested ACP session state dict."""
    return {
        "session_id": f"sess-{writer}-{seq}",
        "created_at": _NOW,
        "metadata": {"writer": writer, "seq": seq, "tags": ["acp", f"w{writer}"]},
        "turns": [{"turn": seq, "text": f"turn {seq} of writer {writer}"}],
    }


def _worker(store: SessionStore, writer: int, errors: list[Exception]) -> None:
    """One concurrent writer: saves its own sessions, loads a neighbour's."""
    for seq in range(_SAVES_PER_WRITER):
        try:
            store.save_session(f"sess-{writer}-{seq}", _state(writer, seq))
            loaded = store.load_session(f"sess-{(writer + 1) % _WRITERS}-{seq}")
            if loaded is not None:
                assert loaded["metadata"]["seq"] == seq  # noqa: S101
        except Exception as exc:
            errors.append(exc)


@pytest.mark.integration
def test_no_storage_unavailable_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Storage-backed saves/loads/list succeed; the error string never appears."""
    pool = ConnectionPool(tmp_path / "state-v2.db")
    try:
        store = SessionStore(root=tmp_path / "sessions", storage=pool)

        state = _state(0, 0)
        assert store.save_session("sess-a", state) == pool.db_path
        # JSON serialization preserved: the loaded state dict round-trips exactly
        assert store.load_session("sess-a") == state
        store.save_session("sess-b", _state(1, 1))
        assert store.list_sessions() == ["sess-a", "sess-b"]
        # a never-saved session is None, not an error
        assert store.load_session("sess-never") is None

        # concurrent save/load churn: no failure of any kind, error string absent
        errors: list[Exception] = []
        threads = [
            threading.Thread(target=_worker, args=(store, w, errors)) for w in range(4)
        ]
        with caplog.at_level(logging.WARNING):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert "session storage unavailable" not in caplog.text
        assert "database is locked" not in caplog.text
        assert "SQLITE_BUSY" not in caplog.text
        assert store.list_sessions() == sorted(
            [f"sess-{w}-{s}" for w in range(4) for s in range(_SAVES_PER_WRITER)]
            + ["sess-a", "sess-b"]
        )
    finally:
        pool.close()


@pytest.mark.integration
def test_concurrent_access_no_wal_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """8 threads through one store+pool: zero WAL/locked errors, all sessions."""
    pool = ConnectionPool(tmp_path / "state-v2.db")
    try:
        store = SessionStore(root=tmp_path / "sessions", storage=pool)
        errors: list[Exception] = []
        threads = [
            threading.Thread(target=_worker, args=(store, w, errors))
            for w in range(_WRITERS)
        ]
        with caplog.at_level(logging.WARNING):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=180)
        assert not any(thread.is_alive() for thread in threads)

        # acceptance: no WAL / SQLITE_BUSY / 'locked' failures under contention
        assert errors == []
        assert "database is locked" not in caplog.text
        assert "SQLITE_BUSY" not in caplog.text
        assert "WAL error" not in caplog.text

        # every write landed and reads back exactly
        expected = [f"sess-{w}-{s}" for w in range(_WRITERS) for s in range(_SAVES_PER_WRITER)]
        assert store.list_sessions() == sorted(expected)
        for writer, seq in ((0, 0), (_WRITERS - 1, _SAVES_PER_WRITER - 1), (3, 12)):
            assert store.load_session(f"sess-{writer}-{seq}") == _state(writer, seq)
    finally:
        pool.close()


@pytest.mark.integration
def test_json_mode_default_untouched(tmp_path: Path) -> None:
    """Without ``storage`` the JSON file mode behaves exactly as before."""
    store = SessionStore(root=tmp_path / "sessions-json")
    state = _state(2, 3)
    path = store.save_session("sess-json-1", state)
    assert path == tmp_path / "sessions-json" / "sess-json-1.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert store.load_session("sess-json-1") == state
    assert store.list_sessions() == ["sess-json-1"]
