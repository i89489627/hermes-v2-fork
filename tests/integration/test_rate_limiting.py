"""P17.T2 integration: MemoryStore + RateLimiter wired end to end.

A denied write must fail *before* any file write or Git commit, leaving
the repository untouched (the pre-write hook runs inside the store lock).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.rate_limiter import RateLimitExceeded, RateLimiter
from hermes.memory.store import MemoryStore


@pytest.mark.integration
def test_session_limit_enforced(tmp_path: Path) -> None:
    """The 4th write of a budget-3 session is denied and persists nothing."""
    store = MemoryStore(
        root=tmp_path / "memory-repo",
        rate_limiter=RateLimiter(max_per_session_per_hour=3),
    )
    first = store.add("fact 0", type="User", session_id="s1")
    second = store.add("fact 1", type="User", session_id="s1")
    # an update counts as a write: it fills the session's 3-write budget
    store.update(first.id, text="fact 0 revised", session_id="s1")

    with pytest.raises(RateLimitExceeded, match="session-hour-limit"):
        store.add("denied fact", type="User", session_id="s1")

    # denial left no file, no index entry and no commit behind
    assert {entry.id for entry in store.list()} == {first.id, second.id}
    assert "denied fact" not in (tmp_path / "memory-repo" / "MEMORY.md").read_text(
        encoding="utf-8"
    )
    assert store.git.is_clean()

    # updates are denied too once the budget is gone, changing nothing
    with pytest.raises(RateLimitExceeded, match="session-hour-limit"):
        store.update(second.id, text="never applied", session_id="s1")
    got = store.get(second.id)
    assert got is not None and got.text == "fact 1"

    # another session keeps its own budget (isolation), same store+limiter
    other = store.add("other session fact", type="Project", session_id="s2")
    assert other.id in {entry.id for entry in store.list()}
