"""P21.T4 integration test: skills search over default paths (#156140).

Drives the DEFAULT persistence layout (``<hermes_home>/memory-repo/skills/``
via ``hermes.config.hermes_home``): a successful refresh persists the cache
and subsequent searches serve it (no silent degradation in the happy path);
a failing remote fetch raises the explicit ``SkillsIndexFetchError`` and the
fallback search serves local skills with a warning — never a quiet empty or
stale result set.
"""

from __future__ import annotations

import pytest

from hermes.config import hermes_home
from hermes.skills.index import SkillsIndex, SkillsIndexFetchError
from hermes.skills.search import SkillsSearch

REMOTE_PAYLOAD = {
    "skills": [
        {
            "name": "quantum-superposition",
            "description": "Demo skill that exists only in the refreshed remote index",
            "category": "demo",
            "version": "9.9.9",
        },
        {
            "name": "commit-message",
            "description": "Write conventional, scoped commit messages from staged diffs",
            "category": "coding",
            "version": "2.0.0",
        },
    ]
}


@pytest.mark.integration
def test_no_silent_degradation() -> None:
    """Happy path persists a cache; degraded path warns AND serves local results."""
    index = SkillsIndex()  # default path under the per-test HERMES_HOME
    assert index.index_path == hermes_home() / "memory-repo" / "skills" / "_skills_index.json"
    search = SkillsSearch(index)

    # Cold cache: bundled records answer immediately (no download, no error).
    cold = search.search_with_fallback("commit message")
    assert cold.source == "index"
    assert cold.warning is None
    assert "commit-message" in {hit.name for hit in cold.hits}
    assert not index.index_path.exists()  # still cold — bundled, not fetched

    # A locally installed skill package exists for the fallback path.
    local_skill = hermes_home() / "memory-repo" / "skills" / "rollback-runbook"
    local_skill.mkdir(parents=True)
    (local_skill / "SKILL.md").write_text(
        "---\n"
        "name: rollback-runbook\n"
        "description: Step-by-step rollback runbook for bad deploys\n"
        "---\n",
        encoding="utf-8",
    )

    # Successful refresh: the remote payload becomes the cached index and the
    # refreshed-only record is now searchable (cache wins, explicitly).
    refreshed = search.search_with_fallback("quantum", fetcher=lambda: REMOTE_PAYLOAD)
    assert refreshed.source == "index"
    assert refreshed.warning is None  # nothing degraded — no warning noise
    assert [hit.name for hit in refreshed.hits] == ["quantum-superposition"]
    assert index.index_path.exists()  # cache persisted by the refresh

    # Now the remote breaks: the failure must be EXPLICIT and the fallback
    # must be VISIBLE (warning flag), never a silent stale/bundled answer.
    def broken_fetcher() -> dict[str, object]:
        raise TimeoutError("marketplace unreachable after 30s")

    with pytest.raises(SkillsIndexFetchError, match="marketplace unreachable"):
        index.refresh(broken_fetcher)

    degraded = search.search_with_fallback("rollback runbook", fetcher=broken_fetcher)
    assert degraded.source == "local_fallback"
    assert degraded.warning is not None
    assert "marketplace unreachable" in degraded.warning
    assert [hit.name for hit in degraded.hits] == ["rollback-runbook"]
    assert all(hit.category == "local" for hit in degraded.hits)


@pytest.mark.integration
def test_error_message_on_fetch_failure() -> None:
    """Fetch failures carry actionable messages and keep the original cause."""
    index = SkillsIndex()

    # Injected fetcher raising: the wrapper message names the operation and
    # chains the original exception as __cause__.
    def failing_fetcher() -> dict[str, object]:
        raise ConnectionError("DNS lookup failed for skills.example.com")

    with pytest.raises(SkillsIndexFetchError) as excinfo:
        index.refresh(failing_fetcher)
    message = str(excinfo.value)
    assert "skills index fetch failed" in message
    assert "DNS lookup failed" in message  # original error text preserved
    assert isinstance(excinfo.value.__cause__, ConnectionError)

    # Malformed payloads are rejected explicitly too (not silently ignored).
    with pytest.raises(SkillsIndexFetchError, match="missing a non-empty"):
        index.fetch_remote(lambda: {"skills": [{"name": "broken"}]})

    # No fetcher configured: explicit error, actionable message.
    with pytest.raises(SkillsIndexFetchError) as noneinfo:
        index.fetch_remote(None)
    assert "no remote skills index source configured" in str(noneinfo.value)
