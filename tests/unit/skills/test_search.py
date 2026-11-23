"""P21.T4 acceptance tests: skills search cold cache + local fallback (#156140).

Covers the spec behaviours: a cold cache (no local index file) searches the
bundled index instantly — 200 fresh queries complete in under 5 seconds
total — and when the remote fetcher fails, the explicit
``SkillsIndexFetchError`` surfaces and the fallback serves local skills-dir
results with a warning flag (no silent degradation).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes.skills.index import (
    BUNDLED_INDEX_RECORD_COUNT,
    SkillsIndex,
    SkillsIndexFetchError,
)
from hermes.skills.search import SkillsSearch


def _raising_fetcher() -> dict[str, object]:
    raise RuntimeError("connection refused by skills marketplace")


@pytest.mark.unit
def test_cold_cache_under_5s_with_bundled_index() -> None:
    """Cold cache serves the bundled index: 200 fresh queries in <5s total."""
    index = SkillsIndex()  # default path; the per-test HERMES_HOME has no cache file
    assert not index.index_path.exists()  # genuinely cold

    records = index.load()
    assert len(records) == BUNDLED_INDEX_RECORD_COUNT
    assert all(
        set(record) == {"name", "description", "category", "version"} for record in records
    )

    search = SkillsSearch(index)  # fresh instance, no warmup
    queries = [
        "commit message",
        "review pull request",
        "sql query plan",
        "secrets scan credentials",
        "cron schedule timezone",
        "docker compose failure",
        "kubernetes rollout",
        "pytest scaffold fixtures",
        "memory duplicate merge",
        "meeting notes digest",
    ] * 20  # 200 queries
    assert len(queries) == 200

    started = time.perf_counter()
    total_hits = 0
    for query in queries:
        total_hits += len(search.search(query, top_k=10))
    elapsed = time.perf_counter() - started

    assert total_hits > 0  # the bundled index actually matches
    assert elapsed < 5.0, f"cold-cache search took {elapsed:.2f}s (>5s budget)"

    # Spot-check relevance and shape on one query.
    top = search.search("secret scan credentials", top_k=3)
    assert top, "expected hits for the secret-scan query"
    assert top[0].name == "secret-scan"
    assert top[0].score >= top[-1].score
    assert all(hit.category and hit.description for hit in top)


@pytest.mark.unit
def test_remote_unavailable_uses_local_fallback(tmp_path: Path) -> None:
    """Fetcher failure: explicit error raised, local fallback served with warning."""
    local_dir = tmp_path / "installed-skills"
    (local_dir / "deploy-helper").mkdir(parents=True)
    (local_dir / "deploy-helper" / "SKILL.md").write_text(
        "---\n"
        "name: deploy-helper\n"
        "description: Walk through the deploy checklist and rollback steps\n"
        "version: 1.2.0\n"
        "---\n",
        encoding="utf-8",
    )
    (local_dir / "hotfix-notes.md").write_text(
        "---\nname: hotfix-notes\ndescription: Draft a hotfix announcement note\n---\n",
        encoding="utf-8",
    )

    index = SkillsIndex()
    search = SkillsSearch(index, local_dir=local_dir)

    # 1) The failure is EXPLICIT: refresh raises, the cause is chained.
    with pytest.raises(SkillsIndexFetchError, match="connection refused"):
        index.refresh(_raising_fetcher)
    with pytest.raises(SkillsIndexFetchError, match="no remote skills index source"):
        index.fetch_remote(None)

    # 2) The fallback path serves local results WITH a warning flag.
    outcome = search.search_with_fallback("deploy checklist", fetcher=_raising_fetcher)
    assert outcome.source == "local_fallback"
    assert outcome.warning is not None
    assert "connection refused" in outcome.warning  # the original error is surfaced
    assert [hit.name for hit in outcome.hits] == ["deploy-helper"]
    assert outcome.hits[0].category == "local"

    # A query matching only the standalone file also resolves via fallback.
    notes = search.search_with_fallback("hotfix announcement", fetcher=_raising_fetcher)
    assert notes.source == "local_fallback"
    assert [hit.name for hit in notes.hits] == ["hotfix-notes"]

    # The failed refresh left no cache file and the bundled index intact.
    assert not index.index_path.exists()
    bundled = search.search_with_fallback("commit message")  # no fetcher: bundled path
    assert bundled.source == "index"
    assert bundled.warning is None
    assert bundled.hits, "bundled index still answers after a failed refresh"
