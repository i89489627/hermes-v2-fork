"""P21.T4 benchmark: cold skills-search latency budget (#156140).

The original bug: 86 seconds on a cold index cache (52MB download). The fix
ships the bundled index, so a fully cold `SkillsSearch` — fresh instances,
no cache file, no warmup — must answer 200 queries in under 5 seconds total
(plan P21.T4 acceptance: cold cache search <5s).
"""

from __future__ import annotations

import time

import pytest

from hermes.skills.index import SkillsIndex
from hermes.skills.search import SkillsSearch

#: Query mix exercising names and descriptions across all index categories.
QUERIES: list[str] = [
    "commit message convention",
    "pull request review correctness",
    "release notes draft",
    "changelog highlights",
    "sql plan index",
    "csv deduplicate normalize",
    "pdf text extract",
    "regex build test",
    "cron expression validate",
    "environment variables audit",
    "secret scan push",
    "dependency cve check",
    "error log cluster root cause",
    "docker compose diagnose",
    "kubernetes rollout status",
    "terraform plan destructive",
    "git bisect automate",
    "pytest scaffold generate",
    "synthetic fixtures edge cases",
    "eval set labeled assemble",
    "memory merge duplicates",
    "trust card provenance audit",
    "prompt template lint",
    "mock api openapi",
    "screenshot regression highlight",
    "browser session replay",
    "email reply draft",
    "meeting transcript decisions",
    "slack channel digest",
    "timezone meeting convert",
]


@pytest.mark.benchmark
def test_cold_search_under_5s() -> None:
    """Cold cache + fresh search: 200 queries (30-record index) <5s total."""
    index = SkillsIndex()  # cold: no cache file under the per-test HERMES_HOME
    assert not index.index_path.exists()
    search = SkillsSearch(index)  # fresh, no warmup

    queries = QUERIES * 7  # 210 queries
    started = time.perf_counter()
    hits = 0
    for query in queries:
        hits += len(search.search(query, top_k=10))
    elapsed = time.perf_counter() - started

    assert hits > 0
    assert elapsed < 5.0, (
        f"cold skills search took {elapsed:.3f}s for {len(queries)} queries "
        f"(budget 5s; original bug: 86s cold cache)"
    )
