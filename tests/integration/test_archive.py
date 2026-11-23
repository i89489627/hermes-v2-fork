"""P8.T6 acceptance: the full archive/recover workflow (plan Phase 8).

Drives the real store end-to-end around the plan's acceptance criteria:

- stale entries (trust < 0.2 AND >1 year unverified — the P8.T6 double rule)
  archive to ``memory-repo/archive/`` with an ``archived_at`` stamp and one
  ``[memory] archive`` commit each;
- archiving does not affect default search results beyond removing the
  archived entry itself; ``search(include_archived=True)`` is the explicit
  opt-in that ranks archived entries again;
- ``recover`` moves the entry back to its original context, where the
  default search surface finds it again (worktree clean throughout).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.memory.decay import archive_stale, recover
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse, serialize


def _rewrite_card(store: MemoryStore, entry_id: str, **changes: object) -> None:
    """Edit an entry's trust-card frontmatter in place (test helper).

    The store API has no "backdate"/"set score without an event" operation by
    design; aging entries for archive tests therefore rewrites the card file
    directly with the public :func:`parse`/:func:`serialize` pair.
    """
    path = store.root / "global" / "facts" / f"{entry_id}.md"
    card, body = parse(path.read_text(encoding="utf-8"))
    for field, value in changes.items():
        setattr(card, field, value)
    path.write_text(serialize(card, body), encoding="utf-8")


@pytest.mark.integration
def test_archive_and_recover_workflow(tmp_path: Path) -> None:
    """Age an entry out, archive it, verify search exclusion, recover it back."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    now = datetime.now(UTC)
    keeper = store.add("Kubernetes deployment flow uses blue-green", type="Project")
    stale = store.add("Legacy heroku deploy pipeline details", type="Project")
    bystander = store.add("Team offsite notes from Lisbon", type="User")
    # A fourth live entry keeps the corpus big enough for BM25's IDF to be
    # positive (with N=2 live docs, a term present in half the corpus scores
    # exactly 0.0 and BM25Okapi-based search returns nothing).
    filler = store.add("Daily standup notes and sprint board links", type="User")
    _rewrite_card(store, stale.id, last_verified_at=now - timedelta(days=400), trust_score=0.1)

    # before archiving: the stale entry is part of the default search surface
    assert [hit.id for hit in store.search("heroku deploy pipeline")] == [stale.id]

    # archive: the double rule (trust < 0.2 AND > 365 days) picks only it
    assert archive_stale(store, now=now) == 1
    archive_file = store.root / "archive" / f"{stale.id}.md"
    assert archive_file.is_file()
    assert store.get(stale.id) is None
    assert [entry.id for entry in store.archived_entries()] == [stale.id]
    archived_text = archive_file.read_text(encoding="utf-8")
    assert "archived_at:" in archived_text  # plan P8.T6 archive metadata
    assert parse(archived_text)[0].id == stale.id

    # archiving does not disturb the default search results beyond removing
    # the archived entry itself; other entries stay findable
    live_hits = [hit.id for hit in store.search("heroku deploy pipeline")]
    assert stale.id not in live_hits
    assert [hit.id for hit in store.search("kubernetes deployment blue-green")] == [keeper.id]

    # explicit inclusion: archived entries rank again when asked for
    included = [hit.id for hit in store.search("heroku deploy pipeline", include_archived=True)]
    assert stale.id in included
    assert keeper.id not in included

    # recover: back to the original context, back in the default search
    restored = recover(stale.id, store)
    assert restored.id == stale.id
    assert restored.context == "global"
    assert restored.text == "Legacy heroku deploy pipeline details"
    assert restored.trust_score == pytest.approx(0.1)
    assert [hit.id for hit in store.search("heroku deploy pipeline")] == [stale.id]
    assert store.archived_entries() == []
    assert not archive_file.exists()

    # untouched bystander never left; every move was committed; tree is clean
    assert store.get(keeper.id) is not None
    assert store.get(bystander.id) is not None
    assert store.get(filler.id) is not None
    assert store.git.is_clean()
