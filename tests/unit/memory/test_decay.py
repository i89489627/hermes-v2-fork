"""P2.T2 acceptance: the forgetting curve (plan Phase 2).

Covers :mod:`hermes.memory.decay`: the exponential formula
``score * exp(-0.01 * days_since_verification)`` (:func:`compute_decay`),
the full-store pass (:func:`apply_decay`, persisted through
``MemoryStore.update``), the "zero after 1 year" acceptance (interpreted as
*effectively zero*: ``0.5 * exp(-3.65) ≈ 0.013 < 0.05`` — exp never reaches a
true 0), the same-day no-op, and the archive/recover lifecycle for stale
(trust < 0.2) entries with their ``[memory] archive``/``[memory] recover``
Git commits.

The ``*_p8`` tests (plan P8.T6) extend the archive lifecycle: the double
staleness rule (trust < 0.2 AND > 365 days unverified), the ``archived_at``
frontmatter stamp on archived files, and search exclusion with explicit
``include_archived`` opt-in around the recover.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes.memory.decay import (
    DECAY_RATE,
    apply_decay,
    archive_stale,
    compute_decay,
    recover,
)
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse, serialize


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory-repo")


def _rewrite_card(store: MemoryStore, entry_id: str, **changes: object) -> None:
    """Edit an entry's trust-card frontmatter in place (test helper).

    The store API has no "backdate"/"set score without an event" operation by
    design; aging entries for decay tests therefore rewrites the card file
    directly with the public :func:`parse`/:func:`serialize` pair.
    """
    path = store.root / "global" / "facts" / f"{entry_id}.md"
    card, body = parse(path.read_text(encoding="utf-8"))
    for field, value in changes.items():
        setattr(card, field, value)
    path.write_text(serialize(card, body), encoding="utf-8")


@pytest.mark.unit
def test_decay_reduces_score_over_time(tmp_path: Path) -> None:
    """A 30-day-old verification decays the score by exp(-0.01*30) and persists."""
    store = _make_store(tmp_path)
    entry = store.add("Gateway restarts wipe the session cache", type="Project")
    original = entry.trust_score  # 1.0: a single confirming event averages to 1.0

    now = datetime.now(UTC)
    verified_at = now - timedelta(days=30)
    _rewrite_card(store, entry.id, last_verified_at=verified_at)

    decayed = apply_decay(store, now=now)
    assert decayed == 1

    after = store.get(entry.id)
    assert after is not None
    expected = original * math.exp(-DECAY_RATE * 30)  # 1.0 * exp(-0.3) ≈ 0.741
    assert after.trust_score == pytest.approx(expected)
    assert after.trust_score < original


@pytest.mark.unit
def test_decay_zero_after_1_year() -> None:
    """365 days at rate 0.01 leaves 0.5 * exp(-3.65) ≈ 0.013 — effectively zero.

    exp never reaches a true 0, so the plan's "zero after 1 year" acceptance
    is interpreted as the documented effectively-zero threshold (< 0.05).
    """
    now = datetime.now(UTC)
    year_ago = now - timedelta(days=365)
    score = compute_decay(0.5, year_ago, now)
    assert score == pytest.approx(0.5 * math.exp(-0.01 * 365), rel=1e-9)
    assert score < 0.05
    assert 0.0 <= score <= 1.0

    # Still-clamped and monotone for very long horizons.
    assert compute_decay(1.0, year_ago, now) == pytest.approx(math.exp(-3.65), rel=1e-9)


@pytest.mark.unit
def test_recently_verified_unchanged(tmp_path: Path) -> None:
    """A same-day verification is left untouched by a decay pass."""
    store = _make_store(tmp_path)
    entry = store.add("Brand-new fact from this session", type="User")
    before = store.get(entry.id)
    assert before is not None
    original = before.trust_score

    decayed = apply_decay(store)  # real "now": the entry is seconds old
    assert decayed == 0

    after = store.get(entry.id)
    assert after is not None
    assert after.trust_score >= 0.99 * original
    # The pass must not re-anchor the curve: last_verified_at is untouched.
    assert after.last_verified_at == before.last_verified_at
    assert len(after.verification_events) == len(before.verification_events)


@pytest.mark.unit
def test_archive_stale_moves_files(tmp_path: Path) -> None:
    """Entries below 0.2 move to memory-repo/archive/ and leave the live surface."""
    store = _make_store(tmp_path)
    fresh = store.add("Still-trusted fact about the deployment flow", type="Project")
    stale = store.add("Old claim nobody verified this year", type="User")
    stale_at = datetime.now(UTC) - timedelta(days=400)
    _rewrite_card(store, stale.id, last_verified_at=stale_at, trust_score=0.1)

    archived_count = archive_stale(store)
    assert archived_count == 1

    archive_file = store.root / "archive" / f"{stale.id}.md"
    assert archive_file.is_file()
    assert not (store.root / "global" / "facts" / f"{stale.id}.md").exists()

    # Out of the live CRUD/search surface, untouched entry still present.
    assert store.get(stale.id) is None
    assert [live.id for live in store.list()] == [fresh.id]
    still_there = store.get(fresh.id)
    assert still_there is not None
    assert still_there.trust_score >= 0.2

    # One archive commit touching the moved file; workdir stays clean.
    history = store.git.log(path=f"archive/{stale.id}.md")
    assert [commit.message for commit in history] == [f"[memory] archive {stale.id}"]
    assert store.git.is_clean()


@pytest.mark.unit
def test_archived_recoverable(tmp_path: Path) -> None:
    """recover() moves an archived entry back to its context with its history."""
    store = _make_store(tmp_path)
    entry = store.add("Archivable fact with preserved text", type="Reference")
    stale_at = datetime.now(UTC) - timedelta(days=400)
    _rewrite_card(store, entry.id, last_verified_at=stale_at, trust_score=0.05)

    assert archive_stale(store) == 1
    assert store.get(entry.id) is None

    restored = recover(entry.id, store)
    assert restored.id == entry.id
    assert restored.text == "Archivable fact with preserved text"
    assert restored.context == "global"

    live_file = store.root / "global" / "facts" / f"{entry.id}.md"
    assert live_file.is_file()
    assert not (store.root / "archive" / f"{entry.id}.md").exists()
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.trust_score == pytest.approx(0.05)

    # Both moves are committed; the archive commit no longer lists the file.
    recover_history = store.git.log(path=f"global/facts/{entry.id}.md")
    assert any(commit.message == f"[memory] recover {entry.id}" for commit in recover_history)
    assert store.git.is_clean()


# ── plan P8.T6: double staleness rule + archived_at stamp + search surface ──


def _rewrite_card_committed(store: MemoryStore, entry_id: str, **changes: object) -> None:
    """_rewrite_card + one Git commit, keeping the worktree clean.

    The P8.T6 test rewrites cards of entries that STAY live (the rule's
    negatives) and asserts ``git.is_clean`` afterwards — so unlike
    :func:`_rewrite_card` (whose uncommitted rewrite rides along inside the
    archived file's own commit), these backdates are committed here.
    """
    _rewrite_card(store, entry_id, **changes)
    path = store.root / "global" / "facts" / f"{entry_id}.md"
    store.git.commit(f"[test] age {entry_id}", [path])


@pytest.mark.unit
def test_archive_stale_moves_files_p8(tmp_path: Path) -> None:
    """P8.T6 double rule: trust < 0.2 AND > 365 days unverified archives.

    Only entries meeting BOTH conditions move to archive/: a low-trust but
    recently verified entry stays live, an old-but-trusted entry stays live,
    and exactly-365-days is not ">1 year" (strict comparison). The archived
    file gains an ``archived_at`` ISO-Z frontmatter stamp while still parsing
    as a trust card, and leaves the default search surface.
    """
    store = _make_store(tmp_path)
    now = datetime.now(UTC)

    stale = store.add("Stale claim from last year", type="Reference")
    _rewrite_card_committed(
        store, stale.id, last_verified_at=now - timedelta(days=400), trust_score=0.1
    )
    low_trust_fresh = store.add("Recently verified low-trust claim", type="User")
    _rewrite_card_committed(
        store, low_trust_fresh.id, last_verified_at=now - timedelta(days=30), trust_score=0.1
    )
    old_but_trusted = store.add("Old but still trusted deployment flow", type="Project")
    _rewrite_card_committed(
        store, old_but_trusted.id, last_verified_at=now - timedelta(days=400), trust_score=0.9
    )
    boundary = store.add("Low-trust entry verified exactly 365 days ago", type="User")
    _rewrite_card_committed(
        store, boundary.id, last_verified_at=now - timedelta(days=365), trust_score=0.1
    )

    assert archive_stale(store, now=now) == 1

    archive_file = store.root / "archive" / f"{stale.id}.md"
    assert archive_file.is_file()
    assert not (store.root / "global" / "facts" / f"{stale.id}.md").exists()
    assert store.get(stale.id) is None
    for live_id in (low_trust_fresh.id, old_but_trusted.id, boundary.id):
        assert store.get(live_id) is not None, live_id

    # archive metadata: ISO-Z archived_at stamp in the frontmatter, and the
    # file still parses as a trust card (parse tolerates the extra key).
    archived_text = archive_file.read_text(encoding="utf-8")
    stamp = re.search(r"^archived_at: (\S+)$", archived_text, re.MULTILINE)
    assert stamp is not None
    assert stamp.group(1).endswith("Z")
    card, body = parse(archived_text)
    assert card.id == stale.id
    assert card.trust_score == pytest.approx(0.1)
    assert body == "Stale claim from last year"

    # default search never returns archived entries; include_archived opts in
    assert stale.id not in [hit.id for hit in store.search("stale claim")]
    assert [hit.id for hit in store.search("stale claim", include_archived=True)] == [stale.id]

    # one archive commit touching the moved file; workdir stays clean
    history = store.git.log(path=f"archive/{stale.id}.md")
    assert [commit.message for commit in history] == [f"[memory] archive {stale.id}"]
    assert store.git.is_clean()


@pytest.mark.unit
def test_archived_recoverable_p8(tmp_path: Path) -> None:
    """A P8-archived entry recovers to its context and re-enters search.

    After the double-rule archive, ``recover`` moves the file back to the
    context recorded in its trust card; the entry is live, listable and
    searchable again without any ``include_archived`` opt-in.
    """
    store = _make_store(tmp_path)
    now = datetime.now(UTC)
    entry = store.add("Recoverable archived fact", type="Reference")
    _rewrite_card(store, entry.id, last_verified_at=now - timedelta(days=400), trust_score=0.1)

    assert archive_stale(store, now=now) == 1
    assert store.get(entry.id) is None
    assert store.search("recoverable archived fact") == []

    restored = recover(entry.id, store)
    assert restored.id == entry.id
    assert restored.text == "Recoverable archived fact"
    assert restored.context == "global"
    assert restored.trust_score == pytest.approx(0.1)

    live_file = store.root / "global" / "facts" / f"{entry.id}.md"
    assert live_file.is_file()
    assert not (store.root / "archive" / f"{entry.id}.md").exists()
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.trust_score == pytest.approx(0.1)
    # back on the default search surface, not just the include_archived one
    assert [hit.id for hit in store.search("recoverable archived fact")] == [entry.id]

    # both moves are committed; workdir stays clean
    recover_history = store.git.log(path=f"global/facts/{entry.id}.md")
    assert any(commit.message == f"[memory] recover {entry.id}" for commit in recover_history)
    assert store.git.is_clean()
