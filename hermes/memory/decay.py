"""Forgetting curve: trust-score decay + stale-entry archiving (plan P2.T2, P8.T6).

Implements the plan Phase 2 forgetting curve — ``score * exp(-decay_rate *
days_since_verification)`` with ``decay_rate = 0.01`` — over the git-backed
memory store (plan Phase 2: *Trust score decay over time (forgetting curve)*).

Public surface
--------------
- :func:`compute_decay` — the pure formula: current score, when it was last
  verified and "now" -> decayed score, clamped to [0, 1].
- :func:`apply_decay` — one pass over every live entry, persisting each
  actually-lowered score via :meth:`~hermes.memory.store.MemoryStore.update`
  (one Git commit per changed entry); returns the number of decayed entries.
  The pass is idempotent within a day: entries verified "today" (or whose
  decay delta is below :data:`MIN_DECAY_DELTA`) are left untouched, so
  running it from a daily cron (plan P2.T4 wires the scheduler; this module
  only exposes :func:`due_for_decay` as the cadence helper) never produces
  churn commits.
- :func:`due_for_decay` — daily-cadence check for the cron wiring.
- :func:`archive_stale` — entries that are stale under the plan P8.T6 double
  rule — ``trust_score < threshold`` (default 0.2, plan Phase 2/P8) **AND**
  unverified for more than ``max_unverified_days`` (default 365 — the plan's
  ">1 year unverified") — are *archived*: their fact file moves to
  ``<memory-repo>/archive/<id>.md`` in one Git commit (``[memory] archive
  <id>``), stamped with an ``archived_at`` frontmatter timestamp. Archived
  entries leave the normal ``list()``/``get()``/``search()`` surface
  automatically, because those scan only the facts directories;
  ``MemoryStore.search(include_archived=True)`` opts back in (plan P8.T6).
- :func:`recover` — the inverse: an archived file moves back to its original
  context's facts directory (the trust card frontmatter carries ``context``)
  in one commit (``[memory] recover <id>``).

Persistence semantics (deliberate trade-off)
--------------------------------------------
Decayed scores persist through
:meth:`~hermes.memory.store.MemoryStore.update` with ``trust_score=<new>``,
which is the store's documented "explicit trust change" path: it appends a
``tool-output`` event recording the new score and moves ``last_verified_at``
forward. That makes consecutive daily passes mathematically equivalent to
the continuous curve (each pass multiplies by ``exp(-0.01 * ~1 day)``) but
means the synthetic ``tool-output`` events stay in the history and slightly
pull later re-verification averages down. This is accepted per the P2.T2
design ("persist via ``store.update``"); a bespoke no-event persistence path
would require store changes, and ``store.py`` is frozen for this wave.

Note on "zero after 1 year": ``exp`` never reaches 0, so the plan's
"decays to zero" acceptance is interpreted as *effectively zero* — a 0.5
score after 365 days at rate 0.01 is ``0.5 * exp(-3.65) ≈ 0.013 < 0.05``.

The archive file-move stages the old path's removal through the same cached
``pygit2`` index the store commits with (same requirement as
``MemoryStore._stage_removal`` — libgit2 never reloads a repository handle's
index from disk), and regenerates the ``MEMORY.md`` index so the archived id
drops out of it in the same commit.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import ARCHIVE_DIRNAME, INDEX_NAME, MemoryStore
from hermes.memory.trust_card import parse
from hermes.memory.types import clamp01, to_utc

__all__ = [
    "ARCHIVE_DIRNAME",
    "DECAY_RATE",
    "DEFAULT_ARCHIVE_THRESHOLD",
    "DEFAULT_MAX_UNVERIFIED_DAYS",
    "MIN_DECAY_DELTA",
    "apply_decay",
    "archive_stale",
    "compute_decay",
    "due_for_decay",
    "recover",
]

_LOGGER = logging.getLogger(__name__)

#: Decay rate of the forgetting curve (plan Phase 2: ``decay_rate=0.01``).
DECAY_RATE: float = 0.01

#: Default archive threshold (plan Phase 2/P8: trust < 0.2 is stale).
DEFAULT_ARCHIVE_THRESHOLD: float = 0.2

#: Second P8.T6 archive condition: an entry must be unverified for MORE than
#: this many days (">1 year unverified") before the archiver considers it.
DEFAULT_MAX_UNVERIFIED_DAYS: float = 365.0

#: Frontmatter key the archiver stamps onto archived files (plan P8.T6).
_ARCHIVED_AT_KEY: str = "archived_at"

#: Smallest score drop :func:`apply_decay` will persist. Below this the
#: entry counts as "verified recently enough" and the pass leaves it alone.
#: The skip is CORRECTNESS, not churn control: the persistence path
#: (:meth:`MemoryStore.update` with ``trust_score=``) re-anchors
#: ``last_verified_at`` to now, so persisting a negligible decay would RESET
#: the forgetting curve instead of advancing it. A ~1e-4 score-point floor
#: (roughly the first half hour of the curve for a 0.5 score) keeps the
#: daily-cron approximation of the continuous formula within 0.02%.
MIN_DECAY_DELTA: float = 1e-4

#: Seconds per day — the decay curve is parameterized in whole days.
_SECONDS_PER_DAY: float = 86400.0

#: Cadence of the decay job (plan Phase 2 acceptance: "runs daily").
_DAILY_CADENCE: timedelta = timedelta(hours=24)


def _utcnow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def compute_decay(
    current_score: float,
    last_verified_at: datetime,
    now: datetime,
) -> float:
    """Forgetting-curve formula (plan Phase 2): ``s * exp(-rate * days)``.

    ``days`` is the elapsed time between ``last_verified_at`` and ``now``
    (fractional days allowed; a ``last_verified_at`` in the future — clock
    skew — counts as zero days). The result is clamped to [0, 1]; naive
    datetimes are rejected like everywhere else in the memory layer.
    """
    verified = to_utc(last_verified_at, "last_verified_at")
    current = to_utc(now, "now")
    days = max(0.0, (current - verified).total_seconds() / _SECONDS_PER_DAY)
    score = float(current_score) * math.exp(-DECAY_RATE * days)
    return clamp01(score)


def due_for_decay(last_run: datetime | None, now: datetime | None = None) -> bool:
    """Daily-cadence helper for the P2.T4 cron wiring.

    True when a decay pass should run: never run before (``last_run`` is
    None) or at least :data:`_DAILY_CADENCE` (24h) elapsed since ``last_run``.
    A ``last_run`` in the future relative to ``now`` is treated as "not due"
    (no catch-up storms from clock skew).
    """
    if last_run is None:
        return True
    current = to_utc(now if now is not None else _utcnow(), "now")
    previous = to_utc(last_run, "last_run")
    return (current - previous) >= _DAILY_CADENCE


def apply_decay(store: MemoryStore, now: datetime | None = None) -> int:
    """Decay every live entry's trust score; returns the number decayed.

    Iterates :meth:`~hermes.memory.store.MemoryStore.list` (all contexts) and
    recomputes each entry's score from its ``last_verified_at``. Only entries
    whose score actually drops by at least :data:`MIN_DECAY_DELTA` are
    persisted (:meth:`~hermes.memory.store.MemoryStore.update` with
    ``trust_score=`` — one commit each); recently verified entries (whose
    decay is below the floor) are skipped — see :data:`MIN_DECAY_DELTA` for
    why the skip is required, not optional. Entries without any
    ``last_verified_at`` are skipped (no anchor to measure days against).
    """
    current = to_utc(now if now is not None else _utcnow(), "now")
    decayed = 0
    for entry in store.list():
        if entry.last_verified_at is None:
            continue
        new_score = compute_decay(entry.trust_score, entry.last_verified_at, current)
        if entry.trust_score - new_score < MIN_DECAY_DELTA:
            continue
        _LOGGER.debug(
            "decay %s: %.4f -> %.4f", entry.id, entry.trust_score, new_score
        )
        store.update(entry.id, trust_score=new_score)
        decayed += 1
    return decayed


# ── archiving (plan Phase 2 acceptance: trust < 0.2 -> memory-repo/archive) ──


def _rel(path: Path, root: Path) -> str:
    """POSIX path of ``path`` relative to the store ``root``."""
    return path.relative_to(root).as_posix()


def _stage_removal(store: MemoryStore, path: Path) -> None:
    """Stage a file deletion in the store repository's Git index.

    Mirrors ``MemoryStore._stage_removal``: the staging must go through the
    SAME ``pygit2.Repository`` handle :meth:`GitBackend.commit` uses, because
    libgit2 keeps one in-memory index per handle and never reloads it from
    disk. A removal applied through a throwaway handle would be invisible to
    the next commit.
    """
    rel = _rel(path, store.root)
    index = store.git._repo.index
    if rel in index:
        index.remove(rel)
        index.write()


def _unstage_removal(store: MemoryStore, rel: str) -> None:
    """Rollback helper: put ``rel`` back into the repository index."""
    index = store.git._repo.index
    index.add(rel)
    index.write()


def _write_store_index(store: MemoryStore) -> None:
    """Regenerate ``MEMORY.md`` after a file move outside the store CRUD API.

    Uses the store's own renderer (``_write_index``) so the index format
    cannot drift from the store's; ``store.py`` is frozen this wave, which
    makes the private helper a stable seam.
    """
    store._write_index()


def _read_prior_index(store: MemoryStore) -> str | None:
    """Pre-mutation ``MEMORY.md`` content (None when the file doesn't exist)."""
    index_path = store.root / INDEX_NAME
    if not index_path.is_file():
        return None
    return index_path.read_text(encoding="utf-8")


def _restore_index(store: MemoryStore, prior: str | None) -> None:
    """Restore ``MEMORY.md`` to its pre-mutation content (None: delete)."""
    index_path = store.root / INDEX_NAME
    if prior is None:
        index_path.unlink(missing_ok=True)
    else:
        index_path.write_text(prior, encoding="utf-8")


def _iso_z(value: datetime) -> str:
    """ISO 8601 UTC with a ``Z`` suffix — the trust-card frontmatter format."""
    return to_utc(value, "archived_at").isoformat().replace("+00:00", "Z")


def _stamp_archived_at(markdown: str, archived_at: datetime) -> str:
    """Inject ``archived_at: <ISO-Z>`` into a trust card's frontmatter block.

    :class:`~hermes.memory.trust_card.TrustCard` is a fixed-field dataclass
    (plan P1.T1) and stays untouched this wave, so the archive timestamp is
    written *additively into the YAML block by hand*: ``parse`` ignores
    unknown frontmatter keys, so a stamped file keeps parsing everywhere the
    store parses cards, and the stamp simply disappears again on the next
    full card rewrite (``store.update``/``record_event``) — harmless once
    the entry is recovered. Re-stamping an already-stamped file replaces the
    value instead of piling up duplicate keys. Content without a frontmatter
    block is returned unchanged (such files fail ``parse`` downstream anyway).
    """
    lines = markdown.split("\n")
    if not lines or lines[0].strip() != "---":
        return markdown
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        return markdown
    stamp = f"{_ARCHIVED_AT_KEY}: {_iso_z(archived_at)}"
    frontmatter = [
        line
        for line in lines[1:closing]
        if not line.strip().startswith(f"{_ARCHIVED_AT_KEY}:")
    ]
    return "\n".join([lines[0], stamp, *frontmatter, *lines[closing:]])


def _archive_one(store: MemoryStore, entry: MemoryEntry, archived_at: datetime) -> Path:
    """Move ``entry``'s fact file to ``archive/`` with one Git commit.

    Returns the archive path. The move stages the old path's removal,
    stamps the archived file with ``archived_at`` (plan P8.T6), regenerates
    ``MEMORY.md`` and commits ``[memory] archive <id>``; on a commit failure
    the file moves back and the index/index-file rollback happens before the
    error propagates (same transactionality as the store).
    """
    source = store._find_path(entry.id)
    if source is None:
        raise KeyError(f"unknown memory entry id: {entry.id!r}")
    archive_dir = store.root / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{entry.id}.md"
    old_rel = _rel(source, store.root)
    prior_index = _read_prior_index(store)
    source.replace(target)  # atomic within one filesystem
    try:
        target.write_text(
            _stamp_archived_at(target.read_text(encoding="utf-8"), archived_at),
            encoding="utf-8",
        )
        _stage_removal(store, source)
        _write_store_index(store)
        store.git.commit(f"[memory] archive {entry.id}", [target, store.root / INDEX_NAME])
    except Exception:
        target.replace(source)
        _unstage_removal(store, old_rel)
        _restore_index(store, prior_index)
        raise
    store._invalidate_search_cache()
    return target


def archive_stale(
    store: MemoryStore,
    threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
    now: datetime | None = None,
    max_unverified_days: float = DEFAULT_MAX_UNVERIFIED_DAYS,
) -> int:
    """Archive every live entry that is stale under the P8.T6 double rule.

    An entry is stale when BOTH conditions hold (plan P8.T6 acceptance:
    "stale entries (>1 year unverified, trust <0.2) archived"):

    * ``trust_score < threshold`` (default 0.2), and
    * it has been unverified for more than ``max_unverified_days`` (default
      365 — the plan's ">1 year"): ``now - anchor`` exceeds the age, where
      the anchor is ``last_verified_at``, falling back to ``created_at`` for
      cards that were never verified. The comparison is strict, so an entry
      verified exactly 365 days ago is not yet stale; ``now`` defaults to
      the current time (tests pass a fixed clock).

    Each archived entry's file moves to ``<root>/archive/<id>.md`` (one
    ``[memory] archive <id>`` commit per entry) carrying an ``archived_at``
    frontmatter stamp, and disappears from the normal
    ``list()``/``get()``/``search()`` surface — those scan only the facts
    directories; :meth:`MemoryStore.search` opts back in with
    ``include_archived=True`` and :meth:`MemoryStore.archived_entries` lists
    the archive contents (plan P8.T6). ``recover`` restores an entry to its
    original context. Entries at or above the threshold, and low-trust
    entries verified within the window, are untouched. Returns the number of
    archived entries. The whole pass runs under the store's exclusive write
    lock.
    """
    current = to_utc(now if now is not None else _utcnow(), "now")
    max_age = timedelta(days=max_unverified_days)
    archived = 0
    with store._lock:
        for entry in store.list():
            if entry.trust_score >= threshold:
                continue
            anchor = entry.last_verified_at or entry.created_at
            if anchor is None or current - anchor <= max_age:
                continue
            _LOGGER.debug("archiving stale entry %s (trust %.3f)", entry.id, entry.trust_score)
            _archive_one(store, entry, current)
            archived += 1
    return archived


def recover(entry_id: str, store: MemoryStore) -> MemoryEntry:
    """Restore an archived entry to its original context (plan P2.T2).

    Moves ``<root>/archive/<entry_id>.md`` back to the facts directory of the
    context recorded in the trust card (``global`` or
    ``contexts/<name>/facts/``) with one ``[memory] recover <id>`` commit.
    Unknown archive ids raise :class:`KeyError`; a live entry with the same
    id raises :class:`ValueError` (ids are unique across the store). Returns
    the restored entry as seen by :meth:`~hermes.memory.store.MemoryStore.get`.
    """
    archive_file = store.root / ARCHIVE_DIRNAME / f"{entry_id}.md"
    if not archive_file.is_file():
        raise KeyError(f"no archived memory entry id: {entry_id!r}")
    if store._find_path(entry_id) is not None:
        raise ValueError(f"memory entry {entry_id!r} already exists outside the archive")
    card, _body = parse(archive_file.read_text(encoding="utf-8"))
    with store._lock:
        destination = store._facts_path(entry_id, card.context)
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_rel = _rel(archive_file, store.root)
        prior_index = _read_prior_index(store)
        archive_file.replace(destination)  # atomic within one filesystem
        try:
            _stage_removal(store, archive_file)
            _write_store_index(store)
            store.git.commit(
                f"[memory] recover {entry_id}", [destination, store.root / INDEX_NAME]
            )
        except Exception:
            destination.replace(archive_file)
            _unstage_removal(store, archive_rel)
            _restore_index(store, prior_index)
            raise
    # The file move bypassed the store CRUD API, so the BM25 cache it manages
    # must be dropped explicitly — the recovered entry is searchable again
    # immediately (plan P8.T6), not one external mutation later.
    store._invalidate_search_cache()
    restored = store.get(entry_id)
    if restored is None:  # pragma: no cover - the file was just moved back
        raise RuntimeError(f"recover of {entry_id!r} succeeded but the entry is not listable")
    _LOGGER.debug("recovered %s to %s", entry_id, destination)
    return restored
