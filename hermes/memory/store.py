"""MemoryStore: git-backed CRUD over trust-card markdown files (plan P1.T3).

One memory entry is one markdown file — trust-card YAML frontmatter (see
:mod:`hermes.memory.trust_card`) plus the entry text as the body — laid out
as::

    <root>/global/facts/<id>.md           the "global" context
    <root>/contexts/<name>/facts/<id>.md  named contexts (mutually isolated)
    <root>/MEMORY.md                      human-readable index, regenerated on
                                          every mutation and part of the same
                                          commit as the mutation itself

Every mutation ends in exactly one Git commit (via
:class:`hermes.memory.git_backend.GitBackend`); when the commit raises, the
file-level change is rolled back before the error propagates, so the store is
never left half-written — add/update/remove are all transactional. Entry ids
are ``<date>-<slug>``: the UTC date plus a text slug and an 8-hex suffix for
uniqueness.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hermes.config import hermes_home
from hermes.memory.entry import MemoryEntry
from hermes.memory.git_backend import GitBackend
from hermes.memory.locking import DEFAULT_LOCK_TIMEOUT, LOCK_NAME, FileLock
from hermes.memory.pii_detector import PIIDetector, PIIResult
from hermes.memory.rate_limiter import RateLimiter, RateLimitExceeded
from hermes.memory.search import BM25Search
from hermes.memory.trust_card import (
    TrustCard,
    add_verification_event,
    compute_trust_score,
    parse,
    serialize,
)
from hermes.memory.types import EntryType, VerificationEvent, VerificationSource, clamp01

#: Name of the index file listing entry ids per context (store root).
INDEX_NAME = "MEMORY.md"

#: Directory holding the named contexts, under the store root.
CONTEXTS_DIR = "contexts"

#: Directory (under the store root) holding archived fact files (plan P8.T6).
#: The canonical constant lives here — next to the search surface that must
#: skip it — and :mod:`hermes.memory.decay` (the archiver) imports it, so
#: writer and reader can never drift apart.
ARCHIVE_DIRNAME = "archive"

#: Context names must be a single safe path component (no traversal).
_CONTEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _utcnow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _validate_context(context: str) -> None:
    """Reject context names that are not a single safe path component."""
    if not _CONTEXT_RE.fullmatch(context):
        raise ValueError(f"context must be a safe single path component, got {context!r}")


def _pii_tags(result: PIIResult) -> list[str]:
    """``pii:<type>`` trust-card tags for the detected PII types (plan P17.T4)."""
    return [f"pii:{pii_type}" for pii_type in result.types]


def _entry_from_card(card: TrustCard, body: str) -> MemoryEntry:
    """Materialize a :class:`MemoryEntry` from a parsed trust card + body."""
    return MemoryEntry(
        id=card.id,
        type=card.type,
        text=body,
        created_at=card.created_at,
        last_verified_at=card.last_verified_at,
        trust_score=card.trust_score,
        verification_events=list(card.verification_events),
        context=card.context,
        scope=list(card.scope),
        ttl_days=card.ttl_days,
        tags=list(card.tags),
        pii_checked=card.pii_checked,
        attachments=list(card.attachments),
    )


class MemoryStore:
    """CRUD facade over the git-backed memory repository (plan P1.T3).

    ``root`` defaults to ``hermes_home() / "memory-repo"``; ``git`` defaults
    to :meth:`GitBackend.init` on that root, so the repository is created on
    first use. ``get``/``list`` scan the facts directories directly — fine at
    the scale the plan targets (Phase 1).
    """

    def __init__(
        self,
        root: Path | None = None,
        git: GitBackend | None = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
        rate_limiter: RateLimiter | None = None,
        pii_detector: PIIDetector | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> None:
        if git is not None and root is None:
            root = git.repo_path
        self.root = Path(root) if root is not None else hermes_home() / "memory-repo"
        self.git = git if git is not None else GitBackend.init(self.root)
        #: Lazily-created BM25 engine (plan P1.T4); created on first search().
        self._search_engine: BM25Search | None = None
        #: Exclusive write lock (plan P1.T5); held across every mutation.
        self._lock = FileLock(self.root / LOCK_NAME, timeout=lock_timeout)
        #: Optional write-rate limiter (plan P17.T2); None = unlimited.
        self.rate_limiter = rate_limiter
        #: Optional PII scanner (plan P17.T4); None = no scanning.
        self.pii_detector = pii_detector
        #: Post-update hook (plan P7.T2): the memory-tool wiring calls
        #: ``store.on_update = hot_reloader.invalidate`` so every committed
        #: update invalidates exactly that entry's prompt segment. Public and
        #: assignable after construction; ``None`` (the default) preserves
        #: the pre-P7.T2 behavior exactly.
        self.on_update = on_update
        self._ensure_lock_ignored()

    # ── layout helpers ──────────────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self.root / INDEX_NAME

    def _facts_path(self, entry_id: str, context: str) -> Path:
        """On-disk path of ``entry_id`` within ``context``."""
        if context == "global":
            return self.root / "global" / "facts" / f"{entry_id}.md"
        return self.root / CONTEXTS_DIR / context / "facts" / f"{entry_id}.md"

    def _scan(self) -> list[tuple[str, Path]]:
        """All fact files as ``(context, path)``; global first, contexts sorted."""
        found: list[tuple[str, Path]] = []
        global_facts = self.root / "global" / "facts"
        if global_facts.is_dir():
            found.extend(("global", path) for path in sorted(global_facts.glob("*.md")))
        contexts_root = self.root / CONTEXTS_DIR
        if contexts_root.is_dir():
            for context_dir in sorted(path for path in contexts_root.iterdir() if path.is_dir()):
                facts_dir = context_dir / "facts"
                if facts_dir.is_dir():
                    found.extend(
                        (context_dir.name, path) for path in sorted(facts_dir.glob("*.md"))
                    )
        return found

    def _load(self, path: Path) -> MemoryEntry:
        """Parse one fact file into a :class:`MemoryEntry`."""
        card, body = parse(path.read_text(encoding="utf-8"))
        return _entry_from_card(card, body)

    def _find_path(self, entry_id: str) -> Path | None:
        """Locate the file for ``entry_id`` (global searched first), or None."""
        for _context, path in self._scan():
            if path.stem == entry_id:
                return path
        return None

    def _new_entry_id(self, text: str, now: datetime) -> str:
        """``<date>-<slug>`` id: UTC date + text slug + 8-hex uniqueness suffix."""
        date_part = now.strftime("%Y-%m-%d")
        words = re.findall(r"[a-z0-9]+", text.lower())
        slug = "-".join(words[:8])[:48].rstrip("-")
        while True:
            suffix = uuid4().hex[:8]
            entry_id = f"{date_part}-{slug}-{suffix}" if slug else f"{date_part}-{suffix}"
            if self._find_path(entry_id) is None:
                return entry_id

    # ── index bookkeeping ───────────────────────────────────────────────────

    def _index_content(self) -> str:
        """Render :data:`INDEX_NAME`: entry ids grouped per context."""
        grouped: dict[str, list[str]] = {}
        for context, path in self._scan():
            grouped.setdefault(context, []).append(path.stem)
        lines = ["# Memory Index", ""]
        if not grouped:
            lines.append("(no entries)")
        else:
            ordered = (["global"] if "global" in grouped else []) + sorted(
                name for name in grouped if name != "global"
            )
            for context in ordered:
                lines.append(f"## {context}")
                lines.append("")
                lines.extend(f"- {entry_id}" for entry_id in sorted(grouped[context]))
                lines.append("")
        return "\n".join(lines).rstrip("\n") + "\n"

    def _write_index(self) -> None:
        _atomic_write(self._index_path, self._index_content())

    def _restore_index(self, prior: str | None) -> None:
        """Restore the index file to its pre-mutation content (None: delete)."""
        if prior is None:
            self._index_path.unlink(missing_ok=True)
        else:
            _atomic_write(self._index_path, prior)

    def _invalidate_search_cache(self) -> None:
        """Drop the BM25 cache after a successful mutation (plan P1.T4)."""
        if self._search_engine is not None:
            self._search_engine.invalidate()

    def _ensure_lock_ignored(self) -> None:
        """Keep the persistent lock file invisible to Git status (plan P1.T5).

        ``<root>/.lock`` must persist while writers hold it open (Windows
        cannot unlink an open file), so it would otherwise show up as an
        untracked file and break ``GitBackend.is_clean``. It is excluded via
        ``.git/info/exclude`` — a repository-local rule that never touches
        the workdir. Skipped when the store root is not itself the Git
        repository (a detached backend owns its own layout).
        """
        git_dir = self.root / ".git"
        if not git_dir.is_dir():
            return
        exclude = git_dir / "info" / "exclude"
        current = ""
        if exclude.is_file():
            try:
                current = exclude.read_text(encoding="utf-8", errors="replace")
            except OSError:
                current = ""
        if re.search(r"^\.lock$", current, re.MULTILINE):
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        separator = "" if current.endswith("\n") or not current else "\n"
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write(f"{separator}.lock\n")

    # ── git staging helpers ─────────────────────────────────────────────────

    def _stage_removal(self, path: Path) -> None:
        """Stage a file deletion in the repository's Git index.

        :meth:`GitBackend.commit` can only stage files that still exist on
        disk (``Index.add`` stats the workdir), so a removal must be recorded
        with :meth:`pygit2.Index.remove` before committing. The staging MUST
        go through the same ``pygit2.Repository`` object GitBackend commits
        with (its ``_repo``): libgit2 keeps one in-memory index per repository
        handle and never reloads it from disk, so a removal applied through a
        throwaway handle would be invisible to the next commit — and would
        leave that handle's cached index stale for all later operations.
        """
        rel = path.relative_to(self.root).as_posix()
        index = self.git._repo.index  # same cached index object commit() stages from
        if rel in index:
            index.remove(rel)
            index.write()

    def _unstage_removal(self, rel: str) -> None:
        """Re-add ``rel`` to the repository's index (remove-rollback helper)."""
        index = self.git._repo.index  # same cached index object commit() stages from
        index.add(rel)
        index.write()

    # ── CRUD API (plan Phase 1 files table) ─────────────────────────────────

    def add(
        self,
        text: str,
        type: EntryType,
        context: str = "global",
        source: VerificationSource = "agent-inferred",
        verifier: str = "main-model",
        evidence: str = "",
        tags: list[str] | None = None,
        session_id: str = "default",
        user_id: str = "default",
    ) -> MemoryEntry:
        """Create an entry: atomic markdown write + one Git commit.

        Builds a single :class:`VerificationEvent` from ``source``/``verifier``
        /``evidence`` and derives ``trust_score`` from it via
        :func:`compute_trust_score`. If the commit raises, the created file is
        deleted (and the previous index restored) before the error propagates.
        The whole mutation runs under the exclusive store lock (plan P1.T5).
        When a rate limiter is wired (plan P17.T2), ``session_id``/``user_id``
        identify the caller to it and a denial raises
        :class:`RateLimitExceeded` before anything is written. When a PII
        detector is wired (plan P17.T4), the text passes through its
        pre-write action first (redact/block/warn); scanned entries are
        marked ``pii_checked=True`` and detected types are recorded as
        ``pii:<type>`` tags.
        """
        with self._lock:
            if self.rate_limiter is not None:
                decision = self.rate_limiter.allow(session_id, user_id)
                if not decision.allowed:
                    raise RateLimitExceeded(decision)
            _validate_context(context)
            pii_result: PIIResult | None = None
            if self.pii_detector is not None:
                text, pii_result = self.pii_detector.process(text)
            now = _utcnow()
            entry_id = self._new_entry_id(text, now)
            event = VerificationEvent(
                event_id=f"ev-{uuid4().hex[:12]}",
                timestamp=now,
                source=source,
                verifier=verifier,
                evidence=evidence,
            )
            card_tags = list(tags) if tags else []
            if pii_result is not None:
                card_tags.extend(tag for tag in _pii_tags(pii_result) if tag not in card_tags)
            card = TrustCard(
                id=entry_id,
                type=type,
                created_at=now,
                last_verified_at=now,
                trust_score=compute_trust_score([event]),
                verification_events=[event],
                context=context,
                tags=card_tags,
                pii_checked=pii_result is not None,
            )
            path = self._facts_path(entry_id, context)
            prior_index = (
                self._index_path.read_text(encoding="utf-8")
                if self._index_path.exists()
                else None
            )
            _atomic_write(path, serialize(card, text))
            try:
                self._write_index()
                self.git.commit(f"[memory] add {entry_id} {context}", [path, self._index_path])
            except Exception:
                path.unlink(missing_ok=True)
                self._restore_index(prior_index)
                raise
            self._invalidate_search_cache()
            return _entry_from_card(card, text)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with ``entry_id`` (global searched first), else None."""
        path = self._find_path(entry_id)
        return None if path is None else self._load(path)

    def update(
        self,
        entry_id: str,
        text: str | None = None,
        trust_score: float | None = None,
        session_id: str = "default",
        user_id: str = "default",
    ) -> MemoryEntry:
        """Update an entry's text and/or trust score; one Git commit.

        Passing ``trust_score`` counts as an explicit trust change: a
        ``tool-output`` verification event is appended (its ``value`` records
        the new score) and the card's ``trust_score`` is set to the clamped
        score. ``text=None`` keeps the current body; the file rewrite is
        atomic and rolled back when the commit fails. Unknown ids raise
        :class:`KeyError`. The whole mutation runs under the exclusive store
        lock (plan P1.T5). After a successful commit the optional
        ``on_update`` hook (plan P7.T2 — the hot-reload wiring) is called
        with the ``entry_id``, so the affected prompt segment is invalidated
        the moment the write lands (memory updates visible within 1 turn,
        prefix cache preserved for every other segment). The hook runs
        POST-commit: it cannot roll the update back, and an exception it
        raises propagates after the update is already durable. When a rate
        limiter is wired (plan P17.T2),
        updates count as writes and a denial raises
        :class:`RateLimitExceeded` before anything is written. When a PII
        detector is wired (plan P17.T4), a new ``text`` passes through its
        pre-write action first (redact/block/warn) and the entry is marked
        ``pii_checked=True`` with ``pii:<type>`` tags for anything found.
        """
        with self._lock:
            if self.rate_limiter is not None:
                decision = self.rate_limiter.allow(session_id, user_id)
                if not decision.allowed:
                    raise RateLimitExceeded(decision)
            path = self._find_path(entry_id)
            if path is None:
                raise KeyError(f"unknown memory entry id: {entry_id!r}")
            prior = path.read_text(encoding="utf-8")
            card, body = parse(prior)
            new_text = body if text is None else text
            if self.pii_detector is not None and text is not None:
                new_text, pii_result = self.pii_detector.process(new_text)
                card.pii_checked = True
                for tag in _pii_tags(pii_result):
                    if tag not in card.tags:
                        card.tags.append(tag)
            if trust_score is not None:
                score = clamp01(trust_score)
                card.verification_events.append(
                    VerificationEvent(
                        event_id=f"ev-{uuid4().hex[:12]}",
                        timestamp=_utcnow(),
                        source="tool-output",
                        verifier="memory-store",
                        evidence=f"trust_score explicitly set to {score}",
                        value=score,
                    )
                )
                card.trust_score = score
                card.last_verified_at = _utcnow()
            _atomic_write(path, serialize(card, new_text))
            try:
                self._write_index()
                self.git.commit(f"[memory] update {entry_id}", [path, self._index_path])
            except Exception:
                _atomic_write(path, prior)
                raise
            self._invalidate_search_cache()
            if self.on_update is not None:
                self.on_update(entry_id)
            return _entry_from_card(card, new_text)

    def record_event(self, entry_id: str, event: VerificationEvent) -> MemoryEntry:
        """Append ``event`` to the entry's trust card; one Git commit (plan P2.T1).

        The event joins ``verification_events``, ``trust_score`` is recomputed
        over the full history via :func:`compute_trust_score` and
        ``last_verified_at`` moves to the event's timestamp (all applied by
        :func:`hermes.memory.trust_card.add_verification_event`). The rewrite
        is atomic and rolled back when the commit fails; unknown ids raise
        :class:`KeyError`. This is the persistence primitive behind
        :func:`hermes.memory.verification.add_verification_event`. The whole
        mutation runs under the exclusive store lock (plan P1.T5).
        """
        with self._lock:
            path = self._find_path(entry_id)
            if path is None:
                raise KeyError(f"unknown memory entry id: {entry_id!r}")
            prior = path.read_text(encoding="utf-8")
            card, body = parse(prior)
            add_verification_event(card, event)
            _atomic_write(path, serialize(card, body))
            try:
                self._write_index()
                self.git.commit(f"[memory] record-event {entry_id}", [path, self._index_path])
            except Exception:
                _atomic_write(path, prior)
                raise
            self._invalidate_search_cache()
            return _entry_from_card(card, body)

    def remove(self, entry_id: str) -> bool:
        """Delete an entry; one Git commit records the removal.

        Returns True when the entry existed (and was removed + committed),
        False when the id is unknown (no commit). A failing commit restores
        the deleted file and index state before the error propagates. The
        whole mutation runs under the exclusive store lock (plan P1.T5).
        """
        with self._lock:
            path = self._find_path(entry_id)
            if path is None:
                return False
            prior = path.read_text(encoding="utf-8")
            prior_index = (
                self._index_path.read_text(encoding="utf-8")
                if self._index_path.exists()
                else None
            )
            rel = path.relative_to(self.root).as_posix()
            path.unlink()
            try:
                self._stage_removal(path)
                self._write_index()
                self.git.commit(f"[memory] remove {entry_id}", [self._index_path])
            except Exception:
                _atomic_write(path, prior)
                self._unstage_removal(rel)
                self._restore_index(prior_index)
                raise
            self._invalidate_search_cache()
            return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        context: str | None = None,
        include_archived: bool = False,
    ) -> list[MemoryEntry]:
        """BM25 keyword search over all entries (plan P1.T4).

        Delegates to :class:`hermes.memory.search.BM25Search` — corpus is
        each entry's text plus its tags; only entries sharing at least one
        query token are returned, best first, at most ``top_k``.
        ``context=None`` spans all contexts, ``context="global"`` only global
        entries, and ``context=<name>`` only that named context (never any
        bleeding). The engine's index is cached per context and invalidated
        on every mutation routed through this store instance, so results
        never lag behind this store's own writes. (Defined *before* the
        ``list`` method so ``list[MemoryEntry]`` annotations inside this
        class resolve to the builtin type, not the method.)

        Archived entries (plan P8.T6) are excluded by default: their files
        sit in :data:`ARCHIVE_DIRNAME`, outside the facts directories the
        live corpus scans, so archiving never changes default search
        results. ``include_archived=True`` opts back in — the live corpus is
        ranked together with the archive directory's entries (a one-off
        engine over the merged corpus; the cached live index is untouched).
        """
        if not include_archived:
            if self._search_engine is None:
                self._search_engine = BM25Search(store=self)
            return self._search_engine.search(query, top_k=top_k, context=context)
        entries = self.list()
        entries.extend(self.archived_entries())
        return BM25Search(entries=entries).search(query, top_k=top_k, context=context)

    def archived_entries(self) -> list[MemoryEntry]:
        """Entries currently in the archive directory (plan P8.T6).

        Archived fact files live outside the facts directories, so they
        never appear in ``get``/``list``/the default ``search``; this reader
        materializes them from their trust cards (``context`` comes from the
        card frontmatter, so context-filtered ``search`` includes/excludes
        them correctly). Returns ``[]`` when nothing was ever archived.
        """
        archive_dir = self.root / ARCHIVE_DIRNAME
        if not archive_dir.is_dir():
            return []
        return [self._load(path) for path in sorted(archive_dir.glob("*.md"))]

    def list(
        self,
        context: str | None = None,
        type: EntryType | None = None,
        tag: str | None = None,
    ) -> list[MemoryEntry]:
        """List entries, ordered by creation time then id.

        ``context=None`` spans ALL contexts; ``context="global"`` returns only
        global entries and ``context=<name>`` only that named context — named
        contexts never bleed into each other or into global. ``type`` and
        ``tag`` additionally filter (``tag`` must be a member of the entry's
        tags).
        """
        selected: list[MemoryEntry] = []
        for entry_context, path in self._scan():
            if context is not None and entry_context != context:
                continue
            entry = self._load(path)
            if type is not None and entry.type != type:
                continue
            if tag is not None and tag not in entry.tags:
                continue
            selected.append(entry)
        selected.sort(key=lambda entry: (entry.created_at, entry.id))
        return selected
