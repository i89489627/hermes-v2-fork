"""Fork-base to v2.0 memory migration (plan P1.T6, P5.T4, P15.T2).

One-time conversion of the fork-base memory directory (``~/.hermes/memories``)
into the v2.0 git-backed trust-card memory repository (plan Section 2.3). The
fork-base format is plain markdown WITHOUT frontmatter:

- ``MEMORY.md`` — personal notes; one entry per non-empty line/bullet
  (entry type ``Feedback``);
- ``USER.md`` — user profile; one entry for the whole file (type ``User``);
- any other ``*.md`` — one entry per file (type ``Reference``).

For ``MEMORY.md``, list-bullet markers (``-``, ``*``, ``+``, ``1.``, ``1)``)
are structural and stripped; everything else on the line is preserved
verbatim. Whole-file entries (``USER.md``, others) keep the file content
byte-exact as the entry body.

Each migrated entry gets a trust card with the **default trust_score 0.5**
(the documented migration default for entries imported without verification
evidence — see :func:`hermes.memory.trust_card.compute_trust_score`) and a
single ``agent-inferred`` verification event whose ``value`` is 0.5, so the
recorded event history re-derives exactly the 0.5 score under the plan's
weighted-average rule (plan Section 2.4) instead of drifting to 1.0 on the
next recomputation. Entry ids are derived from the content slug; collisions
(duplicate bullets, or ids already present in the target repo) get a numeric
``-2``/``-3`` suffix.

Safety contract (implemented on the shared migration base
:class:`hermes.memory.migrations.base.Migration`, plan P18.T4):

- a pre-migration copy of the source directory (``<source>-premigration-backup``)
  is ALWAYS created first — even on a dry run, matching the P21.T11 config
  migration's "backup before any change" rule;
- dry run (``apply=False``) produces the full report with ZERO writes to the
  target repository (the null-writer guarantee: discovery is read-only and
  the mutating step is never called);
- the applied run writes every entry into ``<target>/global/facts/``, writes
  the ``MEMORY.md`` index in the v2 store layout, and ends in a single Git
  commit; if the commit raises, the written files are removed before the
  error propagates;
- :func:`rollback` undoes an applied migration: it removes the target
  repository and restores the source directory byte-exact from the backup,
  so the fork-base format is back exactly as it was (a partial/failed run
  that consumed or damaged the source is repaired from the backup).

The source directory itself is left untouched by a successful migration —
it remains on disk as the historical fork-base data.

Plan P5.T4 makes the target structure explicit: every migrated entry lands
in the ``global/`` context (``global/facts/`` with ``context=global`` trust
cards and a ``MEMORY.md`` index that names the global context);
:func:`migrate_to_global` is the explicit entry point for that contract and
:func:`verify_global_context` re-checks it after any run.

Plan P15.T2 adds :func:`migrate_soul`: it splits the fork-base ``SOUL.md``
into the SoulManager layout — ``soul/soul_core.md`` (safety, alignment,
identity and NEVER/ALWAYS guardrails) plus ``soul/soul_style.md`` (voice,
tone, formatting) — using purely rule-based, first-match-wins block
classification (no LLM), so the split is deterministic and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from hermes.memory.git_backend import GitBackend
from hermes.memory.migrations.base import (
    MIGRATION_BACKUP_SUFFIX,
    Migration,
    MigrationPlan,
    MigrationReport,
    atomic_write,
    rollback_migration,
)
from hermes.memory.soul import SOUL_CORE_FILENAME, SOUL_DIRNAME, SOUL_STYLE_FILENAME
from hermes.memory.store import INDEX_NAME
from hermes.memory.trust_card import TrustCard, compute_trust_score, parse, serialize
from hermes.memory.types import EntryType, VerificationEvent

__all__ = [
    "MIGRATION_BACKUP_SUFFIX",
    "MigrationReport",
    "SoulMigrationResult",
    "migrate",
    "migrate_soul",
    "migrate_to_global",
    "rollback",
    "verify_global_context",
]

#: Fork-base user profile file: one whole-file entry of type ``User``.
USER_MD_NAME = "USER.md"

#: List-marker prefix of a fork-base MEMORY.md bullet/numbered item. The
#: marker is list structure, not content, so it is stripped from the entry
#: body (the rest of the line is preserved verbatim).
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")

#: Fallback slug when a unit's text produces no slug words.
_FALLBACK_SLUG = "entry"


@dataclass(frozen=True)
class _SourceUnit:
    """One candidate entry discovered in the fork-base source."""

    origin: str  # source file name, for the verification event's evidence
    body: str  # entry text (marker-stripped, otherwise verbatim)
    entry_type: EntryType


def _parse_units(path: Path) -> list[_SourceUnit]:
    """Parse one fork-base ``*.md`` file into candidate entries.

    ``MEMORY.md`` yields one unit per non-empty line (bullet markers
    stripped, otherwise verbatim, type ``Feedback``); ``USER.md`` yields one
    whole-file unit of type ``User``; any other file one whole-file unit of
    type ``Reference``. Read errors propagate as ``OSError``.
    """
    content = path.read_text(encoding="utf-8")
    if path.name == INDEX_NAME:
        units: list[_SourceUnit] = []
        for line in content.splitlines():
            if not line.strip():
                continue  # blank lines are layout, not entries
            body = _BULLET_RE.sub("", line, count=1).strip()
            units.append(_SourceUnit(origin=path.name, body=body, entry_type="Feedback"))
        return units
    entry_type: EntryType = "User" if path.name == USER_MD_NAME else "Reference"
    return [_SourceUnit(origin=path.name, body=content, entry_type=entry_type)]


def _slugify(text: str) -> str:
    """Content slug for a migrated entry id (same limits as MemoryStore ids)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:8])[:48].rstrip("-")


def _next_entry_id(body: str, used: set[str]) -> str:
    """``<content-slug>`` id, suffixed ``-2``/``-3``/... on collision."""
    base = _slugify(body) or _FALLBACK_SLUG
    if base not in used:
        return base
    counter = 2
    while f"{base}-{counter}" in used:
        counter += 1
    return f"{base}-{counter}"


def _existing_entry_ids(target_repo: Path) -> set[str]:
    """Ids already present in the target repo (v2 layout), possibly empty."""
    used: set[str] = set()
    global_facts = target_repo / "global" / "facts"
    if global_facts.is_dir():
        used.update(path.stem for path in global_facts.glob("*.md"))
    contexts_root = target_repo / "contexts"
    if contexts_root.is_dir():
        for context_dir in contexts_root.iterdir():
            facts_dir = context_dir / "facts"
            if facts_dir.is_dir():
                used.update(path.stem for path in facts_dir.glob("*.md"))
    return used


def _index_content(entry_ids: list[str]) -> str:
    """Render the v2 ``MEMORY.md`` index, naming the global context (P5.T4)."""
    lines = ["# Memory Index", ""]
    if not entry_ids:
        lines.append("(no entries)")
    else:
        lines.append("## global (context=global)")
        lines.append("")
        lines.extend(f"- {entry_id}" for entry_id in sorted(entry_ids))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _apply_units(units: list[_SourceUnit], target_repo: Path) -> int:
    """Write ``units`` into the target repo and commit; returns the count.

    One entry per unit at ``<target>/global/facts/<id>.md`` — trust card with
    default trust_score 0.5 (single agent-inferred event re-deriving 0.5) and
    the body preserved verbatim — plus the regenerated ``MEMORY.md`` index,
    all in one Git commit. Files written before a failed commit are removed
    before the error propagates; :func:`rollback` removes any remainder.
    """
    git = GitBackend.init(target_repo)
    used = _existing_entry_ids(target_repo)
    now = datetime.now(UTC)
    facts_dir = target_repo / "global" / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    entry_ids: list[str] = []
    written: list[Path] = []
    try:
        for unit in units:
            entry_id = _next_entry_id(unit.body, used)
            used.add(entry_id)
            event = VerificationEvent(
                event_id=f"ev-{uuid4().hex[:12]}",
                timestamp=now,
                source="agent-inferred",
                verifier="fork-to-v2-0",
                evidence=(
                    f"migrated from fork-base {unit.origin}; no explicit "
                    "verification evidence -> default trust_score 0.5"
                ),
                value=0.5,
            )
            card = TrustCard(
                id=entry_id,
                type=unit.entry_type,
                created_at=now,
                last_verified_at=now,
                trust_score=compute_trust_score([event]),
                verification_events=[event],
                context="global",
            )
            path = facts_dir / f"{entry_id}.md"
            atomic_write(path, serialize(card, unit.body))
            written.append(path)
            entry_ids.append(entry_id)
        index_path = target_repo / INDEX_NAME
        atomic_write(index_path, _index_content(entry_ids))
        written.append(index_path)
        git.commit(
            f"[memory] migrate fork-base to v2.0 ({len(units)} entries)", written
        )
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return len(units)


class _ForkMemoryMigration(Migration):
    """The P1.T6 fork-base conversion, on the P18.T4 safety base.

    Discovery is the read-only source walk (candidate units, found/skipped/
    errors); :meth:`write_units` is the only mutating step (entries + index
    + single Git commit). Backup, null-writer dry run and rollback come from
    :class:`hermes.memory.migrations.base.Migration`.
    """

    def _require_source(self) -> None:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"fork-base memory directory does not exist: {self.source_dir}"
            )

    def discover(self) -> MigrationPlan:
        """Walk the fork-base ``*.md`` source files (read-only)."""
        units: list[_SourceUnit] = []
        found = 0
        skipped = 0
        errors: list[str] = []
        for md_path in sorted(self.source_dir.glob("*.md")):
            if not md_path.is_file():
                continue
            try:
                file_units = _parse_units(md_path)
            except OSError as exc:
                skipped += 1
                errors.append(f"{md_path.name}: unreadable, skipped ({exc})")
                continue
            found += len(file_units)
            for unit in file_units:
                if unit.body:
                    units.append(unit)
                else:
                    skipped += 1  # bullet/line with no content after stripping
        return MigrationPlan(units=units, found=found, skipped=skipped, errors=errors)

    def write_units(self, units: list[Any]) -> int:
        """Write the discovered units into the target repo (one Git commit)."""
        return _apply_units(units, self.target_repo)


def migrate(source_dir: Path, target_repo: Path, *, apply: bool = False) -> MigrationReport:
    """Migrate the fork-base memory directory to a v2.0 memory repository.

    Reads every top-level ``*.md`` file in ``source_dir``, converts it into
    trust-card entries (see module docstring for the per-file mapping) and
    writes them into ``target_repo`` as the explicit ``global`` context of
    plan P5.T4: every entry under ``global/facts/`` with a ``context=global``
    trust card, plus a ``MEMORY.md`` index whose global section names
    ``context=global``. A pre-migration backup of the source is ALWAYS
    created first — including on a dry run — and kept. With ``apply=False``
    the target repository is never touched (full report, zero writes); with
    ``apply=True`` the entries land in one Git commit. :func:`rollback` undoes
    an applied run from the backup reported in
    :attr:`MigrationReport.backup_path`; :func:`migrate_to_global` is the
    explicitly named P5.T4 wrapper and :func:`verify_global_context` re-checks
    the global-context contract afterwards.

    Raises :class:`FileNotFoundError` when ``source_dir`` does not exist.
    """
    return _ForkMemoryMigration(Path(source_dir), Path(target_repo)).run(apply=apply)


def rollback(target_repo: Path, backup_path: Path) -> None:
    """Undo an applied migration; restores the fork-base format exactly.

    ``backup_path`` is the ``<source>-premigration-backup`` directory that
    :func:`migrate` created (the source path is derived from its name). The
    target repository — the v2 repo the migration produced — is removed, and
    the source directory is replaced byte-exact with the backup contents, so
    a run that failed halfway (or damaged/consumed the fork-base files) ends
    in the pre-migration state. Raises :class:`FileNotFoundError` when the
    backup does not exist and :class:`ValueError` when the path is not a
    pre-migration backup.
    """
    rollback_migration(target_repo, backup_path)


# ── P15.T2: fork-base SOUL.md split (safety/alignment -> core, rest -> style) ─

#: Ordered, first-match-wins classification rules for SOUL.md blocks (plan
#: P15.T2: rule-based split, no LLM). Rule names carry their destination —
#: ``core:`` blocks are immutable safety/alignment material, ``style:`` blocks
#: are the adaptive communication layer. Keywords are matched whole-word and
#: case-insensitively against the block text (heading + body); the FIRST rule
#: whose any keyword matches wins, and a block matching no rule defaults to
#: the style layer. ``core:hard-rules`` deliberately precedes ``style:style``
#: so an unconditional NEVER/ALWAYS guardrail is protected even when its
#: wording also mentions style vocabulary.
SOUL_CLASSIFICATION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "core:safety",
        (
            "safety",
            "safe",
            "safely",
            "harm",
            "harms",
            "harmful",
            "refuse",
            "refusal",
            "danger",
            "dangerous",
            "injury",
            "injuries",
            "weapon",
            "weapons",
            "self-harm",
            "suicide",
            "violence",
            "violent",
            "abusive",
            "illegal",
        ),
    ),
    (
        "core:alignment",
        (
            "alignment",
            "aligned",
            "honest",
            "honesty",
            "truth",
            "truthful",
            "deceive",
            "deceiving",
            "deceptive",
            "fabricate",
            "fabricated",
            "fabrication",
            "ethic",
            "ethics",
            "ethical",
            "manipulate",
            "manipulating",
            "manipulation",
            "integrity",
            "mislead",
        ),
    ),
    (
        "core:identity-protected",
        (
            "identity",
            "identities",
            "impersonate",
            "impersonating",
            "impersonation",
            "claim to be",
            "pretend to be",
            "who you are",
            "system prompt",
        ),
    ),
    (
        "core:hard-rules",
        (
            "never",
            "always",
            "forbidden",
            "prohibited",
            "mandatory",
            "invariant",
            "invariants",
            "guardrail",
            "guardrails",
            "hard rule",
            "hard rules",
            "at all costs",
            "under no circumstances",
        ),
    ),
    (
        "style:style",
        (
            "tone",
            "voice",
            "style",
            "language",
            "format",
            "formatting",
            "humor",
            "humour",
            "emoji",
            "emojis",
            "concise",
            "friendly",
            "casual",
            "formal",
            "communication",
            "word choice",
            "punctuation",
            "sentence",
            "greeting",
            "respond",
            "responses",
            "phrasing",
            "vocabulary",
            "capitalization",
            "jargon",
        ),
    ),
)

#: Rule recorded for blocks that match no keyword (plan P15.T2: default = style).
SOUL_DEFAULT_RULE = "style:default"

#: Keyword matcher: whole-word, case-insensitive, punctuation-safe boundaries.
_SOUL_KEYWORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _soul_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compiled whole-word pattern for one classification keyword."""
    pattern = _SOUL_KEYWORD_RE_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
        _SOUL_KEYWORD_RE_CACHE[keyword] = pattern
    return pattern


def _split_soul_sections(content: str) -> list[tuple[str, str]]:
    """Split SOUL.md into ``(title, block_text)`` pairs, verbatim.

    Section boundaries are markdown headings (``#`` lines, fork-base memory
    format) or ``§``-prefixed lines; text before the first heading forms the
    ``(preamble)`` block. Each block keeps its heading line and body
    byte-exactly, so the split never rewrites content.
    """
    blocks: list[tuple[str, list[str]]] = []
    title = ""
    lines: list[str] = []
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        is_heading = stripped.startswith("#") or stripped.startswith("§")
        if is_heading and lines:
            blocks.append((title, lines))
            lines = []
        if is_heading:
            heading_text = stripped.lstrip("#§").strip()
            title = heading_text or stripped.strip()
        lines.append(line)
    if lines:
        blocks.append((title, lines))
    return [(block_title, "".join(block_lines)) for block_title, block_lines in blocks]


def _classify_soul_block(block_text: str) -> str:
    """Classify one SOUL.md block: first matching rule wins, default style."""
    for rule, keywords in SOUL_CLASSIFICATION_RULES:
        if any(_soul_keyword_pattern(kw).search(block_text) for kw in keywords):
            return rule
    return SOUL_DEFAULT_RULE


def _split_soul(content: str) -> tuple[str, str, list[str]]:
    """Rule-based split of a fork-base SOUL.md (plan P15.T2).

    Returns ``(core_content, style_content, rules_applied)``: core blocks
    (safety/alignment/identity/guardrails) and style blocks (voice, tone,
    formatting) joined section-wise, plus one ``"<rule> <- <title>"`` record
    per classified block (preamble blocks are titled ``(preamble)``).
    """
    core_blocks: list[str] = []
    style_blocks: list[str] = []
    rules_applied: list[str] = []
    for block_title, block_text in _split_soul_sections(content):
        if not block_text.strip():
            continue  # whitespace-only block: layout, not content
        rule = _classify_soul_block(block_text)
        rules_applied.append(f"{rule} <- {block_title or '(preamble)'}")
        target_blocks = core_blocks if rule.startswith("core:") else style_blocks
        target_blocks.append(block_text.strip("\n"))
    core = "\n\n".join(core_blocks)
    style = "\n\n".join(style_blocks)
    return (
        f"{core}\n" if core else "",
        f"{style}\n" if style else "",
        rules_applied,
    )


@dataclass
class SoulMigrationResult:
    """Outcome of one :func:`migrate_soul` run (plan P15.T2)."""

    #: Path of the written (or, on a dry run, would-be) ``soul_core.md``.
    core_path: Path
    #: Path of the written (or, on a dry run, would-be) ``soul_style.md``.
    style_path: Path
    #: Character count of the classified core (immutable safety) content.
    core_chars: int
    #: Character count of the classified style (adaptive) content.
    style_chars: int
    #: One ``"<rule> <- <section title>"`` record per classified block, in
    #: document order — the audit trail of the rule-based split.
    rules_applied: list[str] = field(default_factory=list)


def migrate_soul(
    source_soul_path: Path, target_repo: Path, *, apply: bool = True
) -> SoulMigrationResult:
    """Split the fork-base ``SOUL.md`` into the SoulManager layout (P15.T2).

    Parses ``source_soul_path`` section-wise (markdown headings or ``§``
    blocks), classifies every block with the ordered keyword rules of
    :data:`SOUL_CLASSIFICATION_RULES` (first match wins, no rule matched =
    style), and writes the split into ``<target_repo>/soul/`` as
    ``soul_core.md`` (safety/alignment/identity/guardrails) and
    ``soul_style.md`` (voice/tone/formatting) — the exact layout and
    filenames :class:`hermes.memory.soul.SoulManager` reads. When
    ``target_repo`` is Git-backed (``.git`` present), both files are
    committed via :class:`hermes.memory.git_backend.GitBackend`; non-Git
    roots get the files only.

    With ``apply=False`` nothing is written anywhere: the returned result
    reports the classification and character counts that an applied run
    would produce.

    Raises :class:`FileNotFoundError` when the source SOUL.md is missing and
    :class:`FileExistsError` on an applied run when ``soul_core.md`` already
    exists — the core is immutable (plan P15.T1), so a migration never
    overwrites it (and never touches the existing style either, since the
    core guard fires first).
    """
    source = Path(source_soul_path)
    if not source.is_file():
        raise FileNotFoundError(f"fork-base SOUL.md does not exist: {source}")
    target_repo = Path(target_repo)
    core_content, style_content, rules_applied = _split_soul(
        source.read_text(encoding="utf-8")
    )
    soul_dir = target_repo / SOUL_DIRNAME
    result = SoulMigrationResult(
        core_path=soul_dir / SOUL_CORE_FILENAME,
        style_path=soul_dir / SOUL_STYLE_FILENAME,
        core_chars=len(core_content),
        style_chars=len(style_content),
        rules_applied=rules_applied,
    )
    if not apply:
        return result
    if result.core_path.exists():
        raise FileExistsError(
            f"refusing to overwrite the immutable {SOUL_CORE_FILENAME} at "
            f"{result.core_path} (plan P15.T1: the core never changes)"
        )
    written: list[Path] = []
    try:
        atomic_write(result.core_path, core_content)
        written.append(result.core_path)
        atomic_write(result.style_path, style_content)
        written.append(result.style_path)
        if (target_repo / ".git").exists():
            GitBackend(target_repo).commit(
                f"[P15.T2] migrate fork-base SOUL.md "
                f"({result.core_chars} core chars, {result.style_chars} style chars)",
                written,
            )
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return result


# ── P5.T4: migration to the explicit global context ──────────────────────────


def migrate_to_global(
    source_dir: Path, target_repo: Path, *, apply: bool = True
) -> MigrationReport:
    """Migrate ALL fork-base memory into the global context (plan P5.T4).

    Explicitly named entry point for the P5.T4 contract — every piece of
    fork-base (v2026.9.24) memory becomes the ``global`` context of the v2
    repository, with zero data loss: entries land under ``global/facts/``,
    their trust cards carry ``context=global``, and the regenerated
    ``MEMORY.md`` index names the global context. Delegates to
    :func:`migrate` (which has produced exactly this structure since P1.T6);
    the dry-run/backup/rollback contract and the report are identical.
    """
    return migrate(source_dir, target_repo, apply=apply)


def verify_global_context(target_repo: Path) -> list[str]:
    """Post-migration check of the P5.T4 global-context contract.

    Returns the list of problems found in ``target_repo`` (an empty list
    means the repository satisfies the contract):

    - the ``global/facts/`` directory is missing (no global context);
    - an entry is stored OUTSIDE the global context (any file under
      ``contexts/<name>/facts/``);
    - an entry's trust-card frontmatter does not say ``context: global``;
    - an entry's trust card is unreadable (cannot be checked).

    Purely read-only — safe to run against a live repository.
    """
    target = Path(target_repo)
    problems: list[str] = []
    global_facts = target / "global" / "facts"
    if not global_facts.is_dir():
        problems.append(
            f"{global_facts}: missing — migration did not create the global context"
        )
    contexts_root = target / "contexts"
    if contexts_root.is_dir():
        for context_dir in sorted(contexts_root.iterdir()):
            facts_dir = context_dir / "facts"
            if not facts_dir.is_dir():
                continue
            for path in sorted(facts_dir.glob("*.md")):
                problems.append(
                    f"{path.relative_to(target).as_posix()}: entry outside the "
                    "global context"
                )
    if global_facts.is_dir():
        for path in sorted(global_facts.glob("*.md")):
            rel = path.relative_to(target).as_posix()
            try:
                card, _body = parse(path.read_text(encoding="utf-8"))
            except Exception as exc:  # a verifier reports problems, never raises
                problems.append(f"{rel}: unreadable trust card ({exc})")
                continue
            if card.context != "global":
                problems.append(
                    f"{rel}: frontmatter context is {card.context!r}, "
                    "expected 'global'"
                )
    return problems
