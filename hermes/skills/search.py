"""SkillsSearch: fast token-overlap search + explicit local fallback (plan P21.T4).

Fixes issue #156140 (86s cold search): searching the bundled/cached index is
pure in-memory token overlap over ``name`` + ``description`` — deterministic
and microseconds per query, so a cold cache searches in well under the 5s
budget (no 52MB download). When the index source fails (remote fetch error,
corrupt cache), :meth:`SkillsSearch.search_with_fallback` serves results
from the LOCAL skills directory listing (``<memory-repo>/skills/`` by
default, injectable via ``local_dir``) and attaches an explicit WARNING to
the result — degradation is never silent (plan P21.T4 acceptance).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from hermes.config import hermes_home
from hermes.skills.index import SkillsIndex, SkillsIndexError

__all__ = ["SkillHit", "SearchResult", "SkillsSearch"]

#: Local-fallback records carry this category (they come from the skills dir,
#: not the remote index taxonomy).
LOCAL_CATEGORY: Final[str] = "local"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SkillHit:
    """One search match (plan P21.T4): record identity plus relevance score."""

    name: str
    description: str
    category: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Outcome of :meth:`SkillsSearch.search_with_fallback`.

    ``source`` records which index served the hits. ``warning`` is ``None``
    when nothing degraded; a non-None warning is the EXPLICIT error text that
    triggered the local fallback (no silent degradation, plan P21.T4).
    """

    hits: list[SkillHit] = field(default_factory=list)
    source: Literal["index", "local_fallback"] = "index"
    warning: str | None = None


def _tokens(text: str) -> set[str]:
    """Word tokens of ``text`` (casefolded, punctuation stripped)."""
    return set(_TOKEN_RE.findall(text.casefold()))


def _score(query_tokens: set[str], name: str, description: str) -> float:
    """Token-overlap score in [0, 1]: name matches weigh double description.

    Deterministic and allocation-light; empty queries match nothing.
    """
    if not query_tokens:
        return 0.0
    name_overlap = len(query_tokens & _tokens(name))
    description_overlap = len(query_tokens & _tokens(description))
    return (2.0 * name_overlap + description_overlap) / (3.0 * len(query_tokens))


class SkillsSearch:
    """Search over the skills index with a local fallback (plan P21.T4).

    ``index`` defaults to a :class:`~hermes.skills.index.SkillsIndex` at the
    default path (bundled snapshot on cold cache); ``local_dir`` is the
    directory scanned by :meth:`fallback_search` (default
    ``<hermes_home>/memory-repo/skills/``).
    """

    def __init__(self, index: SkillsIndex | None = None, local_dir: Path | None = None) -> None:
        self.index = index if index is not None else SkillsIndex()
        self.local_dir = Path(local_dir) if local_dir is not None else (
            hermes_home() / "memory-repo" / "skills"
        )

    def search(self, query: str, top_k: int = 10) -> list[SkillHit]:
        """Token-overlap search over the index (bundled or cached, plan P21.T4).

        Best matches first (score desc, name asc for determinism), at most
        ``top_k``; zero-score records are never returned. A failing index
        SOURCE raises the index error (callers wanting degradation semantics
        use :meth:`search_with_fallback`).
        """
        records = self.index.load()
        return self._rank(query, records, top_k)

    def fallback_search(self, query: str, top_k: int = 10) -> list[SkillHit]:
        """Search the LOCAL skills directory listing (plan P21.T4 fallback path).

        Scans ``local_dir`` for skill packages (``<dir>/SKILL.md`` with YAML
        frontmatter ``name``/``description``) and standalone ``*.md`` files;
        missing or empty directories yield an empty list (never an error — the
        fallback itself is the degradation, already surfaced as a warning by
        :meth:`search_with_fallback`).
        """
        return self._rank(query, self._local_records(), top_k)

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        fetcher: Callable[[], dict[str, Any]] | None = None,
    ) -> SearchResult:
        """Search with explicit no-silent-degradation semantics (plan P21.T4).

        With a ``fetcher`` (a zero-arg callable returning the remote payload
        dict), the index is refreshed FIRST: a fetch failure raises through
        :meth:`~hermes.skills.index.SkillsIndex.refresh` — and here it is
        caught to serve LOCAL results with the full error text as the
        result's ``warning`` (the bug in #156140 was exactly this failure
        being swallowed). Without a fetcher, the bundled/cached index is
        searched directly; only an unusable index source degrades, again with
        an explicit warning.
        """
        try:
            if fetcher is not None:
                self.index.refresh(fetcher)
            return SearchResult(hits=self.search(query, top_k), source="index", warning=None)
        except SkillsIndexError as exc:
            hits = self.fallback_search(query, top_k)
            return SearchResult(
                hits=hits,
                source="local_fallback",
                warning=(
                    f"skills index unavailable — served local fallback from "
                    f"{self.local_dir}: {exc}"
                ),
            )

    # ── internals ─────────────────────────────────────────────────────────────

    def _rank(self, query: str, records: list[dict[str, str]], top_k: int) -> list[SkillHit]:
        """Score ``records`` for ``query``; best-first, deterministic order."""
        if top_k <= 0:
            return []
        query_tokens = _tokens(query)
        hits: list[SkillHit] = []
        for record in records:
            score = _score(query_tokens, record["name"], record["description"])
            if score > 0.0:
                hits.append(
                    SkillHit(
                        name=record["name"],
                        description=record["description"],
                        category=record["category"],
                        score=score,
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.name))
        return hits[:top_k]

    def _local_records(self) -> list[dict[str, str]]:
        """Skill records from ``local_dir`` (deterministic path order)."""
        records: list[dict[str, str]] = []
        if not self.local_dir.is_dir():
            return records
        for path in sorted(self.local_dir.iterdir()):
            if path.is_dir():
                skill_md = path / "SKILL.md"
                if skill_md.is_file():
                    records.append(self._record_from_skill_md(skill_md, path.name))
            elif path.is_file() and path.suffix == ".md":
                records.append(self._record_from_skill_md(path, path.stem))
        return records

    @staticmethod
    def _record_from_skill_md(path: Path, fallback_name: str) -> dict[str, str]:
        """Record from a SKILL.md / markdown file: frontmatter or defaults."""
        meta = _frontmatter(path)
        name = str(meta.get("name") or fallback_name)
        description = str(meta.get("description") or "")
        version = str(meta.get("version") or "local")
        return {"name": name, "description": description, "category": LOCAL_CATEGORY,
                "version": version}


def _frontmatter(path: Path) -> dict[str, str]:
    """Minimal YAML frontmatter reader for local SKILL.md files.

    Deliberately dependency-light (no python-frontmatter round trip): only
    ``name``/``description``/``version`` string keys are read, and a missing
    or malformed block simply yields ``{}`` — a locally broken SKILL.md is
    not allowed to break the fallback listing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}
    lines = stripped.splitlines()[1:]
    meta: dict[str, str] = {}
    for line in lines:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in {"name", "description", "version"} and value:
            meta[key] = value
    return meta
