"""SkillsIndex: bundled base index + explicit remote refresh (plan P21.T4).

Fixes the 86-second cold skills search (issue #156140): the base index ships
with the install, so a cold cache (no local index file yet) serves from
:data:`hermes.skills.data.bundled_index.INDEX` instantly — no 52MB download
on first use. A locally cached index (``<memory-repo>/skills/_skills_index.json``,
written by a successful :meth:`SkillsIndex.refresh`) always wins over the
bundled snapshot because it is fresher.

Bundled-as-python-module deviation (documented)
-----------------------------------------------
The plan mentions bundling the index "with install". pyproject package-data
is outside this task's file list, so the bundled index is an importable
python module (:mod:`hermes.skills.data.bundled_index`) instead of a JSON
data file — identical content, zero packaging wiring, wheels included.

NO silent degradation
---------------------
:mod:`fetch_remote` (fetch + validate) and :meth:`SkillsIndex.refresh`
(fetch + persist + reload) raise :class:`SkillsIndexFetchError` when the
remote source fails or returns a malformed payload — the caller decides what
to do (typically :meth:`SkillsSearch.search_with_fallback`, which surfaces
the error as an explicit warning and serves local results). The default
fetcher is ``None`` and raises the same explicit error: v2 has no built-in
remote endpoint, so "no remote configured" must be an error, never an empty
result set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from hermes.config import hermes_home
from hermes.skills.data.bundled_index import INDEX as _BUNDLED_INDEX

__all__ = [
    "BUNDLED_INDEX_RECORD_COUNT",
    "REQUIRED_RECORD_FIELDS",
    "SkillsIndex",
    "SkillsIndexError",
    "SkillsIndexFetchError",
    "default_index_path",
]

#: Fields every index record must carry (plan P21.T4: name/description/category/version).
REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = ("name", "description", "category", "version")

#: Number of records shipped in the bundled base index.
BUNDLED_INDEX_RECORD_COUNT: Final[int] = len(_BUNDLED_INDEX)


class SkillsIndexError(Exception):
    """Base class for skills index failures (explicit, never silent)."""


class SkillsIndexFetchError(SkillsIndexError):
    """Remote skills index fetch failed or returned a malformed payload."""


def default_index_path() -> Path:
    """Default cache location: ``<hermes_home>/memory-repo/skills/_skills_index.json``."""
    path: Path = hermes_home() / "memory-repo" / "skills" / "_skills_index.json"
    return path


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON atomically (tmp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class SkillsIndex:
    """Skills index source chain: local cache -> bundled snapshot (plan P21.T4).

    ``index_path`` defaults to :func:`default_index_path`. ``load()`` is lazy
    and cached per instance; the cold path (no cache file) reads the bundled
    module with no network I/O. A corrupt cache file raises
    :class:`SkillsIndexError` with the offending path — corruption is never
    silently swallowed.
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = Path(index_path) if index_path is not None else default_index_path()
        self._records: list[dict[str, str]] | None = None

    def load(self) -> list[dict[str, str]]:
        """Current index records (cache file when present, bundled otherwise)."""
        if self._records is None:
            if self.index_path.exists():
                self._records = self._read_cache(self.index_path)
            else:
                self._records = [dict(record) for record in _BUNDLED_INDEX]
        return [dict(record) for record in self._records]

    def fetch_remote(self, fetcher: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Fetch and validate the remote index payload (plan P21.T4).

        ``fetcher`` is injected by the caller (network client, plugin, test
        double). ``None`` raises :class:`SkillsIndexFetchError` immediately —
        there is no built-in remote endpoint to guess at. Any exception the
        fetcher raises is wrapped into :class:`SkillsIndexFetchError` with the
        original cause chained, so the failure is always explicit.
        """
        if fetcher is None:
            raise SkillsIndexFetchError(
                "no remote skills index source configured: pass a fetcher to "
                "refresh()/fetch_remote() or use the bundled index via load()"
            )
        try:
            payload = fetcher()
        except Exception as exc:
            raise SkillsIndexFetchError(
                f"skills index fetch failed: {exc}"
            ) from exc
        self._validate_payload(payload)
        return payload

    def refresh(self, fetcher: Callable[[], dict[str, Any]] | None = None) -> list[dict[str, str]]:
        """Fetch remote, persist as the local cache, reload (plan P21.T4).

        Raises :class:`SkillsIndexFetchError` on any failure (the caller
        decides on fallback — nothing is degraded silently and the previous
        cache file, if any, stays untouched). Returns the refreshed records.
        """
        payload = self.fetch_remote(fetcher)
        records = self._validate_payload(payload)
        _atomic_write_json(self.index_path, {"version": 1, "skills": records})
        self._records = records
        return [dict(record) for record in records]

    # ── internals ─────────────────────────────────────────────────────────────

    def _read_cache(self, path: Path) -> list[dict[str, str]]:
        """Parse and validate the local cache file."""
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillsIndexError(
                f"corrupt local skills index cache {path}: {exc} — delete the "
                "file or refresh() from a working source"
            ) from exc
        return self._validate_payload(payload)

    @staticmethod
    def _validate_payload(payload: Any) -> list[dict[str, str]]:
        """Validate an index payload into normalized records (explicit errors)."""
        if not isinstance(payload, dict):
            raise SkillsIndexFetchError(
                f"malformed skills index payload: expected a mapping, got "
                f"{type(payload).__name__}"
            )
        raw_records = payload.get("skills", payload.get("index"))
        if not isinstance(raw_records, list):
            raise SkillsIndexFetchError(
                "malformed skills index payload: 'skills' must be a list of records"
            )
        records: list[dict[str, str]] = []
        for position, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                raise SkillsIndexFetchError(
                    f"malformed skills index payload: record {position} is not a mapping"
                )
            record: dict[str, str] = {}
            for field in REQUIRED_RECORD_FIELDS:
                value = raw.get(field)
                if not isinstance(value, str) or not value:
                    raise SkillsIndexFetchError(
                        f"malformed skills index payload: record {position} "
                        f"({raw.get('name', '?')!r}) is missing a non-empty "
                        f"{field!r} string"
                    )
                record[field] = value
            records.append(record)
        return records
