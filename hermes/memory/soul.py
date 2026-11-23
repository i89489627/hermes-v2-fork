"""SoulManager — immutable core + bounded adaptive style (plan P15.T1).

Splits the fork-base ``SOUL.md`` into two files under
``<memory-repo>/soul/``:

* ``soul_core.md`` — safety, alignment and identity. IMMUTABLE: the agent
  has no API to write it (:class:`SoulManager` deliberately provides no
  ``update_core``/``write_core``/``set_core`` method — immutability is
  enforced by absence of API), the bundled template carries a
  Git-protected note, and :meth:`SoulManager.verify_core_integrity` lets
  hooks detect any out-of-band change by hash. Changing the core requires
  an explicit manual Git operation by the user (plan P15.T1 acceptance).
* ``soul_style.md`` — adaptive communication style, agent-writable via
  :meth:`SoulManager.update_style`, hard-bounded to
  :data:`SOUL_STYLE_LIMIT` (2000) characters. Every update is an atomic
  write, appended to ``.style_history.log`` (ISO-8601 UTC timestamp + char
  count) and, when the memory repo is Git-backed, committed through
  :class:`hermes.memory.git_backend.GitBackend` (non-Git roots write only).

``ensure_soul`` materializes missing files (core from the bundled
:data:`SAFETY_CORE_TEMPLATE`, style empty) and — in a Git-backed repo —
makes a one-time baseline commit so the core is actually tracked by Git
(the "manual Git operation" override path of the plan acceptance needs a
tracked baseline to diff against). Reads lazily call ``ensure_soul`` so a
fresh install always yields the bundled defaults.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from hermes.config import hermes_home

logger = logging.getLogger(__name__)

#: Directory holding the soul files under the memory-repo root.
SOUL_DIRNAME = "soul"

#: Immutable safety/alignment/identity layer.
SOUL_CORE_FILENAME = "soul_core.md"

#: Adaptive style layer (agent-writable, bounded).
SOUL_STYLE_FILENAME = "soul_style.md"

#: Append-only history of style updates (one ``<ts>\t<char_count>`` per line).
STYLE_HISTORY_FILENAME = ".style_history.log"

#: Hard character bound on ``soul_style.md`` (plan P15.T1 / plan Phase 15).
SOUL_STYLE_LIMIT = 2000

#: Bundled safety/alignment core written on first ``ensure_soul`` (plan
#: P15.T1). The Git-protected note is part of the contract: it documents that
#: the file never changes except through an explicit manual Git override.
SAFETY_CORE_TEMPLATE = """\
<!-- soul_core.md — IMMUTABLE (plan P15.T1). Git-protected: this file never \
changes except through an explicit manual Git operation by the user. The \
agent exposes no API that writes it; hooks verify integrity via \
SoulManager.verify_core_integrity. -->

# Soul — Core (immutable)

Safety and alignment commitments that never drift:

1. A person's life, safety and dignity come first; refuse requests to harm, \
harass or deceive anyone.
2. Be honest: never fabricate facts, results or provenance; say "I don't \
know" when that is the truth.
3. Protect the user's data: no secrets or personal data into memory, logs \
or outgoing messages.
4. Follow the user's explicit, current instructions over stale habits — \
but never over these commitments.
5. Memory is evidence-based: entries carry provenance and lose trust when \
contradicted; never smuggle unverified claims into confident advice.
"""


class StyleLimitError(ValueError):
    """Raised when an update would exceed :data:`SOUL_STYLE_LIMIT` characters.

    The message carries both the attempted length and the limit; the style
    file and its history log are left untouched when this is raised.
    """


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (unique temp + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # pragma: no cover - no-op after successful replace


def _iso_z(now: datetime) -> str:
    """ISO 8601 UTC timestamp with a ``Z`` suffix (history log format)."""
    return now.astimezone(UTC).isoformat().replace("+00:00", "Z")


class SoulManager:
    """Manages the immutable soul core and the bounded adaptive style (P15.T1)."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.root = Path(repo_root) if repo_root is not None else hermes_home() / "memory-repo"
        self._soul_dir = self.root / SOUL_DIRNAME
        self._core_path = self._soul_dir / SOUL_CORE_FILENAME
        self._style_path = self._soul_dir / SOUL_STYLE_FILENAME
        self._history_path = self._soul_dir / STYLE_HISTORY_FILENAME

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _is_git_backed(self) -> bool:
        return (self.root / ".git").exists()

    def _git_commit(self, message: str, files: list[Path]) -> None:
        """Commit ``files`` via GitBackend; a failed commit keeps the write.

        Only called when the repo is Git-backed. The write itself already
        happened atomically, so a Git failure must not fail the update — it
        is logged instead (the next successful update will pick the files up).
        """
        try:
            from hermes.memory.git_backend import GitBackend

            backend = GitBackend(self.root)
            sha = backend.commit(message, files=files)
            logger.debug("soul git commit %s: %s", sha[:8], message)
        except Exception as exc:
            logger.warning("soul git commit failed (write kept): %s", exc)

    def ensure_soul(self) -> None:
        """Create ``soul_core.md`` / ``soul_style.md`` if missing.

        The core is materialized from the bundled :data:`SAFETY_CORE_TEMPLATE`
        (never overwritten when present), the style starts empty. In a
        Git-backed repo a one-time baseline commit tracks both files so the
        core is Git-protected (manual-override workflow) from the start.
        """
        created: list[Path] = []
        if not self._core_path.exists():
            _atomic_write(self._core_path, SAFETY_CORE_TEMPLATE)
            created.append(self._core_path)
            logger.info("soul core created from bundled SAFETY_CORE_TEMPLATE")
        if not self._style_path.exists():
            _atomic_write(self._style_path, "")
            created.append(self._style_path)
        if created and self._is_git_backed():
            self._git_commit(
                "[P15.T1] baseline soul (core immutable, style empty)", files=created
            )

    # ── reads ───────────────────────────────────────────────────────────────

    def get_core(self) -> str:
        """The immutable core content (lazily materialized from the template)."""
        self.ensure_soul()
        return self._core_path.read_text(encoding="utf-8")

    def get_style(self) -> str:
        """The adaptive style content (lazily materialized, empty by default)."""
        self.ensure_soul()
        return self._style_path.read_text(encoding="utf-8")

    # ── writes (STYLE ONLY — there is no update_core, by design) ────────────

    def update_style(self, new_content: str) -> None:
        """Replace the adaptive style within the hard character bound.

        Raises :class:`StyleLimitError` when ``len(new_content)`` exceeds
        :data:`SOUL_STYLE_LIMIT` (2000) — nothing is written or logged in
        that case. On success: atomic write, one history-log line
        (``<ISO-8601 UTC ts>\\t<char_count>``), and a Git commit of the style
        + history files when the repo is Git-backed (write-only otherwise).
        """
        content = str(new_content)
        length = len(content)
        if length > SOUL_STYLE_LIMIT:
            raise StyleLimitError(
                f"soul style update refused: {length} chars exceeds the "
                f"{SOUL_STYLE_LIMIT}-char limit"
            )
        self.ensure_soul()
        _atomic_write(self._style_path, content)
        self._append_history(length)
        if self._is_git_backed():
            self._git_commit(
                f"[P15.T1] update soul style ({length} chars)",
                files=[self._style_path, self._history_path],
            )
        logger.info("soul style updated (%d chars)", length)

    def _append_history(self, char_count: int) -> None:
        """Append ``<ts>\\t<char_count>`` to the style history log."""
        self._soul_dir.mkdir(parents=True, exist_ok=True)
        line = f"{_iso_z(datetime.now(UTC))}\t{char_count}\n"
        with open(self._history_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    # ── integrity helpers (for hooks / manual-override detection) ───────────

    def core_hash(self) -> str:
        """SHA-256 hex digest of the core file bytes (integrity snapshot)."""
        self.ensure_soul()
        return hashlib.sha256(self._core_path.read_bytes()).hexdigest()

    def verify_core_integrity(self, expected_hash: str) -> bool:
        """Compare the current core hash against ``expected_hash``.

        Hook helper: a hook snapshots :meth:`core_hash` and calls this later;
        ``False`` means the immutable core was modified out-of-band (the
        agent has no API for that — only a manual Git override or an external
        writer would cause it) and should be treated as a tamper signal.
        """
        self.ensure_soul()
        return self.core_hash() == expected_hash.strip().lower()
