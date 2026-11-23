"""MultimodalAttachment: image/audio/data assets on memory entries (plan P8.T3).

Attachments are stored as files under ``<memory-repo>/assets/`` —
``images/``, ``audio/`` and ``data/`` subdirectories, one copy per asset with
an ``<original-stem>-<8hex>`` asset id — and referenced from the entry's
trust-card frontmatter ``attachments:`` list (each element is the asset's
repo-relative POSIX path, e.g. ``assets/images/photo-a1b2c3d4.png``; the
field itself stays ``list[str]`` so :class:`TrustCard` parse/serialize
round-trips unchanged). Every attach/detach is one Git commit
(``[memory] attach <asset> -> <entry>``) through the same
:class:`~hermes.memory.git_backend.GitBackend` the store uses, and the
trust-card rewrite is rolled back (plus the copied asset removed) when the
commit fails, mirroring :class:`~hermes.memory.store.MemoryStore`
transactionality.

Per-attachment size limit (plan P8.T3 acceptance): **10 MiB** — larger
sources raise :class:`AttachmentTooLargeError` before anything is copied.

Optional text extraction (plan P8.T4/P8.T5, appended additively)
----------------------------------------------------------------
:meth:`MultimodalAttachment.attach_image` and
:meth:`MultimodalAttachment.attach_audio` accept an optional extractor —
``ocr: Callable[[bytes], str]`` (P8.T4, the :class:`~hermes.memory.ocr.
OcrBackend` seam) or ``transcriber: Callable[[bytes], str]`` (P8.T5, the
:class:`~hermes.memory.transcription.TranscriptionBackend` seam). On
extraction success the text is stored as a ``.txt`` **sidecar** next to the
asset (``assets/images/<asset_id>.txt`` / ``assets/audio/<asset_id>.txt``)
and an ``Attachments (OCR)`` / ``Attachments (Transcription)`` section is
appended to the entry's searchable markdown body, so BM25 search
(:meth:`MemoryStore.search`) finds the entry through the attachment's
content. The body append is done here — in the same card rewrite and the
same commit as the attach — rather than through ``MemoryStore.update``:
:class:`MultimodalAttachment` deliberately holds no ``MemoryStore``
reference (the P8.T3 design), and a single commit keeps the attachment, its
sidecar and the searchable text atomic. Failure semantics (plan P8.T4/P8.T5
acceptance: *extraction failure doesn't break attachment*): an extractor
exception is logged as a warning and the attachment proceeds WITHOUT a
sidecar and without the body section. Extraction results that are not
``str`` or are whitespace-only are treated the same way (no sidecar), with
the non-``str`` case warned. :meth:`MultimodalAttachment.detach` removes the
sidecar together with the asset.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

from hermes.config import hermes_home
from hermes.memory.git_backend import GitBackend
from hermes.memory.trust_card import TrustCard, parse, serialize

__all__ = [
    "ASSETS_DIRNAME",
    "ATTACHMENT_KIND_DIRS",
    "MAX_ATTACHMENT_BYTES",
    "OCR_SECTION_HEADER",
    "SIDECAR_SUFFIX",
    "TRANSCRIPTION_SECTION_HEADER",
    "Attachment",
    "AttachmentKind",
    "AttachmentNotFoundError",
    "AttachmentTooLargeError",
    "MultimodalError",
    "MultimodalAttachment",
]

_LOGGER = logging.getLogger("hermes.memory.multimodal")

#: Per-attachment size limit (plan P8.T3: 10MB).
MAX_ATTACHMENT_BYTES: Final[int] = 10 * 1024 * 1024

#: Directory holding all asset files, under the memory-repo root.
ASSETS_DIRNAME: Final[str] = "assets"

#: Kind of a multimodal attachment (plan P8.T3: images, audio, structured data).
AttachmentKind = Literal["image", "audio", "data"]

#: Storage subdirectory per attachment kind (plan P8.T3 acceptance layout).
ATTACHMENT_KIND_DIRS: Final[dict[str, str]] = {
    "image": "images",
    "audio": "audio",
    "data": "data",
}

#: Suffix of the extracted-text sidecar written next to an asset with a
#: successful OCR (P8.T4) / transcription (P8.T5) run.
SIDECAR_SUFFIX: Final[str] = ".txt"

#: Body-section header under which extracted image text is appended to the
#: entry's searchable markdown (plan P8.T4).
OCR_SECTION_HEADER: Final[str] = "Attachments (OCR)"

#: Body-section header under which transcribed audio text is appended to the
#: entry's searchable markdown (plan P8.T5).
TRANSCRIPTION_SECTION_HEADER: Final[str] = "Attachments (Transcription)"

_ASSET_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


class MultimodalError(Exception):
    """Base class for multimodal attachment failures."""


class AttachmentTooLargeError(MultimodalError):
    """Source file exceeds :data:`MAX_ATTACHMENT_BYTES` (plan P8.T3)."""


class AttachmentNotFoundError(MultimodalError):
    """The entry has no attachment with the requested asset id."""


@dataclass(frozen=True)
class Attachment:
    """One attached asset (plan P8.T3; P8.T4/P8.T5 add the sidecar fields).

    ``stored_path`` is relative to the memory-repo root (POSIX separators) —
    exactly the string recorded in the entry's trust-card ``attachments:``
    list. ``attached_at`` is an aware UTC datetime. The P8.T4/P8.T5 fields
    are populated only when an OCR/transcription run succeeded:
    ``sidecar_path`` is the repo-relative ``.txt`` sidecar and
    ``extracted_text`` the text it holds (the same text appended to the
    entry's searchable body); both stay ``None`` when no extractor ran or
    extraction failed gracefully.
    """

    asset_id: str
    kind: AttachmentKind
    original_name: str
    stored_path: str
    size_bytes: int
    attached_at: datetime
    sidecar_path: str | None = None
    extracted_text: str | None = None


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


def _asset_stem(original_name: str) -> str:
    """Sanitized ``<stem>-`` prefix for a new asset id (fallback: ``asset``)."""
    stem = Path(original_name).stem
    cleaned = _ASSET_STEM_RE.sub("-", stem).strip("-")[:32]
    return f"{cleaned}-" if cleaned else "asset-"


def _body_with_section(body: str, section_header: str | None, text: str) -> str:
    """Append an ``## <header>`` section with ``text`` to the entry body.

    The extracted text becomes part of the entry's searchable markdown so
    BM25 search finds the entry through the attachment's content (plan
    P8.T4/P8.T5 acceptance). An empty body becomes just the section.
    """
    if section_header is None:
        return body
    section = f"## {section_header}\n\n{text.strip()}"
    if body.strip():
        return f"{body.rstrip()}\n\n{section}\n"
    return f"{section}\n"


class MultimodalAttachment:
    """Copy-in asset store wiring files to trust cards (plan P8.T3).

    ``store_root`` defaults to ``hermes_home() / "memory-repo"``; the Git
    backend mirrors :class:`~hermes.memory.store.MemoryStore`: ``GitBackend``
    may be injected, otherwise the repository is created (or re-opened) on
    first use, so every attach/detach leaves exactly one commit behind.
    """

    def __init__(self, store_root: Path | None = None, git: GitBackend | None = None) -> None:
        self.root = Path(store_root) if store_root is not None else hermes_home() / "memory-repo"
        self.git = git if git is not None else GitBackend.init(self.root)

    # ── entry lookup ─────────────────────────────────────────────────────────

    def _find_entry_path(self, entry_id: str) -> Path:
        """Locate the trust-card file of ``entry_id``; ``KeyError`` when unknown.

        Scans the store layout (``global/facts/`` first, then every
        ``contexts/<name>/facts/``) for ``<entry_id>.md``.
        """
        if not entry_id or "/" in entry_id or "\\" in entry_id or entry_id in {".", ".."}:
            raise KeyError(f"unknown memory entry id: {entry_id!r}")
        global_dir = self.root / "global" / "facts"
        candidates: list[Path] = []
        if global_dir.is_dir():
            candidates.append(global_dir / f"{entry_id}.md")
        contexts_root = self.root / "contexts"
        if contexts_root.is_dir():
            for context_dir in sorted(path for path in contexts_root.iterdir() if path.is_dir()):
                facts_dir = context_dir / "facts"
                if facts_dir.is_dir():
                    candidates.append(facts_dir / f"{entry_id}.md")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise KeyError(f"unknown memory entry id: {entry_id!r}")

    # ── attach / detach (plan P8.T3; P8.T4/P8.T5 add the extraction hooks) ───

    def attach(
        self,
        entry_id: str,
        asset_path: str | Path,
        kind: AttachmentKind,
    ) -> Attachment:
        """Copy ``asset_path`` into ``assets/<kind dir>/`` and reference it.

        Enforces the 10 MiB limit (:class:`AttachmentTooLargeError` before
        anything is copied), validates ``kind`` (``ValueError``), resolves the
        entry (:class:`KeyError`), copies the file under
        ``assets/<images|audio|data>/<asset_id><ext>`` and appends the
        repo-relative path to the trust-card ``attachments:`` list — committed
        as ``[memory] attach <asset> -> <entry>``. A failing commit rolls the
        card rewrite and the copied file back before the error propagates.
        """
        return self._attach_with_text(
            entry_id, asset_path, kind, extractor=None, section_header=None
        )

    def attach_image(
        self,
        entry_id: str,
        image_path: str | Path,
        ocr: Callable[[bytes], str] | None = None,
    ) -> Attachment:
        """Attach an image; optionally OCR it into a searchable sidecar (P8.T4).

        ``ocr`` is the :class:`~hermes.memory.ocr.OcrBackend` seam as a plain
        callable (``extract_text(image_bytes) -> str``). On success the text
        lands in ``assets/images/<asset_id>.txt`` and in an
        ``Attachments (OCR)`` section appended to the entry's searchable body
        (also exposed as :attr:`Attachment.extracted_text` /
        :attr:`Attachment.sidecar_path`). On OCR failure the attachment is
        still created — warning logged, no sidecar, no body section.
        """
        return self._attach_with_text(
            entry_id, image_path, "image", extractor=ocr, section_header=OCR_SECTION_HEADER
        )

    def attach_audio(
        self,
        entry_id: str,
        audio_path: str | Path,
        transcriber: Callable[[bytes], str] | None = None,
    ) -> Attachment:
        """Attach audio; optionally transcribe it into a searchable sidecar (P8.T5).

        ``transcriber`` is the :class:`~hermes.memory.transcription.
        TranscriptionBackend` seam as a plain callable (``transcribe(audio_bytes)
        -> str``). Same sidecar + searchable-body mechanics as
        :meth:`attach_image`, under an ``Attachments (Transcription)`` section;
        a transcription failure still attaches the audio (warning, no sidecar).
        """
        return self._attach_with_text(
            entry_id,
            audio_path,
            "audio",
            extractor=transcriber,
            section_header=TRANSCRIPTION_SECTION_HEADER,
        )

    def _extract_safely(self, extractor: Callable[[bytes], str], payload: bytes) -> str | None:
        """Run ``extractor`` with the P8.T4/P8.T5 graceful-failure contract.

        Returns the extracted text, or None when the extractor raised (warning
        logged), returned a non-``str`` (warning logged) or produced
        whitespace-only output (debug log). None means "attach without a
        sidecar" — never an error.
        """
        try:
            result = extractor(payload)
        except Exception:
            _LOGGER.warning(
                "attachment text extraction failed; attaching without a sidecar",
                exc_info=True,
            )
            return None
        if not isinstance(result, str):
            _LOGGER.warning(
                "attachment text extractor returned %s instead of str; ignoring",
                type(result).__name__,
            )
            return None
        if not result.strip():
            _LOGGER.debug("attachment text extractor produced no usable text")
            return None
        return result

    def _attach_with_text(
        self,
        entry_id: str,
        asset_path: str | Path,
        kind: AttachmentKind,
        extractor: Callable[[bytes], str] | None,
        section_header: str | None,
    ) -> Attachment:
        """Shared attach mechanics (P8.T3 copy-in + P8.T4/P8.T5 extraction).

        Order of failure surfaces is preserved from the P8.T3 implementation:
        kind validation, missing source, size limit, unknown entry — then the
        optional extraction (which can never fail the attach). The card
        rewrite carries the attachment reference AND the extracted-text body
        section in the same atomic write and the same single Git commit as
        the asset copy + sidecar; every piece rolls back on commit failure.
        """
        if kind not in ATTACHMENT_KIND_DIRS:
            allowed = ", ".join(sorted(ATTACHMENT_KIND_DIRS))
            raise ValueError(f"attachment kind must be one of {allowed}, got {kind!r}")
        source = Path(asset_path)
        if not source.is_file():
            raise FileNotFoundError(f"attachment source file not found: {source}")
        size_bytes = source.stat().st_size
        if size_bytes > MAX_ATTACHMENT_BYTES:
            raise AttachmentTooLargeError(
                f"attachment {source.name!r} is {size_bytes} bytes, above the "
                f"{MAX_ATTACHMENT_BYTES}-byte (10MB) limit"
            )
        entry_path = self._find_entry_path(entry_id)
        prior = entry_path.read_text(encoding="utf-8")
        card, body = parse(prior)

        extracted: str | None = None
        if extractor is not None and section_header is not None:
            extracted = self._extract_safely(extractor, source.read_bytes())

        target_dir = self.root / ASSETS_DIRNAME / ATTACHMENT_KIND_DIRS[kind]
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        asset_id = self._new_asset_id(_asset_stem(source.name), target_dir)
        target = target_dir / f"{asset_id}{suffix}"
        shutil.copy2(source, target)

        sidecar: Path | None = None
        if extracted is not None:
            sidecar = target.with_suffix(SIDECAR_SUFFIX)
            _atomic_write(sidecar, extracted)

        reference = target.relative_to(self.root).as_posix()
        card.attachments.append(reference)
        attached_at = _utcnow()
        new_body = (
            body if extracted is None else _body_with_section(body, section_header, extracted)
        )
        _atomic_write(entry_path, serialize(card, new_body))
        commit_files: list[Path] = [entry_path, target]
        if sidecar is not None:
            commit_files.append(sidecar)
        try:
            self.git.commit(
                f"[memory] attach {asset_id} -> {entry_id}", commit_files
            )
        except Exception:
            _atomic_write(entry_path, prior)  # restore the pre-attach card
            target.unlink(missing_ok=True)  # remove the copied asset
            if sidecar is not None:
                sidecar.unlink(missing_ok=True)  # remove the text sidecar
            raise
        return Attachment(
            asset_id=asset_id,
            kind=kind,
            original_name=source.name,
            stored_path=reference,
            size_bytes=size_bytes,
            attached_at=attached_at,
            sidecar_path=(
                sidecar.relative_to(self.root).as_posix() if sidecar is not None else None
            ),
            extracted_text=extracted,
        )

    def detach(self, entry_id: str, asset_id: str) -> str:
        """Remove the ``asset_id`` attachment: reference + asset file.

        Drops the matching ``attachments:`` entry (matched on the asset-file
        stem), deletes the stored asset file — plus the P8.T4/P8.T5 ``.txt``
        sidecar when one exists — and commits
        ``[memory] detach <asset> <- <entry>``. Unknown entries raise
        :class:`KeyError`; an entry without this asset raises
        :class:`AttachmentNotFoundError`. Returns the removed repo-relative
        path.
        """
        entry_path = self._find_entry_path(entry_id)
        prior = entry_path.read_text(encoding="utf-8")
        card, body = parse(prior)
        index = next(
            (i for i, ref in enumerate(card.attachments) if Path(ref).stem == asset_id),
            None,
        )
        if index is None:
            raise AttachmentNotFoundError(
                f"entry {entry_id!r} has no attachment with asset id {asset_id!r}"
            )
        reference = str(card.attachments.pop(index))
        asset_file = self.root / reference
        asset_file.unlink(missing_ok=True)
        # The OCR/transcription sidecar shares the asset's stem (P8.T4/P8.T5).
        asset_file.with_suffix(SIDECAR_SUFFIX).unlink(missing_ok=True)
        _atomic_write(entry_path, serialize(card, body))
        try:
            self.git.commit(f"[memory] detach {asset_id} <- {entry_id}", [entry_path])
        except Exception:
            _atomic_write(entry_path, prior)
            raise
        return reference

    def _new_asset_id(self, stem_prefix: str, target_dir: Path) -> str:
        """``<sanitized-stem>-<8hex>`` id, unique within ``target_dir``."""
        while True:
            asset_id = f"{stem_prefix}{uuid4().hex[:8]}"
            if not any(target_dir.glob(f"{asset_id}.*")):
                return asset_id

    # ── read-side helpers ─────────────────────────────────────────────────────

    def entry_card(self, entry_id: str) -> TrustCard:
        """Parsed trust card of ``entry_id`` (``KeyError`` when unknown)."""
        card, _body = parse(self._find_entry_path(entry_id).read_text(encoding="utf-8"))
        return card
