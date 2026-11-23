"""Optional OCR for image attachments (plan P8.T4).

The plan Phase 8 component ``hermes/memory/ocr.py``: extract text from image
bytes so attached images become searchable (plan Phase 8 acceptance: *image
text extracted and added to entry for search*). The extraction seam is the
:class:`OcrBackend` protocol — :meth:`OcrBackend.extract_text` maps image
bytes to text — so :mod:`hermes.memory.multimodal` can accept any backend as
a plain ``Callable[[bytes], str]`` (the pattern its tests inject fakes
through) without importing this module.

:func:`TesseractOcr` is the real backend: it shells out to the ``tesseract``
CLI (``tesseract <image> stdout``) over a temp file. Every failure mode —
binary not on PATH, non-zero exit, timeout — raises
:class:`OcrUnavailableError` (a :class:`~hermes.memory.multimodal.
MultimodalError`), never a bare exception, so the caller's graceful-degrade
path (attachment still created, no sidecar) needs a single catch. The class
is never imported or called by the test suite; tests inject fakes.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from hermes.memory.multimodal import MultimodalError

__all__ = ["OcrBackend", "OcrUnavailableError", "TesseractOcr"]


class OcrUnavailableError(MultimodalError):
    """OCR could not run (backend missing, failed, or timed out)."""


class OcrBackend(Protocol):
    """Extraction seam for image text (plan P8.T4)."""

    def extract_text(self, image_bytes: bytes) -> str:
        """Return the text contained in ``image_bytes`` (may be empty)."""
        ...


class TesseractOcr:
    """Real OCR backend: the ``tesseract`` CLI over a temp image file.

    ``command`` defaults to ``tesseract`` on PATH (override for custom
    installs); ``timeout_seconds`` bounds one invocation. All failure modes
    raise :class:`OcrUnavailableError`.
    """

    def __init__(self, command: str = "tesseract", timeout_seconds: float = 60.0) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds

    def extract_text(self, image_bytes: bytes) -> str:
        """OCR ``image_bytes`` via ``tesseract <tmpfile> stdout``.

        The bytes are written to a temp ``.png`` (leptonica sniffs the real
        format from content); stdout is decoded leniently and stripped.
        Raises :class:`OcrUnavailableError` when the binary is missing, the
        process fails, or it exceeds :attr:`_timeout_seconds`.
        """
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(image_bytes)
                tmp_path = Path(handle.name)
            try:
                completed = subprocess.run(
                    [self._command, str(tmp_path), "stdout"],
                    capture_output=True,
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise OcrUnavailableError(f"tesseract is unavailable: {exc}") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()
                raise OcrUnavailableError(
                    f"tesseract exited with code {completed.returncode}: {stderr[:200]}"
                )
            return completed.stdout.decode("utf-8", errors="replace").strip()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
