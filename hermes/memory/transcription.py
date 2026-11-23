"""Optional audio transcription for attachments (plan P8.T5).

The plan Phase 8 component ``hermes/memory/transcription.py``: transcribe
voice notes so attached audio becomes searchable (plan Phase 8 acceptance:
*audio transcribed and text added to entry*). The seam is the
:class:`TranscriptionBackend` protocol — :meth:`TranscriptionBackend.
transcribe` maps audio bytes to text — so :mod:`hermes.memory.multimodal`
accepts any backend as a plain ``Callable[[bytes], str]`` (the pattern its
tests inject fakes through) without importing this module, mirroring the
P8.T4 OCR design.

:class:`WhisperLocal` is the real backend: it lazily imports
``faster-whisper`` (plan: *transcribe voice notes using whisper*), decodes
the bytes through a temp file and joins the segment texts. A missing
``faster-whisper`` distribution — or any decode failure — raises
:class:`TranscriptionUnavailableError` (a :class:`~hermes.memory.multimodal.
MultimodalError`), never a bare exception, so the caller's graceful-degrade
path (attachment still created, no sidecar) needs a single catch. The class
is never imported or called by the test suite; tests inject fakes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol

from hermes.memory.multimodal import MultimodalError

__all__ = ["TranscriptionBackend", "TranscriptionUnavailableError", "WhisperLocal"]


class TranscriptionUnavailableError(MultimodalError):
    """Transcription could not run (backend missing or failed)."""


class TranscriptionBackend(Protocol):
    """Transcription seam for audio bytes (plan P8.T5)."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """Return the spoken text contained in ``audio_bytes`` (may be empty)."""
        ...


class WhisperLocal:
    """Real transcription backend: ``faster-whisper``, loaded on first use.

    ``model_size`` names the whisper model (default ``base``). The model
    object is loaded lazily on the first :meth:`transcribe` call and cached
    on the instance, so importing this module never pulls the (heavy)
    dependency and a missing install surfaces only when transcription is
    actually requested.
    """

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy ``faster_whisper.WhisperModel`` load (cached per instance)."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415 - lazy by design
            except ImportError as exc:
                raise TranscriptionUnavailableError(
                    "faster-whisper is not installed; install the "
                    "'faster-whisper' distribution to transcribe audio"
                ) from exc
            self._model = WhisperModel(self._model_size)
        return self._model

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe ``audio_bytes``; returns the joined segment texts.

        The bytes are written to a temp ``.wav`` (faster-whisper decodes by
        content via PyAV/ffmpeg, the suffix is only a hint). Decoding or
        model failures raise :class:`TranscriptionUnavailableError`.
        """
        model = self._load_model()
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(audio_bytes)
                tmp_path = Path(handle.name)
            try:
                segments, _info = model.transcribe(str(tmp_path))
                return " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                ).strip()
            except (OSError, RuntimeError, ValueError) as exc:
                raise TranscriptionUnavailableError(f"transcription failed: {exc}") from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
