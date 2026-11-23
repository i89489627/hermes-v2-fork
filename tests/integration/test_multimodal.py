"""P8.T3/P8.T4/P8.T5 integration tests: attachment flows (plan Phase 8).

Drives the DEFAULT persistence layout (memory-repo under the per-test
``HERMES_HOME`` via ``hermes.config.hermes_home``):

- P8.T3: a real 1x1 PNG (bytes generated inline) is attached to a stored
  memory entry, and the test asserts the asset file under
  ``<memory-repo>/assets/images/``, the trust-card frontmatter reference
  (round-tripped through a store reload), and the single
  ``[memory] attach ... -> ...`` Git commit.
- P8.T4/P8.T5: optional OCR / transcription backends are injected as fakes
  (the real ``TesseractOcr`` / ``WhisperLocal`` backends are never imported
  or run here) and their extracted text must land in the ``.txt`` sidecar
  next to the asset AND in the entry's searchable body — so BM25 store
  search finds the entry through the attachment's content. Extraction
  failures must degrade gracefully: attachment still created, warning
  logged, no sidecar.
"""

from __future__ import annotations

import logging
from base64 import b64decode
from pathlib import Path

import pytest

from hermes.memory.multimodal import MultimodalAttachment
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse

#: 1x1 8-bit PNG, base64 (67 bytes) — a real, decodable image.
PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.mark.integration
def test_attach_image_to_entry() -> None:
    """store.add + MultimodalAttachment over default paths, PNG asset."""
    store = MemoryStore()  # default root: <HERMES_HOME>/memory-repo
    attachments = MultimodalAttachment()  # same default root
    assert attachments.root == store.root

    entry = store.add("Architecture diagram of the gateway restart flow", type="Project")
    source = Path(store.root.parent) / "diagram.png"  # outside the repo root
    source.write_bytes(b64decode(PNG_1X1_B64))
    assert source.read_bytes().startswith(b"\x89PNG")

    attachment = attachments.attach(entry.id, source, kind="image")

    stored = store.root / attachment.stored_path
    assert attachment.stored_path.startswith("assets/images/")
    assert attachment.stored_path.endswith(".png")
    assert stored.is_file()
    assert stored.read_bytes() == b64decode(PNG_1X1_B64)
    assert attachment.size_bytes == len(b64decode(PNG_1X1_B64))

    # Trust card references the asset; store reload sees it too.
    entry_file = store.root / "global" / "facts" / f"{entry.id}.md"
    card, _body = parse(entry_file.read_text(encoding="utf-8"))
    assert card.attachments == [attachment.stored_path]
    reloaded = store.get(entry.id)
    assert reloaded is not None
    assert reloaded.attachments == [attachment.stored_path]

    # One attach commit; workdir clean afterwards.
    history = store.git.log(path=attachment.stored_path)
    assert [commit.message for commit in history] == [
        f"[memory] attach {attachment.asset_id} -> {entry.id}"
    ]
    assert store.git.is_clean()


@pytest.mark.integration
def test_image_searchable_via_ocr(tmp_path: Path) -> None:
    """P8.T4: fake OCR text lands in the sidecar and the searchable entry body.

    The entry's own text deliberately does NOT contain the search term, so
    finding it via ``store.search`` proves the OCR output is indexed.
    """
    store = MemoryStore()
    attachments = MultimodalAttachment()
    entry = store.add("Scan of the vendor paperwork for April", type="Reference")
    source = tmp_path / "receipt.png"
    image_bytes = b64decode(PNG_1X1_B64)
    source.write_bytes(image_bytes)

    seen: list[bytes] = []

    def fake_ocr(image: bytes) -> str:
        seen.append(image)
        return "invoice total 42 dollars"

    attachment = attachments.attach_image(entry.id, source, ocr=fake_ocr)
    assert seen == [image_bytes]  # the backend received the image bytes

    # Sidecar: assets/images/<asset_id>.txt holds the extracted text.
    assert attachment.sidecar_path is not None
    assert attachment.sidecar_path.startswith("assets/images/")
    assert attachment.sidecar_path.endswith(".txt")
    sidecar = store.root / attachment.sidecar_path
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8") == "invoice total 42 dollars"
    assert attachment.extracted_text == "invoice total 42 dollars"

    # The entry's searchable body carries an OCR section.
    fetched = store.get(entry.id)
    assert fetched is not None
    assert "## Attachments (OCR)" in fetched.text
    assert "invoice total 42 dollars" in fetched.text

    # BM25 search finds the entry through the OCR text alone.
    hits = store.search("invoice")
    assert [hit.id for hit in hits] == [entry.id]

    # Still exactly one attach commit; workdir clean (sidecar committed too).
    history = store.git.log(path=attachment.stored_path)
    assert [commit.message for commit in history] == [
        f"[memory] attach {attachment.asset_id} -> {entry.id}"
    ]
    assert store.git.is_clean()


@pytest.mark.integration
def test_ocr_failure_handled_gracefully(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """P8.T4: an OCR crash still attaches the image — warning, no sidecar."""
    store = MemoryStore()
    attachments = MultimodalAttachment()
    entry = store.add("Whiteboard photo from the planning session", type="Project")
    source = tmp_path / "whiteboard.png"
    source.write_bytes(b64decode(PNG_1X1_B64))

    def broken_ocr(image: bytes) -> str:
        raise RuntimeError("tesseract binary not found")

    with caplog.at_level(logging.WARNING, logger="hermes.memory.multimodal"):
        attachment = attachments.attach_image(entry.id, source, ocr=broken_ocr)

    # Attachment survived; no sidecar, no extracted text.
    assert (store.root / attachment.stored_path).is_file()
    assert attachment.sidecar_path is None
    assert attachment.extracted_text is None
    assert not list((store.root / "assets" / "images").glob("*.txt"))

    # The entry body is untouched — no OCR section.
    fetched = store.get(entry.id)
    assert fetched is not None
    assert "Attachments (OCR)" not in fetched.text

    # The failure was logged as a warning, not raised.
    assert "extraction failed" in caplog.text
    assert store.git.is_clean()


@pytest.mark.integration
def test_audio_transcribed(tmp_path: Path) -> None:
    """P8.T5: fake transcription lands in the sidecar and the searchable body."""
    store = MemoryStore()
    attachments = MultimodalAttachment()
    entry = store.add("Voice note from the platform standup", type="User")
    source = tmp_path / "note.wav"
    audio_bytes = b"ID3 fake mp3 payload for the transcription backend"
    source.write_bytes(audio_bytes)

    seen: list[bytes] = []

    def fake_transcriber(audio: bytes) -> str:
        seen.append(audio)
        return "standup moved to ten thirty tomorrow"

    attachment = attachments.attach_audio(entry.id, source, transcriber=fake_transcriber)
    assert seen == [audio_bytes]  # the backend received the audio bytes
    assert attachment.kind == "audio"
    assert attachment.stored_path.startswith("assets/audio/")

    # Sidecar: assets/audio/<asset_id>.txt holds the transcript.
    assert attachment.sidecar_path is not None
    assert attachment.sidecar_path.startswith("assets/audio/")
    sidecar = store.root / attachment.sidecar_path
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8") == "standup moved to ten thirty tomorrow"
    assert attachment.extracted_text == "standup moved to ten thirty tomorrow"

    # Searchable body + BM25 reach.
    fetched = store.get(entry.id)
    assert fetched is not None
    assert "## Attachments (Transcription)" in fetched.text
    hits = store.search("standup")
    assert entry.id in [hit.id for hit in hits]
    assert store.git.is_clean()


@pytest.mark.integration
def test_transcription_failure_handled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """P8.T5: a transcriber crash still attaches the audio — warning, no sidecar."""
    store = MemoryStore()
    attachments = MultimodalAttachment()
    entry = store.add("Voicemail from the ops pager", type="User")
    source = tmp_path / "voicemail.wav"
    source.write_bytes(b"RIFF fake wav payload")

    def broken_transcriber(audio: bytes) -> str:
        raise RuntimeError("faster-whisper is not installed")

    with caplog.at_level(logging.WARNING, logger="hermes.memory.multimodal"):
        attachment = attachments.attach_audio(entry.id, source, transcriber=broken_transcriber)

    assert (store.root / attachment.stored_path).is_file()
    assert attachment.sidecar_path is None
    assert attachment.extracted_text is None
    assert not list((store.root / "assets" / "audio").glob("*.txt"))
    fetched = store.get(entry.id)
    assert fetched is not None
    assert "Attachments (Transcription)" not in fetched.text
    assert "extraction failed" in caplog.text
    assert store.git.is_clean()
