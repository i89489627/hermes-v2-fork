"""P8.T3 acceptance tests: MultimodalAttachment (plan Phase 8).

Covers the spec behaviours: image and audio assets copied under
``assets/images/`` / ``assets/audio/`` with the returned ``Attachment``
pointing at the stored file, the 10MB per-attachment limit raising
``AttachmentTooLargeError``, trust-card ``attachments:`` referencing, and
detach removing both the reference and the asset file. Entry fixtures use a
real :class:`~hermes.memory.store.MemoryStore` so the on-disk layout is the
production one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.multimodal import (
    MAX_ATTACHMENT_BYTES,
    AttachmentNotFoundError,
    AttachmentTooLargeError,
    MultimodalAttachment,
)
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse

#: 1x1 8-bit PNG (67 bytes), used as a real image attachment.
PNG_1X1: bytes = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcffc0f01f00050502002dc9f1f3000000000049454e"
    "44ae426082"
)


def _make_repo(tmp_path: Path) -> tuple[MemoryStore, MultimodalAttachment]:
    store = MemoryStore(root=tmp_path / "memory-repo")
    return store, MultimodalAttachment(store_root=store.root)


@pytest.mark.unit
def test_attach_image_creates_asset_file(tmp_path: Path) -> None:
    """An image attachment is copied under assets/images/ and committed."""
    store, attachments = _make_repo(tmp_path)
    entry = store.add("Whiteboard sketch of the deploy topology", type="Project")
    source = tmp_path / "sketch.png"
    source.write_bytes(PNG_1X1)

    attachment = attachments.attach(entry.id, source, kind="image")

    assert attachment.kind == "image"
    assert attachment.original_name == "sketch.png"
    assert attachment.size_bytes == len(PNG_1X1)
    assert attachment.stored_path.startswith("assets/images/")
    assert attachment.asset_id
    assert attachment.attached_at.tzinfo is not None

    stored = store.root / attachment.stored_path
    assert stored.is_file()
    assert stored.read_bytes() == PNG_1X1  # byte-exact copy
    assert stored.parent == store.root / "assets" / "images"

    # Exactly one commit for the attach, naming the asset and the entry.
    history = store.git.log(path=attachment.stored_path)
    assert [commit.message for commit in history] == [
        f"[memory] attach {attachment.asset_id} -> {entry.id}"
    ]
    assert store.git.is_clean()

    # 10MB limit: a larger source raises before anything is copied.
    oversized = tmp_path / "huge.png"
    oversized.write_bytes(b"x" * (MAX_ATTACHMENT_BYTES + 1))
    with pytest.raises(AttachmentTooLargeError, match="10MB"):
        attachments.attach(entry.id, oversized, kind="image")
    assert not any((store.root / "assets" / "images").glob("*huge*"))


@pytest.mark.unit
def test_attach_audio_creates_asset_file(tmp_path: Path) -> None:
    """An audio attachment lands under assets/audio/; asset ids are unique."""
    store, attachments = _make_repo(tmp_path)
    entry = store.add("Voice note about the rollout plan", type="User")
    source = tmp_path / "note.wav"
    source.write_bytes(b"RIFF....WAVEfmt fake-audio-payload")

    first = attachments.attach(entry.id, source, kind="audio")
    second = attachments.attach(entry.id, source, kind="audio")

    assert first.kind == "audio"
    assert first.stored_path.startswith("assets/audio/")
    assert first.stored_path.endswith(".wav")
    stored_first = store.root / first.stored_path
    assert stored_first.is_file()
    assert stored_first.read_bytes() == source.read_bytes()

    # Two attachments of the same source are two independent assets.
    assert first.asset_id != second.asset_id
    assert first.stored_path != second.stored_path
    assert (store.root / second.stored_path).is_file()

    invalid_kind = attachments.attach  # clearer failure below
    with pytest.raises(ValueError, match="attachment kind"):
        invalid_kind(entry.id, source, "video")  # type: ignore[arg-type]


@pytest.mark.unit
def test_attachment_referenced_in_trust_card(tmp_path: Path) -> None:
    """The trust-card frontmatter carries the reference; detach cleans both."""
    store, attachments = _make_repo(tmp_path)
    entry = store.add("Metrics snapshot for the sprint dashboard", type="Project")
    source = tmp_path / "snapshot.json"
    source.write_bytes(b'{"tok_per_s": 42.0}')

    attachment = attachments.attach(entry.id, source, kind="data")

    # Reference lives in the persisted trust-card frontmatter.
    path = store.root / "global" / "facts" / f"{entry.id}.md"
    card, _body = parse(path.read_text(encoding="utf-8"))
    assert card.attachments == [attachment.stored_path]
    reloaded = store.get(entry.id)
    assert reloaded is not None
    assert reloaded.attachments == [attachment.stored_path]
    assert reloaded.attachment_asset_ids() == [attachment.asset_id]

    # detach removes the reference and the asset file, and commits.
    removed = attachments.detach(entry.id, attachment.asset_id)
    assert removed == attachment.stored_path
    assert not (store.root / attachment.stored_path).exists()
    card_after, _body = parse(path.read_text(encoding="utf-8"))
    assert card_after.attachments == []

    detach_history = store.git.log(path=path)
    assert detach_history[0].message == f"[memory] detach {attachment.asset_id} <- {entry.id}"

    # Unknown asset id / unknown entry.
    with pytest.raises(AttachmentNotFoundError, match="no attachment"):
        attachments.detach(entry.id, attachment.asset_id)  # already removed
    with pytest.raises(KeyError, match="unknown memory entry id"):
        attachments.attach("no-such-entry", source, kind="data")
