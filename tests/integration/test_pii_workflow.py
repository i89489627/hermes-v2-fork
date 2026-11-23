"""P17.T4 integration: PII detector wired into MemoryStore pre-write hooks.

Covers the redact action end to end (placeholders persisted, ``pii_checked``
mark and ``pii:<type>`` tags in the trust card frontmatter) and the
configurable action matrix redact | block | warn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes.memory.pii_detector import PIIDetector, PiiBlockedError
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import parse

_SENSITIVE = "Email jane.doe@corp.example for the deploy key sk-abcdef1234567890abcdef123"


@pytest.mark.integration
def test_pii_redacted_before_write(tmp_path: Path) -> None:
    """add/update redact PII and record the scan in the trust card."""
    store = MemoryStore(
        root=tmp_path / "memory-repo",
        pii_detector=PIIDetector(action="redact"),
    )
    entry = store.add(_SENSITIVE, type="User", tags=["ops"])

    # the stored body carries placeholders, never the raw secrets
    assert entry.text == "Email [EMAIL] for the deploy key [API_KEY]"
    assert "jane.doe@corp.example" not in entry.text
    assert "sk-abcdef" not in entry.text

    reread = store.get(entry.id)
    assert reread is not None
    assert reread.text == entry.text
    assert reread.pii_checked is True
    assert "pii:api-key" in reread.tags
    assert "pii:email" in reread.tags

    # frontmatter is schema-safe: pii_checked + tags serialize/parse cleanly
    path = tmp_path / "memory-repo" / "global" / "facts" / f"{entry.id}.md"
    card, body = parse(path.read_text(encoding="utf-8"))
    assert card.pii_checked is True
    assert "pii:email" in card.tags
    assert body == entry.text
    assert [commit.message for commit in store.git.log(path=path)][0] == (
        f"[memory] add {entry.id} global"
    )

    # update path: new text is scanned and redacted too, tags accumulate
    updated = store.update(entry.id, text="reach ops@corp.example or 555-123-4567")
    assert updated.text == "reach [EMAIL] or [PHONE]"
    assert "pii:phone" in updated.tags
    assert store.get(entry.id) is not None
    assert store.get(entry.id).tags == updated.tags  # type: ignore[union-attr]

    # clean text through the same detector: marked as checked, no pii tags
    clean = store.add("no sensitive data here", type="Project")
    assert clean.pii_checked is True
    assert not any(tag.startswith("pii:") for tag in clean.tags)


@pytest.mark.integration
def test_configurable_action_redact_block_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """redact substitutes, block refuses the write, warn keeps and logs."""
    # ── redact: text is substituted before persisting ──
    redact_store = MemoryStore(
        root=tmp_path / "redact-repo",
        pii_detector=PIIDetector(action="redact"),
    )
    redacted = redact_store.add(_SENSITIVE, type="User")
    assert redacted.text == "Email [EMAIL] for the deploy key [API_KEY]"

    # ── block: nothing is written and the error carries the findings ──
    block_store = MemoryStore(
        root=tmp_path / "block-repo",
        pii_detector=PIIDetector(action="block"),
    )
    with pytest.raises(PiiBlockedError, match="PII") as err:
        block_store.add(_SENSITIVE, type="User")
    # findings are ordered by document position (email appears first)
    assert [item["type"] for item in err.value.result.detected] == ["email", "api-key"]
    assert block_store.list() == []
    assert block_store.git.is_clean()  # no file, no index entry, no commit
    # blocked update refuses as well, leaving the entry untouched
    entry = block_store.add("harmless text", type="User")
    with pytest.raises(PiiBlockedError):
        block_store.update(entry.id, text=_SENSITIVE)
    got = block_store.get(entry.id)
    assert got is not None and got.text == "harmless text"

    # ── warn: text stored unchanged, WARNING logged, entry marked ──
    warn_store = MemoryStore(
        root=tmp_path / "warn-repo",
        pii_detector=PIIDetector(action="warn"),
    )
    with caplog.at_level(logging.WARNING, logger="hermes.memory.pii_detector"):
        warned = warn_store.add(_SENSITIVE, type="User")
    assert warned.text == _SENSITIVE  # unchanged
    assert warned.pii_checked is True
    assert "pii:email" in warned.tags

    assert "PII detected" in caplog.text
    assert "email" in caplog.text
    assert "api-key" in caplog.text
