"""P20.T1 chaos: disk full (ENOSPC) during a memory write.

The store's atomic write (temp file + ``os.replace``) is sabotaged with
``OSError(ENOSPC)`` on its FIRST invocation (the chaos "disk full" moment):
``store.add`` must fail CLEANLY — the error propagates, the temp file is
cleaned up by the atomic-write guard, no entry file, no index update, no git
commit, the previous entries intact — and a later write, with the ENOSPC
condition lifted, succeeds and commits (`test_write_fails_gracefully`).
"""

from __future__ import annotations

import errno
import os
from typing import Any

import pytest


@pytest.mark.chaos
def test_write_fails_gracefully(chaos_store, monkeypatch) -> None:
    """ENOSPC on the atomic write: clean failure, no partial state, later add works."""
    store = chaos_store
    survivor = store.add("disk full survivor", type="Feedback", source="user-explicit")
    assert store.get(survivor.id) is not None

    real_replace = os.replace
    attempts = {"count": 0}

    def enospc_then_free(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        """Raise ENOSPC on the FIRST os.replace, then behave (space freed)."""
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", enospc_then_free)

    # The write under ENOSPC raises cleanly (OSError 28), never silently.
    with pytest.raises(OSError):
        store.add("doomed write under ENOSPC", type="Feedback")
    assert attempts["count"] == 1  # the atomic write was really hit, once

    # No partial state anywhere: no temp litter, no entry file, no index or
    # commit side effects — the previous entries are intact.
    facts_dir = store.root / "global" / "facts"
    assert not any(path.name.startswith(".") for path in facts_dir.iterdir())
    assert sorted(entry.text for entry in store.list()) == ["disk full survivor"]
    assert store.get(survivor.id) is not None
    assert store.git.is_clean()

    # Space freed up: the very next write succeeds, is committed, and both
    # entries are readable with their git history intact.
    recovered = store.add(
        "written after space freed", type="Feedback", source="user-explicit"
    )
    assert recovered.id != survivor.id
    assert store.get(recovered.id) is not None
    assert store.get(recovered.id).text == "written after space freed"
    assert sorted(entry.text for entry in store.list()) == [
        "disk full survivor",
        "written after space freed",
    ]
    assert store.git.is_clean()
    assert store.git.log(path=f"global/facts/{recovered.id}.md", limit=5)
