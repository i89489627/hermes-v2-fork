"""hermes-v2 home resolution must agree with the fork-base runtime (regression, 2026-11-23).

``hermes.config.hermes_home()`` used to fall back to ``~/.hermes`` on every platform,
while the fork-base default is ``%LOCALAPPDATA%\\hermes`` on Windows. A v2 call made
without ``HERMES_HOME`` therefore opened a second, empty home beside the real one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hermes_constants import get_hermes_home

from hermes.config import hermes_home


@pytest.mark.unit
def test_matches_fork_base_when_hermes_home_is_set(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-a"))
    assert hermes_home() == get_hermes_home() == tmp_path / "home-a"


@pytest.mark.unit
def test_matches_fork_base_when_hermes_home_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert hermes_home() == get_hermes_home()


@pytest.mark.unit
def test_unset_home_is_never_the_v2_only_fallback_on_windows(monkeypatch) -> None:
    """On Windows the platform default is LOCALAPPDATA, not ``~/.hermes``."""
    import sys

    monkeypatch.delenv("HERMES_HOME", raising=False)
    resolved = hermes_home()
    if sys.platform == "win32":
        assert resolved != Path.home() / ".hermes"
