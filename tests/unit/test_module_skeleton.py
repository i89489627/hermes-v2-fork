"""P0.T3 acceptance: every v2 skeleton module imports without errors (plan Phase 0)."""

from __future__ import annotations

import importlib
import sys

import pytest

SKELETON_MODULES = [
    "hermes",
    "hermes.config",
    "hermes.core",
    "hermes.core.agent",
    "hermes.memory",
    "hermes.mcp",
    "hermes.models",
    "hermes.traces",
    "hermes.optimization",
    "hermes.concurrency",
    "hermes.diagnostics",
    "hermes.skills",
    "hermes.gateway",
]


@pytest.mark.unit
def test_all_modules_import() -> None:
    assert sys.version_info[:2] >= (3, 11), "hermes-v2 requires Python 3.11+"
    for module_name in SKELETON_MODULES:
        importlib.import_module(module_name)
