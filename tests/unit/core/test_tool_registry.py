"""P21.T2 acceptance: registration-time toolset filtering (issue #26568).

Covered: a disabled tool NEVER enters the registry (register_from_config
skips disabled toolsets and disabled tools — the core fix), while enabled
tools register normally and are fully available. Call-time enforcement is
covered by the integration tests (tests/integration/test_tools.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.core.tool_registry import ToolDef, ToolRegistry


def _tool(name: str, toolset: str) -> ToolDef:
    """A minimal ToolDef whose callable records nothing (registration only)."""

    def _impl(*args: Any, **kwargs: Any) -> str:
        return f"{name} ran"

    return ToolDef(name=name, toolset=toolset, callable=_impl, description=f"{name} tool")


@pytest.mark.unit
def test_disabled_tool_not_registered() -> None:
    """Disabled tools never enter the registry (plan P21.T2 core fix).

    Both disable paths are exercised: ``disabled_tools`` (per-tool) and
    ``disabled_toolsets`` (whole toolset). A disabled tool is absent from
    ``get``/``list_tools`` and reports ``is_available() is False`` — it can
    never be advertised to the model or dispatched.
    """
    registry = ToolRegistry(
        config={
            "disabled_tools": ["shell_exec"],
            "disabled_toolsets": ["browser"],
        },
        toolset_map={
            "terminal": [_tool("shell_exec", "terminal"), _tool("run_tests", "terminal")],
            "browser": [_tool("web_search", "browser"), _tool("open_page", "browser")],
            "memory": [_tool("memory_search", "memory")],
        },
    )

    # per-tool disable: never registered at all
    assert registry.get("shell_exec") is None
    assert registry.is_available("shell_exec") is False
    assert "shell_exec" not in [tool.name for tool in registry.list_tools()]

    # whole-toolset disable: none of its tools registered
    assert registry.get("web_search") is None
    assert registry.get("open_page") is None
    assert registry.is_available("web_search") is False
    assert registry.is_available("open_page") is False
    assert not {"web_search", "open_page"} & {tool.name for tool in registry.list_tools()}

    # the surviving registry is exactly the enabled toolsets minus disabled tools
    assert [tool.name for tool in registry.list_tools()] == ["memory_search", "run_tests"]


@pytest.mark.unit
def test_enabled_tool_registered() -> None:
    """Enabled tools register normally and are fully available (P21.T2).

    A tool from an enabled toolset (not named in ``disabled_tools``) lands in
    the registry via ``register_from_config``: retrievable through ``get``,
    listed for the model, available for dispatch.
    """
    tool = _tool("memory_search", "memory")
    registry = ToolRegistry(
        config={"disabled_tools": ["shell_exec"], "disabled_toolsets": ["browser"]},
        toolset_map={"memory": [tool], "terminal": [_tool("shell_exec", "terminal")]},
    )

    fetched = registry.get("memory_search")
    assert fetched is tool
    assert fetched is not None and fetched.toolset == "memory"
    assert registry.is_available("memory_search") is True
    assert [registered.name for registered in registry.list_tools()] == ["memory_search"]

    # allow-list semantics: an explicit enabled_toolsets list restricts
    # registration to exactly those toolsets
    allow_registry = ToolRegistry(
        config={"enabled_toolsets": ["memory"]},
        toolset_map={
            "memory": [_tool("memory_search", "memory")],
            "terminal": [_tool("run_tests", "terminal")],
        },
    )
    assert [registered.name for registered in allow_registry.list_tools()] == ["memory_search"]
    assert allow_registry.is_available("run_tests") is False
