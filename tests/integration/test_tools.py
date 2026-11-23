"""Integration: disabled tools are never invoked, even when requested (P21.T2).

Issue #26568 reproduction: with a toolset disabled in config, a
model-requested call to one of its tools must be refused — the tool never
appears in the registry (registration-time fix) and the guarded invoker
raises :class:`ToolNotAvailableError` without executing the underlying tool
(call-time defense in depth). Enabled tools keep working through the same
guarded path.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.core.tool_registry import ToolDef, ToolNotAvailableError, ToolRegistry


@pytest.mark.integration
def test_disabled_tool_never_called() -> None:
    """A disabled toolset's tool is absent from the registry and never runs."""
    invocations: list[tuple[str, dict[str, Any] | None]] = []

    def read_file(*args: Any, **kwargs: Any) -> str:
        invocations.append(("read_file", kwargs.get("arguments")))
        return "file contents"

    def shell_exec(*args: Any, **kwargs: Any) -> str:
        invocations.append(("shell_exec", kwargs.get("arguments")))
        return "ran a shell command"

    registry = ToolRegistry(
        config={"disabled_toolsets": ["shell"]},
        toolset_map={
            "fs": [ToolDef("read_file", "fs", read_file, "read a file")],
            "shell": [ToolDef("shell_exec", "shell", shell_exec, "run a command")],
        },
    )
    guarded = registry.wrap_invoker(
        lambda name, arguments=None: registry.get(name).callable(arguments=arguments)
    )

    # registration-time fix: the disabled toolset's tool never registered
    assert registry.get("shell_exec") is None
    assert "shell_exec" not in [tool.name for tool in registry.list_tools()]
    assert registry.is_available("shell_exec") is False

    # the model requests the disabled tool -> refused BEFORE any execution
    with pytest.raises(ToolNotAvailableError) as excinfo:
        guarded("shell_exec", {"command": "rm -rf /"})
    assert excinfo.value.tool_name == "shell_exec"
    assert invocations == []  # the underlying tool NEVER ran

    # defense in depth: even force-registering the disabled tool does not
    # make it callable through the guarded invoker
    registry.register(ToolDef("shell_exec", "shell", shell_exec, "run a command"))
    assert registry.get("shell_exec") is not None  # present for introspection...
    assert registry.is_available("shell_exec") is False  # ...but still disabled
    assert "shell_exec" not in [tool.name for tool in registry.list_tools()]
    with pytest.raises(ToolNotAvailableError):
        guarded("shell_exec", {"command": "rm -rf /"})
    assert invocations == []

    # an enabled tool flows through the same guarded path unharmed
    assert guarded("read_file", {"path": "notes.txt"}) == "file contents"
    assert invocations == [("read_file", {"path": "notes.txt"})]
