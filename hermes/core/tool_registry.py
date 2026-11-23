"""Registration-time toolset filtering (plan P21.T2, issue #26568).

Upstream bug (#26568): after the v0.13+ update Hermes called tools the user
had explicitly disabled in ``config.yaml`` — filtering happened (when it
happened at all) at CALL time, so a model requesting a disabled tool still
reached it, and mis-wired filtering let disabled toolsets through entirely.

The fix contract of plan P21.T2 is registration-time enforcement, with
defense in depth at call time:

1. **Registration time (the core fix).** :meth:`ToolRegistry.register_from_config`
   builds the registry ONLY from enabled toolsets minus ``disabled_tools`` —
   a disabled tool never enters the registry, so it cannot be advertised to
   the model (absent from :meth:`ToolRegistry.list_tools`) and cannot be
   dispatched through the normal path (:meth:`ToolRegistry.get` misses it).
2. **Call time (defense in depth).** :meth:`ToolRegistry.wrap_invoker` wraps
   an invoker in a guard that refuses every call whose tool is not currently
   available (:class:`ToolNotAvailableError`) — so even a model-requested
   invocation of a force-registered disabled tool is refused, never executed.

Enable/disable semantics (``config`` keys, all optional):

- ``enabled_toolsets``: allow-list. Missing/``None`` means *every* toolset is
  eligible; a present list (INCLUDING an empty one) means only those toolsets
  are eligible. Explicit-empty deliberately disables everything — use an
  omitted key for "no restriction".
- ``disabled_toolsets``: deny-list on top of the allow-list; a toolset here is
  never registered and its tools report unavailable even if force-registered.
- ``disabled_tools``: per-tool deny-list applied after the toolset rules; a
  named tool is skipped at registration and reports unavailable even if
  force-registered by another path.

:meth:`ToolRegistry.register` stays available for programmatic registration
(tests, plugin seams): it bypasses the config filter (a disabled tool CAN be
force-registered) but never bypasses enforcement — such a tool still reports
``is_available(...) is False``, stays out of :meth:`list_tools`, and is
refused by :meth:`wrap_invoker`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolDef", "ToolNotAvailableError", "ToolRegistry"]


class ToolNotAvailableError(LookupError):
    """A tool call was refused because the tool is not currently available.

    Raised by :meth:`ToolRegistry.wrap_invoker` guards when the requested
    tool name is not registered or is disabled by configuration (issue
    #26568: disabled tools must never be invoked, even when the model
    requests them). Derives from :class:`LookupError` so callers can treat
    "unknown tool" and "disabled tool" uniformly.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"tool {tool_name!r} is not available: {reason}")
        self.tool_name = tool_name
        self.reason = reason


@dataclass
class ToolDef:
    """One registrable tool (plan P21.T2): name, owning toolset, callable.

    ``callable`` is the tool implementation, invoked as
    ``callable(*args, **kwargs)`` by the executor behind a
    :meth:`ToolRegistry.wrap_invoker` guard. ``description`` is the
    model-facing documentation string.
    """

    name: str
    toolset: str
    callable: Callable[..., Any]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolDef.name must be a non-empty string")
        if not isinstance(self.toolset, str) or not self.toolset:
            raise ValueError(f"ToolDef.toolset must be a non-empty string ({self.name!r})")
        if not callable(self.callable):
            raise ValueError(f"ToolDef.callable must be callable ({self.name!r})")


def _string_set(value: Any, key: str) -> frozenset[str]:
    """Normalize a config list into a frozenset of strings (typed, strict)."""
    if value is None:
        return frozenset()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError(f"tool config {key!r} must be a list of strings, got {value!r}")
    members: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"tool config {key!r} must contain only strings, got {item!r}")
        members.append(item)
    return frozenset(members)


class ToolRegistry:
    """Registry whose contents are filtered at REGISTRATION time (P21.T2).

    ``config`` carries the enable/disable rules (see the module docstring);
    ``None`` means everything is enabled. ``toolset_map`` (optional) maps
    toolset name -> its :class:`ToolDef` list; when given, the constructor
    immediately runs :meth:`register_from_config` over it, so a registry built
    from config never contains a disabled tool in the first place (issue
    #26568).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        toolset_map: dict[str, list[ToolDef]] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"config must be a dict or None, got {type(config).__name__}")
        config = config or {}
        #: Allow-list of toolsets; ``None`` = every toolset is eligible.
        enabled: Any = config.get("enabled_toolsets")
        if enabled is None:
            self._enabled_toolsets: frozenset[str] | None = None
        else:
            self._enabled_toolsets = _string_set(enabled, "enabled_toolsets")
        self._disabled_toolsets: frozenset[str] = _string_set(
            config.get("disabled_toolsets"), "disabled_toolsets"
        )
        self._disabled_tools: frozenset[str] = _string_set(
            config.get("disabled_tools"), "disabled_tools"
        )
        self._tools: dict[str, ToolDef] = {}
        if toolset_map is not None:
            self.register_from_config(toolset_map)

    # ── config rule evaluation ────────────────────────────────────────────
    def _toolset_eligible(self, toolset: str) -> bool:
        """True when ``toolset`` passes the allow-list and is not deny-listed."""
        if self._enabled_toolsets is not None and toolset not in self._enabled_toolsets:
            return False
        return toolset not in self._disabled_toolsets

    # ── registration (plan P21.T2 core fix) ───────────────────────────────
    def register_from_config(self, toolset_map: dict[str, list[ToolDef]]) -> None:
        """REBUILD the registry from ``toolset_map`` under the config rules.

        This is the P21.T2 core fix: the registry contents are built ONLY
        from enabled toolsets minus ``disabled_tools`` — disabled tools never
        enter the registry, so they are never advertised to the model and
        never dispatchable. Each ``toolset_map`` key must match its tools'
        :attr:`ToolDef.toolset` (a mismatch is a programming error and raises
        :class:`ValueError`). Calling it again rebuilds from scratch
        (discarding prior contents, including manual registrations).
        """
        for toolset, tools in toolset_map.items():
            for tool in tools:
                if tool.toolset != toolset:
                    raise ValueError(
                        f"toolset_map key {toolset!r} does not match tool "
                        f"{tool.name!r} toolset {tool.toolset!r}"
                    )
        self._tools = {}
        for toolset, tools in toolset_map.items():
            if not self._toolset_eligible(toolset):
                continue  # disabled/not-enabled toolset: its tools NEVER register
            for tool in tools:
                if tool.name in self._disabled_tools:
                    continue  # disabled tool: never registers (issue #26568)
                self._add(tool)

    def register(self, tool: ToolDef) -> None:
        """Register ``tool`` programmatically, bypassing the config filter.

        Explicit registration seam (tests, plugin code): a disabled tool CAN
        be force-registered here, but enforcement still applies — it reports
        :meth:`is_available` ``False``, stays out of :meth:`list_tools`, and
        is refused by :meth:`wrap_invoker`. Duplicate names raise
        :class:`ValueError`.
        """
        self._add(tool)

    def _add(self, tool: ToolDef) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    # ── lookup / availability ─────────────────────────────────────────────
    def get(self, name: str) -> ToolDef | None:
        """The registered tool for ``name``, or ``None`` when never registered.

        Returns force-registered disabled tools too (introspection seam);
        dispatch decisions must go through :meth:`is_available` /
        :meth:`wrap_invoker`, not :meth:`get`.
        """
        return self._tools.get(name)

    def is_available(self, name: str) -> bool:
        """True only when ``name`` is registered AND enabled by config.

        ``False`` for anything disabled — including a force-registered tool
        whose toolset/name the config disables (the P21.T2 enforcement
        invariant), and for names that were never registered at all.
        """
        tool = self._tools.get(name)
        if tool is None:
            return False
        if name in self._disabled_tools:
            return False
        return self._toolset_eligible(tool.toolset)

    def list_tools(self) -> list[ToolDef]:
        """The tools the model may see and call, sorted by name.

        Disabled tools never appear (they never registered via
        :meth:`register_from_config`; force-registered ones are filtered out
        here) — the acceptance "disabled tools never appear in tool
        registry" from issue #26568.
        """
        return sorted(
            (tool for tool in self._tools.values() if self.is_available(tool.name)),
            key=lambda tool: tool.name,
        )

    # ── call-time defense in depth (issue #26568) ─────────────────────────
    def wrap_invoker(self, invoker: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``invoker`` so disabled/unregistered tools are refused.

        The returned guard has the same call shape as ``invoker``; its FIRST
        positional argument is the tool name. Before dispatching it checks
        :meth:`is_available` and raises :class:`ToolNotAvailableError` when
        the tool is not available — even a model-requested call to a
        disabled tool is refused, never executed (issue #26568 defense in
        depth on top of the registration-time fix).
        """

        def guarded(tool_name: str, /, *args: Any, **kwargs: Any) -> Any:
            if not self.is_available(tool_name):
                if tool_name in self._tools:
                    reason = "disabled by tool configuration"
                else:
                    reason = "not registered"
                raise ToolNotAvailableError(tool_name, reason)
            return invoker(tool_name, *args, **kwargs)

        return guarded
