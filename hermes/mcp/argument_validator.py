"""hermes-v2 MCP caller-side argument validation (plan P21.T3, issue #156143).

Upstream bug: ``tools/call`` arguments were intermittently dropped on the
stdio hop, so the server executed the tool with schema defaults and the
caller saw no error — silent wrong results. Part of the v2 fix is to catch
argument defects BEFORE the request is ever written: the client validates
arguments against the tool's input schema with :func:`validate_arguments`
(see :meth:`hermes.mcp.client.McpStdioClient.request`).

**Supported JSON-Schema subset** (documented contract; anything outside it is
ignored rather than guessed at):

- ``type``: ``"string"``, ``"number"``, ``"integer"``, ``"boolean"``,
  ``"object"``, ``"array"`` — as a string or a list (union) of these;
- ``properties``: mapping of name -> subschema, validated recursively;
- ``required``: list of property names that must be present;
- ``additionalProperties``: when exactly ``False``, keys not listed in
  ``properties`` are rejected;
- ``items``: subschema applied to every element of a validated array.

Unsupported ``type`` names (e.g. ``"null"``) and unknown schema keywords
(``enum``, ``format``, ``oneOf``, ...) produce no constraint. Note that JSON
``true`` is an ``integer`` in Python; boolean checks exclude it explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = ["SUPPORTED_TYPES", "validate_arguments"]

#: JSON-Schema type names understood by this validator (documented subset).
SUPPORTED_TYPES: Final[frozenset[str]] = frozenset(
    {"string", "number", "integer", "boolean", "object", "array"}
)


def _type_matches(expected: str, value: Any) -> bool:
    """JSON-Schema type check for one supported type name.

    ``bool`` is excluded from ``integer``/``number``: Python subclasses bool
    from int, but JSON distinguishes ``true`` from ``1``.
    """
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False  # unsupported type names match nothing (documented subset)


def _qualified_name(prefix: str, name: str) -> str:
    """Dotted argument path, e.g. ``user.tags[0]``; top level has no dot."""
    return f"{prefix}.{name}" if prefix else name


def _validate_object(
    prefix: str, schema: Mapping[str, Any], value: Mapping[str, Any], errors: list[str]
) -> None:
    """Validate one object level: required, additionalProperties, properties."""
    properties_raw = schema.get("properties")
    properties: Mapping[str, Any] = (
        properties_raw if isinstance(properties_raw, Mapping) else {}
    )
    required_raw = schema.get("required")
    required: Sequence[Any] = (
        required_raw if isinstance(required_raw, (list, tuple)) else ()
    )

    for name in required:
        if isinstance(name, str) and name not in value:
            errors.append(
                f"missing required argument {_qualified_name(prefix, name)!r}"
            )

    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                errors.append(
                    f"unexpected argument {_qualified_name(prefix, name)!r} "
                    f"(schema sets additionalProperties=false at {prefix or 'top level'})"
                )

    for name, subschema in properties.items():
        if name in value and isinstance(subschema, Mapping):
            _validate_value(_qualified_name(prefix, name), subschema, value[name], errors)


def _validate_value(
    path: str, schema: Mapping[str, Any], value: Any, errors: list[str]
) -> None:
    """Validate one value against its subschema; recurse on objects/arrays."""
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if expected_type not in SUPPORTED_TYPES:
            errors.append(
                f"argument {path!r} uses unsupported schema type {expected_type!r}"
            )
            return
        if not _type_matches(expected_type, value):
            errors.append(
                f"argument {path!r} expected type {expected_type!r}, "
                f"got {type(value).__name__!r}"
            )
            return
    elif isinstance(expected_type, (list, tuple)):
        # Union form, e.g. ["string", "integer"]: at least one must match.
        supported = [t for t in expected_type if t in SUPPORTED_TYPES]
        if supported and not any(_type_matches(t, value) for t in supported):
            errors.append(
                f"argument {path!r} expected one of type {supported!r}, "
                f"got {type(value).__name__!r}"
            )
            return

    if isinstance(value, Mapping):
        _validate_object(path, schema, value, errors)
    elif isinstance(value, (list, tuple)):
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for index, element in enumerate(value):
                _validate_value(f"{path}[{index}]", items_schema, element, errors)


def validate_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> list[str]:
    """Validate ``tools/call`` arguments against a tool's input schema.

    Returns the list of human-readable errors — empty means valid. Never
    raises for schema/argument content (an unusable schema simply constrains
    nothing), so a caller can always distinguish "arguments were checked"
    from "check crashed" via the returned list.
    """
    errors: list[str] = []
    _validate_object("", schema, arguments, errors)
    return errors
