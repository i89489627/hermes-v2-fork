"""Unit tests for caller-side MCP argument validation (plan P21.T3, issue #156143).

``validate_arguments`` is the client-side defense that catches malformed or
dropped arguments BEFORE a ``tools/call`` hits the wire, so a server can
never silently execute with schema defaults (the upstream bug).
"""

from __future__ import annotations

import pytest

from hermes.mcp.argument_validator import validate_arguments

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "exact": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "options": {
            "type": "object",
            "properties": {"lang": {"type": "string"}},
            "required": ["lang"],
            "additionalProperties": False,
        },
    },
    "required": ["query", "limit"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_validates_schema() -> None:
    """Valid arguments (nested objects, arrays, unions) produce no errors."""
    assert validate_arguments(_SEARCH_SCHEMA, {"query": "x", "limit": 5}) == []
    full: dict[str, object] = {
        "query": "hermes mcp stdio",
        "limit": 3,
        "ratio": 0.5,
        "exact": True,
        "tags": ["a", "b"],
        "options": {"lang": "en"},
    }
    assert validate_arguments(_SEARCH_SCHEMA, full) == []

    # Distinct JSON types stay distinct: bool is NOT an integer, 1 is not a string.
    assert validate_arguments(
        _SEARCH_SCHEMA, {"query": "x", "limit": 5, "exact": 1}
    ) != []
    assert validate_arguments(
        _SEARCH_SCHEMA, {"query": 1, "limit": 5}
    ) != []

    # Union form: string OR integer accepted.
    schema = {"type": "object", "properties": {"id": {"type": ["string", "integer"]}}}
    assert validate_arguments(schema, {"id": "abc"}) == []
    assert validate_arguments(schema, {"id": 12}) == []
    assert validate_arguments(schema, {"id": 1.5}) != []

    # Nested object violations are caught with a dotted path.
    errors = validate_arguments(
        _SEARCH_SCHEMA, {"query": "x", "limit": 1, "options": {"lang": 3}}
    )
    assert errors and "'options.lang'" in errors[0]

    # Array element violations are caught with an indexed path.
    errors = validate_arguments(
        _SEARCH_SCHEMA, {"query": "x", "limit": 1, "tags": ["ok", 7]}
    )
    assert errors and "'tags[1]'" in errors[0]

    # additionalProperties=false rejects unexpected keys.
    errors = validate_arguments(
        _SEARCH_SCHEMA, {"query": "x", "limit": 1, "surprise": 1}
    )
    assert errors and "'surprise'" in errors[0]

    # Empty schema constrains nothing.
    assert validate_arguments({}, {"anything": [1, 2, 3]}) == []


@pytest.mark.unit
def test_rejects_missing_required() -> None:
    """Missing required arguments are reported by name, at every level."""
    errors = validate_arguments(_SEARCH_SCHEMA, {})
    assert len(errors) == 2
    assert any("'query'" in e for e in errors)
    assert any("'limit'" in e for e in errors)

    errors = validate_arguments(_SEARCH_SCHEMA, {"query": "x"})  # limit missing
    assert errors and "'limit'" in errors[0]

    # Missing nested required argument.
    errors = validate_arguments(
        _SEARCH_SCHEMA, {"query": "x", "limit": 1, "options": {}}
    )
    assert errors and "'options.lang'" in errors[0]

    # Whitespace-only args still satisfy presence (validator is structural).
    errors = validate_arguments(
        _SEARCH_SCHEMA, {"query": " ", "limit": 0}
    )
    assert errors == []
