"""Property tests: newline-delimited JSON framing roundtrips any arguments
(plan P21.T3, issue #156143).

The wire layer must serialize and re-parse arbitrary JSON-serializable
``tools/call`` arguments losslessly — any loss here is an argument drop.
Roundtrips run through in-memory ``io.BytesIO`` pairs (no subprocesses).
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes.mcp.protocol import FramingError, read_message, write_message

_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_json_values = st.recursive(
    _scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=6,
)
_arguments = st.dictionaries(st.text(min_size=1), _json_values, max_size=6)


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(arguments=_arguments)
def test_roundtrip_any_args(arguments: dict[str, Any]) -> None:
    """write_message -> read_message is the identity for arbitrary arguments."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "echo", "arguments": arguments},
    }
    stream = io.BytesIO()
    write_message(stream, message)
    stream.seek(0)
    assert read_message(stream) == message

    # Two frames back-to-back must stay independent: no leakage across lines.
    second: dict[str, Any] = {"jsonrpc": "2.0", "id": 8, "method": "ping", "params": {}}
    stream = io.BytesIO()
    write_message(stream, message)
    write_message(stream, second)
    stream.seek(0)
    assert read_message(stream) == message
    assert read_message(stream) == second
    with pytest.raises(FramingError):
        read_message(stream)  # EOF is an explicit error, never a silent junk frame
