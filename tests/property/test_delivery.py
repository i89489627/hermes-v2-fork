"""Property tests for gateway delivery (plan P21.T10, issue #157239).

For arbitrary text possibly containing internal markers (any letter case,
split across arbitrary stream chunk boundaries), the delivered output never
contains a marker substring (checked case-insensitively) and delivery never
raises.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes.core.streaming import INTERNAL_MARKERS, filter_stream
from hermes.gateway.delivery import REASON_EMPTY, REASON_NO_REPLY_MARKER, deliver

# Case-mangled spellings of every registered marker: the filter must catch
# all of them, so the property feeds the nastiest variants it can build.
_marker_variants: list[str] = []
for _marker in sorted(INTERNAL_MARKERS):
    for _variant in {_marker, _marker.lower(), _marker.upper(), _marker.title()}:
        _marker_variants.append(_variant)

# Fragments: whole markers plus arbitrary text drawn from an alphabet biased
# toward bracket/no-reply characters, to stress seam and fusion behavior.
_fragment = st.one_of(
    st.sampled_from(_marker_variants),
    st.text(alphabet="<>[]{}()NO_RPLYno_rply \t\nabcXYZ,.-", max_size=15),
)
_messages = st.lists(_fragment, min_size=0, max_size=10).map("".join)


def _assert_marker_free(output: str) -> None:
    lowered = output.lower()
    for marker in INTERNAL_MARKERS:
        assert marker.lower() not in lowered, f"leaked marker {marker!r} in {output!r}"


def _split_into_chunks(message: str, data: st.DataObject) -> list[str]:
    """Cut ``message`` into 1..7 character chunks at hypothesis-chosen seams."""
    chunks: list[str] = []
    position = 0
    while position < len(message):
        step = data.draw(st.integers(min_value=1, max_value=7))
        chunks.append(message[position : position + step])
        position += step
    return chunks


@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(data=st.data(), message=_messages)
def test_no_internal_markers_in_output(data: st.DataObject, message: str) -> None:
    """Delivered output never contains an internal marker substring.

    Both delivery paths are exercised per example: whole-message delivery and
    streaming delivery through arbitrary chunk splits (markers broken across
    chunks must still be caught). Any exception escapes the test and fails it,
    so "never raises" is asserted by construction.
    """
    # 1. Whole-message delivery.
    sink_output: list[str] = []
    result = deliver(message, sink_output.append)
    assert result.delivered == bool(sink_output)
    if result.delivered:
        assert result.reason is None
    else:
        assert result.reason in {REASON_NO_REPLY_MARKER, REASON_EMPTY}
    for output in sink_output:
        _assert_marker_free(output)

    # 2. The same message streamed through arbitrary chunk boundaries.
    streamed = "".join(filter_stream(_split_into_chunks(message, data)))
    streamed_sink: list[str] = []
    streamed_result = deliver(streamed, streamed_sink.append)
    assert streamed_result.delivered == bool(streamed_sink)
    for output in streamed_sink:
        _assert_marker_free(output)
    # And the filter's own output is marker-free even before the delivery gate.
    _assert_marker_free(streamed)
