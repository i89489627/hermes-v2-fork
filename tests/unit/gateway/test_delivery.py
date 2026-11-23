"""Unit tests for the gateway delivery gate (plan P21.T10, issue #157239).

Upstream bug: internal sentinel markers (e.g. ``<NO_REPLY>``) leaked into
gateway-delivered messages. The v2 delivery gate strips every registered
marker before delivery and suppresses NO_REPLY-only messages entirely.
"""

from __future__ import annotations

import pytest

from hermes.core.streaming import INTERNAL_MARKERS, strip_internal_markers
from hermes.gateway.delivery import REASON_NO_REPLY_MARKER, DeliveryResult, deliver


@pytest.mark.unit
def test_no_reply_marker_stripped() -> None:
    """A trailing NO_REPLY sentinel is stripped; a NO_REPLY-only message is
    suppressed entirely (sink never called)."""
    # 1. Real answer with the sentinel appended: answer delivered, marker gone.
    sink_output: list[str] = []
    result = deliver("Here is the answer: 42.\n<NO_REPLY>", sink_output.append)

    assert isinstance(result, DeliveryResult)
    assert result.delivered is True
    assert result.reason is None
    assert sink_output == ["Here is the answer: 42."]
    assert "<NO_REPLY>" not in sink_output[0]

    # 2. NO_REPLY-only message: that is the sentinel's whole point — suppress.
    suppressed_sink: list[str] = []
    suppressed = deliver("<NO_REPLY>", suppressed_sink.append)

    assert suppressed.delivered is False
    assert suppressed.reason == REASON_NO_REPLY_MARKER
    assert suppressed_sink == []  # the sink was never called

    # 3. Whitespace-padded and case-mangled NO_REPLY-only variants too.
    for variant in ("[no_reply]", "  <No_RePlY>  ", "\n[NO_REPLY]\n"):
        sink: list[str] = []
        outcome = deliver(variant, sink.append)
        assert outcome.delivered is False, variant
        assert outcome.reason == REASON_NO_REPLY_MARKER, variant
        assert sink == [], variant


@pytest.mark.unit
def test_internal_sentinels_filtered() -> None:
    """Every registered sentinel is removed — any case, any position — and
    the leftover whitespace/bracket artifacts are cleaned up."""
    message = (
        "alpha <HERMES_INTERNAL> beta [No_RePlY] gamma\n"
        "<<internal_error>> delta\n"
        "<<HERMES_INTERNAL>>"  # nested spelling -> would leave "<>" behind
    )
    sink_output: list[str] = []
    result = deliver(message, sink_output.append)

    assert result.delivered is True
    assert result.reason is None
    assert len(sink_output) == 1
    delivered = sink_output[0]

    # No registered marker survives, in any casing.
    lowered = delivered.lower()
    for marker in INTERNAL_MARKERS:
        assert marker.lower() not in lowered, marker

    # The real content survived; artifacts of removal were cleaned up.
    for word in ("alpha", "beta", "gamma", "delta"):
        assert word in delivered
    assert "  " not in delivered  # doubled whitespace compacted
    assert "<>" not in delivered  # empty bracket leftovers removed
    assert "[]" not in delivered

    # The strip function itself agrees (the gate is a thin wrapper over it).
    stripped = strip_internal_markers(message).strip()
    assert stripped == delivered
    stripped_lowered = stripped.lower()
    for marker in INTERNAL_MARKERS:
        assert marker.lower() not in stripped_lowered, marker
