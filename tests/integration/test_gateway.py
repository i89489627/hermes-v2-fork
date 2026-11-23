"""Integration tests for the hermes-v2 gateway lifecycle.

Covers two gateway-lifecycle fixes:

- plan P21.T8 (issue #156127): shutdown notice retries must abort on network
  loss (host reboot) and the exit must stay clean;
- plan P21.T10 (issue #157239): internal sentinel markers must never leak
  into gateway-delivered messages (see ``test_user_never_sees_sentinel``).
"""

from __future__ import annotations

import logging

import pytest

from hermes.core.streaming import filter_stream
from hermes.gateway.delivery import REASON_NO_REPLY_MARKER, deliver
from hermes.gateway.shutdown import (
    REASON_NETWORK_LOST,
    ShutdownNotifier,
)


@pytest.mark.integration
def test_reboot_scenario_clean(caplog: pytest.LogCaptureFixture) -> None:
    """Host reboot mid-shutdown: notices abort gracefully, exit stays clean.

    Simulated timeline of issue #156127: the gateway tries to tell its peer
    it is shutting down; the first delivery attempt fails because the host is
    already going down; by the time the retry would fire, the network is gone
    entirely. The notifier must stop retrying and report a clean exit — both
    in the flapping phase and in the fully-offline phase.
    """
    send_log: list[str] = []
    network_up = {"value": True}
    fail_first_send = {"count": 1}

    def send_shutdown_notice_to_peer() -> None:
        if fail_first_send["count"] > 0:
            fail_first_send["count"] -= 1
            # The reboot kills the network while this attempt is in flight.
            network_up["value"] = False
            raise ConnectionError("peer unreachable: host is rebooting")
        send_log.append("notice")

    def probe() -> bool:
        return network_up["value"]

    notifier = ShutdownNotifier(
        send_shutdown_notice_to_peer,
        max_attempts=3,
        is_network_available=probe,
        retry_delay_seconds=0.01,
    )

    # Phase 1: network flaps away after the first failed attempt.
    network_up["value"] = True
    with caplog.at_level(logging.DEBUG, logger="hermes.gateway.shutdown"):
        first = notifier.send_shutdown_notice()

    assert first.delivered is False
    assert first.aborted_reason == REASON_NETWORK_LOST
    assert first.attempts == 1  # one failed send, then the network was gone
    assert first.exit_clean is True
    assert send_log == []  # the notice never made it out; nothing re-sent it
    assert "network lost during shutdown; aborting notice retries" in caplog.text

    # Phase 2: the reboot completed — no network at all, no attempts at all.
    network_up["value"] = False
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes.gateway.shutdown"):
        second = ShutdownNotifier(
            send_shutdown_notice_to_peer,
            max_attempts=5,
            is_network_available=probe,
            retry_delay_seconds=0.0,
        ).send_shutdown_notice()

    assert second.delivered is False
    assert second.aborted_reason == REASON_NETWORK_LOST
    assert second.attempts == 0
    assert second.exit_clean is True
    assert send_log == []
    assert "network lost during shutdown" in caplog.text


@pytest.mark.integration
def test_user_never_sees_sentinel() -> None:
    """A streamed turn with a chunk-split sentinel delivers clean text.

    Simulated turn of issue #157239: the model's answer streams in three
    chunks and the ``<NO_REPLY>`` sentinel is broken across the last two.
    The gateway watches the progressive stream through ``filter_stream``
    (what a live-rendering UI would show) and hands the accumulated raw
    output to the delivery gate for the final message; the user-visible
    message contains no sentinel. A follow-up turn that is NO_REPLY-only
    must be suppressed entirely.
    """
    markers = ("<NO_REPLY>", "[NO_REPLY]", "<HERMES_INTERNAL>", "<<INTERNAL_ERROR>>")

    # Turn 1: real answer streamed with the sentinel split across chunks.
    chunks = [
        "Working on it... done. ",
        "Here is your answer: 7\n<NO",
        "_REPLY>",
    ]
    streamed_view = "".join(filter_stream(chunks))
    raw_output = "".join(chunks)

    sink_output: list[str] = []
    result = deliver(raw_output, sink_output.append)

    assert result.delivered is True
    assert result.reason is None
    assert sink_output == ["Working on it... done. Here is your answer: 7"]
    for marker in markers:
        assert marker.lower() not in sink_output[0].lower()
    # The progressive stream view was just as clean (modulo end trimming).
    for marker in markers:
        assert marker.lower() not in streamed_view.lower()
    assert streamed_view.strip() == sink_output[0]

    # Turn 2: the model declines via the sentinel (split across chunks too);
    # nothing user-facing remains, so nothing is delivered.
    decline_chunks = ["<NO", "_REPLY>"]
    decline_view = "".join(filter_stream(decline_chunks))

    decline_sink: list[str] = []
    decline_result = deliver("".join(decline_chunks), decline_sink.append)

    assert decline_result.delivered is False
    assert decline_result.reason == REASON_NO_REPLY_MARKER
    assert decline_sink == []  # the user sees nothing at all
    assert decline_view.strip() == ""  # and the stream view carried no text
