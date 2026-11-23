"""Unit tests for shutdown notification under network loss (plan P21.T8,
issue #156127).

Upstream bug: on host reboot the gateway's shutdown notice kept retrying
after the network was lost, and the shutdown watchdog then forced an unclean
exit. The v2 notifier aborts retries the moment the network is gone and
never lets an exception (including :class:`ShutdownWatchdogError`) escape —
every path returns a clean result.
"""

from __future__ import annotations

import logging

import pytest

from hermes.gateway.shutdown import (
    REASON_DELIVERY_FAILED,
    REASON_NETWORK_LOST,
    REASON_WATCHDOG_SUPPRESSED,
    ShutdownNotifier,
    ShutdownResult,
    ShutdownWatchdogError,
)


@pytest.mark.unit
def test_aborts_on_network_loss(caplog: pytest.LogCaptureFixture) -> None:
    """Network loss aborts immediately: no retry, no further delivery."""
    send_calls: list[int] = []

    def record_send() -> None:
        send_calls.append(1)

    # 1. Network gone from the start: send must never even be called.
    notifier = ShutdownNotifier(
        record_send,
        max_attempts=3,
        is_network_available=lambda: False,
        retry_delay_seconds=0.0,
    )
    with caplog.at_level(logging.DEBUG, logger="hermes.gateway.shutdown"):
        result = notifier.send_shutdown_notice()

    assert isinstance(result, ShutdownResult)
    assert send_calls == []  # abort before the first delivery attempt
    assert result.delivered is False
    assert result.aborted_reason == REASON_NETWORK_LOST
    assert result.attempts == 0
    assert result.exit_clean is True
    assert "network lost during shutdown; aborting notice retries" in caplog.text

    # 2. Network flaps mid-shutdown: one delivery attempt happened, then the
    #    probe reports loss and the retries stop right there.
    flappy_sends: list[int] = []
    states = iter([True, False, False, False])

    def send_once_then_lose_network() -> None:
        flappy_sends.append(1)
        raise ConnectionError("connection reset during host reboot")

    flappy = ShutdownNotifier(
        send_once_then_lose_network,
        max_attempts=3,
        is_network_available=lambda: next(states),
        retry_delay_seconds=0.0,
    )
    flappy_result = flappy.send_shutdown_notice()

    assert flappy_sends == [1]  # exactly one attempt, no retry after the loss
    assert flappy_result.delivered is False
    assert flappy_result.aborted_reason == REASON_NETWORK_LOST
    assert flappy_result.attempts == 1
    assert flappy_result.exit_clean is True


@pytest.mark.unit
def test_clean_exit_no_network(caplog: pytest.LogCaptureFixture) -> None:
    """Every adversarial path still returns a clean result and raises nothing."""
    # 1. Network gone for the whole shutdown: send is never called, exit clean.
    def must_not_send() -> None:
        raise AssertionError("send must never be called without a network")

    notifier = ShutdownNotifier(
        must_not_send,
        max_attempts=5,
        is_network_available=lambda: False,
        clock=lambda: 12345.0,
        retry_delay_seconds=0.0,
    )
    result = notifier.send_shutdown_notice()
    assert result.delivered is False
    assert result.aborted_reason == REASON_NETWORK_LOST
    assert result.attempts == 0
    assert result.exit_clean is True

    # 2. The shutdown watchdog demands an unclean exit — it must never escape.
    def watchdog_send() -> None:
        raise ShutdownWatchdogError("watchdog demands unclean exit")

    watchdog = ShutdownNotifier(
        watchdog_send,
        max_attempts=2,
        is_network_available=lambda: True,
        retry_delay_seconds=0.0,
    )
    with caplog.at_level(logging.DEBUG, logger="hermes.gateway.shutdown"):
        watchdog_result = watchdog.send_shutdown_notice()

    assert watchdog_result.delivered is False
    assert watchdog_result.aborted_reason == REASON_WATCHDOG_SUPPRESSED
    assert watchdog_result.attempts == 1
    assert watchdog_result.exit_clean is True
    assert "shutdown watchdog raised" in caplog.text
    assert "suppressed for a clean exit" in caplog.text

    # 3. Plain delivery failure exhausting bounded retries: still clean.
    def always_failing_send() -> None:
        raise OSError("peer unreachable")

    failing = ShutdownNotifier(
        always_failing_send,
        max_attempts=3,
        is_network_available=lambda: True,
        retry_delay_seconds=0.0,
    )
    failing_result = failing.send_shutdown_notice()
    assert failing_result.delivered is False
    assert failing_result.aborted_reason == REASON_DELIVERY_FAILED
    assert failing_result.attempts == 3
    assert failing_result.exit_clean is True

    # 4. Happy path sanity: delivery succeeds on a live network.
    delivered_sends: list[int] = []

    def working_send() -> None:
        delivered_sends.append(1)

    healthy = ShutdownNotifier(
        working_send,
        max_attempts=3,
        is_network_available=lambda: True,
        retry_delay_seconds=0.0,
    )
    healthy_result = healthy.send_shutdown_notice()
    assert healthy_result.delivered is True
    assert healthy_result.aborted_reason is None
    assert healthy_result.attempts == 1
    assert healthy_result.exit_clean is True
    assert delivered_sends == [1]

    # 5. A broken (raising) probe is fail-closed: treated as network loss.
    def broken_probe() -> bool:
        raise RuntimeError("probe blew up")

    broken = ShutdownNotifier(
        working_send,
        max_attempts=3,
        is_network_available=broken_probe,
        retry_delay_seconds=0.0,
    )
    broken_result = broken.send_shutdown_notice()
    assert broken_result.delivered is False
    assert broken_result.aborted_reason == REASON_NETWORK_LOST
    assert broken_result.exit_clean is True
    assert delivered_sends == [1]  # still nothing delivered
