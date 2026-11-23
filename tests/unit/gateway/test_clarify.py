"""Unit tests for clarify-prompt outcome separation (plan P21.T7, issue #156171).

Upstream bug: an empty free-text submission in a clarify prompt was reported
through the timeout path ("user did not respond within Nm"). The v2 contract
treats an empty submission as a response (retry, then proceed with the empty
default) and reserves the timeout outcome for a genuine no-response.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from hermes.gateway.clarify import (
    ClarifyOutcome,
    ClarifyResponder,
    ClarifyResult,
    ask_clarify,
)


class _FakeClock:
    """Deterministic monotonic-ish clock: one call advances one step."""

    def __init__(self, start: float = 100.0, step: float = 0.1) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        current = self.now
        self.now += self.step
        return current


@contextmanager
def _capture_clarify_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    with caplog.at_level(logging.DEBUG, logger="hermes.gateway.clarify"):
        yield


@pytest.mark.unit
def test_empty_submission_not_timeout(caplog: pytest.LogCaptureFixture) -> None:
    """A blank submission is EMPTY_SUBMISSION, never misreported as timeout.

    Regression for issue #156171: the responder DOES submit (empty string),
    so the result must carry EMPTY_SUBMISSION with no ``timed_out_at`` stamp,
    the empty default is adopted after the attempts are exhausted, and no log
    line may report a timeout for this path.
    """
    clock = _FakeClock()
    prompts: list[str] = []

    def always_empty(prompt: str) -> str | None:
        prompts.append(prompt)
        return ""

    with _capture_clarify_logs(caplog):
        result = ask_clarify(
            "Should I apply the default?",
            always_empty,
            timeout_seconds=5.0,
            max_attempts=2,
            clock=clock,
        )

    assert isinstance(result, ClarifyResult)
    assert result.outcome is ClarifyOutcome.EMPTY_SUBMISSION
    assert result.outcome is not ClarifyOutcome.TIMEOUT  # the #156171 regression
    assert result.text == ""  # proceeded with the empty default
    assert result.attempts == 2  # retried once, then defaulted
    assert result.timed_out_at is None  # nothing timed out here

    # Both attempts saw the same prompt.
    assert prompts == ["Should I apply the default?", "Should I apply the default?"]

    # Logged as an empty submission, with zero timeout wording anywhere.
    lowered = caplog.text.lower()
    assert "empty submission" in lowered
    assert "timeout" not in lowered
    assert "did not respond" not in lowered  # the buggy #156171 phrasing

    # Sanity: a genuine no-response on the same inputs IS a timeout, proving
    # the two paths are truly distinct in the implementation.
    with _capture_clarify_logs(caplog):
        timeout_result = ask_clarify(
            "Should I apply the default?",
            lambda _prompt: None,
            timeout_seconds=5.0,
            max_attempts=2,
            clock=clock,
        )
    assert timeout_result.outcome is ClarifyOutcome.TIMEOUT
    assert timeout_result.text is None
    assert timeout_result.timed_out_at is not None


@pytest.mark.unit
def test_timeout_only_on_no_response(caplog: pytest.LogCaptureFixture) -> None:
    """TIMEOUT is reported only when the responder signals a no-response."""
    clock = _FakeClock(step=0.05)

    with _capture_clarify_logs(caplog):
        result = ask_clarify(
            "Need a channel name",
            lambda _prompt: None,  # the user never submitted anything
            timeout_seconds=0.05,
            max_attempts=2,
            clock=clock,
        )

    assert result.outcome is ClarifyOutcome.TIMEOUT
    assert result.text is None  # there is no response to adopt
    assert result.attempts == 1  # no re-prompt after the user vanished
    assert result.timed_out_at is not None

    # The timeout is announced in the log, with the issue's phrasing kept.
    lowered = caplog.text.lower()
    assert "timeout" in lowered
    assert "no response" in lowered

    # And no empty-submission log line was emitted on this path.
    assert "empty submission" not in lowered
