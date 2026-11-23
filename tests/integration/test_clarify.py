"""Integration tests for clarify prompts over a simulated gateway turn
(plan P21.T7, issue #156171).

These drive :func:`hermes.gateway.clarify.ask_clarify` the way a platform
adapter would: the responder owns the wait and returns ``None`` only once the
deadline passed with no submission. Real (small-scale) wall-clock timing and
pytest's ``caplog`` are used so the *logged* story matches what operators see.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from hermes.gateway.clarify import ClarifyOutcome, ask_clarify

_CLARIFY_LOGGER = "hermes.gateway.clarify"


@contextmanager
def _capture_clarify_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    with caplog.at_level(logging.DEBUG, logger=_CLARIFY_LOGGER):
        yield


@pytest.mark.integration
def test_empty_submission_prompts_again(caplog: pytest.LogCaptureFixture) -> None:
    """First submission empty -> the prompt is retried, then answered.

    Simulated turn: the user taps submit on an empty free-text box, the
    gateway re-prompts, and the second submission carries real text. The
    result is ANSWERED with the second submission — and the empty path's log
    story contains no timeout wording at all (issue #156171).
    """
    submissions = iter(["", "yes"])

    def responder(_prompt: str) -> str | None:
        return next(submissions)

    with _capture_clarify_logs(caplog):
        result = ask_clarify(
            "Proceed with the default workspace?",
            responder,
            timeout_seconds=5.0,
            max_attempts=2,
        )

    assert result.outcome is ClarifyOutcome.ANSWERED
    assert result.text == "yes"
    assert result.attempts == 2
    assert result.timed_out_at is None

    lowered = caplog.text.lower()
    # The blank first submission is visible and explicitly NOT a timeout.
    assert "empty submission" in lowered
    assert "prompting again" in lowered
    assert "timeout" not in lowered
    assert "did not respond" not in lowered


@pytest.mark.integration
def test_timeout_logs_correctly(caplog: pytest.LogCaptureFixture) -> None:
    """A user who never submits produces TIMEOUT with a timeout-flavored log.

    The responder blocks past the (small) deadline, then reports the
    no-response, exactly like a real wait-for-input adapter.
    """
    deadline = time.monotonic() + 0.05

    def never_answers(_prompt: str) -> str | None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining + 0.01)
        return None

    with _capture_clarify_logs(caplog):
        result = ask_clarify(
            "Which profile should I use?",
            never_answers,
            timeout_seconds=0.05,
            max_attempts=2,
        )

    assert result.outcome is ClarifyOutcome.TIMEOUT
    assert result.text is None
    assert result.attempts == 1  # the user is gone; no re-prompt
    assert result.timed_out_at is not None
    assert result.timed_out_at >= deadline - 0.005

    lowered = caplog.text.lower()
    assert "timeout" in lowered
    assert "no response" in lowered
    # And the timeout path never claims an empty submission happened.
    assert "empty submission" not in lowered
