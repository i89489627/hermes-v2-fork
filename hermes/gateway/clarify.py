"""hermes-v2 gateway clarify prompts (plan P21.T7).

Fixes upstream issue #156171: an **empty** free-text submission in a clarify
prompt was misreported as a timeout ("user did not respond within Nm"). An
empty submission IS a response — valid, if unhelpful — while a timeout is the
absence of any response. The v2 fix separates the two outcomes explicitly:

- :attr:`ClarifyOutcome.ANSWERED` — the responder returned usable free text.
- :attr:`ClarifyOutcome.EMPTY_SUBMISSION` — the responder *submitted* a
  response, but it was blank/whitespace. The prompt is retried up to
  ``max_attempts`` times, then the (empty) default is adopted and the caller
  proceeds. This path is logged distinctly and never mentions a timeout.
- :attr:`ClarifyOutcome.TIMEOUT` — the responder reported NO response before
  the deadline (returned ``None``); only this path stamps ``timed_out_at``.

Contract for ``responder`` (a platform adapter's wait-for-input callable):

- returns ``str`` for a user submission (the string MAY be empty — a submit
  of the empty free-text box);
- returns ``None`` ONLY once ``timeout_seconds`` elapsed with no submission;
- is responsible for honoring the deadline itself (it owns the wait), since
  a synchronous callable cannot be preempted from here. ``ask_clarify`` uses
  the injected ``clock`` to stamp ``timed_out_at`` deterministically and to
  report elapsed time in the timeout log line.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ClarifyOutcome", "ClarifyResult", "ClarifyResponder", "ask_clarify"]

logger = logging.getLogger(__name__)

#: A platform wait-for-input callable: prompt text in, submission out.
#: ``None`` means the deadline passed with no submission (issue #156171).
ClarifyResponder = Callable[[str], "str | None"]


class ClarifyOutcome(enum.Enum):
    """Distinct fates of a clarify prompt (issue #156171)."""

    ANSWERED = "answered"
    EMPTY_SUBMISSION = "empty_submission"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ClarifyResult:
    """Outcome of one :func:`ask_clarify` round.

    ``text`` is the adopted free text: the user's submission for ANSWERED,
    the empty default (``""``) for EMPTY_SUBMISSION, and ``None`` for
    TIMEOUT (there was no response to adopt). ``attempts`` counts responder
    invocations. ``timed_out_at`` is the injected-clock stamp of the moment
    the no-response was detected, and is ``None`` unless the outcome is
    TIMEOUT.
    """

    outcome: ClarifyOutcome
    text: str | None
    attempts: int
    timed_out_at: float | None = None


def ask_clarify(
    prompt: str,
    responder: ClarifyResponder,
    *,
    timeout_seconds: float,
    max_attempts: int = 2,
    clock: Callable[[], float] = time.monotonic,
) -> ClarifyResult:
    """Ask the user a clarify question, keeping empty and timeout distinct.

    Loop: call ``responder(prompt)`` up to ``max_attempts`` times.

    - usable text (non-blank submission) -> ANSWERED with that text;
    - blank submission -> logged as an empty submission (distinctly NOT a
      no-response) and retried; after ``max_attempts`` blank submissions the
      empty default is adopted and the caller proceeds with ``""``;
    - ``None`` -> the user never responded within the window: TIMEOUT,
      stamped with ``timed_out_at``. No retry is attempted — the user is
      gone, re-prompting would just burn the deadline again.

    Raises ValueError for ``timeout_seconds <= 0`` or ``max_attempts < 1``.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    started_at = clock()
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        submission = responder(prompt)
        if submission is None:
            timed_out_at = clock()
            logger.warning(
                "clarify prompt: timeout after %d attempt(s); no response within %.2fs "
                "(elapsed %.2fs) [issue #156171]",
                attempts,
                timeout_seconds,
                timed_out_at - started_at,
            )
            return ClarifyResult(
                outcome=ClarifyOutcome.TIMEOUT,
                text=None,
                attempts=attempts,
                timed_out_at=timed_out_at,
            )
        if submission.strip():
            logger.info(
                "clarify prompt answered on attempt %d/%d", attempts, max_attempts
            )
            return ClarifyResult(
                outcome=ClarifyOutcome.ANSWERED, text=submission, attempts=attempts
            )
        # Empty submission: a valid (if unhelpful) response. Log it as such —
        # the log line must stay free of any timeout wording (issue #156171).
        logger.warning(
            "clarify prompt: empty submission on attempt %d/%d; prompting again",
            attempts,
            max_attempts,
        )
    logger.warning(
        "clarify prompt: %d empty submission(s); proceeding with the default empty response",
        attempts,
    )
    return ClarifyResult(
        outcome=ClarifyOutcome.EMPTY_SUBMISSION, text="", attempts=attempts
    )
