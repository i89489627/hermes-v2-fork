"""hermes-v2 gateway delivery gate (plan P21.T10).

Fixes upstream issue #157239: internal sentinel markers (e.g.
``"<NO_REPLY>"``) leaked into gateway-delivered messages. This module is the
choke point every user-facing delivery goes through:

- the message is first run through
  :func:`hermes.core.streaming.strip_internal_markers` (the single marker
  registry lives there);
- if nothing user-facing remains — in particular a message that carried only
  a ``NO_REPLY`` sentinel — the delivery is **suppressed entirely**: the
  channel sink is not called and the result explains why
  (``"no-reply-marker"``). That is the whole point of the sentinel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from hermes.core.streaming import contains_internal_marker, strip_internal_markers

__all__ = ["DeliveryResult", "deliver"]

logger = logging.getLogger(__name__)

#: Reason recorded when a NO_REPLY-only message is suppressed (issue #157239).
REASON_NO_REPLY_MARKER = "no-reply-marker"

#: Reason recorded when a message without any marker is blank after stripping.
REASON_EMPTY = "empty"


@dataclass(frozen=True)
class DeliveryResult:
    """Outcome of one :func:`deliver` call.

    ``delivered`` is True exactly when the channel sink was called.
    ``reason`` is ``None`` on a delivered message and otherwise explains the
    suppression (``"no-reply-marker"`` for a NO_REPLY-only message,
    ``"empty"`` for a message that was blank without any marker).
    """

    delivered: bool
    reason: str | None = None


def deliver(message: str, channel_sink: Callable[[str], None]) -> DeliveryResult:
    """Strip internal markers from ``message`` and deliver what remains.

    The cleaned text is handed to ``channel_sink`` once; blank results
    (including NO_REPLY-only messages) suppress the delivery entirely — the
    sink is never called (issue #157239). Sink failures propagate: a broken
    channel is the caller's problem, not a sentinel leak.
    """
    cleaned = strip_internal_markers(message).strip()
    if not cleaned:
        reason = (
            REASON_NO_REPLY_MARKER
            if contains_internal_marker(message)
            else REASON_EMPTY
        )
        logger.info(
            "gateway delivery suppressed (reason=%s, %d chars in, 0 chars out) "
            "[issue #157239]",
            reason,
            len(message),
        )
        return DeliveryResult(delivered=False, reason=reason)
    channel_sink(cleaned)
    return DeliveryResult(delivered=True, reason=None)
