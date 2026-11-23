"""hermes-v2 gateway shutdown notification (plan P21.T8).

Fixes upstream issue #156127: when the host rebooted mid-shutdown, the
gateway kept retrying the shutdown notice through the network loss, and the
shutdown watchdog then forced an *unclean* exit. The v2 fix makes the
shutdown path fail-safe in three ways:

1. **Network loss aborts retries.** Before every delivery attempt the
   notifier probes the network; if it is gone (or the probe itself fails),
   retries are aborted immediately — no retry storm against a dead network.
2. **Bounded delivery retries.** A plain delivery failure is retried at most
   ``max_attempts`` times with a small delay, then given up.
3. **Nothing escapes; the exit stays clean.** :class:`ShutdownWatchdogError`
   (the watchdog demanding an unclean exit) and any other exception raised by
   ``send`` are caught, logged, and folded into the :class:`ShutdownResult` —
   :meth:`ShutdownNotifier.send_shutdown_notice` always returns a result with
   ``exit_clean=True`` and never raises.

The default availability probe is a plain socket connect (1s timeout) to a
configurable endpoint via :func:`make_socket_network_probe`; tests inject
fakes through ``is_network_available`` instead.
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PROBE_HOST",
    "DEFAULT_PROBE_PORT",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "ShutdownNotifier",
    "ShutdownResult",
    "ShutdownWatchdogError",
    "make_socket_network_probe",
]

logger = logging.getLogger(__name__)

#: Default reachability endpoint for the built-in network probe: a public
#: DNS resolver on port 53. Overridable via :func:`make_socket_network_probe`.
DEFAULT_PROBE_HOST = "1.1.1.1"
DEFAULT_PROBE_PORT = 53
DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0

#: Largest single real sleep performed while pacing retries between clock
#: reads (keeps injected fake clocks responsive).
_MAX_SLEEP_STEP = 0.05

#: ``aborted_reason`` for a delivery attempt skipped because the network is
#: unavailable (issue #156127).
REASON_NETWORK_LOST = "network-lost"

#: ``aborted_reason`` when a shutdown watchdog demanded an unclean exit and
#: was suppressed instead (issue #156127).
REASON_WATCHDOG_SUPPRESSED = "watchdog-suppressed"

#: ``aborted_reason`` when every bounded delivery attempt failed on a live
#: network.
REASON_DELIVERY_FAILED = "delivery-failed"


class ShutdownWatchdogError(RuntimeError):
    """Raised by a shutdown watchdog demanding an unclean/forced exit.

    Issue #156127: this error used to escape the shutdown path and force the
    gateway out uncleanly. :meth:`ShutdownNotifier.send_shutdown_notice`
    guarantees it never escapes again.
    """


@dataclass(frozen=True)
class ShutdownResult:
    """Outcome of one :meth:`ShutdownNotifier.send_shutdown_notice` round.

    ``attempts`` counts delivery attempts actually *started* (a network
    check that fails before a send is not an attempt). ``aborted_reason`` is
    ``None`` exactly when ``delivered`` is ``True``. ``exit_clean`` is always
    ``True`` from :meth:`send_shutdown_notice` — the invariant the fix
    establishes; callers use it to select the process exit code.
    """

    delivered: bool
    attempts: int
    aborted_reason: str | None = None
    exit_clean: bool = True


def make_socket_network_probe(
    host: str = DEFAULT_PROBE_HOST,
    port: int = DEFAULT_PROBE_PORT,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> Callable[[], bool]:
    """Build a network-availability probe: a plain outbound socket connect.

    Returns True when ``connect((host, port))`` succeeds within
    ``timeout_seconds``, False on any :class:`OSError` (the usual signal of
    the network being gone, e.g. during a host reboot).
    """

    def probe() -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except OSError:
            return False

    return probe


class ShutdownNotifier:
    """Deliver the shutdown notice with network-loss aborts and clean exits."""

    def __init__(
        self,
        send: Callable[[], None],
        *,
        max_attempts: int = 3,
        is_network_available: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        retry_delay_seconds: float = 0.1,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self._send = send
        self._max_attempts = max_attempts
        self._is_network_available = (
            is_network_available
            if is_network_available is not None
            else make_socket_network_probe()
        )
        self._clock = clock
        self._retry_delay_seconds = retry_delay_seconds

    def send_shutdown_notice(self) -> ShutdownResult:
        """Deliver the notice; abort on network loss; never raise.

        Per attempt: probe the network first (abort immediately with
        ``network-lost`` when it is gone — no retry), then call ``send``.
        Delivery failures retry up to ``max_attempts`` with
        ``retry_delay_seconds`` pacing. :class:`ShutdownWatchdogError` and
        any other exception are caught and logged; every return path is a
        clean exit (issue #156127).
        """
        attempts = 0
        for attempt in range(1, self._max_attempts + 1):
            if not self._network_available():
                logger.warning(
                    "network lost during shutdown; aborting notice retries "
                    "(attempt %d/%d, %d delivery attempt(s) made) [issue #156127]",
                    attempt,
                    self._max_attempts,
                    attempts,
                )
                return ShutdownResult(
                    delivered=False,
                    attempts=attempts,
                    aborted_reason=REASON_NETWORK_LOST,
                    exit_clean=True,
                )
            attempts = attempt
            try:
                self._send()
            except ShutdownWatchdogError:
                logger.error(
                    "shutdown watchdog raised during notice delivery; suppressed for a "
                    "clean exit (attempt %d/%d) [issue #156127]",
                    attempt,
                    self._max_attempts,
                )
                return ShutdownResult(
                    delivered=False,
                    attempts=attempts,
                    aborted_reason=REASON_WATCHDOG_SUPPRESSED,
                    exit_clean=True,
                )
            except Exception:
                logger.warning(
                    "shutdown notice delivery failed on attempt %d/%d; %s",
                    attempt,
                    self._max_attempts,
                    "retrying" if attempt < self._max_attempts else "giving up",
                    exc_info=True,
                )
                if attempt < self._max_attempts:
                    self._pause_before_retry()
                continue
            logger.info(
                "shutdown notice delivered on attempt %d/%d", attempt, self._max_attempts
            )
            return ShutdownResult(
                delivered=True, attempts=attempts, aborted_reason=None, exit_clean=True
            )
        logger.warning(
            "shutdown notice undelivered after %d attempt(s); exiting clean "
            "[issue #156127]",
            attempts,
        )
        return ShutdownResult(
            delivered=False,
            attempts=attempts,
            aborted_reason=REASON_DELIVERY_FAILED,
            exit_clean=True,
        )

    def _network_available(self) -> bool:
        """Probe availability; a failing probe counts as network loss.

        Fail-closed on purpose (issue #156127): a broken probe during a
        reboot must not trigger a retry storm, and skipping the notice still
        exits clean.
        """
        try:
            return bool(self._is_network_available())
        except Exception:
            logger.warning(
                "network availability probe failed during shutdown; treating the "
                "network as lost",
                exc_info=True,
            )
            return False

    def _pause_before_retry(self) -> None:
        """Pace retries by ``retry_delay_seconds`` on the injected clock.

        Sleeps in bounded steps so an injected fake clock that jumps forward
        does no real sleeping at all, and a frozen fake clock cannot spin
        forever.
        """
        delay = self._retry_delay_seconds
        if delay <= 0:
            return
        deadline = self._clock() + delay
        while (remaining := deadline - self._clock()) > 0:
            before = self._clock()
            time.sleep(min(remaining, _MAX_SLEEP_STEP))
            if self._clock() == before:
                # Frozen clock: no amount of sleeping makes progress.
                return
