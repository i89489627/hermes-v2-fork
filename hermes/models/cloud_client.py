"""Cloud model availability client (plan P19.T3).

One tiny seam the deprecation detector (``hermes.models.cloud_deprecation``)
probes through: "is this model id still servable, and is the provider
warning about its retirement?".

The default :class:`CloudModelClient` has NO provider wiring by design —
real SDK integration (provider clients, auth, rate limits) is out of scope
for the v2 pipeline core. Callers inject a
``check_availability(model_id) -> AvailabilityResult`` callable that wraps
their provider SDK of choice (e.g. a ``client.models.retrieve`` lookup
mapped onto :class:`AvailabilityResult`); tests inject fakes. The injected
callable raising is treated as a probe failure by the caller, never
swallowed here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["AvailabilityProbe", "AvailabilityResult", "CloudModelClient"]


@dataclass(frozen=True)
class AvailabilityResult:
    """Outcome of one availability probe for ``model_id``.

    ``deprecation_warning`` carries the provider's retirement notice text
    when the model still works but is scheduled to go away;
    ``estimated_removal`` is the provider-stated removal date string (any
    format the provider uses — the detector surfaces it verbatim).
    """

    model_id: str
    available: bool
    deprecation_warning: str | None = None
    estimated_removal: str | None = None


#: Injected probe signature: model id -> availability report.
AvailabilityProbe = Callable[[str], AvailabilityResult]


class CloudModelClient:
    """Availability probe seam with an injectable provider implementation.

    Without an injected probe the client refuses to guess: queries raise
    ``NotImplementedError`` with wiring guidance, so a forgotten dependency
    injection fails loudly instead of silently reporting "available".
    """

    def __init__(self, check_availability: AvailabilityProbe | None = None) -> None:
        self._probe = check_availability

    def check_availability(self, model_id: str) -> AvailabilityResult:
        """Probe ``model_id`` via the injected provider callable.

        Raises ``NotImplementedError`` when no probe was injected.
        """
        if self._probe is None:
            raise NotImplementedError(
                "CloudModelClient has no availability probe configured: inject "
                "check_availability=<callable(model_id) -> AvailabilityResult> that "
                "wraps your provider SDK (real provider wiring is out of scope for "
                "the v2 core, plan P19.T3); tests inject fakes"
            )
        return self._probe(model_id)
