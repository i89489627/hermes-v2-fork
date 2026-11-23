"""Stop-event post-turn hook: fire the memory pipeline async (plan P3.T5).

Registered via :func:`hermes.core.agent.install_memory_pipeline` so every
completed turn (the Stop equivalent hook point) triggers
:meth:`~hermes.memory.pipeline.MemoryPipeline.run_after_turn` with
``sync=False`` — the pipeline runs on a daemon thread and the hook returns
immediately (plan acceptance: "Stop hook fires pipeline async"; "Pipeline
completes within 10s for typical turn (async)").

Payload contract (documented keys, tolerant of missing keys)
------------------------------------------------------------------------------
The hook receives the completed-turn payload dict from
:meth:`hermes.core.agent.V2Agent.run_post_turn_hooks`:

- ``"turn_id"``: str — the completed turn's id; a missing/empty/non-str
  value is replaced by a synthesized ``"turn-<8 hex>"`` so trace records
  stay distinguishable across turns.
- ``"turn_text"``: str — the user prompt / turn text; missing or non-str
  degrades to ``""`` (extraction then runs on an empty turn and yields no
  drafts).
- ``"session_id"``: str — defaults to ``"default"``.
- ``"context"``: str — active memory context name; defaults to ``"global"``.

Failure contract (plan P3.T5 acceptance / plan Section 5.5)
------------------------------------------------------------------------------
The hook NEVER raises: the whole dispatch is wrapped in a try/except that
logs with ``logger.exception`` and continues — the hook contract is advisory
per :class:`hermes.core.agent.V2Agent` ("hooks are advisory: a failing hook
is logged and skipped, never allowed to break the turn"). A synchronous
dispatch failure (e.g. thread spawn) is therefore contained here; pipeline
failures themselves surface on the returned future (the ``sync=False`` error
channel), never in the turn thread.

Trace capture (plan P11.T2, additive)
------------------------------------------------------------------------------
With ``trace_store`` injected, the hook records a pipeline-run MARKER trace
for the turn — but only when its trace store is NOT the pipeline's own (read
via :attr:`MemoryPipeline.trace_store`): the one-trace-per-turn contract
means the pipeline appends the FULL turn trace (extraction/review/... stage
payloads, plus the pre-step injection record) when it holds this store, and
a hook-side marker would duplicate the turn. The marker exists for the
pipeline-without-trace-store wiring: the turn is still visible on a store
the operator wired into the hook alone. Marker shape: ``turn_id`` /
``session_id`` / ``user_prompt`` / ``models_used=["stop-hook"]`` /
``extraction={"marker": "pipeline-run-dispatched"}``. Marker failures are
logged and never break the dispatch (plan Section 5.5).
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes.memory.pipeline import MemoryPipeline, PipelineResult
    from hermes.traces.store import TraceStore

logger = logging.getLogger(__name__)

__all__ = ["StopHook", "build_stop_hook"]

#: Payload defaults for missing keys (documented in the module docstring).
_DEFAULT_SESSION_ID = "default"
_DEFAULT_CONTEXT = "global"

#: Models_used label of this hook's own pipeline-run MARKER trace (P11.T2 —
#: the hook appends the marker only on a trace store the pipeline itself does
#: not cover; the pipeline's full turn trace is the non-marker capture).
_MARKER_MODEL = "stop-hook"

#: Marker payload key inside the marker trace's ``extraction`` component.
_MARKER_KEY = "pipeline-run-dispatched"


def _payload_str(payload: dict[str, Any], key: str, default: str) -> str:
    """Tolerant string extraction: missing/empty/non-str -> ``default``."""
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


class StopHook:
    """Callable post-turn hook dispatching the memory pipeline async (P3.T5).

    ``futures`` records one :class:`~concurrent.futures.Future` per dispatched
    run, oldest first — the bounded-time test/observability seam for waiting
    on the async pipeline runs (``futures[-1]`` is the most recent turn).
    """

    def __init__(self, pipeline: MemoryPipeline, trace_store: TraceStore | None = None) -> None:
        self._pipeline = pipeline
        self._trace_store = trace_store
        self.futures: list[Future[PipelineResult]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        """Dispatch the pipeline for the completed turn; never raises."""
        try:
            turn_id = payload.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                turn_id = f"turn-{uuid.uuid4().hex[:8]}"
            turn_text = payload.get("turn_text")
            if not isinstance(turn_text, str):
                turn_text = ""
            session_id = _payload_str(payload, "session_id", _DEFAULT_SESSION_ID)
            context = _payload_str(payload, "context", _DEFAULT_CONTEXT)
            dispatched = self._pipeline.run_after_turn(
                turn_id,
                turn_text,
                session_id=session_id,
                context=context,
                sync=False,
            )
            if isinstance(dispatched, Future):  # sync=False -> always a Future
                self.futures.append(dispatched)
            self._record_pipeline_run_marker(turn_id, turn_text, session_id)
            logger.debug("stop hook: async pipeline run dispatched for turn %s", turn_id)
        except Exception:
            # Advisory hook contract (plan P3.T5): a memory pipeline failure
            # must never crash the session.
            logger.exception("stop hook: memory pipeline dispatch failed; session continues")

    # ── trace capture (plan P11.T2, additive) ──────────────────────────────
    def _record_pipeline_run_marker(
        self, turn_id: str, turn_text: str, session_id: str
    ) -> None:
        """Append the pipeline-run marker trace (plan P11.T2; never raises).

        Skipped when this hook's trace store IS the pipeline's own: the
        pipeline appends the full per-turn trace on that store and a marker
        would duplicate the turn (one-trace-per-turn contract, module
        docstring). A marker append failure is logged and the dispatch stays
        complete (plan Section 5.5).
        """
        if self._trace_store is None:
            return
        if self._pipeline.trace_store is self._trace_store:
            return  # the pipeline's full trace covers this turn on this store
        try:
            from hermes.traces.schema import Trace

            self._trace_store.append(
                Trace(
                    turn_id=turn_id,
                    session_id=session_id,
                    user_prompt=turn_text,
                    models_used=[_MARKER_MODEL],
                    extraction={"marker": _MARKER_KEY},
                )
            )
        except Exception:
            logger.warning(
                "stop hook: pipeline-run marker trace capture failed", exc_info=True
            )


def build_stop_hook(
    pipeline: MemoryPipeline, trace_store: TraceStore | None = None
) -> StopHook:
    """Build the post-turn Stop hook for ``pipeline`` (plan P3.T5).

    Returns a :class:`StopHook` — the ``Callable[[dict], None]`` post-turn
    hook contract: tolerant of missing payload keys (see the module
    docstring), dispatching ``run_after_turn(..., sync=False)`` and never
    raising. ``trace_store`` (plan P11.T2, additive) enables the pipeline-run
    marker capture on stores the pipeline itself does not trace (module
    docstring).
    """
    return StopHook(pipeline, trace_store)
