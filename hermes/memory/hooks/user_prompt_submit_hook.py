"""UserPromptSubmit pre-step hook: per-turn CBDC injection (plan P4.T4).

The pre-step hook point fired by
:meth:`hermes.core.agent.V2Agent.run_pre_step_hooks` before every model call
(the UserPromptSubmit equivalent): the hook runs the Phase 4 injection chain
— BM25 retrieval (P4.T1) -> batched CBDC gating (P4.T2) -> composition
(P4.T3 :class:`~hermes.memory.injector.ContextInjector`) — and returns the
``additionalContext`` payload merged into the model call. Registered via
:func:`hermes.core.agent.install_prompt_injection` (the install twin of
:func:`hermes.core.agent.install_memory_pipeline`).

Payload contract (documented keys, tolerant of missing keys)
------------------------------------------------------------------------------
The hook receives the pre-step payload dict:

- ``"user_prompt"``: str — the submitted prompt; missing or non-str degrades
  to ``""`` (an empty turn retrieves nothing, so the hook then returns ``{}``
  and the prompt is not polluted).
- ``"session_metadata"``: dict — context-detection input for the stable block
  (``cwd`` / ``gateway_chat_id`` / ``explicit``; see
  :meth:`~hermes.memory.context.ContextManager.detect_context`); missing or
  non-dict degrades to ``{}`` (detection then falls back to ``global``).
- ``"context"``: str — explicit active-context override; missing/blank means
  "use the context detected from ``session_metadata``".
- ``"session_id"``: str (optional) — session identity for the stable-context
  cache; falls back to ``session_metadata["session_id"]`` then ``"default"``.

Stable/dynamic split (plan P4.T3 session contract)
------------------------------------------------------------------------------
The stable block (global + active-context ``MEMORY.md``) must be loaded ONCE
per session and then handed back byte-identically on every turn (prefix cache
preservation). The hook owns that contract for the pre-step path: it caches
one :class:`~hermes.memory.injector.StableContextPayload` per session id
(bounded; oldest entry evicted) and never re-reads the stable documents
within a session — a new session id starts a new session boundary and
reloads.

Failure + latency contract (plan Phase 4 acceptance / plan Section 5.5)
------------------------------------------------------------------------------
- The injection runs on a bounded worker thread: the hook joins it for
  ``timeout_seconds`` (default 5.0 — the plan's hard hook budget). On
  timeout the hook logs a warning, abandons the (daemon) worker and returns
  ``{}`` — the turn proceeds without injected context and NEVER crashes.
- An injector failure (any exception inside the worker, e.g. an unsafe
  context name from :meth:`ContextInjector.build_dynamic_context`) is logged
  and degrades to ``{}`` — never raised into the turn.
- When the gate injected nothing (no ``use`` and no ``needs_verification``
  entries) the hook returns ``{}`` WITHOUT formatting — an empty wrapper
  must not pollute the prompt (plan P4.T4: "No injection when no relevant
  facts").
- The hook itself never raises: any unexpected failure is contained by the
  top-level guard (same advisory policy as
  :class:`hermes.memory.hooks.stop_hook.StopHook`).

Trace capture (plan P11.T2, additive)
------------------------------------------------------------------------------
With ``trace_store`` injected, a completed injection chain RECORDS its gating
+ injection stage summaries via
:func:`hermes.memory.pipeline.record_turn_injection` — the pipeline merges
them into the turn's single trace when it appends it (one trace per turn; the
hook never appends the turn trace itself). Recording happens only when the
injector chain actually RAN (a timeout/failure leaves no record, so the
turn's trace keeps ``gating``/``injection`` ``None``) and is failure-safe:
a trace-capture error is logged and never breaks the turn.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from hermes.memory.injector import format_additional_context

if TYPE_CHECKING:
    from hermes.memory.injector import ContextInjector, InjectedContext, StableContextPayload
    from hermes.traces.store import TraceStore

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_HOOK_TIMEOUT_SECONDS", "PromptSubmitHook", "build_prompt_submit_hook"]

#: Plan Phase 4 acceptance: hard hook budget of 5s (typical <2s); the gate
#: itself is bounded at 3.5s inside the injector, so the hook budget only
#: ever guards provider stalls.
DEFAULT_HOOK_TIMEOUT_SECONDS: float = 5.0

#: Stable-payload cache bound (per distinct session id; oldest evicted).
_STABLE_CACHE_LIMIT = 32

#: Payload defaults for missing keys (documented in the module docstring).
_DEFAULT_SESSION_ID = "default"


def _payload_str(payload: dict[str, Any], key: str) -> str | None:
    """Tolerant string extraction: missing/non-str/blank -> ``None``."""
    value = payload.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _has_injected_facts(injected: InjectedContext) -> bool:
    """True when the gate injected at least one claim or verification entry."""
    dynamic = injected.get("dynamic")
    if not isinstance(dynamic, dict):
        return False
    claims = dynamic.get("claims")
    verification = dynamic.get("needs_verification")
    return bool(
        (isinstance(claims, list) and claims)
        or (isinstance(verification, list) and verification)
    )


class PromptSubmitHook:
    """Callable pre-step hook injecting per-turn memory context (P4.T4).

    The hook type registered on :class:`hermes.core.agent.V2Agent`: takes
    the pre-step payload, returns the additionalContext contribution (or
    ``None``; ``{}`` and ``None`` both mean "nothing to merge"). See the
    module docstring for the payload, stable-cache and failure contracts.
    """

    def __init__(
        self,
        injector: ContextInjector,
        timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
        trace_store: TraceStore | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        self._injector = injector
        self._timeout_seconds = timeout_seconds
        self._trace_store = trace_store
        self._stable_cache: dict[str, StableContextPayload] = {}

    # ── stable block: loaded ONCE per session (plan P4.T3 contract) ────────
    def _stable_for(
        self, session_id: str, session_metadata: dict[str, Any]
    ) -> StableContextPayload:
        """Cached stable payload for ``session_id``; loads on first sight.

        A bounded per-session cache implements the plan's "stable context
        changes <=1 time per session": within a session the payload (and
        therefore the stable block bytes) never changes; a new session id
        reloads at its boundary. Detection errors from
        :meth:`ContextInjector.load_stable_context` propagate to the hook's
        degrade-to-``{}`` guard (an ``explicit`` context that does not exist
        must stay visible, per the ContextManager contract).
        """
        stable = self._stable_cache.get(session_id)
        if stable is None:
            stable = self._injector.load_stable_context(session_metadata)
            if len(self._stable_cache) >= _STABLE_CACHE_LIMIT:
                self._stable_cache.pop(next(iter(self._stable_cache)))
            self._stable_cache[session_id] = stable
        return stable

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Run the injection chain for one submitted prompt; never raises."""
        try:
            return self._run(payload)
        except Exception:
            logger.exception(
                "user prompt submit hook: injection failed; the turn proceeds "
                "without injected memory context"
            )
            return {}

    def _run(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Payload -> additionalContext contribution (``{}`` when nothing to inject)."""
        prompt = payload.get("user_prompt")
        turn_text = prompt if isinstance(prompt, str) else ""
        metadata = payload.get("session_metadata")
        session_metadata = metadata if isinstance(metadata, dict) else {}
        session_id = (
            _payload_str(payload, "session_id")
            or _payload_str(session_metadata, "session_id")
            or _DEFAULT_SESSION_ID
        )
        stable = self._stable_for(session_id, session_metadata)
        context_name = _payload_str(payload, "context") or stable.context_name

        injected = self._build_with_timeout(
            turn_text=turn_text, context_name=context_name, stable=stable
        )
        if injected is not None and self._trace_store is not None:
            # P11.T2: the injection chain ran to completion — record its
            # gating/injection stage summaries for the turn's single trace.
            self._record_injection_trace(
                self._trace_store, session_id, context_name, injected
            )
        if injected is None or not _has_injected_facts(injected):
            # Nothing the gate deemed worth injecting: return {} (falsy) so
            # the prompt is never polluted with an empty wrapper (plan P4.T4).
            return {}
        additional_context = format_additional_context(injected)
        if not additional_context:
            return {}
        return {"additionalContext": additional_context}

    def _build_with_timeout(
        self, *, turn_text: str, context_name: str, stable: StableContextPayload
    ) -> InjectedContext | None:
        """Run the injector on a bounded worker thread; ``None`` on timeout/failure.

        The worker is a daemon thread: on timeout the hook abandons it (a
        hung provider call must never block the turn or the interpreter
        exit) and returns ``None``. The worker's result/exception is relayed
        through cells so the turn thread never sees a raise.
        """
        result_cell: dict[str, InjectedContext] = {}
        error_cell: list[BaseException] = []

        def _worker() -> None:
            try:
                result_cell["injected"] = self._injector.build_dynamic_context(
                    turn_text, context_name, stable=stable
                )
            except BaseException as exc:  # relayed to the turn thread below
                error_cell.append(exc)

        worker = threading.Thread(target=_worker, name="prompt-submit-injection", daemon=True)
        worker.start()
        worker.join(self._timeout_seconds)
        if worker.is_alive():
            logger.warning(
                "user prompt submit hook: memory injection timed out after %.2fs; "
                "continuing without injected context",
                self._timeout_seconds,
            )
            return None
        if error_cell:
            logger.warning(
                "user prompt submit hook: memory injection failed (%s); "
                "continuing without injected context",
                error_cell[0],
            )
            return None
        return result_cell.get("injected")

    # ── trace capture (plan P11.T2, additive) ──────────────────────────────
    def _record_injection_trace(
        self,
        trace_store: TraceStore,
        session_id: str,
        context_name: str,
        injected: InjectedContext,
    ) -> None:
        """Record the gating + injection stage summaries for the pending turn.

        Builds the P11.T2 injection record from the payload this hook actually
        received (module docstring): ``gate_model`` (the documented
        ``model_name`` label of the injector's gate model, else
        ``"anonymous"``), ``gating`` (use/verify decision counts +
        ``fallback_used``) and ``injection`` (claims/verification/hot counts,
        injected chars, ``fallback_used``). Handed to
        :func:`hermes.memory.pipeline.record_turn_injection` — the pipeline
        merges it into the turn's SINGLE trace; this hook never appends a
        trace itself. Never raises: a trace-capture failure is logged and the
        turn proceeds (plan Section 5.5).
        """
        try:
            from hermes.memory.pipeline import model_label, record_turn_injection

            dynamic_raw = injected.get("dynamic")
            dynamic = dynamic_raw if isinstance(dynamic_raw, dict) else {}
            claims = dynamic.get("claims")
            claims_list = claims if isinstance(claims, list) else []
            verification = dynamic.get("needs_verification")
            verification_list = verification if isinstance(verification, list) else []
            hot = dynamic.get("hot_memory")
            hot_list = hot if isinstance(hot, list) else []
            budget_raw = dynamic.get("budget_used")
            budget = budget_raw if isinstance(budget_raw, dict) else {}
            fallback_used = bool(dynamic.get("fallback_used", False))
            hot_use_count = sum(
                1 for item in hot_list if isinstance(item, dict) and not item.get(
                    "needs_verification"
                )
            )
            gate = getattr(self._injector, "_gate", None)
            record_turn_injection(
                trace_store,
                session_id,
                {
                    "gate_model": model_label(
                        getattr(gate, "_gate_model", None)
                    ),
                    "gating": {
                        "context": context_name,
                        "decisions_by_type": {
                            "use": len(claims_list) + hot_use_count,
                            "verify": len(verification_list),
                        },
                        "fallback_used": fallback_used,
                    },
                    "injection": {
                        "claims": len(claims_list),
                        "needs_verification": len(verification_list),
                        "hot_memory": len(hot_list),
                        "chars": budget.get("chars", 0),
                        "fallback_used": fallback_used,
                    },
                },
            )
        except Exception:
            logger.warning(
                "user prompt submit hook: injection trace capture failed", exc_info=True
            )


def build_prompt_submit_hook(
    injector: ContextInjector,
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    trace_store: TraceStore | None = None,
) -> PromptSubmitHook:
    """Build the pre-step UserPromptSubmit hook for ``injector`` (plan P4.T4).

    Returns a :class:`PromptSubmitHook` — the ``Callable[[dict], dict | None]``
    pre-step hook contract: tolerant of missing payload keys (module
    docstring), bounded by ``timeout_seconds`` (default 5.0), returning
    ``{"additionalContext": ...}`` only when the gate injected facts and
    ``{}`` otherwise. The hook never raises. ``trace_store`` (plan P11.T2,
    additive) enables the injection-stage trace capture: a completed
    injection chain records its gating/injection summaries for the pipeline's
    single per-turn trace (module docstring).
    """
    return PromptSubmitHook(injector, timeout_seconds, trace_store)
