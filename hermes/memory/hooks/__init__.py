"""hermes-v2 hook adapters: memory components wired into agent hook points.

Currently: the Stop-event post-turn hook (:mod:`hermes.memory.hooks.stop_hook`,
plan P3.T5), registered on :class:`hermes.core.agent.V2Agent` via
:func:`hermes.core.agent.install_memory_pipeline`. The UserPromptSubmit
pre-step hook (plan P4.T4) lands in its own module here later.
"""

from hermes.memory.hooks.stop_hook import StopHook, build_stop_hook

__all__ = ["StopHook", "build_stop_hook"]
