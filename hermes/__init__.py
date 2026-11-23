"""hermes-v2 — memory-first re-architecture of Hermes Agent (plan v2.1-r7).

New top-level package holding the v2 memory pipeline (atomic markdown graph in
Git, trust cards, extractor/reviewer pipeline, CBDC gating, GEPA optimization,
diagnostics). The fork-base code (``run_agent.py``, ``hermes_cli``) stays
untouched and is extended from here behind feature flags.
"""

__version__ = "2.0.0a0"
