"""hermes-v2 session state persistence (plan P21.T12).

Holds the durable session store used by the v2 ACP server: atomic JSON
session files under ``<hermes_home>/sessions`` so a turn can be recovered
(or at least correctly reported) after an editor crash (issue #156173).
"""
