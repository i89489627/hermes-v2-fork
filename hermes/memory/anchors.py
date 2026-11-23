"""Automatic ground-truth anchors — rule-based test/build hooks (plan P2.T5).

Implements the plan Phase 2 "who verifies the verifier" element: when a test
suite or build completes, the memory entries RELATED to that run are anchored
automatically — a verification event is appended to every related entry's
trust card without any manual ``verify`` call. Matching is deliberately
RULE-BASED, never LLM-based (plan Phase 2: ``hermes/memory/anchors.py``).

Relatedness rule — an entry is RELATED when any of the following holds:

(a) **Session citation**: the entry's id is a member of the session citation
    set, or a citation string occurs in the entry's text, or the citation set
    names one of the entry's tags.
(b) **Changed-file overlap**: the entry's text or tags mention a path that
    overlaps one of the run's changed files under normalized path-suffix
    matching on segment boundaries (``memory/store.py`` matches
    ``hermes/memory/store.py`` but NOT ``hermes/memory/soul_store.py``).
(c) **Test-module tag**: the entry carries a tag equal to a test module slug
    of the run (``tests/unit/memory/test_anchors.py`` -> ``test_anchors``).

Event mapping (IMPORTANT DEVIATION NOTE — see also
:mod:`hermes.memory.verification`)
-----------------------------------------------------------------------
The plan's verification-source literal set
(:data:`hermes.memory.types.VerificationSource`) has no ``test-red`` member,
so a RED (failing) run cannot use its natural source. Red is recorded as the
P1.T1 NEGATIVE anchor — source ``tool-output``, weight 0.85, ``value=0.0``
(plan Section 2.4 already models ``value=0.0`` as the negative-anchor
verdict) — with the event ``evidence`` prefixed ``test-red:`` so the origin
stays machine-readable in the trust card. Green uses the native
``test-green`` source (weight 1.0, value 1.0). Unrelated entries receive NO
event: an anchor run never touches entries it cannot tie to the run
(plan P2.T5 acceptance criterion).

Agent wiring
------------
:func:`build_command_hook` returns the PostToolUse-equivalent callable the
agent registers via
:meth:`hermes.core.agent.V2Agent.register_command_execution_hook` and the
agent fires through :meth:`hermes.core.agent.V2Agent.on_command_execution`.
The hook accepts the completed command result (a mapping or any object
carrying ``command`` / ``exit_code`` / optional ``changed_files`` /
``session_cited_entry_ids``), decides whether the command looks like a test
run (pytest / make test / npm test / ... patterns), derives the outcome from
the exit code (``green`` iff exit code 0; a missing/unparsable exit code
counts ``red`` — trust is never granted without evidence) and records the
anchor via :func:`record_test_anchor`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from hermes.memory.entry import MemoryEntry
from hermes.memory.store import MemoryStore
from hermes.memory.verification import build_test_green_event, build_test_red_event

__all__ = [
    "AnchorMatcher",
    "AnchorOutcome",
    "AnchorResult",
    "build_command_hook",
    "record_test_anchor",
]

#: Outcome of an anchored test/build run (plan P2.T5: green or red).
AnchorOutcome = Literal["green", "red"]

#: Substrings that mark a command as a test run (case-insensitive contains).
_TEST_COMMAND_MARKERS: Final[tuple[str, ...]] = (
    "pytest",
    "python -m unittest",
    "unittest",
    "make test",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "cargo test",
    "go test",
    "tox",
    "gradle test",
    "mvn test",
    "rake test",
    "dotnet test",
    "mix test",
)

#: Path-like tokens inside entry text/tags: slash-carrying paths plus bare
#: extension-bearing filenames (``store.py``, ``MEMORY.md``) — the latter
#: still suffix-match ``hermes/memory/store.py`` on a segment boundary.
_PATH_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9_.\-]+(?:[/\\][A-Za-z0-9_.\-]+)+"
    r"|[A-Za-z0-9_.\-]+\.[A-Za-z0-9]{1,8}"
)

#: Test module file names inside a command line (stem -> the rule-(c) slug).
_TEST_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9_.\-/\\]*test[A-Za-z0-9_.\-]*\.py"
)


def _normalize_path(value: str) -> str:
    """Normalize a path token to forward slashes without a leading ``./``."""
    return value.replace("\\", "/").lstrip("./")


def _paths_overlap(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` name the same file up to path suffixes.

    Segment-boundary suffix matching: one path equals the other or ends with
    ``/other`` (``hermes/memory/store.py`` and ``memory/store.py`` overlap;
    ``hermes/memory/soul_store.py`` does not match ``store.py``).
    """
    na, nb = _normalize_path(a), _normalize_path(b)
    if not na or not nb:
        return False
    return na == nb or na.endswith("/" + nb) or nb.endswith("/" + na)


def _path_tokens(text: str) -> set[str]:
    """All path-like tokens in ``text`` (slash paths + bare filenames)."""
    return {token for token in _PATH_TOKEN_RE.findall(text) if token.strip(".")}


class AnchorMatcher:
    """Rule-based relatedness decision for one anchor run (plan P2.T5).

    Pure predicate object: constructed with the run's citation set, changed
    files and test module slugs, then asked per entry via :meth:`is_related`
    (or :meth:`match_reason` for the audit trail of WHICH rule fired). No
    store access, no LLM — the matching is fully deterministic.
    """

    def __init__(
        self,
        *,
        session_cited_entry_ids: Sequence[str] | None = None,
        changed_files: Sequence[str] | None = None,
        test_modules: Sequence[str] | None = None,
    ) -> None:
        #: Citation set: entry ids, text fragments or tag names (rule a).
        self.citations: frozenset[str] = frozenset(
            token.strip() for token in (session_cited_entry_ids or ()) if token.strip()
        )
        #: Changed files of the run (rule b).
        self.changed_files: tuple[str, ...] = tuple(
            str(value) for value in (changed_files or ()) if str(value).strip()
        )
        #: Test module slugs of the run (rule c).
        self.test_modules: frozenset[str] = frozenset(
            token.strip() for token in (test_modules or ()) if token.strip()
        )

    def match_reason(self, entry: MemoryEntry) -> str | None:
        """Which relatedness rule fires for ``entry``, or ``None``.

        Returns ``"cited"`` (rule a), ``"changed-files"`` (rule b) or
        ``"test-module"`` (rule c) — first match wins in that order.
        """
        if entry.id in self.citations:
            return "cited"
        if any(citation in entry.text for citation in self.citations):
            return "cited"
        if self.citations & frozenset(entry.tags):
            return "cited"
        if self.changed_files:
            tokens = _path_tokens(entry.text)
            for tag in entry.tags:
                tokens |= _path_tokens(tag)
            for token in tokens:
                if any(_paths_overlap(token, changed) for changed in self.changed_files):
                    return "changed-files"
        if self.test_modules & frozenset(entry.tags):
            return "test-module"
        return None

    def is_related(self, entry: MemoryEntry) -> bool:
        """True when any relatedness rule fires for ``entry``."""
        return self.match_reason(entry) is not None


@dataclass
class AnchorResult:
    """Outcome of one :func:`record_test_anchor` call (plan P2.T5)."""

    #: Number of verification events appended (one per related entry).
    events_added: int
    #: Ids of the related entries that received an event, in scan order.
    related_ids: list[str] = field(default_factory=list)


def record_test_anchor(
    store: MemoryStore,
    *,
    session_cited_entry_ids: Sequence[str] | None = None,
    changed_files: Sequence[str] | None = None,
    outcome: AnchorOutcome,
    evidence: str,
    test_modules: Sequence[str] | None = None,
) -> AnchorResult:
    """Append one ground-truth anchor event per RELATED entry (plan P2.T5).

    Scans every entry of ``store`` (all contexts), selects the related ones
    via :class:`AnchorMatcher` (session citations, changed-file overlap,
    test-module tags) and appends for each exactly one event — ``green``:
    source ``test-green``, weight 1.0, value 1.0; ``red``: the documented
    negative-anchor mapping (source ``tool-output``, weight 0.85, value 0.0,
    evidence prefixed ``test-red:``; see the module docstring). Each event is
    persisted with :meth:`MemoryStore.record_event` (atomic rewrite + one
    Git commit per entry). Unrelated entries are untouched.
    """
    matcher = AnchorMatcher(
        session_cited_entry_ids=session_cited_entry_ids,
        changed_files=changed_files,
        test_modules=test_modules,
    )
    related = [entry for entry in store.list() if matcher.is_related(entry)]
    touched: list[str] = []
    for entry in related:
        event = (
            build_test_green_event(evidence)
            if outcome == "green"
            else build_test_red_event(evidence)
        )
        store.record_event(entry.id, event)
        touched.append(entry.id)
    return AnchorResult(events_added=len(touched), related_ids=touched)


def _result_field(result: Any, names: Sequence[str]) -> Any:
    """First non-None field ``names`` on a mapping-shaped or object-shaped result."""
    if isinstance(result, Mapping):
        for name in names:
            value = result.get(name)
            if value is not None:
                return value
        return None
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return value
    return None


def _looks_like_test_run(command: str) -> bool:
    """Rule-based test-run detection (pytest / make test / npm test / ...)."""
    lowered = command.lower()
    return any(marker in lowered for marker in _TEST_COMMAND_MARKERS)


def _test_module_slugs(command: str) -> list[str]:
    """Stems of the ``*test*.py`` files named by ``command`` (rule-c slugs).

    ``pytest tests/unit/memory/test_anchors.py::test_x`` -> ``["test_anchors"]``.
    """
    slugs: list[str] = []
    for raw in _TEST_FILE_RE.findall(command):
        stem = Path(raw.replace("\\", "/")).stem
        if stem and stem not in slugs:
            slugs.append(stem)
    return slugs


def build_command_hook(
    store: MemoryStore,
    *,
    session_cited_entry_ids: Sequence[str] | None = None,
) -> Callable[[Any], AnchorResult | None]:
    """Build the PostToolUse-equivalent anchor hook for ``store`` (plan P2.T5).

    The returned callable takes one completed command result (a mapping with
    ``command`` / ``exit_code`` / optional ``changed_files`` /
    ``session_cited_entry_ids`` keys, or any object with the same
    attributes), and:

    - returns ``None`` for commands that do not look like a test run
      (:data:`_TEST_COMMAND_MARKERS` patterns) — no event for non-test work;
    - derives the outcome from the exit code (``green`` iff 0; a missing or
      unparsable exit code counts ``red`` — trust is never granted without
      evidence);
    - falls back to the ``session_cited_entry_ids`` given here when the
      command result does not carry its own citation set;
    - records the anchor via :func:`record_test_anchor` and returns its
      :class:`AnchorResult`.

    Register the callable on the agent with
    :meth:`hermes.core.agent.V2Agent.register_command_execution_hook`; the
    agent fires it from :meth:`hermes.core.agent.V2Agent.on_command_execution`
    (advisory — hook failures never break the command).
    """
    fallback_citations = session_cited_entry_ids

    def hook(cmd_result: Any) -> AnchorResult | None:
        command = _result_field(cmd_result, ("command", "cmd"))
        if not isinstance(command, str) or not _looks_like_test_run(command):
            return None
        exit_raw = _result_field(cmd_result, ("exit_code", "returncode", "exit"))
        try:
            exit_code = int(exit_raw) if exit_raw is not None else None
        except (TypeError, ValueError):
            exit_code = None
        outcome: AnchorOutcome = "green" if exit_code == 0 else "red"
        changed: Sequence[str] | None = _result_field(
            cmd_result, ("changed_files", "files")
        )
        if isinstance(changed, str):
            changed = (changed,)
        cited = _result_field(
            cmd_result,
            ("session_cited_entry_ids", "cited_entry_ids", "citations"),
        )
        if cited is None:
            cited = fallback_citations
        evidence = command if exit_code is None else f"{command} (exit={exit_code})"
        return record_test_anchor(
            store,
            session_cited_entry_ids=cited,
            changed_files=changed,
            outcome=outcome,
            evidence=evidence,
            test_modules=_test_module_slugs(command),
        )

    return hook
