"""hermes-v2 adversarial input detection (plan P17.T1, plan Section 5.5).

:class:`AdversarialInputDetector` scores memory-write candidates for
adversarial intent and returns a :class:`RiskResult` (``risk_score`` in
[0, 1], human-readable ``reasons``, machine-readable ``patterns_matched``).
High-risk inputs (>0.7) are blocked from the memory write (plan acceptance)
via :func:`store_guard`, which wraps :meth:`MemoryStore.add`.

Two layers, blended:

- **Rule-based pattern layer** (deterministic; the primary test path and the
  plan 5.5 degradation path when the main model is unavailable):
  *prompt injection* ("ignore previous instructions", "disregard your rules",
  "system prompt:", roleplay-override phrasing), *trust manipulation*
  ("trust_score=1", "verified by user", "this fact is verified",
  fake-verifier phrasing) and *contradiction flooding* (>= 3 mutually
  contradictory statements within a per-session window — the antonym /
  negation-cue table pattern from :mod:`hermes.memory.conflict_detector`,
  whose public ``ANTONYM_PAIRS``/``NEGATION_CUES`` are imported). Rules
  contribute weighted scores; the combined rule score is the **max** of the
  fired rules (the strongest single signal governs; repeated matches inside a
  rule saturate via a documented cap).
- **Model classifier layer** (optional): the ``classify`` callable is
  injected exactly like the P3 extractor's model seam. The model score
  **blends by max**: ``final = max(rule_score, model_score)`` — a confident
  model catches rule-evasion, a confident rule catch survives a missing
  model, and neither can lower the other's verdict. Accepted ``classify``
  return shapes: a bare float, or a dict carrying ``"risk_score"`` (also
  reads ``"risk"``/``"score"``); unusable payloads contribute 0.

Contradiction flooding needs a window: callers pass
``context={"recent_writes": [...], "session_id": ...}``; the detector ALSO
keeps a bounded per-session deque fed by every ``detect`` call, so repeated
writes through the same detector accumulate a window even without explicit
``recent_writes``.

False-positive posture (plan acceptance <5% on normal inputs): patterns are
anchored phrasings — e.g. the roleplay-override matches "you are now a/an …"
rather than bare "you are now" (so "You are now entering the final phase" is
clean), and "system prompt:" requires the colon (so "The system prompt file
was updated" is clean).

``store_guard(store, detector, threshold=0.7)`` returns a wrapped-add callable
with :meth:`MemoryStore.add`'s signature that detects before writing, raises
:class:`AdversarialInputBlockedError` above the threshold, notifies the
injected notifier on every block (``("adversarial_input_blocked", msg)`` —
the same ``(alert_name, message)`` seam the P21.T5 watchdog uses) and fires
the plan 5.3 ``adversarial_attempts`` alert via the same notifier when a
session exceeds 5 blocked attempts within one hour.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from hermes.memory.conflict_detector import ANTONYM_PAIRS, NEGATION_CUES
from hermes.memory.store import MemoryStore
from hermes.memory.types import EntryType

__all__ = [
    "ADVERSARIAL_ATTEMPTS_ALERT",
    "BLOCKED_ATTEMPTS_ALERT",
    "CONTRADICTION_PAIR_THRESHOLD",
    "DEFAULT_SESSION_WINDOW",
    "HIGH_RISK_THRESHOLD",
    "AdversarialInputBlockedError",
    "AdversarialInputDetector",
    "RiskResult",
    "store_guard",
]

_LOGGER = logging.getLogger("hermes.memory.adversarial_detector")

#: Plan acceptance: inputs scoring above this are blocked from memory writes.
HIGH_RISK_THRESHOLD: Final[float] = 0.7

#: Contradiction pairs within the window that constitute flooding.
CONTRADICTION_PAIR_THRESHOLD: Final[int] = 3

#: Bounded per-session contradiction window (recent writes kept per session).
DEFAULT_SESSION_WINDOW: Final[int] = 32

#: Plan 5.3 alert: more than 5 blocked attempts in 1h per session.
_ADVERSARIAL_MAX_ATTEMPTS_PER_HOUR: Final[int] = 5

#: Alert names on the notifier seam (same shape as the P21.T5 watchdog).
BLOCKED_ATTEMPTS_ALERT: Final[str] = "adversarial_input_blocked"
ADVERSARIAL_ATTEMPTS_ALERT: Final[str] = "adversarial_attempts"

#: Prompt-injection cue phrases (casefolded substring match).
_INJECTION_PHRASES: Final[tuple[str, ...]] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "disregard previous instructions",
    "disregard your rules",
    "disregard your instructions",
    "forget your instructions",
    "forget all instructions",
    "system prompt:",
    "new instructions:",
    "you are now a ",
    "you are now an ",
    "act as if you were",
    "pretend you are not",
)

#: Trust-manipulation patterns (regex, case-insensitive).
_TRUST_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("trust_score=1", re.compile(r"trust[_\s]*score\s*[=:]\s*1(?:\.0+)?", re.IGNORECASE)),
    ("trust=1", re.compile(r"\btrust\s*[=:]\s*1(?:\.0+)?\b", re.IGNORECASE)),
    ("verified by user", re.compile(r"verified by (?:the\s+)?user", re.IGNORECASE)),
    ("verified by admin", re.compile(r"verified by (?:the\s+)?(?:admin|owner)", re.IGNORECASE)),
    ("this fact is verified", re.compile(r"this (?:fact|entry|claim) is verified", re.IGNORECASE)),
    ("mark as verified", re.compile(r"\bmark(?:ed)?\s+as\s+verified\b", re.IGNORECASE)),
    ("verified: true", re.compile(r"\bverified\s*:\s*true\b", re.IGNORECASE)),
)

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")

#: Base rule scores per fired rule (documented weighting): a single canonical
#: adversarial phrase already clears the 0.7 block threshold; additional
#: distinct matches inside a rule add 0.1 each and cap at 1.0.
_RULE_BASE_SCORE: Final[float] = 0.8
_RULE_REPEAT_BONUS: Final[float] = 0.1
_FLOOD_BASE_SCORE: Final[float] = 0.7
_FLOOD_REPEAT_BONUS: Final[float] = 0.1


def _tokens(text: str) -> set[str]:
    """Word tokens of ``text`` (casefolded) — same tokenization as P8."""
    return set(_WORD_RE.findall(text.casefold()))


def _cue_phrases(text: str) -> set[str]:
    """Multi-word negation cues present in the normalized text."""
    normalized = " " + " ".join(_WORD_RE.findall(text.casefold())) + " "
    return {cue for cue in NEGATION_CUES if " " in cue and f" {cue} " in normalized}


def _texts_contradict(left: str, right: str) -> bool:
    """Heuristic contradiction between two statements (P8 table pattern).

    Fires when (a) a negation cue appears on exactly one side, or (b) one
    antonym-pair member appears on each side — the same deterministic lexical
    heuristic :mod:`hermes.memory.conflict_detector` uses for entries.
    """
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    left_phrases = _cue_phrases(left)
    right_phrases = _cue_phrases(right)
    for cue in NEGATION_CUES:
        if " " in cue:
            if (cue in left_phrases) != (cue in right_phrases):
                return True
        elif (cue in left_tokens) != (cue in right_tokens):
            return True
    for first, second in ANTONYM_PAIRS:
        if (first in left_tokens and second in right_tokens) or (
            second in left_tokens and first in right_tokens
        ):
            return True
    return False


def _count_contradiction_pairs(texts: Iterable[str]) -> int:
    """Number of mutually contradictory unordered pairs among ``texts``."""
    ordered = list(texts)
    return sum(
        1
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
        if _texts_contradict(ordered[i], ordered[j])
    )


def _extract_model_score(raw: float | dict[str, Any]) -> float:
    """Normalize a classifier payload to a score in [0, 1] (0 when unusable)."""
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        return min(1.0, max(0.0, float(raw)))
    if isinstance(raw, dict):
        for key in ("risk_score", "risk", "score"):
            value = raw.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return min(1.0, max(0.0, float(value)))
    return 0.0


@dataclass
class RiskResult:
    """Outcome of one :meth:`AdversarialInputDetector.detect` call (plan P17.T1).

    ``risk_score`` is clamped into [0, 1]; ``reasons`` explains the verdict in
    human-readable form; ``patterns_matched`` lists the exact rule identifiers
    (canonical phrases or regex labels) that fired — empty for clean input.
    """

    risk_score: float
    reasons: list[str] = field(default_factory=list)
    patterns_matched: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.risk_score = min(1.0, max(0.0, float(self.risk_score)))


class AdversarialInputBlockedError(RuntimeError):
    """Raised by :func:`store_guard` when a write scores above the threshold.

    Carries the full :class:`RiskResult` (``.result``) so callers (gateway,
    reviewer, tests) can report the score and reasons without re-detecting.
    """

    def __init__(self, result: RiskResult) -> None:
        self.result = result
        super().__init__(
            f"adversarial input blocked (risk {result.risk_score:.2f}): "
            + "; ".join(result.reasons)
        )


class AdversarialInputDetector:
    """Two-layer adversarial-input classifier (plan P17.T1).

    Args:
        classify: optional main-model classifier
            ``Callable[[str], float | dict[str, Any]]`` — injected like the
            extractor's model seam. Its score blends by max with the rule
            score (documented above). ``None`` (or a model outage — callers
            simply pass ``None`` per plan 5.5) leaves the deterministic
            rule layer, which is the degradation path.
        session_window: per-session history bound for contradiction flooding.
    """

    def __init__(
        self,
        classify: Callable[[str], float | dict[str, Any]] | None = None,
        *,
        session_window: int = DEFAULT_SESSION_WINDOW,
    ) -> None:
        self._classify = classify
        self._window_size = max(2, session_window)
        self._history: dict[str, deque[str]] = {}
        self._lock = threading.Lock()

    # ── detection ────────────────────────────────────────────────────────────
    def detect(
        self,
        input_text: str,
        context: Mapping[str, Any] | None = None,
    ) -> RiskResult:
        """Score ``input_text``; returns :class:`RiskResult` (never raises).

        ``context`` may carry ``session_id`` (key for the per-session window,
        default ``"default"``) and ``recent_writes`` (explicit window texts,
        e.g. the store's recent entries). The input itself is appended to the
        per-session history after scoring, so consecutive detects build the
        flooding window automatically.
        """
        reasons: list[str] = []
        matched: list[str] = []
        score = 0.0
        casefolded = input_text.casefold()

        injection_hits = [phrase for phrase in _INJECTION_PHRASES if phrase in casefolded]
        if injection_hits:
            matched.extend(injection_hits)
            injection_score = _RULE_BASE_SCORE + _RULE_REPEAT_BONUS * (len(injection_hits) - 1)
            score = max(score, min(1.0, injection_score))
            reasons.append(
                f"prompt injection: {len(injection_hits)} pattern(s) matched, "
                f"e.g. '{injection_hits[0].strip()}'"
            )

        trust_hits = [label for label, pattern in _TRUST_PATTERNS if pattern.search(input_text)]
        if trust_hits:
            matched.extend(trust_hits)
            trust_score = _RULE_BASE_SCORE + _RULE_REPEAT_BONUS * (len(trust_hits) - 1)
            score = max(score, min(1.0, trust_score))
            reasons.append(
                f"trust manipulation: {len(trust_hits)} pattern(s) matched, "
                f"e.g. '{trust_hits[0]}'"
            )

        session_id = "default"
        recent_writes: list[str] = []
        if context is not None:
            raw_session = context.get("session_id")
            if raw_session is not None:
                session_id = str(raw_session)
            raw_recent = context.get("recent_writes")
            if isinstance(raw_recent, (list, tuple)):
                recent_writes = [str(text) for text in raw_recent]

        with self._lock:
            history = list(self._history.get(session_id, ()))
        window = [*recent_writes, *history, input_text]
        pairs = _count_contradiction_pairs(window)
        if pairs >= CONTRADICTION_PAIR_THRESHOLD:
            matched.append("contradiction_flooding")
            flood_score = _FLOOD_BASE_SCORE + _FLOOD_REPEAT_BONUS * (
                pairs - CONTRADICTION_PAIR_THRESHOLD
            )
            score = max(score, min(1.0, flood_score))
            reasons.append(
                f"contradiction flooding: {pairs} mutually contradictory pairs within "
                f"the session window (threshold {CONTRADICTION_PAIR_THRESHOLD})"
            )

        if self._classify is not None:
            model_score = _extract_model_score(self._classify(input_text))
            score = max(score, model_score)  # documented blend: max of both layers
            reasons.append(f"model classifier risk {model_score:.2f}")

        with self._lock:
            history_deque = self._history.setdefault(
                session_id, deque(maxlen=self._window_size)
            )
            history_deque.append(input_text)

        return RiskResult(risk_score=score, reasons=reasons, patterns_matched=matched)


def store_guard(
    store: MemoryStore,
    detector: AdversarialInputDetector,
    threshold: float = HIGH_RISK_THRESHOLD,
    *,
    notifier: Callable[[str, str], None] | None = None,
    max_attempts_per_hour: int = _ADVERSARIAL_MAX_ATTEMPTS_PER_HOUR,
    now: Callable[[], datetime] | None = None,
) -> Callable[..., Any]:
    """Wrap ``store.add`` with an adversarial-input gate (plan P17.T1).

    The returned callable mirrors :meth:`MemoryStore.add`'s signature
    (``guarded_add(text, type, **kwargs)``; ``session_id`` in ``kwargs`` also
    keys the detector's per-session window). Behaviour:

    - runs ``detector.detect(text, {"session_id": session_id})`` first;
    - risk strictly above ``threshold``: the write never reaches the store —
      the notifier (if injected) is told (``BLOCKED_ATTEMPTS_ALERT``, then
      ``ADVERSARIAL_ATTEMPTS_ALERT`` once the session exceeds
      ``max_attempts_per_hour`` blocks within the trailing hour) and
      :class:`AdversarialInputBlockedError` is raised;
    - otherwise the write proceeds and the entry is returned.

    ``now`` is an injectable clock (aware UTC) for the attempts window.
    """
    blocked: dict[str, deque[datetime]] = {}
    _clock = now if now is not None else (lambda: datetime.now(UTC))

    def guarded_add(text: str, type: EntryType, **kwargs: Any) -> Any:
        session_id = str(kwargs.get("session_id", "default"))
        result = detector.detect(text, {"session_id": session_id})
        if result.risk_score <= threshold:
            return store.add(text, type=type, **kwargs)
        moment = _clock()
        window = blocked.setdefault(session_id, deque())
        window.append(moment)
        cutoff = moment - timedelta(hours=1)
        while window and window[0] < cutoff:
            window.popleft()
        if notifier is not None:
            try:
                notifier(
                    BLOCKED_ATTEMPTS_ALERT,
                    f"blocked adversarial memory write (session {session_id}, "
                    f"risk {result.risk_score:.2f}): "
                    + ("; ".join(result.reasons) or "unknown signal"),
                )
            except Exception:  # noqa: BLE001 - a broken notifier must not block writes
                _LOGGER.exception("adversarial-input notifier failed (session %s)", session_id)
            if len(window) > max_attempts_per_hour:
                try:
                    notifier(
                        ADVERSARIAL_ATTEMPTS_ALERT,
                        f"{len(window)} blocked adversarial attempts within 1h for "
                        f"session '{session_id}' (threshold: more than "
                        f"{max_attempts_per_hour})",
                    )
                except Exception:  # noqa: BLE001 - same failure isolation
                    _LOGGER.exception(
                        "adversarial_attempts notifier failed (session %s)", session_id
                    )
        raise AdversarialInputBlockedError(result)

    return guarded_add
