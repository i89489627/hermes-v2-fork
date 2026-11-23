"""Sensitive-data (PII) detection for memory writes (plan P17.T4).

Regex-based detection of email addresses, phone numbers, SSNs, credit
card numbers (Luhn-validated) and API keys/tokens, plus caller-supplied
extra patterns. Three actions, per plan Phase 17:

- ``redact`` (default): matched spans are replaced with ``[TYPE]``
  placeholders before the entry is written;
- ``block``: any finding raises :class:`PiiBlockedError` and nothing is
  written;
- ``warn``: the text is written unchanged and a WARNING is logged.

:func:`PIIDetector.process` is the single entry point the store's pre-write
hook uses: it returns the text-to-store plus a :class:`PIIResult` whose
``detected`` list records ``{"type", "match_span", "sample_redacted"}`` per
finding and whose ``risk`` score sums per-type severities (capped at 1.0).
The store marks scanned entries ``pii_checked=True`` and records the
detected types as ``pii:<type>`` trust-card tags (frontmatter-safe — no
schema change).

Findings may not overlap: built-in patterns run in decreasing priority
(credit-card, SSN, api-key, email, phone) and a span claimed by a
higher-priority type is not re-reported by a lower one. Phone/credit-card
patterns are separator-structured and length-bounded so ISO dates
(``2026-11-18``) and ordinary 10-digit numbers stay undetected; credit
cards additionally require a Luhn-valid digit string.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "PIIAction",
    "PIIDetector",
    "PIIFinding",
    "PIIPattern",
    "PIIResult",
    "PiiBlockedError",
]

PIIAction = Literal["redact", "block", "warn"]

#: All valid :data:`PIIAction` values.
ACTIONS: tuple[str, ...] = ("redact", "block", "warn")

logger = logging.getLogger(__name__)

# Domain's first label must start with a letter — keeps retina-style asset
# names ("image@2x.png") from matching as emails (plan FPR < 2% target).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)+")

_PHONE_RE = re.compile(
    r"(?<![\w.])"  # not mid-number/mid-word and not right after a dot
    r"(?:\+\d{1,3}[\s.-])?"  # optional country code
    r"(?:\(\d{1,4}\)[\s.-]?|\d{2,4}[\s.-])"  # area code, dashed/spaced/parenthesised
    r"\d{3}[\s.-]?\d{4}\b"  # subscriber number
)

_SSN_RE = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")

_CREDIT_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")

_API_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),  # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),  # GitHub tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),  # Slack tokens
    re.compile(
        r"\b(?:api[_-]?key|apikey|token|secret|password)"
        r'["\']?\s*[:=]\s*["\']?[A-Za-z0-9+/_-]{12,}',
    ),  # generic assignment: <keyword> = <value>
)


def _luhn_ok(raw: str) -> bool:
    """True when ``raw``'s digits form a Luhn-valid 13-19 digit number."""
    digits = re.sub(r"\D", "", raw)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _ssn_ok(raw: str) -> bool:
    """True when ``raw`` is a structurally valid US SSN (no 000/666/9xx)."""
    area, group, serial = int(raw[:3]), int(raw[4:6]), int(raw[7:11])
    return area not in (0, 666) and area < 900 and group != 0 and serial != 0


def _mask(raw: str) -> str:
    """Keep the outer characters of ``raw``, mask the middle."""
    if len(raw) <= 6:
        return "*" * len(raw)
    return raw[:2] + "*" * (len(raw) - 4) + raw[-2:]


@dataclass
class PIIPattern:
    """One caller-supplied detection rule (plan P17.T4 "configurable").

    ``pattern`` may be a compiled regex or a pattern string (compiled on
    construction). ``validator`` post-checks a match (e.g. checksums) and
    rejects it by returning False; ``severity`` feeds the risk score.
    """

    type: str
    pattern: re.Pattern[str] | str
    severity: float = 0.4
    validator: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("PIIPattern.type must be a non-empty string")
        if not 0.0 < self.severity <= 1.0:
            raise ValueError(f"severity must be in (0, 1], got {self.severity}")
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)
        if not isinstance(self.pattern, re.Pattern):
            raise TypeError("pattern must be a regex string or compiled pattern")

    @property
    def placeholder(self) -> str:
        """Replacement token used by :func:`PIIDetector.redact`."""
        return f"[{re.sub(r'[^A-Za-z0-9]', '_', self.type).upper()}]"


@dataclass(frozen=True)
class PIIFinding:
    """One accepted match (public view: :attr:`PIIResult.detected`)."""

    type: str
    start: int
    end: int
    sample_redacted: str
    severity: float


@dataclass
class PIIResult:
    """Outcome of :func:`PIIDetector.scan` (plan P17.T4).

    ``detected`` holds one mapping per finding — ``type``, ``match_span``
    (a ``(start, end)`` tuple), ``sample_redacted`` (the match with its
    middle masked, never the raw secret) — and ``risk`` is in [0, 1].
    """

    detected: list[dict[str, Any]] = field(default_factory=list)
    risk: float = 0.0

    @property
    def types(self) -> list[str]:
        """Sorted unique PII types found (drives the ``pii:<type>`` tags)."""
        return sorted({str(item["type"]) for item in self.detected})


class PiiBlockedError(RuntimeError):  # noqa: N818 - name mandated by plan P17.T4
    """A memory write was blocked because it contains PII (plan P17.T4)."""

    def __init__(self, result: PIIResult) -> None:
        self.result = result
        super().__init__(
            f"memory write blocked: PII detected (types: {', '.join(result.types)})"
        )


@dataclass(frozen=True)
class _BuiltinSpec:
    """A built-in pattern: priority-ordered, with optional validation."""

    type: str
    regex: re.Pattern[str]
    severity: float
    validator: Callable[[str], bool] | None = None
    placeholder: str = ""

    @property
    def token(self) -> str:
        return self.placeholder or f"[{re.sub(r'[^A-Za-z0-9]', '_', self.type).upper()}]"


_BUILTINS: tuple[_BuiltinSpec, ...] = (
    _BuiltinSpec("credit-card", _CREDIT_CARD_RE, 0.6, _luhn_ok),
    _BuiltinSpec("ssn", _SSN_RE, 0.6, _ssn_ok),
    _BuiltinSpec("api-key", _API_KEY_RES[0], 0.5),
    _BuiltinSpec("api-key", _API_KEY_RES[1], 0.5),
    _BuiltinSpec("api-key", _API_KEY_RES[2], 0.5),
    _BuiltinSpec("api-key", _API_KEY_RES[3], 0.5),
    _BuiltinSpec("api-key", _API_KEY_RES[4], 0.5),
    _BuiltinSpec("email", _EMAIL_RE, 0.25),
    _BuiltinSpec("phone", _PHONE_RE, 0.25),
)


class PIIDetector:
    """Scans memory-entry text for PII and applies the configured action."""

    def __init__(
        self,
        action: PIIAction = "redact",
        extra_patterns: Sequence[PIIPattern] | None = None,
    ) -> None:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        self.action = action
        self.extra_patterns = list(extra_patterns) if extra_patterns else []

    # ── detection ───────────────────────────────────────────────────────────

    def scan(self, text: str) -> PIIResult:
        """Return the non-overlapping findings in ``text`` plus a risk score."""
        accepted: list[PIIFinding] = []
        occupied: list[tuple[int, int]] = []

        def claim(spec_type: str, start: int, end: int, raw: str, severity: float) -> None:
            occupied.append((start, end))
            accepted.append(
                PIIFinding(spec_type, start, end, _mask(raw), severity)
            )

        for spec in _BUILTINS:
            for match in spec.regex.finditer(text):
                if spec.validator is not None and not spec.validator(match.group(0)):
                    continue
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                claim(spec.type, span[0], span[1], match.group(0), spec.severity)

        for extra in self.extra_patterns:
            pattern = extra.pattern
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            for match in pattern.finditer(text):
                if extra.validator is not None and not extra.validator(match.group(0)):
                    continue
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                claim(extra.type, span[0], span[1], match.group(0), extra.severity)

        accepted.sort(key=lambda finding: (finding.start, finding.end))
        risk = min(1.0, sum(finding.severity for finding in accepted))
        detected = [
            {
                "type": finding.type,
                "match_span": (finding.start, finding.end),
                "sample_redacted": finding.sample_redacted,
            }
            for finding in accepted
        ]
        return PIIResult(detected=detected, risk=risk)

    def redact(self, text: str) -> str:
        """Replace every detected span with its ``[TYPE]`` placeholder."""
        return self._redact(text, self.scan(text))

    def _redact(self, text: str, result: PIIResult) -> str:
        """Placeholder substitution for a precomputed :class:`PIIResult`."""
        placeholders: dict[str, str] = {spec.type: spec.token for spec in _BUILTINS}
        for extra in self.extra_patterns:
            placeholders[extra.type] = extra.placeholder
        clean = text
        for item in reversed(result.detected):
            start, end = item["match_span"]
            token = placeholders[str(item["type"])]
            clean = clean[:start] + token + clean[end:]
        return clean

    # ── store-facing action ─────────────────────────────────────────────────

    def process(self, text: str) -> tuple[str, PIIResult]:
        """Apply the configured action to ``text`` (store pre-write hook).

        Returns ``(text_to_store, result)``. ``redact`` returns the
        placeholder-substituted text; ``warn`` returns the text unchanged
        and logs a WARNING naming the detected types; ``block`` raises
        :class:`PiiBlockedError` when anything was detected (identity
        otherwise).
        """
        result = self.scan(text)
        if not result.detected:
            return text, result
        if self.action == "block":
            raise PiiBlockedError(result)
        if self.action == "warn":
            logger.warning(
                "PII detected in memory write (types: %s, risk: %.2f) - "
                "stored unredacted with pii_checked mark",
                ", ".join(result.types),
                result.risk,
            )
            return text, result
        return self._redact(text, result), result
