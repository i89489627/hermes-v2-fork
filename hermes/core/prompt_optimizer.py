"""PromptOptimizer: prompt size analysis + per-op token usage ledger (plan P22.T3).

The plan Phase 22 token-reduction component — analyze prompt sizes, suggest
reductions, and track token usage per operation (extraction, gating, review,
retrieval prompt assembly, ...).

SUGGESTIONS ONLY — PROMPTS ARE NEVER MUTATED
--------------------------------------------
Every method here is read-only over the prompts it receives.
:func:`suggest_reduction` returns quantified, rule-based advice; the actual
rewriting is deliberately out of scope until GEPA (plan Phase 9/P10) deploys
its optimized templates — an auto-rewriter here would fight the very loop
that is supposed to own prompt text.

Token accounting
----------------
The default tokenizer is the documented chars/4 heuristic (ceiling), the
same approximation :func:`hermes.optimization.performance.estimate_tokens`
uses for the P22.T1 profiler — duplicated locally (3 lines of arithmetic)
rather than imported so this core module does not drag the optimization
package (and its BM25 dependencies) into every core import. A real
tokenizer callable can be injected via the ``tokenizer`` parameter.

Usage ledger
------------
:func:`record_usage(op, prompt_text)` appends one record per memory
operation to an in-memory bounded deque (the last
:data:`_LEDGER_MAX` records) AND writes it through to
``<hermes_home()>/diagnostics/prompt_usage.jsonl`` (append-only JSONL, one
JSON object per line with ``op`` / ``tokens`` / ``recorded_at``). A new
:class:`PromptOptimizer` instance loads the persisted tail back in, so the
ledger survives process restarts; the JSONL file keeps the full history
while the deque bound only trims what is held in memory.

Reduction rules (each suggestion quantified with est. tokens saved)
--------------------------------------------------------------------
1. **Duplicate lines** — exact duplicate non-blank lines: "remove N
   duplicate lines" for the N removable (count-1 per line) occurrences.
2. **Boilerplate section headers** — headers (``#``-style markdown
   headings, or short ``:``-terminated section labels) the optimizer has
   seen in OTHER prompts too (tracked from :func:`analyze` /
   :func:`record_usage` / prior :func:`suggest_reduction` calls).
3. **Verbose connectors** — a small phrase table ("please note that",
   "in order to", ...) with terse replacements.
4. **Over-long examples** — lines over 200 chars ("compress example"),
   aggregated, with the est. saving at a 50% compression target.

Suggestions are generated for OVERSIZED prompts only (tokens above
``limit_tokens``): a prompt within budget has nothing to reduce per the
P22.T3 acceptance.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hermes.config import hermes_home

__all__ = [
    "OversizedPrompt",
    "PromptOptimizer",
    "PromptSizeReport",
    "UsageRecord",
]

#: In-memory ledger bound (the JSONL file itself is append-only and keeps
#: the full history; see the module docstring).
_LEDGER_MAX = 1000

#: Ledger location under the Hermes home (plan P22.T3): a JSONL file in the
#: ``diagnostics`` directory, resolved via :func:`hermes.config.hermes_home`.
_LEDGER_DIRNAME = "diagnostics"
_LEDGER_FILENAME = "prompt_usage.jsonl"

#: Over-long example threshold (plan P22.T3 design: ">200 chars").
_OVERLONG_LINE_CHARS = 200

#: Est. fraction of an over-long example's tokens recoverable by compression.
#: A conservative documented heuristic: examples typically halve when
#: rewritten tersely.
_OVERLONG_SAVING_FRACTION = 0.5

#: Verbose connector phrase table (phrase -> terse replacement). Matched
#: case-insensitively as exact substrings; longer phrases are listed first
#: so a phrase that contains another is counted before the shorter one can
#: double-count its span.
_VERBOSE_PHRASES: tuple[tuple[str, str], ...] = (
    ("it is important to note that", ""),
    ("it should be noted that", ""),
    ("please note that", ""),
    ("as a matter of fact", ""),
    ("due to the fact that", "because"),
    ("at this point in time", "now"),
    ("for the purpose of", "for"),
    ("in the event that", "if"),
    ("in the near future", "soon"),
    ("with regard to", "about"),
    ("in order to", "to"),
)

#: Markdown-style section heading (``#``..``######`` followed by text).
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def _default_token_count(text: str) -> int:
    """Documented chars/4 token heuristic, ceiling (0 for empty text).

    Mirrors :func:`hermes.optimization.performance.estimate_tokens` (plan
    P22.T1) — see the module docstring for why it is duplicated here.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def _headers_of(text: str) -> set[str]:
    """Section-header lines of ``prompt_text`` (markdown headings and short
    ``:``-terminated labels), normalized by stripping surrounding whitespace."""
    headers: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(stripped) or (
            stripped.endswith(":") and len(stripped) <= 60
        ):
            headers.add(stripped)
    return headers


@dataclass(frozen=True)
class UsageRecord:
    """One persisted token-usage observation (plan P22.T3 ledger entry)."""

    op: str
    tokens: int
    recorded_at: str


@dataclass(frozen=True)
class OversizedPrompt:
    """One prompt over the token limit, with its over-by count (P22.T3)."""

    name: str
    tokens: int
    limit_tokens: int

    @property
    def over_by(self) -> int:
        """How many tokens the prompt exceeds the limit by (>= 1)."""
        return self.tokens - self.limit_tokens


@dataclass
class PromptSizeReport:
    """Result of :func:`PromptOptimizer.analyze` (plan P22.T3).

    ``tokens_per_prompt`` carries every analyzed prompt's token count,
    ``total_tokens`` their sum, ``limit_tokens`` the configured limit and
    ``oversized`` the prompts above it (with their over-by counts).
    """

    tokens_per_prompt: dict[str, int]
    total_tokens: int
    limit_tokens: int
    oversized: list[OversizedPrompt] = field(default_factory=list)


class PromptOptimizer:
    """Analyzes prompt sizes, tracks per-op token usage, suggests reductions.

    ``limit_tokens`` flags prompts above this many tokens as oversized
    (default 2048 — plan P22.T3); ``tokenizer`` defaults to the documented
    chars/4 heuristic and can be replaced with a real tokenizer callable.
    The optimizer never mutates prompts — see the module docstring.
    """

    def __init__(
        self,
        limit_tokens: int = 2048,
        tokenizer: Callable[[str], int] = _default_token_count,
    ) -> None:
        if limit_tokens < 1:
            raise ValueError(f"limit_tokens must be >= 1, got {limit_tokens}")
        if not callable(tokenizer):
            raise TypeError(f"tokenizer must be callable, got {tokenizer!r}")
        self._limit_tokens = int(limit_tokens)
        self._tokenizer = tokenizer
        #: Bounded per-op usage ledger (the last _LEDGER_MAX records).
        self._ledger: deque[UsageRecord] = self._load_ledger()
        #: Section headers seen per prompt name, for cross-prompt
        #: boilerplate detection (P22.T3 rule 2).
        self._headers: dict[str, set[str]] = {}

    # ── persistence ─────────────────────────────────────────────────────────
    def _ledger_path(self) -> Path:
        path: Path = hermes_home() / _LEDGER_DIRNAME / _LEDGER_FILENAME
        return path

    def _load_ledger(self) -> deque[UsageRecord]:
        """Load the persisted ledger tail; malformed lines are skipped."""
        records: deque[UsageRecord] = deque(maxlen=_LEDGER_MAX)
        path = self._ledger_path()
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # corrupt tail line (e.g. crashed mid-write)
            if (
                isinstance(data, dict)
                and isinstance(data.get("op"), str)
                and isinstance(data.get("tokens"), int)
                and not isinstance(data.get("tokens"), bool)
            ):
                records.append(
                    UsageRecord(
                        op=data["op"],
                        tokens=data["tokens"],
                        recorded_at=str(data.get("recorded_at", "")),
                    )
                )
        return records

    def _persist(self, record: UsageRecord) -> None:
        """Append one record to the JSONL ledger (write-through)."""
        path = self._ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "op": record.op,
                        "tokens": record.tokens,
                        "recorded_at": record.recorded_at,
                    }
                )
                + "\n"
            )

    # ── token usage tracking (plan P22.T3 acceptance) ───────────────────────
    def record_usage(self, op: str, prompt_text: str) -> int:
        """Record one ``op`` invocation's prompt token cost; returns the count.

        Appends to the bounded in-memory ledger and persists the record to
        ``<hermes_home()>/diagnostics/prompt_usage.jsonl``. Also registers
        the prompt's section headers under ``op`` for boilerplate detection.
        """
        if not isinstance(op, str) or not op:
            raise TypeError(f"op must be a non-empty str, got {op!r}")
        if not isinstance(prompt_text, str):
            raise TypeError(f"prompt_text must be a str, got {prompt_text!r}")
        tokens = self._tokenizer(prompt_text)
        record = UsageRecord(
            op=op, tokens=tokens, recorded_at=datetime.now(UTC).isoformat()
        )
        self._ledger.append(record)
        self._persist(record)
        self._headers.setdefault(op, set()).update(_headers_of(prompt_text))
        return tokens

    def usage_report(self) -> dict[str, dict[str, float]]:
        """Aggregate the ledger per operation: calls, token totals, p50.

        ``{op: {"calls": n, "tokens_total": t, "tokens_p50": p}}`` — the p50
        is the median of that op's per-call token samples (0.0 for an op
        with no samples, though an op only exists in the report once it has
        at least one recorded call).
        """
        grouped: dict[str, list[int]] = {}
        for record in self._ledger:
            grouped.setdefault(record.op, []).append(record.tokens)
        report: dict[str, dict[str, float]] = {}
        for op, samples in grouped.items():
            report[op] = {
                "calls": len(samples),
                "tokens_total": sum(samples),
                "tokens_p50": float(statistics.median(samples)),
            }
        return report

    # ── prompt size analysis (plan P22.T3 acceptance) ───────────────────────
    def analyze(self, prompts: dict[str, str]) -> PromptSizeReport:
        """Token-count every prompt; flag the ones over ``limit_tokens``.

        Every analyzed prompt's section headers are registered for the
        cross-prompt boilerplate rule, so a subsequent
        :func:`suggest_reduction` on any one of them can name headers the
        other prompts repeat.
        """
        if not isinstance(prompts, dict):
            raise TypeError(f"prompts must be a dict[str, str], got {type(prompts)!r}")
        for name, text in prompts.items():
            if not isinstance(name, str) or not isinstance(text, str):
                raise TypeError(
                    f"prompts must map str -> str, got key {type(name)!r} "
                    f"with value type {type(text)!r}"
                )
        tokens_per_prompt = {
            name: self._tokenizer(text) for name, text in prompts.items()
        }
        for name, text in prompts.items():
            self._headers.setdefault(name, set()).update(_headers_of(text))
        oversized = [
            OversizedPrompt(
                name=name,
                tokens=tokens,
                limit_tokens=self._limit_tokens,
            )
            for name, tokens in sorted(tokens_per_prompt.items())
            if tokens > self._limit_tokens
        ]
        return PromptSizeReport(
            tokens_per_prompt=tokens_per_prompt,
            total_tokens=sum(tokens_per_prompt.values()),
            limit_tokens=self._limit_tokens,
            oversized=oversized,
        )

    # ── reduction suggestions (plan P22.T3 acceptance) ──────────────────────
    def suggest_reduction(self, prompt_name: str, prompt_text: str) -> list[str]:
        """Quantified, rule-based reduction suggestions for an oversized prompt.

        Returns ``[]`` for a prompt within ``limit_tokens`` (nothing to
        reduce) and never mutates ``prompt_text`` — auto-rewrite is out of
        scope until GEPA deploys its optimized templates (module docstring).
        """
        if not isinstance(prompt_name, str) or not prompt_name:
            raise TypeError(f"prompt_name must be a non-empty str, got {prompt_name!r}")
        if not isinstance(prompt_text, str):
            raise TypeError(f"prompt_text must be a str, got {prompt_text!r}")
        tokens = self._tokenizer(prompt_text)
        if tokens <= self._limit_tokens:
            return []

        # Register this prompt's headers so later suggestions on OTHER
        # prompts can detect boilerplate shared with this one.
        self._headers.setdefault(prompt_name, set()).update(_headers_of(prompt_text))

        suggestions: list[str] = []

        # Rule 1: exact duplicate non-blank lines ("remove N duplicate lines").
        line_counts = Counter(
            line for line in prompt_text.splitlines() if line.strip()
        )
        removable = sum(count - 1 for count in line_counts.values() if count > 1)
        if removable > 0:
            saved = sum(
                self._tokenizer(line) * (count - 1)
                for line, count in line_counts.items()
                if count > 1
            )
            suggestions.append(
                f"remove {removable} duplicate lines (~{saved} est. tokens saved)"
            )

        # Rule 2: section headers this prompt shares with OTHER prompts.
        headers = _headers_of(prompt_text)
        for header in sorted(headers):
            others = sum(
                1
                for name, seen in self._headers.items()
                if name != prompt_name and header in seen
            )
            if others > 0:
                suggestions.append(
                    f"remove boilerplate section header '{header}' "
                    f"(repeated in {others} other prompt(s), "
                    f"~{self._tokenizer(header)} est. tokens saved)"
                )

        # Rule 3: verbose connectors -> terse replacements.
        folded = prompt_text.casefold()
        for phrase, replacement in _VERBOSE_PHRASES:
            count = folded.count(phrase.casefold())
            if count > 0:
                saved = count * (
                    self._tokenizer(phrase) - self._tokenizer(replacement)
                )
                suggestions.append(
                    f"replace '{phrase}' with '{replacement}' "
                    f"({count} occurrence(s), ~{saved} est. tokens saved)"
                )

        # Rule 4: over-long example lines (>200 chars) -> compress example.
        overlong = [line for line in prompt_text.splitlines() if len(line) > _OVERLONG_LINE_CHARS]
        if overlong:
            saved = sum(
                int(self._tokenizer(line) * _OVERLONG_SAVING_FRACTION)
                for line in overlong
            )
            suggestions.append(
                f"compress {len(overlong)} over-long example line(s) "
                f"(>{_OVERLONG_LINE_CHARS} chars each, "
                f"~{saved} est. tokens saved at 50% compression)"
            )

        return suggestions
