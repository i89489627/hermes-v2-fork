"""P22.T3 acceptance: PromptOptimizer (plan Phase 22 token reduction).

Covered: prompt size analysis (per-prompt tokens, total, oversized flagged
with counts) feeding rule-based reduction suggestions (a 3000+-token fake
prompt yields >= 2 quantified suggestions), and per-memory-operation token
usage tracking (bounded in-memory ledger + JSONL persistence under
``<hermes_home()>/diagnostics/prompt_usage.jsonl``, proven by a reload
roundtrip and by the deque bound trimming old records while the file keeps
the full history).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import hermes.core.prompt_optimizer as prompt_optimizer_module
from hermes.config import hermes_home
from hermes.core.prompt_optimizer import PromptOptimizer

#: The exact duplicate line planted several times in the oversized fixture.
_DUP_LINE = "Review every draft fact before it is committed to the memory store."


def _oversized_prompt() -> str:
    """A deterministic ~3200-token fake prompt (chars/4) with removable fat.

    Composition: one filler body (the bulk), 7 copies of one instruction
    line (exactly 6 removable duplicates), and 5 UNIQUE sentences each
    carrying two verbose connectors ("please note that", "in order to") so
    they do not themselves count as duplicate lines.
    """
    filler = (
        "Detailed background paragraph explaining memory pipeline behavior "
        "for the benchmark run. " * 130
    )
    connector_sentences = "".join(
        f"please note that turn {i} must be processed in order to keep the "
        f"ledger at revision {i} consistent.\n"
        for i in range(5)
    )
    return (
        "# Extraction Instructions\n"
        + filler
        + "\n"
        + (_DUP_LINE + "\n") * 4
        + connector_sentences
        + (_DUP_LINE + "\n") * 3
    )


@pytest.mark.unit
def test_prompt_size_report() -> None:
    """analyze() token-counts every prompt and flags oversized ones; a
    3000+-token fake prompt gets >= 2 quantified reduction suggestions."""
    optimizer = PromptOptimizer()  # default limit 2048, chars/4 tokenizer
    oversized_text = _oversized_prompt()
    small_text = "You are a memory gate. Return JSON."
    report = optimizer.analyze({"extraction_system": oversized_text, "gate_system": small_text})

    assert report.tokens_per_prompt["gate_system"] == (len(small_text) + 3) // 4
    assert (
        report.tokens_per_prompt["extraction_system"]
        == (len(oversized_text) + 3) // 4
    )
    assert report.total_tokens == sum(report.tokens_per_prompt.values())
    assert report.limit_tokens == 2048

    # Only the 3000+-token prompt is oversized, with its over-by count.
    assert [item.name for item in report.oversized] == ["extraction_system"]
    oversized = report.oversized[0]
    assert oversized.tokens >= 3000
    assert oversized.over_by == oversized.tokens - 2048 >= 1

    # Rule-based suggestions for the oversized prompt: duplicates plus the
    # two verbose connectors -> >= 2 suggestions, each quantified.
    suggestions = optimizer.suggest_reduction("extraction_system", oversized_text)
    assert len(suggestions) >= 2
    assert any("remove 6 duplicate lines" in s for s in suggestions)
    assert any("replace 'please note that'" in s for s in suggestions)
    assert any("replace 'in order to'" in s for s in suggestions)
    assert all("est. tokens saved" in s for s in suggestions)

    # Suggestions target oversized prompts only; prompts are never mutated.
    assert optimizer.suggest_reduction("gate_system", small_text) == []
    assert oversized_text == _oversized_prompt()


@pytest.mark.unit
def test_token_usage_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    """record_usage() tracks tokens per memory op, persists to JSONL under
    hermes_home()/diagnostics, and the reload roundtrip + deque bound hold."""
    monkeypatch.setattr(prompt_optimizer_module, "_LEDGER_MAX", 5)

    optimizer = PromptOptimizer()
    a = optimizer.record_usage("extraction", "a" * 40)  # 10 tokens (chars/4)
    b = optimizer.record_usage("extraction", "b" * 80)  # 20 tokens
    c = optimizer.record_usage("review", "c" * 12)  # 3 tokens
    assert (a, b, c) == (10, 20, 3)

    report = optimizer.usage_report()
    assert report["extraction"]["calls"] == 2
    assert report["extraction"]["tokens_total"] == 30
    assert report["extraction"]["tokens_p50"] == 15.0  # median of 10, 20
    assert report["review"]["calls"] == 1
    assert report["review"]["tokens_total"] == 3

    # Write-through persistence under the per-test hermes home.
    ledger_path = hermes_home() / "diagnostics" / "prompt_usage.jsonl"
    assert ledger_path.exists()
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first_record: dict[str, Any] = json.loads(lines[0])
    assert first_record["op"] == "extraction"
    assert first_record["tokens"] == 10
    assert first_record["recorded_at"]  # ISO-8601 timestamp present

    # Roundtrip: a fresh optimizer resumes the ledger from the JSONL tail.
    reloaded = PromptOptimizer()
    assert reloaded.usage_report() == report

    # The deque bound trims in-memory history to the last _LEDGER_MAX
    # records while the JSONL file keeps the full append-only history.
    for _ in range(3):
        reloaded.record_usage("gating", "d" * 8)  # 2 tokens each
    trimmed = reloaded.usage_report()
    assert trimmed["gating"] == {"calls": 3, "tokens_total": 6, "tokens_p50": 2.0}
    # 6 records total, window of 5: the FIRST extraction record fell out.
    assert trimmed["extraction"] == {"calls": 1, "tokens_total": 20, "tokens_p50": 20.0}
    assert trimmed["review"] == {"calls": 1, "tokens_total": 3, "tokens_p50": 3.0}
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 6

    final = PromptOptimizer()
    assert final.usage_report()["gating"]["calls"] == 3
    assert final.usage_report()["review"]["calls"] == 1
    assert final.usage_report()["extraction"]["calls"] == 1
