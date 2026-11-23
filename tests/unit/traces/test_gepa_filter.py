"""P11.T4 acceptance: GepaTraceFilter — GEPA-ready trace curation (plan Phase 11).

Covered: :class:`hermes.traces.gepa_filter.GepaTraceFilter`.

- only ``quality_label="good"`` traces are selected (``bad`` / ``acceptable``
  / ``needs-review`` labels and stored ``pii_flagged`` traces are excluded);
- PII detection filters sensitive data: a fresh scan over the prompt +
  serialized payloads catches PII the store never flagged (skip mode
  excludes the trace; redact mode exports it with ``[EMAIL]``-style
  placeholders substituted);
- export writes the GEPA eval-set JSONL schema of
  ``tests/fixtures/eval_sets/*.json`` (``{input, expected_output, context}``
  plus the ``trace_id`` provenance key), atomically, counting
  written/skipped records.

Traces are seeded with realistic pipeline payloads (extraction drafts +
review actions, as ``hermes/memory/pipeline.py`` records them) so the
expected-output pairing is exercised against the real shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.memory.pii_detector import PIIDetector
from hermes.traces.gepa_filter import GepaTraceFilter
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

_TS = 1_700_000_000.0  # fixed epoch for deterministic store ordering

#: The exported email placeholder (PIIDetector's redaction token for `email`).
_EMAIL = "alice@example.com"
_EMAIL_TOKEN = "[EMAIL]"


def _trace(
    prompt: str,
    label: str | None,
    *,
    session: str = "sess-1",
    pii_flagged: bool = False,
    offset: float = 0.0,
    draft_text: str = "User prefers oat milk in coffee",
    action: str = "add",
) -> Trace:
    """One pipeline-shaped trace: extraction drafts + review actions."""
    return Trace(
        turn_id=f"turn-{prompt[:8]}",
        session_id=session,
        timestamp=_TS + offset,
        user_prompt=prompt,
        models_used=["main-model"],
        extraction={
            "drafts_count": 1,
            "draft_ids": ["d1"],
            "drafts": [
                {
                    "draft_id": "d1",
                    "text": draft_text,
                    "type": "User",
                    "proposed_source": "user-explicit",
                    "confidence": 0.9,
                }
            ],
        },
        review={
            "actions": [
                {"action": action, "draft_id": "d1", "rationale": "ok"}
            ],
            "actions_by_type": {action: 1},
            "rationales": ["ok"],
            "applied": (
                [{"action": action, "draft_id": "d1", "entry_id": "e1", "rationale": "ok"}]
                if action in ("add", "update")
                else []
            ),
            "elapsed_ms": 1.0,
        },
        quality_label=label,
        pii_flagged=pii_flagged,
    )


def _seed(store: TraceStore, *traces: Trace) -> None:
    for trace in traces:
        store.append(trace)


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.unit
def test_filters_bad_quality(tmp_path: Path) -> None:
    """select() keeps ONLY good-labeled traces; stored PII flags also exclude."""
    store = TraceStore(root=tmp_path / "traces")
    clean_prompt = "what did we decide about oat milk"
    _seed(
        store,
        _trace(clean_prompt, "good", offset=0.0),
        _trace("the deploy failed again", "bad", offset=1.0),
        _trace("casual small talk, nothing learned", "acceptable", offset=2.0),
        _trace("my budget contradicts last week", "needs-review", offset=3.0),
        _trace("unlabeled turn", None, offset=4.0),
        # good label but the store already flagged it (upstream P17.T4 scan)
        _trace("flagged upstream", "good", pii_flagged=True, offset=5.0),
    )

    fitter = GepaTraceFilter(store)
    selected = fitter.select()

    assert [trace.quality_label for trace in selected] == ["good"]
    assert selected[0].user_prompt == clean_prompt
    # scan_pii is the second exclusion layer, separate from the stored flag
    assert fitter.scan_pii(selected[0]) is False


@pytest.mark.unit
def test_filters_pii(tmp_path: Path) -> None:
    """A good trace whose prompt holds an email: skipped (default) or redacted."""
    store = TraceStore(root=tmp_path / "traces")
    clean_prompt = "remind me what the deploy pipeline does"
    dirty_prompt = f"email me the deploy log at {_EMAIL} please"
    _seed(
        store,
        _trace(clean_prompt, "good", offset=0.0),
        # the store never scanned the prompt: pii_flagged stays False here,
        # the filter's own detector must catch it
        _trace(dirty_prompt, "good", offset=1.0),
    )

    # skip mode (default): the flagged trace never enters the set / the file
    skip_filter = GepaTraceFilter(store, PIIDetector(action="warn"))
    assert skip_filter.scan_pii(skip_filter.select()[0]) is False
    selected_prompts = [trace.user_prompt for trace in skip_filter.select()]
    assert selected_prompts == [clean_prompt]
    assert _EMAIL not in " ".join(selected_prompts)

    out_skip = tmp_path / "eval-skip.jsonl"
    counts = skip_filter.export(out_skip)
    assert counts.written == 1 and counts.skipped == 1 and counts.redacted == 0
    assert _EMAIL not in out_skip.read_text(encoding="utf-8")

    # redact mode: the flagged trace IS exported with the email placeholder-substituted
    redact_filter = GepaTraceFilter(store, PIIDetector(action="warn"), export_mode="redact")
    out_redact = tmp_path / "eval-redact.jsonl"
    counts = redact_filter.export(out_redact)
    assert counts.written == 2 and counts.skipped == 0 and counts.redacted == 1

    records = _records(out_redact)
    assert [record["input"] for record in records] == [clean_prompt, dirty_prompt.replace(_EMAIL, _EMAIL_TOKEN)]
    assert all(_EMAIL not in str(record["input"]) for record in records)
    # the exported prompt must not match the detector even after substitution
    assert redact_filter.scan_pii(redact_filter.select()[0]) is False


@pytest.mark.unit
def test_export_to_jsonl(tmp_path: Path) -> None:
    """export() writes the fixture eval-set schema, atomically, with counts."""
    store = TraceStore(root=tmp_path / "traces")
    _seed(
        store,
        _trace("first good turn about coffee", "good", session="sess-a", offset=0.0),
        _trace("second good turn about deploys", "good", session="sess-b", offset=1.0,
               draft_text="Deploy runs through GitHub Actions", action="update"),
        _trace("a bad turn never exports", "bad", offset=2.0),
    )

    target = tmp_path / "eval" / "exported.jsonl"
    counts = GepaTraceFilter(store).export(target)

    assert counts == (2, 0, 0) or (counts.written, counts.skipped, counts.redacted) == (2, 0, 0)
    assert target.exists()
    # atomic write: no temp remnants next to the target
    assert [p.name for p in target.parent.iterdir()] == ["exported.jsonl"]

    records = _records(target)
    assert len(records) == 2
    for record in records:
        # the GEPA eval-set schema of tests/fixtures/eval_sets/*.json
        assert set(record) == {"input", "expected_output", "context", "trace_id"}
        assert isinstance(record["expected_output"], list)
        stored = store.get(str(record["trace_id"]))
        assert stored is not None
        assert record["input"] == stored.user_prompt
        assert record["context"] == stored.session_id

    # oldest first (deterministic curation order)
    assert [record["input"] for record in records] == [
        "first good turn about coffee",
        "second good turn about deploys",
    ]

    # expected_output pairs each draft with its final review decision
    assert records[0]["expected_output"] == [
        {"text": "User prefers oat milk in coffee", "type": "User", "action": "add"}
    ]
    assert records[1]["expected_output"] == [
        {
            "text": "Deploy runs through GitHub Actions",
            "type": "User",
            "action": "update",
        }
    ]

    # re-export replaces the file atomically (idempotent, no duplication)
    counts = GepaTraceFilter(store).export(target)
    assert (counts.written, counts.skipped, counts.redacted) == (2, 0, 0)
    assert len(_records(target)) == 2

    # only "jsonl" is the supported interchange format
    with pytest.raises(ValueError, match="format"):
        GepaTraceFilter(store).export(tmp_path / "x.json", format="json")
