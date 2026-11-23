"""P10.T2 acceptance tests: eval set fixtures (plan Phase 10).

The four static fixture files under ``tests/fixtures/eval_sets/`` label 55
hand-crafted examples per memory-pipeline component (extractor, retrieval,
gating, reviewer). These tests validate the shape only: component count in
the 50-100 band, the ``{input, expected_output, context}`` record schema
(extra metadata fields allowed), the documented ``_meta`` header, and
uniqueness of ``input`` within each file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

EVAL_SET_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "eval_sets"

#: fixture filename -> expected ``_meta.component`` value
EVAL_SET_FILES: dict[str, str] = {
    "extractor_labeled.json": "extractor",
    "retrieval_labeled.json": "retrieval",
    "gating_labeled.json": "gating",
    "reviewer_labeled.json": "reviewer",
}

#: keys every labeled example must carry (extras are explicitly allowed)
REQUIRED_RECORD_KEYS = frozenset({"input", "expected_output", "context"})

#: keys the documented ``_meta`` header must carry
REQUIRED_META_KEYS = frozenset({"component", "version", "created_at", "description"})


def _load(name: str) -> dict[str, Any]:
    """Load one eval-set fixture and return its parsed JSON document."""
    path = EVAL_SET_DIR / name
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical_input(value: Any) -> str:
    """Canonical JSON form of an example's ``input`` for duplicate detection."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@pytest.mark.integration
def test_eval_sets_have_50_plus_examples() -> None:
    """Each component eval set carries between 50 and 100 labeled examples."""
    for name in EVAL_SET_FILES:
        examples = _load(name)["examples"]
        assert 50 <= len(examples) <= 100, (
            f"{name}: {len(examples)} examples, expected 50-100"
        )


@pytest.mark.integration
def test_eval_sets_schema_valid() -> None:
    """Every example carries {input, expected_output, context} and a valid _meta.

    Extra metadata fields are allowed on both levels (the schema is an
    open-world record). No file may contain two examples with the same
    ``input`` (compared canonically, since gating/reviewer inputs are
    structured objects).
    """
    for name, component in EVAL_SET_FILES.items():
        doc = _load(name)
        assert isinstance(doc, dict), f"{name}: top level must be an object"

        meta = doc.get("_meta")
        assert isinstance(meta, dict), f"{name}: missing _meta header"
        missing_meta = REQUIRED_META_KEYS - set(meta)
        assert not missing_meta, f"{name}: _meta missing keys {sorted(missing_meta)}"
        assert meta["component"] == component, (
            f"{name}: _meta.component {meta['component']!r} != {component!r}"
        )
        assert isinstance(meta["version"], str) and meta["version"], (
            f"{name}: _meta.version must be a non-empty string"
        )
        assert isinstance(meta["description"], str) and meta["description"], (
            f"{name}: _meta.description must be a non-empty string"
        )

        examples = doc["examples"]
        assert isinstance(examples, list) and examples, (
            f"{name}: examples must be a non-empty list"
        )
        seen_inputs: list[str] = []
        for example in examples:
            ex_id = example.get("id", "<no-id>")
            missing = REQUIRED_RECORD_KEYS - set(example)
            assert not missing, f"{name}/{ex_id}: missing record keys {sorted(missing)}"
            assert example["input"] is not None, f"{name}/{ex_id}: input is null"
            assert example["expected_output"] is not None, (
                f"{name}/{ex_id}: expected_output is null"
            )
            assert isinstance(example["context"], str) and example["context"], (
                f"{name}/{ex_id}: context must be a non-empty string"
            )
            assert isinstance(example["id"], str) and example["id"], (
                f"{name}/{ex_id}: id must be a non-empty string"
            )
            seen_inputs.append(_canonical_input(example["input"]))

        duplicates = {value for value in seen_inputs if seen_inputs.count(value) > 1}
        assert not duplicates, f"{name}: duplicate inputs by (input): {duplicates}"
