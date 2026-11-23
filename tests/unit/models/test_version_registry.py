"""P19.T1 acceptance tests: model version registry (plan Phase 19).

Covers: registration with metadata and atomic persistence, current-version
queries (last registration wins, None when never registered), and history
retention per component and overall, including across registry restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.config import hermes_home
from hermes.models.version_registry import ModelVersionRegistry


@pytest.mark.unit
def test_register_version(tmp_path: Path) -> None:
    registry = ModelVersionRegistry()  # default path under per-test HERMES_HOME
    expected_path = hermes_home() / "model_versions.json"
    assert registry.path == expected_path

    record = registry.register(
        "main-model",
        "main-2026-11-20",
        provider="openai",
        model_id="gpt-5.2",
        eval_score=0.87,
    )
    assert record.component == "main-model"
    assert record.version_id == "main-2026-11-20"
    assert record.provider == "openai"
    assert record.model_id == "gpt-5.2"
    assert record.eval_score == 0.87
    assert record.deployed_at  # stamped by register()
    assert record.gepa_run_id is None

    # Persisted atomically at the default location.
    doc = json.loads(expected_path.read_text(encoding="utf-8"))
    assert doc["entries"][0]["version_id"] == "main-2026-11-20"

    gepa_record = registry.register(
        "prompt.compaction",
        "gepa-run-42-compaction",
        eval_score=0.91,
        gepa_run_id="gepa-run-42",
    )
    assert gepa_record.gepa_run_id == "gepa-run-42"
    assert gepa_record.eval_score == 0.91

    # Invalid registrations are rejected.
    with pytest.raises(ValueError):
        registry.register("", "v1")
    with pytest.raises(ValueError):
        registry.register("main-model", "")


@pytest.mark.unit
def test_get_current_version(tmp_path: Path) -> None:
    store = tmp_path / "model_versions.json"
    registry = ModelVersionRegistry(path=store)

    assert registry.get_current("main-model") is None  # nothing registered yet

    registry.register("main-model", "v1")
    registry.register("prompt.compaction", "p1")
    registry.register("main-model", "v2")

    current = registry.get_current("main-model")
    assert current is not None
    assert current.version_id == "v2"  # last registration wins
    prompt_current = registry.get_current("prompt.compaction")
    assert prompt_current is not None
    assert prompt_current.version_id == "p1"
    assert registry.get_current("never-registered") is None

    # Current version survives a restart via the persisted file.
    reloaded = ModelVersionRegistry(path=store)
    reloaded_current = reloaded.get_current("main-model")
    assert reloaded_current is not None
    assert reloaded_current.version_id == "v2"
    assert reloaded_current.deployed_at == current.deployed_at


@pytest.mark.unit
def test_list_history(tmp_path: Path) -> None:
    store = tmp_path / "model_versions.json"
    registry = ModelVersionRegistry(path=store)
    registry.register("main-model", "v1", provider="openai", model_id="m1")
    registry.register("prompt.compaction", "p1", eval_score=0.9, gepa_run_id="run-1")
    registry.register("main-model", "v2", provider="openai", model_id="m2")

    main_history = registry.get_history("main-model")
    assert [entry.version_id for entry in main_history] == ["v1", "v2"]
    assert all(entry.component == "main-model" for entry in main_history)

    prompt_history = registry.get_history("prompt.compaction")
    assert [entry.version_id for entry in prompt_history] == ["p1"]
    assert prompt_history[0].gepa_run_id == "run-1"
    assert prompt_history[0].eval_score == 0.9

    everything = registry.get_history()
    assert [entry.version_id for entry in everything] == ["v1", "p1", "v2"]

    # History is retained across restarts.
    reloaded = ModelVersionRegistry(path=store)
    assert [entry.version_id for entry in reloaded.get_history()] == ["v1", "p1", "v2"]
    assert [entry.version_id for entry in reloaded.get_history("main-model")] == [
        "v1",
        "v2",
    ]
