"""P20.T6 acceptance: Phase 21 re-mutant validation (plan Section 4.7).

Plan Section 4.7, workflow step 5: "each bug-fix regression test must kill a
mutant reintroducing the original bug — 'tests that would have caught the
bug' becomes a verifiable criterion (validated in P20.T6)". For each critical
Phase 21 bug (P21.T1, P21.T2, P21.T11) this file REINTRODUCES the original
bug's behavior via monkeypatching, runs the bug's regression scenario, and
asserts the invariant the regression suite enforces now returns the WRONG
result — i.e. the regression suite would FAIL, so the mutant is killed by the
existing tests:

- P21.T1 (issue #35309): the fork-base interrupt state keyed by worker-thread
  ID with no cleanup. Regression suite:
  ``tests/integration/test_tool_executor.py::test_crash_does_not_poison_next_call``.
- P21.T2 (issue #26568): disabled tools registered/advertised/usable —
  filtering removed at registration AND call time. Regression suite:
  ``tests/unit/core/test_tool_registry.py::test_disabled_tool_not_registered``.
- P21.T11 (issue #25272): the update migration overwriting the user's config
  (current-wins merge semantics removed). Regression suite:
  ``tests/integration/test_migration.py::test_no_config_loss_on_update``.

Honest framing (P20.T6 acceptance "Re-mutants of P21.T1/T2/T11 bugs killed by
regression tests"): these tests prove DETECTABILITY of the reintroduced bugs
by demonstrating the exact invariant violation. They complement — not replace
— the real ``mutmut run`` (which generates the same mutations mechanically
and runs the whole suite per mutant): mutmut 3.8.1 refuses to run on native
Windows (upstream mutmut issue #397), so real mutation passes execute in the
nightly CI job (``.github/workflows/mutation.yml``), and this file is the
local, always-runnable validation of the same property.

The final test records the mutation-score trend through the diagnostics
metrics' EXISTING generic ``counters`` interface (plan Section 5.2 / P20.T6
acceptance "mutation score trend visible in diagnostics metrics");
``hermes/diagnostics/metrics.py`` itself needs no schema change — its
``MetricsSnapshot.counters`` dict plus ``append_snapshot``/``load_history``
JSONL time series already carry arbitrary named counters.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import hermes.core.execution_context as execution_context
import hermes.core.tool_executor as tool_executor
from hermes.config import CONFIG_SCHEMA_VERSION_KEY, load_raw_config
from hermes.core.execution_context import ExecutionContext, current_execution
from hermes.core.tool_executor import ToolExecutor
from hermes.core.tool_registry import ToolDef, ToolRegistry
from hermes.diagnostics.metrics import MetricsCollector, MetricsSnapshot
from hermes.install import migrations as install_migrations
from hermes.install.migrations import run_migration

# ── P21.T1 re-mutant: per-thread-ID interrupt leak (#35309) ──────────────────


class _WorkerAbort(BaseException):
    """BaseException-derived crash, like the upstream mid-run worker abort."""


def _crash_then_followup(executor: ToolExecutor) -> tuple[bool, bool]:
    """The issue-#35309 regression scenario: crash on a worker, recycle it.

    Returns ``(followup_healthy, thread_recycled)``. ``followup_healthy`` is
    the invariant enforced by
    ``tests/integration/test_tool_executor.py::test_crash_does_not_poison_next_call``:
    the tool scheduled onto the recycled worker thread must observe its own
    non-interrupted execution state (True on the fixed code).
    """
    seen_idents: list[int] = []

    def crashing_tool() -> None:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        assert ctx is not None, "tool must see its own ExecutionContext"
        ctx.request_interrupt()  # mark interrupted, then die mid-run
        time.sleep(0.05)
        raise _WorkerAbort("worker aborted mid-run")

    crashing = executor.submit("flaky-tool", crashing_tool)
    with pytest.raises(_WorkerAbort):
        crashing.result(timeout=10)

    def healthy_tool() -> bool:
        seen_idents.append(threading.get_ident())
        ctx = current_execution()
        return ctx is not None and not ctx.is_interrupted()

    followup = executor.submit("healthy-tool", healthy_tool)
    healthy = followup.result(timeout=10)
    executor.shutdown(wait=True)
    recycled = len(seen_idents) == 2 and seen_idents[0] == seen_idents[1]
    return healthy, recycled


@pytest.mark.unit
def test_remutant_p21_t1_thread_poisoning_reintroduced_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reintroduce the #35309 thread-ID-keyed interrupt leak; the invariant fails.

    Control: the v2 per-execution context keeps a recycled thread clean (the
    regression invariant holds). Mutant: interrupt state is keyed by thread
    ID in a registry that is never cleared (the fork-base leak — including a
    neutered ``clear_current_execution``), so the follow-up execution on the
    recycled thread sees ``is_interrupted() == True``. The regression suite's
    assertion (``followup.result() is True``) would FAIL -> the mutant is
    killed by the existing tests.
    """
    # control: the fix holds — the recycled thread starts clean
    with ToolExecutor(max_workers=1) as executor:
        healthy, recycled = _crash_then_followup(executor)
    assert healthy is True, "the P21.T1 fix must keep the invariant green"
    assert recycled is True, "the scenario must recycle the worker thread"

    # mutant: per-thread-ID interrupt registry, never cleared (fork-base bug)
    poisoned: dict[int, bool] = {}

    def _buggy_request_interrupt(self: ExecutionContext) -> None:
        poisoned[threading.get_ident()] = True  # keyed by THREAD id (#35309)

    def _buggy_is_interrupted(self: ExecutionContext) -> bool:
        return poisoned.get(threading.get_ident(), False)  # stale thread state

    monkeypatch.setattr(
        execution_context.ExecutionContext, "request_interrupt", _buggy_request_interrupt
    )
    monkeypatch.setattr(
        execution_context.ExecutionContext, "is_interrupted", _buggy_is_interrupted
    )
    # the leak's second half: the finally-clear is neutered
    monkeypatch.setattr(tool_executor, "clear_current_execution", lambda: None)

    with ToolExecutor(max_workers=1) as executor:
        mutant_healthy, mutant_recycled = _crash_then_followup(executor)

    assert mutant_recycled is True  # same scenario shape: thread recycled
    assert poisoned, "the mutant's thread-keyed registry recorded the crash"
    # THE INVARIANT NOW RETURNS THE WRONG RESULT: under the reintroduced bug
    # the next call on the recycled thread believes it is interrupted, so
    # test_crash_does_not_poison_next_call's `result() is True` assertion
    # fails -> the regression suite kills this re-mutant.
    assert mutant_healthy is False


# ── P21.T2 re-mutant: disabled tools registered/advertised/usable (#26568) ───


def _tool(name: str, toolset: str) -> ToolDef:
    """A minimal ToolDef (same shape as the P21.T2 acceptance fixtures)."""

    def _impl(*args: Any, **kwargs: Any) -> str:
        return f"{name} ran"

    return ToolDef(name=name, toolset=toolset, callable=_impl, description=f"{name} tool")


_TOOLSET_MAP = {
    "terminal": [_tool("shell_exec", "terminal"), _tool("run_tests", "terminal")],
    "browser": [_tool("web_search", "browser"), _tool("open_page", "browser")],
    "memory": [_tool("memory_search", "memory")],
}

_CONFIG = {"disabled_tools": ["shell_exec"], "disabled_toolsets": ["browser"]}


@pytest.mark.unit
def test_remutant_p21_t2_disabled_tool_registration_reintroduced_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reintroduce #26568: config filtering removed; the invariant fails.

    Control: the registry built from config contains only enabled toolsets
    minus disabled tools (the P21.T2 registration-time fix). Mutant:
    ``register_from_config`` skips the enabled/disabled filtering AND
    ``is_available`` degrades to "registered = available" (the pre-fix
    call-time behavior), so the disabled toolset's tool enters the registry,
    is advertised through ``list_tools``, and dispatches. The regression
    suite's assertions (``get(...) is None``, absent from ``list_tools``,
    ``is_available() is False``, ``ToolNotAvailableError``) would FAIL ->
    the mutant is killed by the existing tests.
    """
    # control: the fix holds — disabled tools never enter the registry
    registry = ToolRegistry(config=_CONFIG, toolset_map=_TOOLSET_MAP)
    assert registry.get("web_search") is None
    assert registry.is_available("web_search") is False
    assert [tool.name for tool in registry.list_tools()] == ["memory_search", "run_tests"]

    # mutant: registration-time and call-time filtering both removed
    def _buggy_register_from_config(
        self: ToolRegistry, toolset_map: dict[str, list[ToolDef]]
    ) -> None:
        for toolset, tools in toolset_map.items():
            for tool in tools:
                if tool.toolset != toolset:
                    raise ValueError(f"toolset mismatch for {tool.name!r}")
        self._tools = {}
        for tools in toolset_map.values():
            for tool in tools:
                self._tools[tool.name] = tool  # BUG: no config filtering (#26568)

    monkeypatch.setattr(ToolRegistry, "register_from_config", _buggy_register_from_config)
    monkeypatch.setattr(
        ToolRegistry, "is_available", lambda self, name: name in self._tools
    )

    mutant = ToolRegistry(config=_CONFIG, toolset_map=_TOOLSET_MAP)
    advertised = [tool.name for tool in mutant.list_tools()]

    # THE INVARIANTS NOW RETURN THE WRONG RESULT: the disabled toolset's tool
    # is registered, advertised to the model, and dispatches — exactly the
    # #26568 behavior; test_disabled_tool_not_registered asserts the opposite,
    # so the regression suite fails under the mutant -> the mutant is killed.
    assert mutant.get("web_search") is not None  # was None under the fix
    assert "web_search" in advertised  # was absent under the fix
    assert mutant.is_available("web_search") is True  # was False under the fix
    guarded = mutant.wrap_invoker(lambda name, *args, **kwargs: f"{name} ran")
    assert guarded("web_search") == "web_search ran"  # a disabled call EXECUTES


# ── P21.T11 re-mutant: update migration overwrites the user config (#25272) ──


def _custom_models(count: int) -> dict[str, Any]:
    """Build ``count`` custom model config entries like a real user's setup."""
    return {
        f"custom-{index}": {
            "provider": "openai-compatible",
            "base_url": f"https://inference.example-{index}.internal/v1",
            "model": f"teamFineTune-{index}",
            "api_key_env": f"TEAM_MODEL_{index}_KEY",
        }
        for index in range(1, count + 1)
    }


_CURRENT_CONFIG: dict[str, Any] = {
    CONFIG_SCHEMA_VERSION_KEY: 2,
    "custom_models": _custom_models(5),
    "v2": {"memory_pipeline": {"retry": {"max_retries": 9}}},
}

_INCOMING_CONFIG: dict[str, Any] = {
    CONFIG_SCHEMA_VERSION_KEY: 3,
    "models": {"built-in-a": {"provider": "nous"}, "built-in-b": {"provider": "openai"}},
    "v2": {"memory_pipeline": {"gating": {"mode": "batched"}}},
}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.unit
def test_remutant_p21_t11_overwrite_migration_reintroduced_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reintroduce #25272: the merge overwrites; 5 custom models vanish.

    Control: ``run_migration``'s current-wins merge keeps all 5 custom models
    (the P21.T11 zero-loss guarantee). Mutant: the merge step is replaced by
    an overwrite (``dict(incoming)`` — keep-current/merge semantics removed),
    reproducing the fork-base update that dropped every custom model. The
    regression suite's assertion (``after["custom_models"] == custom_models``)
    would FAIL -> the mutant is killed by the existing tests.
    """
    current_path = tmp_path / "config.yaml"
    _write_yaml(current_path, _CURRENT_CONFIG)

    # control: the fix holds — every custom model survives the update
    result = run_migration(current_path, _INCOMING_CONFIG)
    after = load_raw_config(current_path)
    assert after is not None
    assert after["custom_models"] == _CURRENT_CONFIG["custom_models"]
    assert result.merged_config["custom_models"] == _CURRENT_CONFIG["custom_models"]
    assert set(result.merged_config["models"]) == {"built-in-a", "built-in-b"}

    # mutant: the merge step overwrites with the incoming template
    def _buggy_merge(incoming: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        return dict(incoming)  # BUG: user config discarded (#25272)

    monkeypatch.setattr(install_migrations, "deep_merge_config", _buggy_merge)

    overwritten_path = tmp_path / "config-overwritten.yaml"
    _write_yaml(overwritten_path, _CURRENT_CONFIG)
    mutant_result = run_migration(overwritten_path, _INCOMING_CONFIG)
    lost = load_raw_config(overwritten_path)
    assert lost is not None

    # THE INVARIANT NOW RETURNS THE WRONG RESULT: all 5 custom models are
    # GONE from the merged config and from the written file — the exact
    # #25272 data loss; test_no_config_loss_on_update asserts the opposite,
    # so the regression suite fails under the mutant -> the mutant is killed.
    assert "custom_models" not in mutant_result.merged_config
    assert "custom_models" not in lost
    assert set(lost["models"]) == {"built-in-a", "built-in-b"}


# ── P20.T6: mutation-score trend through the diagnostics metrics counters ────


@pytest.mark.unit
def test_mutation_score_trend_recorded_in_diagnostics_metrics(tmp_path: Path) -> None:
    """Nightly mutation scores trend through the metrics counters time series.

    P20.T6 acceptance "mutation score trend visible in diagnostics metrics"
    is satisfied WITHOUT a ``hermes/diagnostics/metrics.py`` schema change:
    ``MetricsSnapshot.counters`` is a generic named-counter dict and the
    collector persists snapshots as a JSONL time series, so each nightly
    mutation job appends one snapshot carrying the per-tier mutation scores
    (recorded as integer per-mille to respect the ``dict[str, int]`` counter
    type). ``load_history`` reads the trend back in order and survives the
    ``to_dict``/``from_dict`` roundtrip.
    """
    collector = MetricsCollector(store_path=tmp_path / "metrics.jsonl")

    nightly_scores = (
        ("2026-11-18", 940, 840),
        ("2026-11-19", 948, 851),
        ("2026-11-20", 955, 860),
    )
    for day, tier1_permille, tier2_permille in nightly_scores:
        collector.append_snapshot(
            MetricsSnapshot(
                timestamp=datetime.fromisoformat(f"{day}T03:00:00+00:00"),
                counters={
                    "mutation_score_tier1_permille": tier1_permille,
                    "mutation_score_tier2_permille": tier2_permille,
                },
            )
        )

    history = collector.load_history()
    assert len(history) == 3
    tier1_trend = [
        snapshot.counters["mutation_score_tier1_permille"] for snapshot in history
    ]
    tier2_trend = [
        snapshot.counters["mutation_score_tier2_permille"] for snapshot in history
    ]
    assert tier1_trend == [940, 948, 955]  # rising trend, oldest first
    assert tier2_trend == [840, 851, 860]
    # the Section 4.7 thresholds are visible as the trend's target line
    assert tier1_trend[-1] >= 950  # Tier-1 bar: >=95%
    assert tier2_trend[-1] >= 850  # Tier-2 bar: >=85%

    # the trend survives the JSON roundtrip (restart-safe time series)
    roundtripped = MetricsSnapshot.from_dict(history[-1].to_dict())
    assert roundtripped.counters == history[-1].counters
    assert roundtripped.counters["mutation_score_tier1_permille"] == 955
