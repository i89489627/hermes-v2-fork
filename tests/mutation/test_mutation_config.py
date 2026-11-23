"""P20.T5 acceptance tests: mutation-testing infrastructure (plan Phase 20).

pyproject ``[tool.mutmut]`` holds the in-scope deterministic-logic modules
per plan Section 4.7 that exist today. This file carries the FULL intended
Tier-1/Tier-2 manifest as constants so the scope self-tightens: whenever a
later phase lands one of the planned modules on disk,
``test_in_scope_modules_exist`` fails until it joins ``paths_to_mutate``.

Tier-1 modules carry a 0.95 mutation-score bar and Tier-2 modules 0.85
(Section 4.7). The thresholds are mirrored into pyproject as custom
``[tool.mutmut]`` keys — ``tier1_score_threshold`` / ``tier2_score_threshold``
(mutmut reads only the keys it knows, so the extras are inert) — asserted
equal to the constants here, and the CI workflow (``.github/workflows/
mutation.yml``) is pinned to a nightly full ``mutmut run`` plus a per-PR
diff-scope job.

No ``mutmut run`` is executed here (slow by design); the nightly job in the
workflow covers the real run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "mutation.yml"

#: plan Section 4.7, Tier 1: core deterministic logic (score bar 0.95).
EXPECTED_TIER1: tuple[str, ...] = (
    "hermes/cron/watchdog.py",
    "hermes/memory/trust_card.py",
    "hermes/optimization/metrics.py",
    "hermes/optimization/pareto.py",
    "hermes/skills/lifecycle.py",
)

#: plan Section 4.7, Tier 2: supporting deterministic logic (score bar 0.85).
EXPECTED_TIER2: tuple[str, ...] = (
    "hermes/config.py",
    "hermes/install/migrations.py",
    "hermes/mcp/argument_validator.py",
    "hermes/models/version_registry.py",
    "hermes/traces/schema.py",
)

TIER1_THRESHOLD = 0.95
TIER2_THRESHOLD = 0.85


def _mutmut_config() -> dict[str, Any]:
    """Parse ``[tool.mutmut]`` out of pyproject.toml."""
    with open(PYPROJECT_PATH, "rb") as fh:
        doc = tomllib.load(fh)
    return doc["tool"]["mutmut"]


def _workflow() -> dict[str, Any]:
    """Parse the mutation workflow YAML (handling pyyaml's bare-``on`` bool)."""
    doc: dict[str, Any] = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return doc


@pytest.mark.unit
def test_in_scope_modules_exist() -> None:
    """Every configured scope path exists, and every landed planned module is in it.

    (a) every path in ``paths_to_mutate`` exists on disk (a renamed/moved
    module must be updated in the scope, not left to nightly CI to trip over);
    (b) every EXPECTED tier module that exists on disk is included in
    ``paths_to_mutate`` — so when later phases land the remaining Section 4.7
    modules, this test forces them into the mutation scope.
    """
    config = _mutmut_config()
    paths = list(config["paths_to_mutate"])
    assert paths, "[tool.mutmut] paths_to_mutate must not be empty"

    missing_on_disk = [p for p in paths if not (REPO_ROOT / p).is_file()]
    assert not missing_on_disk, (
        f"[tool.mutmut] paths_to_mutate contains paths that do not exist: "
        f"{missing_on_disk}"
    )

    not_in_scope = [
        module
        for tier in (EXPECTED_TIER1, EXPECTED_TIER2)
        for module in tier
        if (REPO_ROOT / module).is_file() and module not in paths
    ]
    assert not not_in_scope, (
        f"Section 4.7 tier modules exist on disk but are missing from "
        f"[tool.mutmut] paths_to_mutate: {not_in_scope}"
    )

    # The Tier-1 set is the plan's core deterministic logic: it must exist now.
    absent_tier1 = [m for m in EXPECTED_TIER1 if not (REPO_ROOT / m).is_file()]
    assert not absent_tier1, f"plan Section 4.7 Tier-1 modules missing: {absent_tier1}"


@pytest.mark.unit
def test_tier_thresholds_match_section_4_7() -> None:
    """pyproject carries machine-readable copies of the Section 4.7 score bars.

    ``tier1_score_threshold`` / ``tier2_score_threshold`` are custom keys in
    ``[tool.mutmut]`` (TOML has no comment-as-data mechanism, and mutmut
    ignores unknown keys); they must equal the constants this test file
    publishes for the score-raising work in P20.T6.
    """
    config = _mutmut_config()
    assert float(config["tier1_score_threshold"]) == TIER1_THRESHOLD
    assert float(config["tier2_score_threshold"]) == TIER2_THRESHOLD


@pytest.mark.unit
def test_ci_workflow_has_nightly_mutation_job() -> None:
    """mutation.yml pins the nightly full run and the per-PR diff-scope job."""
    doc = _workflow()
    # pyyaml parses the bare `on:` trigger key as boolean True.
    triggers: dict[str, Any] = doc.get("on") or doc.get(True) or {}

    # Nightly schedule: cron trigger with a valid five-field expression.
    schedule = triggers.get("schedule")
    assert isinstance(schedule, list) and schedule, (
        "mutation.yml must have a `schedule:` trigger"
    )
    crons = [entry["cron"] for entry in schedule]
    assert crons and all(isinstance(c, str) and len(c.split()) == 5 for c in crons), (
        f"schedule cron expressions must be five-field cron strings, got {crons}"
    )
    assert "workflow_dispatch" in triggers, "manual dispatch must stay available"

    # Full-scope nightly job: named `mutation`, runs `mutmut run`, publishes
    # the report artifact, and never burns per-PR minutes.
    jobs: dict[str, Any] = doc["jobs"]
    assert "mutation" in jobs, "a job named `mutation` must run the full scope"
    mutation = jobs["mutation"]
    assert "pull_request" in str(mutation.get("if", "")), (
        "the full mutation job must be gated away from pull_request events"
    )
    full_run_steps = [
        step
        for step in mutation["steps"]
        if "mutmut run" in str(step.get("run", "")) and "if" not in step
    ]
    assert full_run_steps, "job `mutation` must run `mutmut run`"
    assert any(
        str(step.get("uses", "")).startswith("actions/upload-artifact")
        for step in mutation["steps"]
    ), "job `mutation` must publish a report artifact"

    # Per-PR diff job: computes changed files vs the base with `git diff`,
    # runs mutmut only when an in-scope module changed, within a 10-min budget.
    assert "pull_request" in triggers, "mutation.yml must trigger on pull_request"
    assert "mutation-diff" in jobs, "a per-PR diff-scope mutation job must exist"
    diff_job = jobs["mutation-diff"]
    assert diff_job.get("timeout-minutes", 10**6) <= 10, (
        "the diff-scope mutation job targets <10 minutes"
    )
    assert any("git diff" in str(step.get("run", "")) for step in diff_job["steps"]), (
        "the diff job must compute changed files with `git diff`"
    )
    gated_steps = [
        step
        for step in diff_job["steps"]
        if "mutmut run" in str(step.get("run", "")) and step.get("if")
    ]
    assert gated_steps, "the diff job must run mutmut behind a changed-files gate"
    assert any(
        "steps.changed.outputs.count" in str(step.get("if", "")) for step in gated_steps
    ), "the mutmut run must be gated on the changed in-scope module count"
