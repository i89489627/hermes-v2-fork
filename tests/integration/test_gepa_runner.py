"""P10.T5 acceptance tests: GEPARunner weekly optimization (plan Phase 10).

Drives the full loop end-to-end with deterministic fake runners (outputs
keyed by the prompt's last ``## Revision constraints`` marker, so candidate
variants score differently) over the real P10.T2 eval-set fixtures:

- a weekly run produces the seed + 3 generations x 3 variants = 10 candidates
  per component (acceptance: >=3) and populates each Pareto front;
- the Pareto-front best is deployed by rewriting the prompt template file —
  the deployment surface the pipeline reads at P10.T6 cron time — including
  the default ``<hermes_home>/optimization/prompts/<component>.txt`` path,
  with the drift-registry ``on_deploy`` hook wired to
  ``ModelVersionRegistry.register`` (the documented production pattern);
- every deployment leaves a ``[gepa] deploy <component> <generation>
  score=<s>`` commit in the audit Git repository;
- the structural proxy for the plan's 1-hour budget: the full 4-component
  run with fake runners completes in well under 60s (the real-run budget
  reasoning lives in the module docstring of ``hermes.optimization.gepa_runner``).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hermes.config import hermes_home
from hermes.memory.git_backend import GitBackend
from hermes.models.version_registry import ModelVersionRegistry
from hermes.optimization.adapters.extractor_adapter import ExtractorAdapter
from hermes.optimization.adapters.gate_adapter import GateAdapter
from hermes.optimization.adapters.reviewer_adapter import ReviewerAdapter
from hermes.optimization.adapters.rewriter_adapter import RewriterAdapter
from hermes.optimization.gepa_runner import (
    ComponentSpec,
    DeploymentRecord,
    GEPARunner,
    default_prompt_template_path,
)
from hermes.optimization.pareto import ParetoFrontStore
from hermes.traces.store import TraceStore

EVAL_SETS = Path(__file__).resolve().parents[1] / "fixtures" / "eval_sets"

#: The fake runner returns perfect output only for this marker — the last
#: mutation of the default loop (generation 3, variant 1). Unique best =>
#: deterministic deployment.
_PERFECT_MARKER = "generation 3, variant 1"


def _last_marker(prompt: str) -> str | None:
    """The prompt's most recent ``Revision constraints`` header, or None."""
    sections = prompt.split("## Revision constraints (")
    if len(sections) < 2:
        return None
    return sections[-1].split(")", 1)[0].strip()


def _perfect_output(expected: Any) -> dict[str, Any]:
    """Output matching the labeled expected_output, per adapter contract."""
    if isinstance(expected, list):  # extractor: {"drafts": [fact, ...]}
        return {"drafts": [str(fact) for fact in expected]}
    if isinstance(expected, dict):
        if "action" in expected:  # reviewer: {"action": verdict}
            return {"action": str(expected["action"])}
        if "decisions" in expected:  # gate: {"decisions": [{"id", "action"}]}
            return {
                "decisions": [
                    {"id": str(d["id"]), "action": str(d["action"])}
                    for d in expected["decisions"]
                ]
            }
        if "key_points" in expected:  # rewriter: {"summary": text}
            points = [str(point) for point in expected["key_points"]]
            return {"summary": " ".join(points)}
    raise AssertionError(f"unexpected expected_output shape: {expected!r}")


def _partial_output(expected: Any) -> dict[str, Any]:
    """Deliberately imperfect output (scores below the perfect variant)."""
    if isinstance(expected, list):
        return {"drafts": [str(expected[0])] if expected else []}
    if isinstance(expected, dict):
        if "action" in expected:
            return {"action": str(expected["action"]) + "-mutated"}
        if "decisions" in expected:
            return {
                "decisions": [
                    {
                        "id": str(d["id"]),
                        "action": "skip" if str(d["action"]) == "use" else "use",
                    }
                    for d in expected["decisions"]
                ]
            }
        if "key_points" in expected:
            points = [str(point) for point in expected["key_points"]]
            return {"summary": points[0] if points else ""}
    raise AssertionError(f"unexpected expected_output shape: {expected!r}")


def _fake_runner(prompt: str, example: dict[str, Any]) -> dict[str, Any]:
    """Deterministic outputs keyed by prompt content so variants differ."""
    if _last_marker(prompt) == _PERFECT_MARKER:
        return _perfect_output(example["expected_output"])
    return _partial_output(example["expected_output"])


def _seed(component: str) -> str:
    """Seed prompt template content (the pre-optimization deployment surface)."""
    return (
        f"You are the {component} stage of the hermes-v2 memory pipeline.\n"
        "Produce exactly the labeled output schema for every example."
    )


def _write_rewriter_eval_set(path: Path) -> Path:
    """A small rewriter-shaped fixture ({key_points, char_limit} expected outputs)."""
    turns = [
        (
            "Quick recap: I switched my coffee to oat milk, standup moved to Thursday, "
            "and I owe Priya the design deck by Friday.",
            ["user prefers oat milk in coffee", "standup moved to thursday", "deck owed to priya by friday"],
        ),
        (
            "Note: my laptop battery drains fast now, and I booked the flights to Lisbon "
            "for the March 3 conference.",
            ["laptop battery drains fast", "flights to lisbon booked", "conference is march 3"],
        ),
        (
            "Reminder that the budget review is Monday morning and the shared doc is "
            "already updated with the Q4 numbers.",
            ["budget review is monday morning", "q4 numbers in shared doc"],
        ),
        (
            "My sister Priya's birthday is March 14, and I still need to buy a gift.",
            ["priya's birthday is march 14", "gift not bought yet"],
        ),
        (
            "Started physical therapy for my shoulder on Tuesdays; the doctor said "
            "no overhead lifting for six weeks.",
            ["physical therapy tuesdays for shoulder", "no overhead lifting six weeks"],
        ),
        (
            "The payments team onboarded two engineers and our sprint now ends Wednesday.",
            ["two engineers onboarded to payments", "sprint ends wednesday"],
        ),
    ]
    examples = [
        {
            "id": f"rw-{index:03d}",
            "input": text,
            "expected_output": {"key_points": points, "char_limit": 400},
            "context": "compaction summary for the session tail",
        }
        for index, (text, points) in enumerate(turns, start=1)
    ]
    doc = {
        "_meta": {
            "component": "rewriter",
            "version": "1.0.0",
            "description": "test fixture: compaction-summary examples",
        },
        "examples": examples,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _build_components(
    templates_dir: Path, rewriter_eval_set: Path
) -> dict[str, ComponentSpec]:
    """The four plan components wired to their adapters + real P10.T2 fixtures."""
    return {
        "extractor": ComponentSpec(
            component="extractor",
            adapter=ExtractorAdapter(_fake_runner),
            eval_set_path=EVAL_SETS / "extractor_labeled.json",
            prompt_template_path=templates_dir / "extractor.txt",
        ),
        "gate": ComponentSpec(
            component="gate",
            adapter=GateAdapter(_fake_runner),
            eval_set_path=EVAL_SETS / "gating_labeled.json",
            prompt_template_path=templates_dir / "gate.txt",
        ),
        "reviewer": ComponentSpec(
            component="reviewer",
            adapter=ReviewerAdapter(_fake_runner),
            eval_set_path=EVAL_SETS / "reviewer_labeled.json",
            prompt_template_path=templates_dir / "reviewer.txt",
        ),
        "rewriter": ComponentSpec(
            component="rewriter",
            adapter=RewriterAdapter(_fake_runner),
            eval_set_path=rewriter_eval_set,
            prompt_template_path=templates_dir / "rewriter.txt",
        ),
    }


def _seed_templates(components: dict[str, ComponentSpec]) -> None:
    for spec in components.values():
        path = spec.resolve_template_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_seed(spec.component), encoding="utf-8")


@pytest.mark.integration
def test_weekly_run_produces_candidates(tmp_path: Path) -> None:
    """run_weekly evaluates seed + 9 variants per component and populates fronts."""
    rewriter_set = _write_rewriter_eval_set(tmp_path / "rewriter_labeled.json")
    components = _build_components(tmp_path / "templates", rewriter_set)
    _seed_templates(components)
    pareto = ParetoFrontStore(path=tmp_path / "pareto.json")
    trace_store = TraceStore(root=tmp_path / "traces")
    runner = GEPARunner(components, trace_store=trace_store, pareto=pareto)

    result = runner.run_weekly(now=datetime(2026, 11, 15, 3, 0, tzinfo=UTC))

    assert result.components_run == ["extractor", "gate", "reviewer", "rewriter"]
    assert result.run_id == "gepa-20261115T030000Z"
    assert result.elapsed_seconds > 0
    for name, count in result.candidates_per_component.items():
        # acceptance: >=3 candidates per component (default loop: seed + 3x3)
        assert count >= 3
        assert count == 1 + 3 * 3
        front = pareto.get_front(name)
        assert front, f"empty Pareto front for {name}"
        best = pareto.get_best(name)
        assert result.deployed[name] == best.prompt
        assert result.deployed[name].strip()
    # every component's summary trace landed in the trace store
    for name in result.components_run:
        assert trace_store.get(f"{result.run_id}-{name}") is not None

    # subset selection: only the requested component runs (and deploys)
    extractor_only = runner.run_weekly(components="extractor")
    assert extractor_only.components_run == ["extractor"]
    assert set(extractor_only.deployed) == {"extractor"}
    with pytest.raises(KeyError, match="unknown component"):
        runner.run_weekly(components=["extractor", "nope"])


@pytest.mark.integration
def test_deploy_updates_prompt_template(tmp_path: Path) -> None:
    """The Pareto-best candidate is deployed to the prompt template file."""
    # DEFAULT deployment surface (no explicit path on the spec):
    # <hermes_home>/optimization/prompts/<component>.txt — the file the
    # pipeline reads at P10.T6 cron time.
    template_path = default_prompt_template_path("extractor")
    assert template_path == hermes_home() / "optimization" / "prompts" / "extractor.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    seed = _seed("extractor")
    template_path.write_text(seed, encoding="utf-8")

    # drift-registry hook wiring (documented production pattern for on_deploy)
    registry = ModelVersionRegistry(path=tmp_path / "model_versions.json")
    deployments: list[DeploymentRecord] = []

    def on_deploy(record: DeploymentRecord) -> None:
        deployments.append(record)
        registry.register(
            record.component,
            record.version_id,
            eval_score=record.score,
            gepa_run_id=record.run_id,
        )

    spec = ComponentSpec(
        component="extractor",
        adapter=ExtractorAdapter(_fake_runner),
        eval_set_path=EVAL_SETS / "extractor_labeled.json",
    )
    runner = GEPARunner(
        {"extractor": spec},
        pareto=ParetoFrontStore(path=tmp_path / "pareto.json"),
        on_deploy=on_deploy,
    )
    result = runner.run_weekly(now=datetime(2026, 11, 15, 3, 0, tzinfo=UTC))

    deployed = template_path.read_text(encoding="utf-8")
    assert deployed != seed  # the template was actually updated
    assert deployed == result.deployed["extractor"]
    assert _last_marker(deployed) == _PERFECT_MARKER  # the best variant won

    assert len(deployments) == 1
    record = deployments[0]
    expected_version = hashlib.sha256(deployed.encode("utf-8")).hexdigest()[:12]
    assert record.version_id == f"extractor-prompt-{expected_version}"
    assert record.prompt == deployed
    assert record.score == pytest.approx(1.0)
    assert record.generation == 3
    assert record.run_id == result.run_id
    assert record.deployed_at == "2026-11-15T03:00:00+00:00"

    # the hook registered the deployment as the component's current version
    current = registry.get_current("extractor")
    assert current is not None
    assert current.version_id == record.version_id
    assert current.eval_score == pytest.approx(1.0)
    assert current.gepa_run_id == result.run_id


@pytest.mark.integration
def test_audit_trail_in_git(tmp_path: Path) -> None:
    """Every deployment leaves a ``[gepa] deploy ...`` commit in the audit repo."""
    repo_root = tmp_path / "audit-repo"
    git = GitBackend.init(repo_root)
    prompts_dir = repo_root / "prompts"
    prompts_dir.mkdir(parents=True)

    components: dict[str, ComponentSpec] = {}
    for name, adapter, fixture in (
        ("extractor", ExtractorAdapter(_fake_runner), "extractor_labeled.json"),
        ("gate", GateAdapter(_fake_runner), "gating_labeled.json"),
    ):
        template = prompts_dir / f"{name}.txt"
        template.write_text(_seed(name), encoding="utf-8")
        components[name] = ComponentSpec(
            component=name,
            adapter=adapter,
            eval_set_path=EVAL_SETS / fixture,
            prompt_template_path=template,
        )
    pareto = ParetoFrontStore(path=repo_root / "optimization" / "pareto-front.json")
    runner = GEPARunner(components, pareto=pareto, git=git)

    result = runner.run_weekly(now=datetime(2026, 11, 15, 3, 0, tzinfo=UTC))

    for name in ("extractor", "gate"):
        commits = git.log(path=f"prompts/{name}.txt", limit=10)
        assert commits, f"no audit commit for {name}"
        message = commits[0].message
        assert message.startswith(f"[gepa] deploy {name} ")
        assert re.search(r"score=\d\.\d{4}", message), message
        # the committed deployment content matches what landed on disk
        committed = prompts_dir / f"{name}.txt"
        assert committed.read_text(encoding="utf-8") == result.deployed[name]
        assert committed.read_text(encoding="utf-8") != _seed(name)

    # the Pareto front store is part of the audit trail too
    assert git.log(path="optimization/pareto-front.json", limit=10)
    assert len(git.log(limit=10)) >= 2  # one commit per component deployment


@pytest.mark.integration
def test_completes_within_1_hour(tmp_path: Path) -> None:
    """Structural proxy for the plan's 1-hour budget: full run < 60s with fakes.

    The plan's acceptance criterion ("weekly run completes within 1 hour") is
    structural: the default loop performs 4 components x (3 generations x
    3 variants + 1 seed) = 40 adapter evaluations plus 12 reflection batches
    over the 55-example eval sets. With fake runners this is seconds; with
    the real pipeline (and optionally an LLM-backed reflective rewriter) the
    budget is 9 reflection-model calls per component — far inside the hour.
    """
    rewriter_set = _write_rewriter_eval_set(tmp_path / "rewriter_labeled.json")
    components = _build_components(tmp_path / "templates", rewriter_set)
    _seed_templates(components)
    pareto = ParetoFrontStore(path=tmp_path / "pareto.json")
    runner = GEPARunner(components, pareto=pareto)

    started = time.monotonic()
    result = runner.run_weekly()
    wall_seconds = time.monotonic() - started

    assert wall_seconds < 60.0
    assert result.elapsed_seconds < 60.0
    assert len(result.components_run) == 4
    assert all(count >= 3 for count in result.candidates_per_component.values())
    assert all(prompt.strip() for prompt in result.deployed.values())
