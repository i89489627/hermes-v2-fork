"""P11.T2 acceptance: trace capture wired into the pipeline (plan Phase 11).

Covered: one turn through the v2 memory stack produces exactly ONE trace in
the :class:`~hermes.traces.store.TraceStore`; a FULL turn (pre-step CBDC
injection over the shared store + Stop-hook pipeline run) merges every stage
component into that single trace — ``turn_id``, ``user_prompt``, ``models_used``
(the documented ``model_name`` labels of the extractor/reviewer/gate model
callables), ``extraction``, ``review``, ``gating``, ``injection`` — and the
Stop hook writes no duplicate marker when the pipeline covers the same store
(one-trace-per-turn contract, plan P11.T2). Trace-capture failures never
break the turn (plan Section 5.5). The ``memory_ops`` MCP server
(``hermes/mcp/memory_ops_server.py``) is driven through its dispatcher layers
and a real stdio subprocess (bootstrap-compatible zero-arg ``serve()``).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hermes.mcp.client import McpStdioClient
from hermes.mcp.memory_ops_server import (
    MEMORY_GET_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_STATS_TOOL_NAME,
    build_request_handler,
    register_memory_ops_server,
)
from hermes.memory.context import ContextManager
from hermes.memory.extractor import ExtractorAgent
from hermes.memory.gating import CBDCGate
from hermes.memory.hooks.stop_hook import build_stop_hook
from hermes.memory.hooks.user_prompt_submit_hook import build_prompt_submit_hook
from hermes.memory.injector import ContextInjector
from hermes.memory.pipeline import MemoryPipeline, PipelineResult
from hermes.memory.retrieval import RetrievalEngine
from hermes.memory.reviewer import ReviewerAgent
from hermes.memory.store import MemoryStore
from hermes.traces.quality_labeler import (
    auto_label,
    label_source,
    label_trace,
    manual_override,
)
from hermes.traces.schema import Trace
from hermes.traces.store import TraceStore

_PIPELINE_LOGGER = "hermes.memory.pipeline"

_TURN_TEXT = "how do I deploy the acme service today?"
_SESSION = "s1"

_DEPLOY_FACT = "acme deploy pipeline runs via GitHub Actions"
_DRAFT_FACT = "User asked how to deploy the acme service"
#: Filler corpus so the target terms keep positive BM25 idf (a term hitting
#: the only document of a tiny corpus scores 0 and is dropped by P4.T1).
_FILLER_FACTS = (
    "quarterly budget review spreadsheet numbers",
    "gardening log tomatoes basil balcony",
    "weekly team sync offsite notes",
    "grandma apple pie recipe cinnamon nutmeg",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Zero-argument subprocess bootstrap (memory-ops server over stdio).
_MEMORY_OPS_BOOTSTRAP = (
    "from hermes.mcp.memory_ops_server import serve\nserve()\n"
)


class _NamedModel:
    """Fake single-model callable carrying the documented ``model_name``."""

    def __init__(self, model_name: str, response: str) -> None:
        self.model_name = model_name
        self._response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


class _ScriptedGateModel:
    """Fake gate model: scripted decision per candidate id found in the prompt."""

    def __init__(self, decisions: dict[str, str]) -> None:
        self.model_name = "fake-gate"
        self._decisions = decisions
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        # the gate prompt embeds its own instruction example ("<id>") — skip it
        found = [
            candidate_id
            for candidate_id in re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if candidate_id != "<id>"
        ]
        return json.dumps([
            {
                "candidate_id": candidate_id,
                "decision": self._decisions.get(candidate_id, "ignore"),
                "reason": "scripted decision",
            }
            for candidate_id in found
        ])


def _json_draft() -> str:
    """Extractor model output: one draft fact for the turn."""
    return json.dumps([
        {
            "type": "Project",
            "text": _DRAFT_FACT,
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ])


def _json_add() -> str:
    """Reviewer model output: always add (no existing entries to match)."""
    return json.dumps({
        "action": "add",
        "target_id": None,
        "rationale": "new durable fact",
        "merged_text": None,
    })


def _seed_store(tmp_path: Path) -> MemoryStore:
    """Tmp-rooted store seeded with the relevant fact + the idf filler corpus."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    store.add(_DEPLOY_FACT, type="Project")
    for filler in _FILLER_FACTS:
        store.add(filler, type="Reference")
    return store


def _make_pipeline(
    store: MemoryStore,
    trace_store: TraceStore | None,
    extractor_model: _NamedModel | None = None,
    reviewer_model: _NamedModel | None = None,
) -> MemoryPipeline:
    """Real pipeline over scripted single-model fakes (never called live)."""
    extractor = ExtractorAgent(
        model=(
            extractor_model
            if extractor_model is not None
            else _NamedModel("fake-extractor", _json_draft())
        )
    )
    reviewer = ReviewerAgent(
        model=(
            reviewer_model
            if reviewer_model is not None
            else _NamedModel("fake-reviewer", _json_add())
        ),
        search_fn=lambda _query: [],
    )
    return MemoryPipeline(extractor, reviewer, store, trace_store=trace_store)


def _jsonl_trace_count(trace_store: TraceStore) -> int:
    """Number of trace records across the store's JSONL day files."""
    day_dir = trace_store.root / "v1.0"
    if not day_dir.is_dir():
        return 0
    return sum(
        1
        for path in day_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


# ── P11.T2: test_turn_creates_trace ───────────────────────────────────────────


@pytest.mark.integration
def test_turn_creates_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One pipeline turn -> exactly ONE trace; capture failure never breaks it.

    Pipeline-only turn (no injection ran): the trace carries turn_id, session,
    prompt, models_used, extraction and review; gating/injection stay None
    (plan P11.T2: "gating summary (if injection ran — else None)"). A
    TraceStore.append that raises is logged and the turn still completes with
    its store writes.
    """
    store = _seed_store(tmp_path)
    trace_store = TraceStore(root=tmp_path / "traces")
    pipeline = _make_pipeline(store, trace_store)

    result = pipeline.run_after_turn(
        "t1", "From now on I prefer dark mode in all my IDEs", session_id=_SESSION
    )
    assert isinstance(result, PipelineResult)
    assert [record.action for record in result.applied] == ["add"]

    traces = trace_store.search("dark mode", limit=10)
    assert len(traces) == 1
    assert _jsonl_trace_count(trace_store) == 1  # exactly one trace for one turn
    trace = traces[0]
    assert trace.turn_id == "t1"
    assert trace.session_id == _SESSION
    assert trace.user_prompt == "From now on I prefer dark mode in all my IDEs"
    assert trace.trace_id  # assigned by the store on append
    assert trace.timestamp > 0
    # models_used: documented model_name labels, extractor then reviewer
    assert trace.models_used == ["fake-extractor", "fake-reviewer"]
    # extraction component: draft count + ids + full drafts
    assert trace.extraction is not None
    assert trace.extraction["drafts_count"] == 1
    assert trace.extraction["draft_ids"] == [result.drafts[0].draft_id]
    assert [draft["text"] for draft in trace.extraction["drafts"]] == [_DRAFT_FACT]
    # review component: actions by type + rationales + applied + elapsed
    assert trace.review is not None
    assert trace.review["actions_by_type"] == {"add": 1, "update": 0, "skip": 0, "conflict": 0}
    assert trace.review["rationales"] == ["new durable fact"]
    assert [record["entry_id"] for record in trace.review["applied"]] == [
        result.applied[0].entry_id
    ]
    assert trace.review["elapsed_ms"] >= 0
    # no injection ran for this turn -> gating/injection stay None
    assert trace.gating is None
    assert trace.injection is None

    # ── failure safety: a raising trace store must not break the turn ────
    def _raising_append(_self: TraceStore, _trace: Any) -> str:
        raise RuntimeError("trace store down")

    monkeypatch.setattr(TraceStore, "append", _raising_append)
    with caplog.at_level(logging.WARNING, logger=_PIPELINE_LOGGER):
        second = pipeline.run_after_turn(
            "t2", "Also the build cache lives in .cache/acme", session_id=_SESSION
        )
    assert isinstance(second, PipelineResult)
    assert [record.action for record in second.applied] == ["add"]
    assert store.get(second.applied[0].entry_id) is not None  # the write happened
    assert "trace capture failed" in caplog.text


# ── P11.T2: test_trace_includes_all_components ────────────────────────────────


@pytest.mark.integration
def test_trace_includes_all_components(tmp_path: Path) -> None:
    """Full turn (injection + pipeline, one shared store) -> one complete trace.

    The pre-step hook (gate candidates seeded) records the gating/injection
    stage summaries for the pipeline to merge, so the single per-turn trace
    has ALL components non-None: turn_id, user_prompt, models_used
    (extractor/reviewer/gate), extraction, review, gating, injection.
    """
    store = _seed_store(tmp_path)
    deploy_id = store.list(context="global")[0].id
    trace_store = TraceStore(root=tmp_path / "traces")

    gate_model = _ScriptedGateModel({deploy_id: "use"})
    injector = ContextInjector(
        retrieval=RetrievalEngine(store),
        gate=CBDCGate(gate_model=gate_model),
        context_manager=ContextManager(repo_root=tmp_path / "memory-repo"),
    )
    prompt_hook = build_prompt_submit_hook(injector, trace_store=trace_store)
    pipeline = _make_pipeline(store, trace_store)
    stop_hook = build_stop_hook(pipeline, trace_store=trace_store)

    pre_step_payload: dict[str, Any] = {
        "user_prompt": _TURN_TEXT,
        "session_metadata": {"session_id": _SESSION},
        "session_id": _SESSION,
        "context": "global",
    }
    merged = prompt_hook(pre_step_payload)
    assert "additionalContext" in merged  # the gate injected the seeded claim
    assert _DEPLOY_FACT in merged["additionalContext"]

    stop_hook({
        "turn_id": "t1",
        "turn_text": _TURN_TEXT,
        "session_id": _SESSION,
        "context": "global",
    })
    stop_hook.futures[-1].result(timeout=30)  # the async pipeline run finished

    traces = trace_store.search("deploy acme", limit=10)
    assert len(traces) == 1
    assert _jsonl_trace_count(trace_store) == 1  # no duplicate stop-hook marker
    trace = traces[0]
    assert trace.turn_id == "t1"
    assert trace.user_prompt == _TURN_TEXT
    assert trace.session_id == _SESSION
    # ALL components non-None (plan P11.T2 acceptance)
    assert trace.models_used == ["fake-extractor", "fake-reviewer", "fake-gate"]
    assert trace.extraction is not None and trace.extraction["drafts_count"] == 1
    assert trace.review is not None and trace.review["actions_by_type"]["add"] == 1
    assert trace.gating is not None
    assert trace.gating["context"] == "global"
    assert trace.gating["decisions_by_type"] == {"use": 1, "verify": 0}
    assert trace.gating["fallback_used"] is False
    assert trace.injection is not None
    assert trace.injection["claims"] == 1
    assert trace.injection["chars"] > 0
    assert trace.injection["fallback_used"] is False
    # the gated claim really came from the seeded entry this turn retrieved
    assert gate_model.calls == 1  # single batched gate call per turn (plan P4.T2)


# ── P11.T2 additive: the Stop-hook marker on an UNcovered store ───────────────


@pytest.mark.integration
def test_stop_hook_marker_on_uncovered_store(tmp_path: Path) -> None:
    """Pipeline without a trace store -> the hook's own marker trace lands.

    The P11.T2 Stop-hook marker records that a pipeline run was dispatched
    for the turn on a store the pipeline itself does not trace; when the
    pipeline holds the SAME store no marker is written (the full turn trace
    covers it — one-trace-per-turn).
    """
    store = _seed_store(tmp_path)
    trace_store = TraceStore(root=tmp_path / "traces")
    pipeline = _make_pipeline(store, None)  # the pipeline does NOT trace
    stop_hook = build_stop_hook(pipeline, trace_store=trace_store)

    stop_hook({"turn_id": "t9", "turn_text": _TURN_TEXT, "session_id": _SESSION})
    stop_hook.futures[-1].result(timeout=30)

    assert _jsonl_trace_count(trace_store) == 1
    marker = trace_store.search("deploy acme", limit=10)[0]
    assert marker.turn_id == "t9"
    assert marker.session_id == _SESSION
    assert marker.user_prompt == _TURN_TEXT
    assert marker.models_used == ["stop-hook"]
    assert marker.extraction == {"marker": "pipeline-run-dispatched"}


# ── P11.T2 files_to_create: hermes/mcp/memory_ops_server.py ───────────────────


@pytest.mark.integration
def test_memory_ops_server_tools(tmp_path: Path) -> None:
    """The dispatcher serves stats/search/get with the echo contract (#156143)."""
    store = MemoryStore(root=tmp_path / "memory-repo")
    deploy = store.add("Deployments run via GitHub Actions", type="Project")
    store.add("Gardening log: tomatoes and basil on the balcony", type="Reference")
    # idf filler corpus: with a 2-doc corpus the target terms score exactly 0
    # (a term in half the docs has zero BM25Okapi idf) and would be dropped
    for filler in _FILLER_FACTS:
        store.add(filler, type="Reference")

    dispatch = register_memory_ops_server(store=store)
    listing = dispatch("tools/list", {})
    names = [tool["name"] for tool in listing["tools"]]
    assert names == [MEMORY_STATS_TOOL_NAME, MEMORY_SEARCH_TOOL_NAME, MEMORY_GET_TOOL_NAME]

    stats = dispatch("tools/call", {"name": MEMORY_STATS_TOOL_NAME, "arguments": {}})
    assert stats["stats"]["total_entries"] == 2 + len(_FILLER_FACTS)
    assert stats["stats"]["by_type"] == {
        "Project": 1,
        "Reference": 1 + len(_FILLER_FACTS),
    }
    assert stats["stats"]["by_context"] == {"global": 2 + len(_FILLER_FACTS)}
    assert 0.0 <= stats["stats"]["average_trust"] <= 1.0
    assert stats["arguments_received"] == {}  # echo contract

    search = dispatch(
        "tools/call", {"name": MEMORY_SEARCH_TOOL_NAME, "arguments": {"query": "GitHub Actions"}}
    )
    assert [result["id"] for result in search["results"]] == [deploy.id]
    assert search["results"][0]["text"] == "Deployments run via GitHub Actions"
    assert search["arguments_received"] == {"query": "GitHub Actions"}

    fetched = dispatch(
        "tools/call", {"name": MEMORY_GET_TOOL_NAME, "arguments": {"entry_id": deploy.id}}
    )
    assert fetched["found"] is True
    assert fetched["entry"]["type"] == "Project"
    assert fetched["entry"]["trust_score"] == deploy.trust_score
    assert fetched["entry"]["verification_events"] == 1

    miss = dispatch(
        "tools/call", {"name": MEMORY_GET_TOOL_NAME, "arguments": {"entry_id": "no-such-id"}}
    )
    assert miss["found"] is False and miss["entry"] is None

    # invalid arguments and unknown tools raise (the stdio layer turns them
    # into JSON-RPC error responses)
    with pytest.raises(ValueError):
        dispatch("tools/call", {"name": MEMORY_SEARCH_TOOL_NAME, "arguments": {"query": ""}})
    with pytest.raises(ValueError):
        dispatch("tools/call", {"name": "bogus", "arguments": {}})

    # same dispatcher behind the serve_stdio-shaped request handler
    request_handler = build_request_handler(dispatch)
    response = request_handler({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": MEMORY_STATS_TOOL_NAME, "arguments": {}},
    })
    assert response is not None
    assert response["stats"]["total_entries"] == 2 + len(_FILLER_FACTS)


@pytest.mark.integration
def test_memory_ops_server_serves_over_stdio(tmp_path: Path) -> None:
    """Zero-arg ``serve()`` subprocess: tools/list + tools/call end to end.

    The subprocess serves the DEFAULT store (``hermes_home()/memory-repo`` —
    the per-test HERMES_HOME), so an entry added by the parent is visible to
    the child; ``McpStdioClient.call_tool`` enforces the echo contract.
    """
    store = MemoryStore()  # default root under the per-test HERMES_HOME
    entry = store.add("The nightly backup cron runs at 03:00", type="Project")

    with McpStdioClient([sys.executable, "-c", _MEMORY_OPS_BOOTSTRAP], cwd=_REPO_ROOT) as client:
        listing = client.request("tools/list")
        names = [tool.get("name") for tool in listing.get("tools", [])]
        assert MEMORY_SEARCH_TOOL_NAME in names
        assert MEMORY_GET_TOOL_NAME in names
        assert MEMORY_STATS_TOOL_NAME in names

        result = client.call_tool(MEMORY_GET_TOOL_NAME, {"entry_id": entry.id})
        assert result["found"] is True
        assert result["entry"]["id"] == entry.id
        assert result["arguments_received"] == {"entry_id": entry.id}

        search = client.call_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "backup cron nightly"})
        assert [hit["id"] for hit in search["results"]] == [entry.id]


# ── P11.T3: auto-quality labeling (plan Phase 11) ─────────────────────────────


_GOOD_TURN_TEXT = "Always sign release builds with the acme release key"
_PARSE_FAILURE_TURN_TEXT = "um idk whatever lol"
_CONFLICT_TURN_TEXT = "Maybe we should email the standup summary instead of meetings"
#: Deliberately lexically disjoint from the seeded deploy fact so the
#: ConflictDetector's deterministic trigram backend stays silent and the
#: conflict comes from the MODEL decision, not the P8.T2 detector rule.
_CONFLICT_DRAFT_TEXT = "User prefers email summaries over standup meetings"


def _json_meeting_draft() -> str:
    """Extractor model output for the conflict turn: one unrelated draft."""
    return json.dumps([
        {
            "type": "User",
            "text": _CONFLICT_DRAFT_TEXT,
            "proposed_source": "user-explicit",
            "confidence": 0.9,
        }
    ])


def _json_conflict(target_id: str) -> str:
    """Reviewer model output: conflict against a KNOWN existing entry id."""
    return json.dumps({
        "action": "conflict",
        "target_id": target_id,
        "rationale": "contradicts the recorded deployment policy",
        "merged_text": None,
    })


def _make_label_pipeline(
    store: MemoryStore,
    trace_store: TraceStore,
    *,
    extractor_model: _NamedModel,
    reviewer_model: _NamedModel,
    search_fn: Callable[[str], list[Any]],
) -> MemoryPipeline:
    """Real pipeline over scripted fakes with an explicit candidate source."""
    extractor = ExtractorAgent(model=extractor_model)
    reviewer = ReviewerAgent(model=reviewer_model, search_fn=search_fn)
    return MemoryPipeline(extractor, reviewer, store, trace_store=trace_store)


def _review_only_trace(
    actions: list[dict[str, Any]], applied: list[dict[str, Any]] | None = None
) -> Any:
    """Synthetic trace carrying just a review payload (rule-table input)."""
    return Trace(review={"actions": actions, "applied": applied or []})


@pytest.mark.integration
def test_quality_auto_labeled(tmp_path: Path) -> None:
    """Pipeline turns auto-label per the P11.T3 rule table (plan Phase 11).

    One good add turn, one extraction parse-failure turn and one conflict
    turn run through real pipelines over scripted fakes; every trace is
    auto-labeled by :func:`hermes.traces.quality_labeler.label_trace`
    (3/3 = 100%, satisfying the 80%+ acceptance) and each label matches the
    documented rule table: add applied -> ``good``, nothing learned ->
    ``acceptable``, conflict dropped by the termination rule ->
    ``needs-review``. The rule table's failure refinements (conservative
    degradation skips / store-error signature -> ``bad``, deliberate
    dedup-skips -> ``acceptable``) are asserted directly on synthetic
    traces. Labels and their ``auto`` source survive a store reload.
    """
    store = _seed_store(tmp_path)
    trace_store = TraceStore(root=tmp_path / "traces")
    deploy_id = next(
        entry.id for entry in store.list(context="global") if entry.text == _DEPLOY_FACT
    )
    deploy_entry = store.get(deploy_id)
    assert deploy_entry is not None

    # ── turn 1: good add ─────────────────────────────────────────────────
    good_pipeline = _make_label_pipeline(
        store,
        trace_store,
        extractor_model=_NamedModel("fake-extractor", _json_draft()),
        reviewer_model=_NamedModel("fake-reviewer", _json_add()),
        search_fn=lambda _query: [],
    )
    good_result = good_pipeline.run_after_turn("t-good", _GOOD_TURN_TEXT, session_id=_SESSION)
    assert isinstance(good_result, PipelineResult)
    assert [record.action for record in good_result.applied] == ["add"]

    # ── turn 2: extraction parse failure (malformed model output) ────────
    fail_pipeline = _make_label_pipeline(
        store,
        trace_store,
        extractor_model=_NamedModel("fake-extractor", "this is not a JSON array"),
        reviewer_model=_NamedModel("fake-reviewer", _json_add()),
        search_fn=lambda _query: [],
    )
    fail_result = fail_pipeline.run_after_turn(
        "t-parse", _PARSE_FAILURE_TURN_TEXT, session_id=_SESSION
    )
    assert isinstance(fail_result, PipelineResult)
    assert fail_result.drafts == [] and fail_result.actions == []  # nothing learned

    # ── turn 3: model-declared conflict, unresolved after the final pass ─
    conflict_pipeline = _make_label_pipeline(
        store,
        trace_store,
        extractor_model=_NamedModel("fake-extractor", _json_meeting_draft()),
        reviewer_model=_NamedModel("fake-reviewer", _json_conflict(deploy_id)),
        search_fn=lambda _query: [deploy_entry],
    )
    conflict_result = conflict_pipeline.run_after_turn(
        "t-conflict", _CONFLICT_TURN_TEXT, session_id=_SESSION
    )
    assert isinstance(conflict_result, PipelineResult)
    assert [action.action for action in conflict_result.actions] == ["conflict"]
    assert conflict_result.applied == []  # termination rule: dropped, never written

    # ── find the one trace per turn (distinctive prompts, FTS over store) ─
    good_trace = trace_store.search("sign release builds", limit=10)
    parse_trace = trace_store.search("whatever", limit=10)
    conflict_trace = trace_store.search("standup meetings", limit=10)
    assert len(good_trace) == len(parse_trace) == len(conflict_trace) == 1
    good_trace, parse_trace, conflict_trace = (
        good_trace[0],
        parse_trace[0],
        conflict_trace[0],
    )
    trace_ids = [good_trace.trace_id, parse_trace.trace_id, conflict_trace.trace_id]

    # payload audit: each label really follows the rule-table inputs
    assert good_trace.review is not None
    assert good_trace.review["actions_by_type"]["add"] == 1
    assert [record["action"] for record in good_trace.review["applied"]] == ["add"]
    assert parse_trace.extraction is not None
    assert parse_trace.extraction["drafts_count"] == 0
    assert parse_trace.review is not None and parse_trace.review["actions"] == []
    assert conflict_trace.review is not None
    assert [action["action"] for action in conflict_trace.review["actions"]] == ["conflict"]
    assert conflict_trace.review["applied"] == []

    # ── auto-label every trace; each label matches the rule table ────────
    labeled = {
        "t-good": label_trace(trace_store, good_trace.trace_id),
        "t-parse": label_trace(trace_store, parse_trace.trace_id),
        "t-conflict": label_trace(trace_store, conflict_trace.trace_id),
    }
    assert labeled == {
        "t-good": "good",
        "t-parse": "acceptable",
        "t-conflict": "needs-review",
    }

    # acceptance (plan P11.T3): 80%+ of traces auto-labeled — here 3/3
    relabeled = [trace_store.get(trace_id) for trace_id in trace_ids]
    assert all(trace is not None for trace in relabeled)
    labeled_count = sum(1 for trace in relabeled if trace.quality_label is not None)
    assert labeled_count == len(relabeled) == 3
    assert labeled_count / len(relabeled) >= 0.8

    # rule-table failure refinements on synthetic traces (documented mapping)
    # - conservative degradation skips are bad (malformed model output)
    assert (
        auto_label(_review_only_trace([{"action": "skip", "rationale": "reviewer failed: boom"}]))
        == "bad"
    )
    assert (
        auto_label(
            _review_only_trace([{"action": "skip", "rationale": "conflict re-review failed: x"}])
        )
        == "bad"
    )
    assert (
        auto_label(
            _review_only_trace([
                {"action": "skip", "rationale": "unparseable reviewer output"}
            ])
        )
        == "bad"
    )
    # - store-error signature: a decided add that never landed -> bad
    assert (
        auto_label(_review_only_trace([{"action": "add", "rationale": "new fact"}], applied=[]))
        == "bad"
    )
    # - deliberate dedup-skip is a correct decision -> acceptable
    assert (
        auto_label(
            _review_only_trace([
                {"action": "skip", "rationale": "duplicate of existing entry e1 (similarity 0.93)"}
            ])
        )
        == "acceptable"
    )
    # - stop-hook marker trace (no review stage): nothing learned -> acceptable
    assert auto_label(Trace(extraction={"marker": "pipeline-run-dispatched"})) == "acceptable"

    # labels + auto source survive a reload (fresh store over the same root)
    reloaded = TraceStore(root=trace_store.root)
    assert reloaded.get(good_trace.trace_id) is not None
    assert reloaded.get(good_trace.trace_id).quality_label == "good"  # type: ignore[union-attr]
    assert reloaded.get(conflict_trace.trace_id) is not None
    assert reloaded.get(conflict_trace.trace_id).quality_label == "needs-review"  # type: ignore[union-attr]
    assert label_source(reloaded, good_trace.trace_id) == "auto"
    assert label_source(reloaded, conflict_trace.trace_id) == "auto"


@pytest.mark.integration
def test_manual_label_overrides_auto(tmp_path: Path) -> None:
    """Manual override changes the label, wins over auto, survives reload.

    A good add turn auto-labels ``good``; ``manual_override`` re-labels it
    ``bad`` (plan P11.T3: manual override via /trace label <id>
    <good|bad|needs-review>), records the ``manual`` source so later
    :func:`label_trace` runs never clobber it, and both the label and its
    source survive a reload. ``acceptable`` is not manual-eligible, unknown
    labels raise ``ValueError``, unknown trace ids raise ``KeyError``.
    """
    store = _seed_store(tmp_path)
    trace_store = TraceStore(root=tmp_path / "traces")
    pipeline = _make_label_pipeline(
        store,
        trace_store,
        extractor_model=_NamedModel("fake-extractor", _json_draft()),
        reviewer_model=_NamedModel("fake-reviewer", _json_add()),
        search_fn=lambda _query: [],
    )
    pipeline.run_after_turn("t1", _GOOD_TURN_TEXT, session_id=_SESSION)
    found = trace_store.search("sign release builds", limit=10)
    assert len(found) == 1
    trace_id = found[0].trace_id

    # auto labeling first
    assert label_trace(trace_store, trace_id) == "good"
    assert trace_store.get(trace_id) is not None
    assert trace_store.get(trace_id).quality_label == "good"  # type: ignore[union-attr]
    assert label_source(trace_store, trace_id) == "auto"

    # manual override changes the label and marks the source manual
    assert manual_override(trace_store, trace_id, "bad") == "bad"
    assert trace_store.get(trace_id).quality_label == "bad"  # type: ignore[union-attr]
    assert label_source(trace_store, trace_id) == "manual"

    # a later auto-labeling run must NOT clobber the manual override
    assert label_trace(trace_store, trace_id) == "bad"
    assert trace_store.get(trace_id).quality_label == "bad"  # type: ignore[union-attr]
    assert label_source(trace_store, trace_id) == "manual"

    # the label and its source survive a reload (fresh store + sidecar read)
    reloaded = TraceStore(root=trace_store.root)
    assert reloaded.get(trace_id) is not None
    assert reloaded.get(trace_id).quality_label == "bad"  # type: ignore[union-attr]
    assert label_source(reloaded, trace_id) == "manual"

    # the remaining manual-eligible labels are accepted on a reloaded store
    assert manual_override(reloaded, trace_id, "needs-review") == "needs-review"
    assert reloaded.get(trace_id).quality_label == "needs-review"  # type: ignore[union-attr]

    # validation: acceptable is auto-only, unknown labels and ids rejected
    with pytest.raises(ValueError):
        manual_override(trace_store, trace_id, "acceptable")
    with pytest.raises(ValueError):
        manual_override(trace_store, trace_id, "excellent")
    with pytest.raises(KeyError):
        manual_override(trace_store, "no-such-trace", "good")
    with pytest.raises(KeyError):
        label_trace(trace_store, "no-such-trace")
