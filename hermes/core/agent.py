"""hermes-v2 agent core (plan P0.T3).

Extends the fork-base ``AIAgent`` (``run_agent.py``, kept temporarily per plan
Phase 0 Files) with the hook points the v2 memory pipeline needs:

- ``register_pre_step_hook`` — fired before each model call (UserPromptSubmit
  equivalent; P4.T4 wires CBDC injection here).
- ``register_post_turn_hook`` — fired after each completed turn (Stop
  equivalent; P3.T5 wires the async memory pipeline here).
- ``register_command_execution_hook`` — fired after each completed command
  (PostToolUse equivalent; P2.T5 wires the automatic ground-truth anchors
  built by ``hermes.memory.anchors.build_command_hook`` here).

Hooks are advisory: a failing hook is logged and skipped, never allowed to
break the turn (plan Section 5.5 graceful-degradation principle).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from hermes.memory.injector import ContextInjector

from hermes.config import hermes_home
from hermes.memory.conflict_detector import ConflictDetector
from hermes.memory.context import (
    CONTEXTS_DIRNAME,
    GLOBAL_CONTEXT,
    GLOBAL_DIRNAME,
    METADATA_FILENAME,
    ContextManager,
    UnknownContextError,
)
from hermes.memory.entry import MemoryEntry
from hermes.memory.hooks.stop_hook import StopHook, build_stop_hook
from hermes.memory.hooks.user_prompt_submit_hook import (
    PromptSubmitHook,
    build_prompt_submit_hook,
)
from hermes.memory.pipeline import MemoryPipeline
from hermes.memory.soul import (
    SOUL_DIRNAME,
    SOUL_STYLE_FILENAME,
    SOUL_STYLE_LIMIT,
    SoulManager,
    StyleLimitError,
)
from hermes.memory.store import INDEX_NAME, MemoryStore
from hermes.memory.trust_card import TrustCard, parse, serialize
from hermes.optimization.pareto import ParetoFrontStore
from hermes.skills.auto_deprecator import AutoDeprecator
from hermes.skills.lifecycle import (
    InvalidTransitionError,
    Skill,
    SkillLifecycle,
    SkillState,
)
from hermes.skills.registry import SkillRegistry
from hermes.traces.store import TraceStore

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from run_agent import AIAgent as _ForkBaseAgent
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "fork-base run_agent.py is not importable; run hermes-v2 from the "
        "repository root (or install the hermes-agent package)"
    ) from exc

logger = logging.getLogger(__name__)

#: Hook receiving the pre-step payload (user prompt + session metadata) and
#: returning additionalContext to merge into the model call, or ``None``.
PreStepHook = Callable[[dict[str, Any]], "dict[str, Any] | None"]

#: Hook receiving the completed-turn payload (turn id, transcript reference).
PostTurnHook = Callable[[dict[str, Any]], None]

#: Hook receiving the completed command result (PostToolUse equivalent,
#: plan P2.T5); returns the hook's outcome payload or ``None``.
CommandExecutionHook = Callable[[Any], Any]


class V2Agent(_ForkBaseAgent):  # type: ignore[misc]  # fork-base AIAgent is untyped
    """Fork-base ``AIAgent`` + v2 memory-pipeline hook points.

    Construction is a pure passthrough: every fork-base constructor argument
    keeps its meaning, so existing sessions and callers are unaffected until a
    v2 feature flag opts the agent into the memory pipeline.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pre_step_hooks: list[PreStepHook] = []
        self._post_turn_hooks: list[PostTurnHook] = []
        self._command_execution_hooks: list[CommandExecutionHook] = []

    def register_pre_step_hook(self, hook: PreStepHook) -> None:
        """Register a pre-step hook (order of registration is execution order)."""
        self._pre_step_hooks.append(hook)

    def register_post_turn_hook(self, hook: PostTurnHook) -> None:
        """Register a post-turn hook (order of registration is execution order)."""
        self._post_turn_hooks.append(hook)

    def register_command_execution_hook(self, hook: CommandExecutionHook) -> None:
        """Register a command-execution hook (plan P2.T5 PostToolUse equivalent).

        Order of registration is execution order. Typical hook: the automatic
        ground-truth anchor built by
        :func:`hermes.memory.anchors.build_command_hook`.
        """
        self._command_execution_hooks.append(hook)

    def run_pre_step_hooks(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run all pre-step hooks; merge their additionalContext into one dict.

        A hook that raises is logged and skipped — the turn proceeds without
        its contribution (additionalContext from earlier hooks is kept).
        """
        additional_context: dict[str, Any] = {}
        for hook in self._pre_step_hooks:
            hook_name = getattr(hook, "__name__", repr(hook))
            try:
                contribution = hook(payload)
            except Exception:
                logger.exception("pre-step hook %r failed; skipped", hook_name)
                continue
            if contribution:
                additional_context.update(contribution)
        return additional_context

    def run_post_turn_hooks(self, payload: dict[str, Any]) -> None:
        """Run all post-turn hooks; failures are logged, never propagated."""
        for hook in self._post_turn_hooks:
            hook_name = getattr(hook, "__name__", repr(hook))
            try:
                hook(payload)
            except Exception:
                logger.exception("post-turn hook %r failed; skipped", hook_name)

    def on_command_execution(self, cmd_result: Any) -> list[Any]:
        """Run all command-execution hooks (plan P2.T5 PostToolUse equivalent).

        ``cmd_result`` is the completed command result (typically a mapping
        with ``command`` / ``exit_code`` / ``changed_files`` /
        ``session_cited_entry_ids``). Returns the non-``None`` hook outcomes
        in execution order; a failing hook is logged and skipped — never
        propagated to the command caller (same advisory policy as the other
        hook families).
        """
        outcomes: list[Any] = []
        for hook in self._command_execution_hooks:
            hook_name = getattr(hook, "__name__", repr(hook))
            try:
                outcome = hook(cmd_result)
            except Exception:
                logger.exception(
                    "command-execution hook %r failed; skipped", hook_name
                )
                continue
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes


__all__ = [
    "CommandExecutionHook",
    "CONTEXT_COMMANDS",
    "LEARNING_COMMANDS",
    "MEMORY_COMMANDS",
    "PostTurnHook",
    "PreStepHook",
    "SKILL_COMMANDS",
    "SOUL_COMMANDS",
    "V2Agent",
    "handle_context_command",
    "handle_learning_command",
    "handle_memory_command",
    "install_memory_pipeline",
    "install_prompt_injection",
    "handle_skills_command",
    "handle_soul_command",
    "list_effective_entries",
]


# ── /skills slash command (plan P13.T4) ─────────────────────────────────────
#
# Thin orchestration over SkillLifecycle (the state machine, plan P13.T1) and
# SkillRegistry (the version store, plan P13.T2). The agent PROPOSES skills
# autonomously (lifecycle.propose); the user governs them here: list, inspect,
# approve, deprecate, rollback. Spec doc: commands/skills.md (plan Appendix A).
#
# Rollback semantics: the registry re-activates the target version's content
# as a NEW latest version (non-destructive; plan P13.T2), and the lifecycle
# record is synced via lifecycle.version so both views of the shared
# _skill_registry.json agree. Version bumps require the approved/active state
# (lifecycle transition table), which the rollback branch pre-checks so a
# rejected command never leaves a half-applied store mutation behind.

#: Subcommand dispatch registry for ``/skills`` (plan P13.T4).
SKILL_COMMANDS: Final[tuple[str, ...]] = (
    "list",
    "inspect",
    "approve",
    "deprecate",
    "rollback",
)

_SKILLS_USAGE: Final[str] = (
    "usage: /skills <subcommand>\n"
    "  list                        show every skill (state, version, last_used_at)\n"
    "  inspect <name>              detailed view of one skill\n"
    "  approve <id>                approve a proposed skill (state -> active)\n"
    "  deprecate <name> [reason]   retire an active skill (state -> deprecated)\n"
    "  rollback <name> <version>   restore a prior version's content as a new version"
)


def _resolve_skill_name(lifecycle: SkillLifecycle, name: str) -> Skill | None:
    """Exact skill id first, then the slugified display name (mirrors lifecycle)."""
    skill = lifecycle.get(name)
    if skill is not None:
        return skill
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return lifecycle.get(slug) if slug else None


def _last_used_display(last_used: dict[str, datetime] | None, skill_id: str) -> str:
    """ISO timestamp for ``skill_id`` from the usage mapping, or ``never``."""
    if not last_used:
        return "never"
    stamp = last_used.get(skill_id)
    return stamp.isoformat() if isinstance(stamp, datetime) else "never"


def _skills_list(
    lifecycle: SkillLifecycle, last_used: dict[str, datetime] | None
) -> str:
    """One line per skill: id, state, version, last_used_at."""
    skills = sorted(lifecycle.list_skills(), key=lambda skill: skill.skill_id)
    if not skills:
        return (
            "no skills registered yet (the agent proposes them; "
            "approve with /skills approve <id>)"
        )
    lines = ["skills:"]
    for skill in skills:
        lines.append(
            f"  {skill.skill_id}  {skill.state.value}  v{skill.version}  "
            f"last_used_at={_last_used_display(last_used, skill.skill_id)}"
        )
    return "\n".join(lines)


def _skills_inspect(
    lifecycle: SkillLifecycle,
    name: str,
    last_used: dict[str, datetime] | None,
) -> str:
    """Detailed view of one skill (state, version, usage, version history)."""
    skill = _resolve_skill_name(lifecycle, name)
    if skill is None:
        return f"error: unknown skill {name!r}"
    versions = (
        ", ".join(str(entry.version) for entry in skill.versions) or str(skill.version)
    )
    lines = [
        f"{skill.skill_id}: {skill.state.value} v{skill.version} "
        f"({len(skill.versions)} versions)",
        f"  name: {skill.name}",
        f"  rationale: {skill.rationale}",
        f"  created_at: {skill.created_at}",
        f"  updated_at: {skill.updated_at}",
        f"  last_used_at: {_last_used_display(last_used, skill.skill_id)}",
        f"  versions: {versions}",
    ]
    if skill.deprecation_reason:
        lines.append(f"  deprecation_reason: {skill.deprecation_reason}")
    return "\n".join(lines)


def _skills_approve(lifecycle: SkillLifecycle, skill_id: str) -> str:
    """approve <id>: proposed -> active (user approval, plan Phase 13)."""
    try:
        skill = lifecycle.approve(skill_id)
    except KeyError:
        return f"error: unknown skill id {skill_id!r}"
    except InvalidTransitionError as exc:
        return f"error: {exc}"
    return f"approved {skill.skill_id}: state={skill.state.value} version={skill.version}"


def _skills_deprecate(
    lifecycle: SkillLifecycle, name: str, reason: str | None
) -> str:
    """deprecate <name> [reason]: active -> deprecated."""
    skill = _resolve_skill_name(lifecycle, name)
    if skill is None:
        return f"error: unknown skill {name!r}"
    try:
        deprecated = lifecycle.deprecate(skill.skill_id, reason or "deprecated via /skills")
    except InvalidTransitionError as exc:
        return f"error: {exc}"
    return (
        f"deprecated {deprecated.skill_id}: state={deprecated.state.value} "
        f"(reason: {deprecated.deprecation_reason})"
    )


def _skills_rollback(
    lifecycle: SkillLifecycle, registry: SkillRegistry, name: str, version_raw: str
) -> str:
    """rollback <name> <version>: restore a prior version's content as a new version."""
    skill = _resolve_skill_name(lifecycle, name)
    if skill is None:
        return f"error: unknown skill {name!r}"
    try:
        version = int(version_raw)
    except ValueError:
        return f"error: version must be an integer, got {version_raw!r}"
    record = lifecycle.get(skill.skill_id)
    if record is not None and record.state not in (SkillState.APPROVED, SkillState.ACTIVE):
        return (
            f"error: cannot roll back skill {skill.skill_id!r} while in state "
            f"{record.state.value!r} (version bumps need approved/active)"
        )
    # Re-read the shared store through a fresh registry: the registry caches
    # the file independently of the lifecycle writer, so a long-lived
    # instance may be stale relative to lifecycle-side mutations.
    fresh_registry = SkillRegistry(store_path=registry.path)
    try:
        latest_before = fresh_registry.get_latest_version(skill.skill_id).version
        entry = fresh_registry.rollback(skill.skill_id, version)
    except KeyError as exc:
        return f"error: {exc}"
    # Sync the lifecycle record (its cache predates the registry's write) so
    # subsequent list/inspect/deprecate calls see the rolled-back state.
    if record is not None and record.version < entry.version:
        lifecycle.version(skill.skill_id, entry.yaml)
    if entry.version == latest_before:
        return f"{skill.skill_id} already at version {entry.version}: content unchanged"
    return (
        f"rolled back {skill.skill_id} to version {version}: content restored "
        f"as version {entry.version}"
    )


def handle_skills_command(
    agent: Any,
    args: list[str],
    lifecycle: SkillLifecycle,
    registry: SkillRegistry,
    last_used: dict[str, datetime] | None = None,
) -> str:
    """Dispatch ``/skills <subcommand>`` (plan P13.T4).

    ``agent`` is the harness context (reserved — the command is pure
    orchestration over ``lifecycle`` + ``registry`` today; kept first for
    dispatch-signature symmetry with the other slash commands). ``last_used``
    optionally maps skill ids to their last-used timestamps (from the usage
    ledger, plan P13.T3) for the ``list``/``inspect`` displays. Returns the
    user-facing reply text; expected failures (unknown skill, bad version,
    illegal transition) come back as ``error: ...`` strings instead of
    exceptions so the slash-command harness never crashes.
    """
    if not args:
        return _SKILLS_USAGE
    subcommand, *rest = args
    if subcommand not in SKILL_COMMANDS:
        return f"unknown /skills subcommand {subcommand!r}\n{_SKILLS_USAGE}"
    if subcommand == "list":
        return _skills_list(lifecycle, last_used)
    if subcommand == "inspect":
        if not rest:
            return "usage: /skills inspect <name>"
        return _skills_inspect(lifecycle, rest[0], last_used)
    if subcommand == "approve":
        if not rest:
            return "usage: /skills approve <id>"
        return _skills_approve(lifecycle, rest[0])
    if subcommand == "deprecate":
        if not rest:
            return "usage: /skills deprecate <name> [reason]"
        return _skills_deprecate(lifecycle, rest[0], " ".join(rest[1:]) or None)
    # rollback <name> <version>
    if len(rest) < 2:
        return "usage: /skills rollback <name> <version>"
    return _skills_rollback(lifecycle, registry, rest[0], rest[1])


# ── /context slash command (plan P5.T2) ─────────────────────────────────────
#
# Thin orchestration over ContextManager (plan P5.T1). ``ContextManager``
# owns detect/switch/create/list; this command adds the user-facing surface
# (list / switch / create / inspect / delete). The delete branch is
# implemented HERE (not in context.py) because plan P5.T2's file list extends
# only agent.py: directory removal plus a metadata (contexts.json) update
# using the public layout constants of hermes.memory.context, following the
# same atomic-write convention the memory stores use. Deleting a context
# refuses the implicit ``global`` root and any context with attached active
# sessions unless ``--force`` is given; children whose ``parent`` pointed at
# the deleted context fall back to direct global inheritance (their parent
# link is cleared in the metadata).

#: Subcommand dispatch registry for ``/context`` (plan P5.T2).
CONTEXT_COMMANDS: Final[tuple[str, ...]] = (
    "list",
    "switch",
    "create",
    "inspect",
    "delete",
)

_CONTEXT_USAGE: Final[str] = (
    "usage: /context <subcommand>\n"
    "  list                          show every context (parent, sessions, facts)\n"
    "  switch <name>                 switch the session to <name> (reloads the stable context)\n"
    "  create <name> [--parent <p>]  create a named context (lowercase slug)\n"
    "  inspect [name]                details for one context (default: global)\n"
    "  delete <name> [--force]       remove a context + its facts; --force overrides"
    " the active-session guard"
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomic text write (unique temp + ``os.replace``), store-file convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _context_facts_dir(context_manager: ContextManager, name: str) -> Path:
    """Facts directory backing context ``name`` (aligned with MemoryStore)."""
    base = (
        context_manager.root / GLOBAL_DIRNAME
        if name == GLOBAL_CONTEXT
        else context_manager.root / CONTEXTS_DIRNAME / name
    )
    return base / "facts"


def _context_fact_count(context_manager: ContextManager, name: str) -> int:
    """Number of fact files stored in context ``name`` (0 when absent)."""
    facts_dir = _context_facts_dir(context_manager, name)
    return len(list(facts_dir.glob("*.md"))) if facts_dir.is_dir() else 0


def _context_list(context_manager: ContextManager) -> str:
    """One line per context: name, parent, active sessions, fact count."""
    contexts = context_manager.list_contexts()
    lines = ["contexts:"]
    for context in contexts:
        sessions = ", ".join(context.active_sessions) or "none"
        lines.append(
            f"  {context.name}  parent={context.parent or '-'}  "
            f"sessions={len(context.active_sessions)} ({sessions})  "
            f"facts={_context_fact_count(context_manager, context.name)}"
        )
    return "\n".join(lines)


def _context_inspect(context_manager: ContextManager, name: str) -> str:
    """Details for one context: parent, path, active sessions, fact count."""
    if not context_manager.context_exists(name):
        return f"error: unknown context {name!r}"
    context = context_manager.switch_context(name)
    sessions = ", ".join(context.active_sessions) or "none"
    return "\n".join(
        [
            f"context: {context.name}",
            f"  parent: {context.parent or '-'}",
            f"  path: {context.path}",
            f"  active_sessions: {len(context.active_sessions)} ({sessions})",
            f"  facts: {_context_fact_count(context_manager, name)}",
        ]
    )


def _context_delete(context_manager: ContextManager, name: str, force: bool) -> str:
    """Remove a context directory + its metadata entry (plan P5.T2).

    Refuses the implicit ``global`` root and any context with attached
    active sessions unless ``force``. Children pointing at the deleted
    context as parent get their parent link cleared (they fall back to
    direct global inheritance).
    """
    if name == GLOBAL_CONTEXT:
        return "error: the 'global' context is the implicit root and cannot be deleted"
    if not context_manager.context_exists(name):
        return f"error: unknown context {name!r}"
    context = context_manager.switch_context(name)
    if context.active_sessions and not force:
        return (
            f"error: context {name!r} has {len(context.active_sessions)} active "
            f"session(s) ({', '.join(context.active_sessions)}); re-run with --force"
        )
    context_dir = context_manager.root / CONTEXTS_DIRNAME / name
    if context_dir.is_dir():
        shutil.rmtree(context_dir)
    # Metadata cleanup (best-effort): drop the context entry and clear
    # dangling parent links. A corrupt metadata file never blocks deletion —
    # directory presence is the authoritative existence signal (plan P5.T1).
    metadata_path = context_manager.root / METADATA_FILENAME
    if metadata_path.exists():
        try:
            doc: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("contexts"), dict):
            contexts: dict[str, Any] = doc["contexts"]
            contexts.pop(name, None)
            for entry in contexts.values():
                if isinstance(entry, dict) and entry.get("parent") == name:
                    entry["parent"] = None
            _atomic_write_text(metadata_path, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return (
        f"deleted context {name!r} (facts directory and metadata entry removed; "
        f"its facts are gone — global memory is untouched)"
    )


def handle_context_command(
    agent: Any,
    args: list[str],
    context_manager: ContextManager,
) -> str:
    """Dispatch ``/context <subcommand>`` (plan P5.T2).

    Subcommands: ``list``, ``switch <name>``, ``create <name> [--parent <p>]``,
    ``inspect [name]`` (default ``global``), ``delete <name> [--force]``.
    ``agent`` is the reserved harness context (unused by the dispatch — kept
    first for signature symmetry with the other slash commands). Expected
    failures (unknown context, bad slug, guarded delete) come back as
    ``error: ...`` strings instead of exceptions so the harness never crashes.
    """
    if not args:
        return _CONTEXT_USAGE
    subcommand, *rest = args
    if subcommand not in CONTEXT_COMMANDS:
        return f"unknown /context subcommand {subcommand!r}\n{_CONTEXT_USAGE}"
    if subcommand == "list":
        return _context_list(context_manager)
    if subcommand == "switch":
        if not rest:
            return "usage: /context switch <name>"
        try:
            context = context_manager.switch_context(rest[0])
        except UnknownContextError as exc:
            return f"error: {exc}"
        return (
            f"switched to context {context.name!r} "
            f"(parent={context.parent or '-'}; rebuild the stable context at this boundary)"
        )
    if subcommand == "create":
        if not rest:
            return "usage: /context create <name> [--parent <p>]"
        name, *flags = rest
        # Default parent is the explicit "global" root (the metadata then
        # shows the inheritance link; the effective view reaches global
        # either way).
        parent: str | None = GLOBAL_CONTEXT
        if flags and flags[0] == "--parent":
            if len(flags) < 2:
                return "usage: /context create <name> [--parent <p>]"
            parent = flags[1]
        try:
            context = context_manager.create_context(name, parent=parent)
        except (ValueError, UnknownContextError) as exc:
            return f"error: {exc}"
        return (
            f"created context {context.name!r} (parent={context.parent or '-'}); "
            f"facts inherit global unless overridden"
        )
    if subcommand == "inspect":
        return _context_inspect(context_manager, rest[0] if rest else GLOBAL_CONTEXT)
    # delete <name> [--force]
    if not rest:
        return "usage: /context delete <name> [--force]"
    return _context_delete(context_manager, rest[0], "--force" in rest[1:])


# ── effective fact view (plan P5.T2 acceptance: contexts inherit global) ────
#
# MemoryStore.list is strictly per-context (named contexts never bleed);
# the /context and injection surfaces need the MERGED view: the context's
# own facts plus everything inherited up the parent chain, with ``global``
# as the chain root (plan P5.T1: "the root context chain always reaches
# global"). An entry id present both in the context and in an ancestor is
# reported once from the nearest (context) copy — "inherits global facts
# unless overridden". Listing logic intentionally lives here (a small
# helper over store.list), NOT in hermes/memory/context.py.


def list_effective_entries(
    store: MemoryStore,
    context: str,
    context_manager: ContextManager | None = None,
) -> list[MemoryEntry]:
    """Merged fact view for ``context``: own facts + inherited (plan P5.T2).

    Walks the parent chain via ``context_manager`` when provided (unknown
    names and cycles degrade gracefully — the walk just stops); ``global``
    is always included as the chain root even when the metadata does not
    reach it. Entries are ordered nearest-first (own context, then
    ancestors, global last) and de-duplicated by id on the nearest copy.
    """
    chain: list[str] = [context]
    seen: set[str] = {context}
    if context_manager is not None and context != GLOBAL_CONTEXT:
        contexts = {c.name: c for c in context_manager.list_contexts()}
        current = context
        while current != GLOBAL_CONTEXT:
            parent = contexts.get(current)
            if parent is None:
                break
            next_name = parent.parent
            if next_name is None or next_name in seen:
                break
            chain.append(next_name)
            seen.add(next_name)
            current = next_name
    if GLOBAL_CONTEXT not in seen:
        chain.append(GLOBAL_CONTEXT)
    entries: list[MemoryEntry] = []
    seen_ids: set[str] = set()
    for name in chain:
        for entry in store.list(context=name):
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                entries.append(entry)
    return entries


# ── /memory slash command (plan P15.T3) ─────────────────────────────────────
#
# Introspection + bulk transfer over MemoryStore (plan Phase 15): list,
# search (BM25 via store.search), stats (counts + trust distribution),
# export, import. Export/import fidelity: an entry is exported as its FULL
# markdown file (trust-card frontmatter + text body, exactly what
# MemoryStore writes) — parse/serialize round-trip byte-exactly, so a
# re-imported entry carries its whole trust card including every
# verification event. Import therefore cannot go through MemoryStore.add
# (it would mint a new id and a fresh event): the import helper writes the
# entry file directly the same way the store does (documented store-layout
# use: root/global/facts/<id>.md or root/contexts/<name>/facts/<id>.md,
# index regenerated + one batch Git commit), as the plan task pre-approves.
# Import conflict policy: an id that already exists is skipped (default) or
# overwritten (--overwrite); the optional ConflictDetector (plan P8.T1)
# flags near-duplicate/contradicting ADDs in the summary without blocking
# them — the policy governs, the detector informs.

#: Subcommand dispatch registry for ``/memory`` (plan P15.T3).
MEMORY_COMMANDS: Final[tuple[str, ...]] = ("list", "search", "stats", "export", "import")

#: Import conflict policies (plan P15.T3 acceptance: skip or overwrite).
_MEMORY_CONFLICT_POLICIES: Final[tuple[str, ...]] = ("skip", "overwrite")

_MEMORY_USAGE: Final[str] = (
    "usage: /memory <subcommand>\n"
    "  list [context]              entries of one context (or all)\n"
    "  search <query>              BM25 keyword search, best first\n"
    "  stats                       counts by type/context + trust distribution\n"
    "  export [out_path]           dump every entry (full trust cards) to JSON\n"
    "  import <path> [--skip|--overwrite]   import an export file (default: skip existing ids)"
)

#: Import JSON envelope version.
_MEMORY_EXPORT_VERSION: Final[int] = 1

#: Context-name rule aligned with MemoryStore's private validation.
_CONTEXT_SAFE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _memory_entry_card(entry: MemoryEntry) -> TrustCard:
    """Project a :class:`MemoryEntry` onto its trust card (export helper)."""
    return TrustCard(
        id=entry.id,
        type=entry.type,
        created_at=entry.created_at,
        last_verified_at=entry.last_verified_at,
        trust_score=entry.trust_score,
        verification_events=list(entry.verification_events),
        context=entry.context,
        scope=list(entry.scope),
        ttl_days=entry.ttl_days,
        tags=list(entry.tags),
        pii_checked=entry.pii_checked,
        attachments=list(entry.attachments),
    )


def _memory_snippet(text: str, width: int = 64) -> str:
    """First line of ``text`` truncated to ``width`` chars (display helper)."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line if len(first_line) <= width else first_line[: width - 1] + "…"


def _memory_list(store: MemoryStore, context: str | None) -> str:
    """Formatted entry table for ``context`` (``None`` = all contexts)."""
    entries = store.list(context=context)
    header = f"entries ({context or 'all contexts'}): {len(entries)}"
    if not entries:
        return f"{header}\n  (no entries)"
    lines = [header]
    for entry in entries:
        lines.append(
            f"  {entry.id}  {entry.type:<9} ctx={entry.context}  "
            f"trust={entry.trust_score:.2f}  {_memory_snippet(entry.text)}"
        )
    return "\n".join(lines)


def _memory_search(store: MemoryStore, query: str) -> str:
    """BM25 search results, best first (plan P15.T3 ``search <query>``)."""
    results = store.search(query, top_k=10)
    if not results:
        return f"no entries match {query!r}"
    lines = [f"search {query!r}: {len(results)} match(es)"]
    for entry in results:
        lines.append(
            f"  {entry.id}  {entry.type:<9} ctx={entry.context}  "
            f"trust={entry.trust_score:.2f}  {_memory_snippet(entry.text)}"
        )
    return "\n".join(lines)


def _memory_stats(store: MemoryStore) -> str:
    """Counts by type/context + trust distribution (plan P15.T3 ``stats``)."""
    entries = store.list()
    by_type: dict[str, int] = {}
    by_context: dict[str, int] = {}
    for entry in entries:
        by_type[entry.type] = by_type.get(entry.type, 0) + 1
        by_context[entry.context] = by_context.get(entry.context, 0) + 1
    scores = [entry.trust_score for entry in entries]
    lines = [f"total: {len(entries)}"]
    lines.append(
        "by type: "
        + (
            ", ".join(f"{kind}={count}" for kind, count in sorted(by_type.items()))
            if by_type
            else "-"
        )
    )
    lines.append(
        "by context: "
        + (
            ", ".join(f"{name}={count}" for name, count in sorted(by_context.items()))
            if by_context
            else "-"
        )
    )
    if scores:
        below = sum(1 for score in scores if score < 0.3)
        lines.append(
            f"trust: avg={sum(scores) / len(scores):.2f} min={min(scores):.2f} "
            f"below_0.3={below}"
        )
    else:
        lines.append("trust: avg=- min=- below_0.3=0 (no entries)")
    return "\n".join(lines)


def _memory_export(store: MemoryStore, out_path: Path | None) -> str:
    """Serialize ALL entries (full trust cards) to one JSON file.

    Each entry is stored as its complete markdown file content under
    ``entries[].markdown`` (trust-card frontmatter + body) so the import
    path can restore ids, trust cards and verification histories exactly.
    """
    target = out_path if out_path is not None else (
        hermes_home() / f"memory-export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    entries = []
    for entry in store.list():
        entries.append(
            {
                "id": entry.id,
                "context": entry.context,
                "markdown": serialize(_memory_entry_card(entry), entry.text),
            }
        )
    doc = {
        "version": _MEMORY_EXPORT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    _atomic_write_text(target, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return f"exported {len(entries)} entries -> {target}"


def _memory_import_entry_path(store: MemoryStore, context: str) -> Path:
    """On-disk facts path for an imported entry (the store's own layout)."""
    if context == "global":
        return store.root / "global" / "facts"
    return store.root / CONTEXTS_DIRNAME / context / "facts"


def _memory_import(
    store: MemoryStore,
    source: Path,
    policy: str,
    conflict_detector: ConflictDetector | None,
) -> str:
    """Import an export file with skip/overwrite conflict handling (plan P15.T3)."""
    if not source.is_file():
        return f"error: export file not found: {source}"
    try:
        doc: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"error: unreadable export file {source}: {exc}"
    raw_entries = doc.get("entries") if isinstance(doc, dict) else None
    if not isinstance(raw_entries, list):
        return f"error: export file must carry an 'entries' list: {source}"

    added = skipped = overwritten = flagged = errors = 0
    written_paths: list[Path] = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("markdown"), str):
            errors += 1
            continue
        try:
            card, body = parse(raw["markdown"])
        except ValueError:
            errors += 1
            continue
        if not _CONTEXT_SAFE_RE.fullmatch(card.context):
            errors += 1
            continue
        existing_path = store._find_path(card.id)  # store-layout use (documented)
        path = _memory_import_entry_path(store, card.context) / f"{card.id}.md"
        if existing_path is not None and policy == "skip":
            skipped += 1
            continue
        if existing_path is not None and existing_path != path:
            # The id already lives in another context: clear that copy
            # (staging the removal so the batch commit records it).
            existing_path.unlink()
            store._stage_removal(existing_path)  # store-layout use (documented)
        if existing_path is None and conflict_detector is not None and body.strip():
            existing = [
                (entry.id, entry.text) for entry in store.list(context=card.context)
            ]
            result = conflict_detector.check_conflict(body, existing)
            if result.is_conflict:
                flagged += 1
        _atomic_write_text(path, serialize(card, body))
        written_paths.append(path)
        if existing_path is not None:
            overwritten += 1
        else:
            added += 1
    if written_paths:
        store._write_index()  # store-layout use (documented)
        store._invalidate_search_cache()
        store.git.commit(
            f"[memory] import from {source.name} "
            f"(added={added}, overwritten={overwritten})",
            [*written_paths, store.root / INDEX_NAME],
        )
    return (
        f"imported from {source}: added={added} skipped={skipped} "
        f"overwritten={overwritten} conflicts_flagged={flagged} errors={errors}"
    )


def handle_memory_command(
    agent: Any,
    args: list[str],
    store: MemoryStore,
    conflict_detector: ConflictDetector | None = None,
) -> str:
    """Dispatch ``/memory <subcommand>`` (plan P15.T3).

    Subcommands: ``list [context]``, ``search <query>``, ``stats``,
    ``export [out_path]`` (default path under ``hermes_home()``),
    ``import <path> [--skip|--overwrite]`` (default skip). ``agent`` is the
    reserved harness context (unused). ``conflict_detector`` (plan P8.T1)
    optionally flags near-duplicate/contradicting imports in the summary.
    Expected failures come back as ``error: ...`` strings, never exceptions.
    """
    if not args:
        return _MEMORY_USAGE
    subcommand, *rest = args
    if subcommand not in MEMORY_COMMANDS:
        return f"unknown /memory subcommand {subcommand!r}\n{_MEMORY_USAGE}"
    if subcommand == "list":
        return _memory_list(store, rest[0] if rest else None)
    if subcommand == "search":
        if not rest:
            return "usage: /memory search <query>"
        return _memory_search(store, " ".join(rest))
    if subcommand == "stats":
        return _memory_stats(store)
    if subcommand == "export":
        return _memory_export(store, Path(rest[0]) if rest else None)
    # import <path> [--skip|--overwrite] (also --conflict <policy> form)
    policy: str = "skip"
    tokens: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token in ("--skip", "--overwrite"):
            policy = token[2:]
            index += 1
            continue
        if token == "--conflict":
            if index + 1 >= len(rest) or rest[index + 1] not in _MEMORY_CONFLICT_POLICIES:
                return "usage: /memory import <path> [--skip|--overwrite]"
            policy = rest[index + 1]
            index += 2
            continue
        if token.startswith("--conflict="):
            value = token.split("=", 1)[1]
            if value not in _MEMORY_CONFLICT_POLICIES:
                return "usage: /memory import <path> [--skip|--overwrite]"
            policy = value
            index += 1
            continue
        tokens.append(token)
        index += 1
    if not tokens:
        return _MEMORY_USAGE
    return _memory_import(store, Path(tokens[0]), policy, conflict_detector)


# ── /soul slash command (plan P15.T4) ───────────────────────────────────────
#
# User surface over SoulManager (plan P15.T1): inspect (immutable core +
# bounded adaptive style with char counts), edit-style (the ONLY write path
# — the core has no write API by design; the style is hard-bounded to
# SOUL_STYLE_LIMIT) and history. History prefers the Git log of
# ``soul/soul_style.md`` (every style update is one commit in a Git-backed
# memory repo); without Git the append-only ``.style_history.log`` content
# is shown instead (same information, weaker provenance).

#: Subcommand dispatch registry for ``/soul`` (plan P15.T4).
SOUL_COMMANDS: Final[tuple[str, ...]] = ("inspect", "edit-style", "history")

_SOUL_USAGE: Final[str] = (
    "usage: /soul <subcommand>\n"
    "  inspect            show the immutable core and the adaptive style (char counts)\n"
    "  edit-style <text|->   replace the style ('-' reads the new text from stdin)\n"
    "  history            style change history (git log, or .style_history.log without git)"
)


def _soul_inspect(soul_manager: SoulManager) -> str:
    """Core + style with char counts and the style limit (plan P15.T4)."""
    core = soul_manager.get_core()
    style = soul_manager.get_style()
    lines = [
        f"soul core (immutable, {len(core)} chars):",
        core.rstrip("\n"),
        f"soul style ({len(style)}/{SOUL_STYLE_LIMIT} chars):",
        style.rstrip("\n") if style.strip() else "(empty)",
    ]
    return "\n".join(lines)


def _soul_edit_style(soul_manager: SoulManager, text_tokens: list[str]) -> str:
    """Replace the adaptive style via SoulManager.update_style (plan P15.T4).

    ``-`` reads the replacement from stdin (one trailing newline stripped);
    otherwise the tokens are joined with spaces. Over-limit updates come
    back as ``error: ...`` with the style left untouched (the manager
    raises :class:`StyleLimitError` before writing anything).
    """
    if text_tokens and text_tokens[0] == "-":
        text = sys.stdin.read()
        if text.endswith("\r\n"):
            text = text[:-2]
        elif text.endswith("\n"):
            text = text[:-1]
    else:
        text = " ".join(text_tokens)
    try:
        soul_manager.update_style(text)
    except StyleLimitError as exc:
        return f"error: {exc}"
    return f"style updated ({len(text)}/{SOUL_STYLE_LIMIT} chars)"


def _soul_history(soul_manager: SoulManager, git_backend: Any) -> str:
    """Style change history: git log of the style file, or the history log.

    ``git_backend`` (when provided) must be a GitBackend over
    ``soul_manager.root``; without one, a Git-backed repo gets a lazily
    constructed backend (the same lazy-import pattern SoulManager uses).
    """
    style_rel_path = f"{SOUL_DIRNAME}/{SOUL_STYLE_FILENAME}"
    if git_backend is None and (soul_manager.root / ".git").exists():
        from hermes.memory.git_backend import GitBackend  # lazy (pygit2)

        git_backend = GitBackend(soul_manager.root)
    if git_backend is not None:
        commits = git_backend.log(path=style_rel_path, limit=50)
        if not commits:
            return f"no style history yet (git log of {style_rel_path} is empty)"
        lines = [f"style history (git log of {style_rel_path}):"]
        for commit in commits:
            first_line = commit.message.strip().splitlines()[0] if commit.message.strip() else ""
            lines.append(
                f"  {commit.sha[:12]}  {commit.timestamp.isoformat()}  {first_line}"
            )
        return "\n".join(lines)
    # Non-Git root: the append-only history log is the only provenance.
    history_path = soul_manager.root / SOUL_DIRNAME / ".style_history.log"
    if not history_path.is_file():
        return "no style history yet (no git repository and no .style_history.log)"
    lines = ["style history (.style_history.log):"]
    for line in history_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            lines.append(f"  {parts[0]}  {parts[1]} chars")
        else:
            lines.append(f"  {line}")
    return "\n".join(lines)


def handle_soul_command(
    agent: Any,
    args: list[str],
    soul_manager: SoulManager,
    git_backend: Any = None,
) -> str:
    """Dispatch ``/soul <subcommand>`` (plan P15.T4).

    Subcommands: ``inspect``, ``edit-style <text|->`` (manual style
    override, 2000-char enforcement via :meth:`SoulManager.update_style`),
    ``history`` (git log of ``soul/soul_style.md`` when Git-backed, else the
    ``.style_history.log`` content). ``agent`` is the reserved harness
    context (unused); ``git_backend`` optionally injects the
    :class:`hermes.memory.git_backend.GitBackend` for the history view.
    Expected failures come back as ``error: ...`` strings, never exceptions.
    """
    if not args:
        return _SOUL_USAGE
    subcommand, *rest = args
    if subcommand not in SOUL_COMMANDS:
        return f"unknown /soul subcommand {subcommand!r}\n{_SOUL_USAGE}"
    if subcommand == "inspect":
        return _soul_inspect(soul_manager)
    if subcommand == "edit-style":
        if not rest:
            return "usage: /soul edit-style <text|->"
        return _soul_edit_style(soul_manager, rest)
    # history
    return _soul_history(soul_manager, git_backend)


# ── /learning slash command — the dashboard (plan P13.T5) ───────────────────
#
# Read-only visibility over the learning-loop components (plan Phase 13):
# SkillLifecycle state counts + SkillRegistry size + TraceStore volume +
# pending approvals (proposed-state skills AND AutoDeprecator proposals) +
# GEPA history. Two documented proxy details:
#
# - TraceStore (plan P11.T1) exposes append/search/label/get but NO count or
#   recent query, so the dashboard reads its JSONL source of truth
#   (``<root>/v1.0/<day>.jsonl``, the layout TraceStore itself writes)
#   read-only. JSONL may hold duplicate lines after an id re-append; the
#   dashboard de-duplicates by trace_id (last line wins) for both the count
#   and the recent list.
# - GEPA run history is proxied by ParetoFrontStore front sizes per
#   component until the P10.T5 optimization runner exists (plan note:
#   "get_front sizes as GEPA history proxy"); components are enumerated
#   from the store's JSON file, fronts resolved via the public get_front.

#: Subcommand dispatch registry for ``/learning`` (plan P13.T5).
LEARNING_COMMANDS: Final[tuple[str, ...]] = ("stats", "traces", "pending", "history")

#: How many recent traces the ``traces`` subcommand shows (plan P13.T5: N=5).
_LEARNING_RECENT_LIMIT: Final[int] = 5

_LEARNING_USAGE: Final[str] = (
    "usage: /learning <subcommand>\n"
    "  stats     skills (total/active/deprecated), pending approvals, traces count\n"
    "  traces    recent 5 traces with quality labels\n"
    "  pending   skills awaiting approval (proposed + auto-deprecation proposals)\n"
    "  history   GEPA run history (pareto front sizes until the P10.T5 runner exists)"
)


def _trace_records(trace_store: TraceStore) -> dict[str, dict[str, Any]]:
    """Unique trace records from the TraceStore JSONL source of truth.

    Read-only scan of ``<root>/v1.0/<day>.jsonl`` (day-file names sort
    chronologically; within a file, later lines are later writes). Keyed by
    trace_id with last-write-wins so an upserted trace is counted once.
    """
    records: dict[str, dict[str, Any]] = {}
    day_dir = trace_store.root / "v1.0"
    if not day_dir.is_dir():
        return records
    for path in sorted(day_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                doc: Any = json.loads(stripped)
            except ValueError:
                continue  # corrupted line: the dashboard never crashes on it
            if isinstance(doc, dict) and doc.get("trace_id"):
                records[str(doc["trace_id"])] = doc
    return records


def _learning_stats(
    lifecycle: SkillLifecycle | None,
    registry: SkillRegistry | None,
    trace_store: TraceStore | None,
    auto_deprecator: AutoDeprecator | None,
) -> str:
    """stats: total/active/deprecated skills, pending approvals, traces count."""
    total = active = deprecated = proposed = approved = archived = 0
    if lifecycle is not None:
        for skill in lifecycle.list_skills():
            total += 1
            if skill.state is SkillState.ACTIVE:
                active += 1
            elif skill.state is SkillState.DEPRECATED:
                deprecated += 1
            elif skill.state is SkillState.PROPOSED:
                proposed += 1
            elif skill.state is SkillState.APPROVED:
                approved += 1
            else:
                archived += 1
    auto_pending = len(auto_deprecator.pending()) if auto_deprecator is not None else 0
    traces = len(_trace_records(trace_store)) if trace_store is not None else 0
    lines = [
        "learning dashboard:",
        (
            f"  skills: total={total} active={active} deprecated={deprecated} "
            f"(proposed={proposed} approved={approved} archived={archived})"
        ),
        f"  pending approvals: {proposed + auto_pending} "
        f"({proposed} proposed skill(s), {auto_pending} auto-deprecation proposal(s))",
        f"  traces: {traces}",
    ]
    if registry is not None:
        lines.insert(2, f"  registry: {len(registry.list_skills())} skill record(s)")
    return "\n".join(lines)


def _learning_traces(trace_store: TraceStore | None) -> str:
    """traces: the recent N traces with quality labels (plan P13.T5)."""
    if trace_store is None:
        return "no trace store wired (pass trace_store to the dashboard)"
    records = list(_trace_records(trace_store).values())
    if not records:
        return "no traces recorded yet"
    records.sort(key=lambda doc: float(doc.get("timestamp", 0.0)), reverse=True)
    lines = [f"recent traces (up to {_LEARNING_RECENT_LIMIT} newest, quality label):"]
    for doc in records[:_LEARNING_RECENT_LIMIT]:
        label = str(doc.get("quality_label") or "unlabeled")
        stamp = datetime.fromtimestamp(float(doc.get("timestamp", 0.0)), tz=UTC)
        prompt = str(doc.get("user_prompt", ""))
        lines.append(
            f"  {str(doc.get('trace_id', ''))[:12]}  {label:<12} "
            f"{stamp.isoformat()}  {_memory_snippet(prompt)}"
        )
    return "\n".join(lines)


def _learning_pending(
    lifecycle: SkillLifecycle | None, auto_deprecator: AutoDeprecator | None
) -> str:
    """pending: proposed-state skills + auto-deprecator proposals, if any."""
    proposed = (
        [
            skill
            for skill in lifecycle.list_skills()
            if skill.state is SkillState.PROPOSED
        ]
        if lifecycle is not None
        else []
    )
    deprecations = (
        auto_deprecator.pending() if auto_deprecator is not None else []
    )
    if not proposed and not deprecations:
        return (
            "nothing pending (no proposed skills, no auto-deprecation proposals; "
            "approve proposed skills with /skills approve <id>)"
        )
    lines = ["pending approvals:"]
    if proposed:
        lines.append("  proposed skills (awaiting /skills approve):")
        lines.extend(f"    - {skill.skill_id} ({skill.name})" for skill in proposed)
    if deprecations:
        lines.append("  auto-deprecation proposals (awaiting /skills deprecate):")
        lines.extend(
            f"    - {proposal.get('skill_id')} "
            f"(last_used_at={proposal.get('last_used_at')})"
            for proposal in deprecations
        )
    return "\n".join(lines)


def _learning_history(pareto: ParetoFrontStore | None) -> str:
    """history: GEPA run history via pareto front sizes (documented proxy)."""
    if pareto is None:
        return "no pareto store wired (GEPA history unavailable)"
    components: set[str] = set()
    if pareto.path.is_file():
        try:
            doc: Any = json.loads(pareto.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        if isinstance(doc, dict):
            for raw in doc.get("candidates", []):
                if isinstance(raw, dict) and isinstance(raw.get("component"), str):
                    components.add(raw["component"])
    lines = [
        "GEPA run history (pareto front sizes as proxy until the P10.T5 runner exists):"
    ]
    if not components:
        lines.append("  (no candidates recorded yet)")
    for name in sorted(components):
        front = pareto.get_front(name)
        best = f"{front[0].mean_score():.2f}" if front else "-"
        lines.append(f"  {name}: front={len(front)} best_mean={best}")
    return "\n".join(lines)


def handle_learning_command(
    agent: Any,
    args: list[str],
    *,
    lifecycle: SkillLifecycle | None = None,
    registry: SkillRegistry | None = None,
    trace_store: TraceStore | None = None,
    pareto: ParetoFrontStore | None = None,
    auto_deprecator: AutoDeprecator | None = None,
) -> str:
    """Dispatch ``/learning <subcommand>`` — the dashboard (plan P13.T5).

    Subcommands: ``stats`` (total/active/deprecated skills, pending
    approvals, traces count), ``traces`` (recent 5 traces with quality
    labels), ``pending`` (proposed skills + auto-deprecation proposals),
    ``history`` (GEPA run history via pareto front sizes — the documented
    proxy until P10.T5's runner exists). All components are optional; an
    unwired component renders a placeholder instead of failing. ``agent``
    is the reserved harness context (unused).
    """
    if not args:
        return _LEARNING_USAGE
    subcommand, *rest = args
    if subcommand not in LEARNING_COMMANDS:
        return f"unknown /learning subcommand {subcommand!r}\n{_LEARNING_USAGE}"
    if subcommand == "stats":
        return _learning_stats(lifecycle, registry, trace_store, auto_deprecator)
    if subcommand == "traces":
        return _learning_traces(trace_store)
    if subcommand == "pending":
        return _learning_pending(lifecycle, auto_deprecator)
    # history
    return _learning_history(pareto)


# ── Stop-event post-turn hook wiring (plan P3.T5) ────────────────────────────


def install_memory_pipeline(agent: V2Agent, pipeline: MemoryPipeline) -> StopHook:
    """Register the P3.T5 Stop-event post-turn hook for ``pipeline``.

    The registered hook fires on every completed turn (Stop equivalent) and
    dispatches :meth:`~hermes.memory.pipeline.MemoryPipeline.run_after_turn`
    ASYNC (``sync=False``): the pipeline runs on a daemon thread, the turn
    thread never blocks, and the hook never raises — a dispatch failure is
    logged and the session continues (plan acceptance: "Stop hook fires
    pipeline async", "Pipeline failure logged but doesn't crash session").
    Returns the registered :class:`~hermes.memory.hooks.stop_hook.StopHook`;
    its ``futures`` list is the test/observability seam for bounded waits on
    the async runs.
    """
    hook = build_stop_hook(pipeline)
    agent.register_post_turn_hook(hook)
    return hook


# ── UserPromptSubmit pre-step hook wiring (plan P4.T4) ───────────────────────


def install_prompt_injection(
    agent: V2Agent,
    injector: ContextInjector,
) -> PromptSubmitHook:
    """Register the P4.T4 UserPromptSubmit pre-step hook for ``injector``.

    The registered hook fires before every model call (UserPromptSubmit
    equivalent) and runs the Phase 4 injection chain — BM25 retrieval
    (P4.T1) -> batched CBDC gating (P4.T2) ->
    :meth:`~hermes.memory.injector.ContextInjector.build_dynamic_context`
    (P4.T3) — returning the ``additionalContext`` the agent merges into the
    model call (:meth:`V2Agent.run_pre_step_hooks` merges every hook's
    contribution). Mirrors :func:`install_memory_pipeline`: the wiring is a
    one-liner for the composition root, the hook itself never raises, and
    the plan's graceful-degradation contract holds at the hook level — a
    timeout (5s budget) or injector failure degrades to an empty
    additionalContext instead of breaking the turn (plan Phase 4
    acceptance: "Timeout returns empty additionalContext, doesn't crash").

    Returns the registered
    :class:`~hermes.memory.hooks.user_prompt_submit_hook.PromptSubmitHook`.
    """
    hook = build_prompt_submit_hook(injector)
    agent.register_pre_step_hook(hook)
    return hook
