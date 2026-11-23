"""hermes-v2 auto-repair for the top 5 common issues (plan P14.T4, plan 5.4).

The plan's five auto-repairs, each a :meth:`AutoRepair` method returning a
:class:`RepairResult` (``issue``/``action``/``status``/``details``):

1. **dirty Git** -> auto-commit with the plan's exact message
   "auto-repair: commit pending changes" (only when the tree is dirty);
2. **missing trust cards** -> re-run the migration for affected entries —
   pragmatic implementation: write the migration-default trust card
   (trust 0.5 via one ``agent-inferred`` event, per
   :func:`hermes.memory.trust_card.compute_trust_score`'s documented
   migration default) preserving the entry body;
3. **trace store full** -> archive daily JSONL trace files older than
   90 days into ``<traces>/archive/``;
4. **Pareto front stale** (missing or older than 8 days, matching the
   ``gepa_stale`` alert/P14.T1 check) -> call the injected ``gepa_trigger``;
5. **backup stale** (no dated archive in 25 h, matching ``backup_stale``) ->
   call the injected ``backup_trigger``.

Expensive/external actions (GEPA run, backup run) are injected callables on
:class:`RepairContext` — auto-repair decides WHEN, the wiring decides HOW.
Every attempt is appended as a JSON line to
``<hermes_home>/diagnostics/repairs.log``; a ``failed`` result escalates
through the ``on_failure`` notifier seam ``Callable[[alert_name, message],
None]`` — the same shape the P21.T5 cron watchdog uses, wired by the caller
into the AlertEngine/gateway path (plan acceptance: "failed repairs escalate
to alert"; alert name :data:`AUTO_REPAIR_FAILED`).

:obj:`status` semantics: ``repaired`` = the issue was present and is now
fixed, ``skipped`` = the issue was not present (nothing to do), ``failed`` =
the issue is present but the repair attempt errored (escalated).

Together the five repairs cover the plan 5.1 auto-repair column for every
check that has one (memory store, git state, trace store size, pareto
freshness, backup freshness) — the "resolves 80%+ of common issues"
acceptance.

Plan P18.T3 adds a sixth, opt-in repair: :meth:`AutoRepair.repair_git_corruption`
escalates detect -> repair (stale locks + ``git gc`` + fsck re-check, via
:mod:`hermes.backup.git_recovery`) -> restore-from-backup through the
injected ``recover_git`` callable on :class:`RepairContext` (wired to
:func:`hermes.backup.git_recovery.recover_from_backup`). It joins
:meth:`AutoRepair.repair_all` only when ``context.recover_git`` is
configured, so the plan-5.4 five-result contract of existing callers is
untouched; the issue key is ``git_corruption`` (matching the P18.T3 health
check name), kept out of :data:`REPAIR_ISSUES` — that constant is the plan
5.4 list.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

from hermes.backup.git_recovery import attempt_repair, detect_corruption
from hermes.config import hermes_home
from hermes.diagnostics.health_checks import _scan_fact_files
from hermes.memory.git_backend import GitBackend
from hermes.memory.trust_card import TrustCard, add_verification_event, parse, serialize
from hermes.memory.types import VerificationEvent
from hermes.optimization.pareto import default_pareto_path

__all__ = [
    "AUTO_REPAIR_COMMIT_MESSAGE",
    "AUTO_REPAIR_FAILED",
    "BACKUP_STALE_HOURS",
    "DEFAULT_TRUST_SCORE",
    "PARETO_STALE_DAYS",
    "REPAIR_ISSUES",
    "TRACE_ARCHIVE_MAX_AGE_DAYS",
    "AutoRepair",
    "RepairContext",
    "RepairResult",
    "default_repairs_log_path",
]

_LOGGER = logging.getLogger("hermes.diagnostics.auto_repair")

#: Plan 5.4: exact commit message for the dirty-git auto-repair.
AUTO_REPAIR_COMMIT_MESSAGE: Final[str] = "auto-repair: commit pending changes"

#: Alert name used when a failed repair escalates through ``on_failure``.
AUTO_REPAIR_FAILED: Final[str] = "auto_repair_failed"

#: Plan 5.4: archive traces older than 90 days.
TRACE_ARCHIVE_MAX_AGE_DAYS: Final[int] = 90

#: Pareto staleness threshold (matches the gepa_stale alert, plan 5.3).
PARETO_STALE_DAYS: Final[float] = 8.0

#: Backup staleness threshold (matches the backup_stale alert, plan 5.3).
BACKUP_STALE_HOURS: Final[float] = 25.0

#: Migration-default trust score for repaired orphan entries (plan Section 2.4:
#: an unverified entry scores 0.5 — ``compute_trust_score``'s empty default).
DEFAULT_TRUST_SCORE: Final[float] = 0.5

RepairStatus = Literal["repaired", "skipped", "failed"]
_STATUSES: Final[frozenset[str]] = frozenset({"repaired", "skipped", "failed"})

#: The plan 5.4 issue keys, in repair_all() execution order. The P18.T3
#: ``git_corruption`` repair (see module docstring) runs after these when
#: ``RepairContext.recover_git`` is configured — it is plan-18 wiring, not a
#: plan-5.4 issue, so it is not part of this tuple.
REPAIR_ISSUES: Final[tuple[str, ...]] = (
    "dirty_git",
    "missing_trust_cards",
    "trace_store_full",
    "stale_pareto",
    "stale_backup",
)

_SECONDS_PER_HOUR: Final[float] = 3600.0
_SECONDS_PER_DAY: Final[float] = 86400.0

_JSONL_SUFFIX: Final[str] = ".jsonl"
_DAY_FORMAT: Final[str] = "%Y-%m-%d"


def default_repairs_log_path() -> Path:
    """Default repair log: ``<hermes_home>/diagnostics/repairs.log``."""
    path: Path = hermes_home() / "diagnostics" / "repairs.log"
    return path


def _default_repo_path() -> Path:
    """Default memory-repo path (the store IS the Git repository)."""
    home: Path = hermes_home() / "memory-repo"
    return home


def _default_traces_root() -> Path:
    """Default trace-store root (plan P11.T1 layout)."""
    home: Path = hermes_home() / "traces"
    return home


def _default_backups_dir() -> Path:
    """Default backups directory (plan P18.T1 layout)."""
    home: Path = hermes_home() / "backups"
    return home


@dataclass
class RepairContext:
    """Paths + injected actions for :class:`AutoRepair` (hermes-home defaults).

    ``gepa_trigger``/``backup_trigger`` are the expensive external actions:
    auto-repair only decides WHEN to run them (issue detected), the wiring
    decides HOW (GEPA pipeline, :class:`~hermes.backup.manager.BackupManager`).
    They are ``None`` by default — the corresponding repairs then report
    ``failed`` with an explicit "not configured" detail and escalate, instead
    of silently doing nothing. ``recover_git`` (plan P18.T3) is the same
    seam for the git-corruption escalation: ``Callable[[], Path | None]``
    restoring the repo from the latest backup (production wires it to
    :func:`hermes.backup.git_recovery.recover_from_backup`); it also gates
    whether :meth:`AutoRepair.repair_all` includes the corruption repair.
    """

    repo_path: Path = field(default_factory=_default_repo_path)
    store_root: Path = field(default_factory=_default_repo_path)
    traces_root: Path = field(default_factory=_default_traces_root)
    pareto_path: Path = field(default_factory=default_pareto_path)
    backups_dir: Path = field(default_factory=_default_backups_dir)
    gepa_trigger: Callable[[], None] | None = None
    backup_trigger: Callable[[], Path] | None = None
    recover_git: Callable[[], Path | None] | None = None


@dataclass
class RepairResult:
    """Outcome of one repair attempt (plan P14.T4 design).

    ``issue`` is one of :data:`REPAIR_ISSUES`; ``action`` is the short verb
    phrase of what was done (or would have been done); ``status`` is
    ``repaired``/``skipped``/``failed``; ``details`` carries counts, paths
    and error text for the repairs.log line and the diagnostics report.
    """

    issue: str
    action: str
    status: RepairStatus
    details: str = ""

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            allowed = sorted(_STATUSES)
            raise ValueError(f"repair status must be one of {allowed}, got {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection (repairs.log line / diagnostics report)."""
        return {
            "issue": self.issue,
            "action": self.action,
            "status": self.status,
            "details": self.details,
        }


def _body_without_frontmatter(content: str) -> str:
    """Drop a frontmatter block (even a broken one), keeping the body.

    Used by the trust-card repair: when the existing frontmatter is invalid
    the body starts after the *closing* ``---`` (or the opening one when the
    block is unterminated); content without frontmatter is kept whole.
    """
    stripped = content.lstrip("\ufeff").lstrip()
    if not stripped.startswith("---"):
        return content
    lines = stripped.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[index + 1 :]).lstrip("\n")
    return content  # unterminated block — conservatively keep everything


def _day_stamp_of(path: Path) -> date | None:
    """The ``YYYY-MM-DD`` day encoded in a JSONL file stem, or ``None``."""
    if path.suffix != _JSONL_SUFFIX:
        return None
    try:
        return datetime.strptime(path.stem, _DAY_FORMAT).date()
    except ValueError:
        return None


class AutoRepair:
    """Executes the plan 5.4 top-5 repairs with logging + escalation.

    Args:
        context: paths and injected triggers (:class:`RepairContext`,
            hermes-home defaults).
        on_failure: escalation seam for failed repairs,
            ``Callable[[alert_name, message], None]`` — the same notifier
            shape as the P21.T5 cron watchdog, so the DiagnosticsSystem can
            route both into the AlertEngine/gateway path.
        log_path: JSON-lines repair log (default
            ``<hermes_home>/diagnostics/repairs.log``).
        now: injectable clock (aware-UTC by convention) for age thresholds.
    """

    def __init__(
        self,
        context: RepairContext | None = None,
        *,
        on_failure: Callable[[str, str], None] | None = None,
        log_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.context = context if context is not None else RepairContext()
        self.on_failure = on_failure
        self.log_path: Path = log_path if log_path is not None else default_repairs_log_path()
        self._now = now if now is not None else (lambda: datetime.now(UTC))
        self._lock = threading.Lock()

    # ── plan 5.4 repair #1: dirty Git -> auto-commit ─────────────────────────
    def repair_dirty_git(self) -> RepairResult:
        """Commit pending changes with the plan's exact message (only if dirty).

        Untracked files are included; tracked deletions are staged through the
        index before the single auto-repair commit so the tree ends clean.
        """
        issue = "dirty_git"
        repo_path = self.context.repo_path
        try:
            if not repo_path.is_dir() or not (repo_path / ".git").exists():
                return self._record(
                    RepairResult(issue, "none", "skipped", f"no git repository at {repo_path}")
                )
            backend = GitBackend(repo_path)
            status: dict[str, int] = backend._repo.status()
            if not status:
                return self._record(
                    RepairResult(issue, "none", "skipped", "working tree already clean")
                )
            dirty = sorted(status)
            missing = [path for path in dirty if not (repo_path / path).exists()]
            if missing:
                index = backend._repo.index
                index.remove_all(missing)
                index.write()
            existing = [path for path in dirty if (repo_path / path).exists()]
            sha = backend.commit(AUTO_REPAIR_COMMIT_MESSAGE, existing)
            return self._record(
                RepairResult(
                    issue,
                    "auto-commit pending changes",
                    "repaired",
                    f"committed {len(dirty)} pending path(s) in {sha[:12]}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "auto-commit pending changes", "failed",
                             f"auto-commit failed: {exc}")
            )

    # ── plan 5.4 repair #2: missing trust cards -> re-run migration ──────────
    def repair_missing_trust_cards(self) -> RepairResult:
        """Write the migration-default trust card for entries missing one.

        A fact file "missing a trust card" is one whose frontmatter does not
        parse (missing/invalid) or whose card carries no verification events.
        The pragmatic migration re-run replaces the card with the default
        (id = file stem, type ``Reference``, trust 0.5 via one
        ``agent-inferred`` event of value 0.5) and preserves the body byte
        exactly. Repaired files are committed to the repo (one commit) when
        the store is Git-backed.
        """
        issue = "missing_trust_cards"
        store_root = self.context.store_root
        try:
            if not store_root.is_dir():
                return self._record(
                    RepairResult(
                        issue, "none", "skipped",
                        f"memory store directory not found: {store_root}",
                    )
                )
            fact_files = _scan_fact_files(store_root)
            if not fact_files:
                return self._record(
                    RepairResult(issue, "none", "skipped", "no entries stored yet")
                )
            moment = self._now()
            event = VerificationEvent(
                event_id=f"ev-{uuid.uuid4().hex[:12]}",
                timestamp=moment,
                source="agent-inferred",
                verifier="auto-repair",
                evidence="auto-repair: migration re-run wrote the default trust card "
                "(plan Section 5.4)",
                value=DEFAULT_TRUST_SCORE,
            )
            repaired: list[Path] = []
            for path in fact_files:
                content = path.read_text(encoding="utf-8")
                try:
                    card, body = parse(content)
                except (OSError, ValueError):
                    # missing/invalid frontmatter: build the default card fresh
                    new_card = TrustCard(
                        id=path.stem,
                        type="Reference",
                        created_at=moment,
                        trust_score=DEFAULT_TRUST_SCORE,
                        verification_events=[event],
                        context="global",
                    )
                    path.write_text(
                        serialize(new_card, _body_without_frontmatter(content)),
                        encoding="utf-8",
                    )
                    repaired.append(path)
                    _LOGGER.warning(
                        "auto-repair wrote default trust card for %s (frontmatter "
                        "was missing/invalid)",
                        path,
                    )
                    continue
                if card.verification_events:
                    continue  # valid card with evidence: nothing to repair
                add_verification_event(card, event)
                path.write_text(serialize(card, body), encoding="utf-8")
                repaired.append(path)
                _LOGGER.warning(
                    "auto-repair seeded default verification event for %s "
                    "(card had no events)",
                    path,
                )
            if not repaired:
                return self._record(
                    RepairResult(
                        issue, "none", "skipped",
                        f"all {len(fact_files)} entries already carry valid trust cards",
                    )
                )
            rels = [str(path.relative_to(store_root).as_posix()) for path in repaired]
            details = (
                f"re-ran trust-card migration for {len(repaired)} of "
                f"{len(fact_files)} entries: {', '.join(rels)}"
            )
            if (store_root / ".git").exists():
                backend = GitBackend(store_root)
                sha = backend.commit(
                    f"auto-repair: re-run trust-card migration for {len(repaired)} entries",
                    rels,
                )
                details = f"{details} (commit {sha[:12]})"
            return self._record(RepairResult(issue, "re-run trust-card migration",
                                             "repaired", details))
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "re-run trust-card migration", "failed",
                             f"trust-card migration re-run failed: {exc}")
            )

    # ── plan 5.4 repair #3: trace store full -> archive >90 days ─────────────
    def repair_trace_store_full(self) -> RepairResult:
        """Move daily JSONL trace files older than 90 days to ``traces/archive/``.

        The ``YYYY-MM-DD`` day stamp in the filename is authoritative (it is
        what :mod:`hermes.traces.store` writes); files whose stem is not a day
        stamp (index files, unknown artifacts) stay untouched.
        """
        issue = "trace_store_full"
        traces_root = self.context.traces_root
        version_dir = traces_root / "v1.0"
        try:
            if not version_dir.is_dir():
                return self._record(
                    RepairResult(issue, "none", "skipped", f"no trace days under {version_dir}")
                )
            cutoff = (self._now() - timedelta(days=TRACE_ARCHIVE_MAX_AGE_DAYS)).date()
            archive_dir = traces_root / "archive"
            moved: list[str] = []
            for path in sorted(version_dir.glob(f"*{_JSONL_SUFFIX}")):
                day = _day_stamp_of(path)
                if day is None or day >= cutoff:
                    continue
                archive_dir.mkdir(parents=True, exist_ok=True)
                target = archive_dir / path.name
                if target.exists():
                    continue  # idempotent: already archived on a previous pass
                shutil.move(str(path), str(target))
                moved.append(path.name)
            if not moved:
                return self._record(
                    RepairResult(
                        issue, "none", "skipped",
                        f"no trace day files older than {TRACE_ARCHIVE_MAX_AGE_DAYS} days",
                    )
                )
            return self._record(
                RepairResult(
                    issue, "archive old traces", "repaired",
                    f"archived {len(moved)} trace day file(s) to {archive_dir}: "
                    f"{', '.join(moved)}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "archive old traces", "failed",
                             f"trace archiving failed: {exc}")
            )

    # ── plan 5.4 repair #4: stale Pareto front -> trigger GEPA ───────────────
    def repair_stale_pareto(self) -> RepairResult:
        """Trigger a GEPA run when the Pareto front is missing or >8 days old.

        Freshness comes from the file mtime (ParetoFrontStore rewrites it on
        every run, so mtime == last run — the same source the P14.T1 check
        uses). A missing front counts as stale: the GEPA run has simply never
        completed. No configured trigger -> ``failed`` + escalation.
        """
        issue = "stale_pareto"
        pareto_path = self.context.pareto_path
        try:
            if pareto_path.is_file():
                modified = datetime.fromtimestamp(pareto_path.stat().st_mtime, tz=UTC)
                age_days = (self._now() - modified).total_seconds() / _SECONDS_PER_DAY
                if age_days <= PARETO_STALE_DAYS:
                    return self._record(
                        RepairResult(
                            issue, "none", "skipped",
                            f"Pareto front fresh (updated {age_days:.1f} days ago)",
                        )
                    )
                base = f"Pareto front stale (updated {age_days:.1f} days ago)"
            else:
                base = "no Pareto front yet (no GEPA run has completed)"
            if self.context.gepa_trigger is None:
                return self._record(
                    RepairResult(issue, "trigger GEPA run", "failed",
                                 f"{base}; no gepa_trigger configured")
                )
            self.context.gepa_trigger()
            return self._record(
                RepairResult(issue, "trigger GEPA run", "repaired",
                             f"{base}; GEPA run triggered")
            )
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "trigger GEPA run", "failed",
                             f"GEPA trigger failed: {exc}")
            )

    # ── plan 5.4 repair #5: stale backup -> trigger backup ───────────────────
    def repair_stale_backup(self) -> RepairResult:
        """Trigger a backup when the newest archive is older than 25 hours.

        Looks for the ``YYYY-MM-DD.tar.gz`` archives the BackupManager writes;
        none at all counts as stale ("never backed up", like the P14.T1 check
        and the ``backup_stale`` alert). No configured trigger -> ``failed`` +
        escalation.
        """
        issue = "stale_backup"
        backups_dir = self.context.backups_dir
        try:
            archives: list[Path] = []
            if backups_dir.is_dir():
                archives = list(backups_dir.glob("*.tar.gz"))
            if archives:
                newest = max(archives, key=lambda path: path.stat().st_mtime)
                age_hours = (
                    self._now()
                    - datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
                ).total_seconds() / _SECONDS_PER_HOUR
                if age_hours <= BACKUP_STALE_HOURS:
                    return self._record(
                        RepairResult(
                            issue, "none", "skipped",
                            f"latest backup {newest.name} is {age_hours:.1f} h old",
                        )
                    )
                base = (
                    f"latest backup {newest.name} is {age_hours:.1f} h old "
                    f"(limit {BACKUP_STALE_HOURS:g} h)"
                )
            else:
                base = f"no backups found in {backups_dir} (never backed up)"
            if self.context.backup_trigger is None:
                return self._record(
                    RepairResult(issue, "trigger backup", "failed",
                                 f"{base}; no backup_trigger configured")
                )
            archive = self.context.backup_trigger()
            return self._record(
                RepairResult(issue, "trigger backup", "repaired",
                             f"{base}; backup triggered -> {archive}")
            )
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "trigger backup", "failed",
                             f"backup trigger failed: {exc}")
            )

    # ── plan P18.T3: git corruption -> locks + gc -> restore from backup ─────
    def repair_git_corruption(
        self, trigger_recover: Callable[[], Path | None] | None = None
    ) -> RepairResult:
        """Detect corruption, attempt the gc repair, escalate to backup restore.

        Escalation (plan P18.T3): :func:`detect_corruption` flags the repo;
        :func:`attempt_repair` removes stale locks and runs ``git gc
        --prune=now`` with an fsck re-check; when the re-check stays
        corrupted the injected recover callable (``trigger_recover``, else
        ``context.recover_git`` — production wires
        :func:`hermes.backup.git_recovery.recover_from_backup`) restores the
        latest backup. No corruption (or no repository) -> ``skipped``;
        unrepairable without a recover callable -> ``failed`` + escalation.
        """
        issue = "git_corruption"
        repo_path = self.context.repo_path
        recover = trigger_recover if trigger_recover is not None else self.context.recover_git
        try:
            if not repo_path.is_dir() or not (repo_path / ".git").exists():
                return self._record(
                    RepairResult(
                        issue, "none", "skipped", f"no git repository at {repo_path}"
                    )
                )
            report = detect_corruption(repo_path)
            if not report.corrupted:
                return self._record(
                    RepairResult(
                        issue, "none", "skipped",
                        "no git corruption detected (fsck clean)",
                    )
                )
            outcome = attempt_repair(repo_path)
            if outcome.repaired:
                return self._record(
                    RepairResult(
                        issue, "remove stale locks + git gc + fsck re-check", "repaired",
                        "; ".join(outcome.actions),
                    )
                )
            base = f"unrepairable corruption: {'; '.join(outcome.actions)}"
            if recover is None:
                return self._record(
                    RepairResult(
                        issue, "restore from backup", "failed",
                        f"{base}; no recover_git configured",
                    )
                )
            restored = recover()
            if restored is None:
                return self._record(
                    RepairResult(
                        issue, "restore from backup", "failed",
                        f"{base}; recovery produced no backup archive",
                    )
                )
            return self._record(
                RepairResult(
                    issue, "restore from backup", "repaired",
                    f"restored from {Path(restored).name} after unrepairable corruption",
                )
            )
        except Exception as exc:  # noqa: BLE001 - repairs must escalate, not raise
            return self._record(
                RepairResult(issue, "restore from backup", "failed",
                             f"git corruption repair failed: {exc}")
            )

    # ── orchestrator ──────────────────────────────────────────────────────────
    def repair_all(self) -> list[RepairResult]:
        """Run all five plan 5.4 repairs in order; every attempt is logged.

        Each repair is failure-isolated: a crashing repair is captured as a
        ``failed`` :class:`RepairResult` (and escalated) instead of aborting
        the remaining repairs. When ``context.recover_git`` is configured
        (plan P18.T3 wiring), the ``git_corruption`` repair runs after the
        five, appended additively.
        """
        results = [
            self.repair_dirty_git(),
            self.repair_missing_trust_cards(),
            self.repair_trace_store_full(),
            self.repair_stale_pareto(),
            self.repair_stale_backup(),
        ]
        if self.context.recover_git is not None:
            results.append(self.repair_git_corruption())
        return results

    # ── internals ─────────────────────────────────────────────────────────────
    def _record(self, result: RepairResult) -> RepairResult:
        """Log one attempt; escalate to the on_failure seam when it failed."""
        self._append_log(result)
        if result.status == "failed":
            self._escalate(result)
        return result

    def _append_log(self, result: RepairResult) -> None:
        """Append one JSON line to repairs.log (atomic single write + fsync)."""
        record = {"ts": self._now().isoformat(), **result.to_dict()}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _escalate(self, result: RepairResult) -> None:
        """Notify the on_failure seam (wired to the AlertEngine sink path)."""
        if self.on_failure is None:
            return
        message = f"auto-repair failed for '{result.issue}': {result.details}"
        try:
            self.on_failure(AUTO_REPAIR_FAILED, message)
        except Exception:  # noqa: BLE001 - a broken notifier must not raise
            _LOGGER.exception("auto-repair escalation notifier failed for %s", result.issue)
