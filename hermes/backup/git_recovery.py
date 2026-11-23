"""Git repository corruption detection, repair and backup recovery (plan P18.T3).

The plan's corruption workflow — "git fsck check on every health check; if
corruption detected: alert + attempt repair (git gc, fsck); if unrepairable:
restore from latest backup" — as three functions:

- :func:`detect_corruption` runs ``git fsck --no-progress --full`` via
  ``subprocess`` (pygit2 has no fsck binding — the same reason
  :class:`~hermes.backup.manager.BackupManager` shells out) and flags
  corruption when the fsck exit code is non-zero or its output reports
  ``missing`` / ``broken link`` / ``error`` lines. It additionally verifies
  that HEAD resolves to an existing commit object (via pygit2) and that no
  stale ``.git/index.lock`` blocks git write operations — a lock older than
  :data:`LOCK_STALE_SECONDS` (5 minutes, the health-check cadence of plan
  Section 5.1) is treated as corruption because every memory-store commit
  would fail while it exists; the age threshold keeps a live writer's lock
  from ever being flagged or removed.
- :func:`attempt_repair` performs the auto-repair escalation: remove the
  stale lock file (if any), run ``git gc --prune=now`` (the plan's "git gc,
  fsck" repair — git has no ``fsck --fix`` flag; gc's prune/repack is the
  fix-up step), then re-run :func:`detect_corruption`. Clean re-check ->
  ``repaired=True``; anything else is reported honestly as unrepairable —
  object bytes that are gone cannot be regenerated, only restored.
- :func:`recover_from_backup` restores the latest ``YYYY-MM-DD.tar.gz``
  archive (newest date name wins) over the corrupted repository through
  :meth:`hermes.backup.recovery.Recovery.restore` with
  ``pre_restore_backup=False`` — the archive being restored from IS the
  safety copy, and copying a corrupted (possibly locked, read-only-object)
  repository first would only duplicate the damage.

Every step is exception-safe: missing ``git`` binary, fsck timeouts and
unreadable mtimes degrade into findings ("integrity unknown") instead of
raised exceptions, so the P14.T1-style health checks built on top never
crash (plan acceptance: corruption detected within 5 minutes — the next
health-check cadence).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pygit2

from hermes.backup.manager import BackupManager
from hermes.backup.recovery import Recovery, RecoveryResult

__all__ = [
    "FSCK_TIMEOUT_SECONDS",
    "GC_TIMEOUT_SECONDS",
    "LOCK_STALE_SECONDS",
    "CorruptionReport",
    "RepairOutcome",
    "attempt_repair",
    "detect_corruption",
    "find_latest_backup",
    "recover_from_backup",
]

#: Seconds after which a ``.git/index.lock`` counts as stale. Matches the
#: 5-minute health-check cadence (plan Section 5.1 / P18.T3 acceptance):
#: within one cadence a lock may belong to a live writer; across one it is
#: blocking every commit and is treated as corruption.
LOCK_STALE_SECONDS: Final[float] = 300.0

#: Wall-clock cap for one ``git fsck`` run (well under the 5-minute cadence).
FSCK_TIMEOUT_SECONDS: Final[float] = 120.0

#: Wall-clock cap for the ``git gc`` repair run.
GC_TIMEOUT_SECONDS: Final[float] = 120.0

#: Output markers that indicate corruption (plan P18.T3 design: "missing",
#: "broken link", "error" in fsck output), matched case-insensitively.
_CORRUPTION_MARKERS: Final[tuple[str, ...]] = ("missing", "broken link", "error")

#: Cap on the fsck lines reported as findings (details stay bounded).
_MAX_FSCK_FINDINGS: Final[int] = 10

#: Backup archives are named ``YYYY-MM-DD.tar.gz`` (plan P18.T1 layout).
_ARCHIVE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}\.tar\.gz")


@dataclass
class CorruptionReport:
    """Outcome of :func:`detect_corruption` (plan P18.T3 design).

    ``corrupted`` is True only on positive corruption evidence (fsck exit
    code or marker lines, unresolvable HEAD, a stale index lock).
    ``findings`` lists the evidence lines — plus "integrity unknown" notes
    (git binary missing, fsck timeout) when corruption could not be ruled
    out; callers treat non-empty findings with ``corrupted=False`` as warn.
    """

    corrupted: bool
    findings: list[str] = field(default_factory=list)


@dataclass
class RepairOutcome:
    """Outcome of :func:`attempt_repair` (plan P18.T3 design).

    ``repaired`` True means the fsck re-check came back clean;
    ``actions`` records every step taken (lock removal, gc result, the
    re-check verdict) for the repairs.log / diagnostics report.
    """

    repaired: bool
    actions: list[str] = field(default_factory=list)


def _as_utc(value: datetime) -> datetime:
    """Normalize ``value`` to aware UTC (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_git_repo(repo_path: Path) -> bool:
    """True when ``repo_path`` exists and carries a ``.git`` entry."""
    return repo_path.is_dir() and (repo_path / ".git").exists()


def _fsck_output(result: subprocess.CompletedProcess[bytes]) -> str:
    """Combined fsck stdout+stderr, decoded with replacement."""
    combined = (result.stdout or b"") + (result.stderr or b"")
    return combined.decode("utf-8", errors="replace")


def _head_resolution_finding(repo_path: Path) -> str | None:
    """Finding when HEAD cannot be resolved; ``None`` when it resolves.

    A fresh repository's unborn HEAD (symbolic ref to a branch that does
    not exist yet) is a legitimate early state, not corruption — only a
    branch ref that EXISTS but points nowhere (or an unopenable repository)
    is a finding.
    """
    try:
        repo = pygit2.Repository(str(repo_path))
    except Exception as exc:  # noqa: BLE001 - any open failure is a finding
        return f"repository cannot be opened by libgit2: {exc}"
    try:
        head = repo.head
        repo[head.target]  # raises when the commit object is missing
        return None
    except Exception:  # noqa: BLE001 - fall through to the unborn check
        pass
    try:
        head_ref = repo.lookup_reference("HEAD")
    except Exception:  # noqa: BLE001
        return None
    target = str(head_ref.target)  # e.g. "refs/heads/master" (symbolic HEAD)
    try:
        repo.lookup_reference(target)
    except Exception:  # noqa: BLE001 - unborn branch: legitimate fresh state
        return None
    return f"HEAD does not resolve to a commit object (symbolic target {target})"


def _lock_age_seconds(repo_path: Path, now: datetime | None) -> float | None:
    """Age (seconds) of ``.git/index.lock``, or ``None`` when absent/unreadable."""
    lock = repo_path / ".git" / "index.lock"
    try:
        if not lock.exists():
            return None
        modified = datetime.fromtimestamp(lock.stat().st_mtime, tz=UTC)
    except OSError:
        return None
    moment = _as_utc(now) if now is not None else datetime.now(UTC)
    return (moment - modified).total_seconds()


def _stale_lock_finding(
    repo_path: Path, lock_stale_seconds: float, now: datetime | None
) -> str | None:
    """Finding for an index lock older than ``lock_stale_seconds``."""
    age = _lock_age_seconds(repo_path, now)
    if age is None or age < lock_stale_seconds:
        return None  # absent, unreadable, or possibly a live writer
    return (
        f"stale lock file .git/index.lock (age {age:.0f}s >= "
        f"{lock_stale_seconds:g}s) blocks git write operations"
    )


def detect_corruption(
    repo_path: Path | str,
    *,
    lock_stale_seconds: float = LOCK_STALE_SECONDS,
    now: datetime | None = None,
) -> CorruptionReport:
    """Detect Git repository corruption: fsck + HEAD resolution + locks.

    ``git fsck --no-progress --full`` runs via subprocess in ``repo_path``;
    a non-zero exit or ``missing`` / ``broken link`` / ``error`` output
    lines mark the repository corrupted with those lines as findings. HEAD
    must resolve to an existing commit object (unborn HEAD on a fresh
    repository is not corruption), and a ``.git/index.lock`` older than
    ``lock_stale_seconds`` counts as corruption (it blocks every write).
    A missing git binary or an fsck timeout degrades to an "integrity
    unknown" finding with ``corrupted=False``; a non-repository path
    reports a single "no git repository" finding.
    """
    repo_path = Path(repo_path)
    if not _is_git_repo(repo_path):
        return CorruptionReport(
            corrupted=False,
            findings=[f"no git repository at {repo_path}; nothing to check"],
        )
    findings: list[str] = []
    corrupted = False
    try:
        result = subprocess.run(
            ["git", "fsck", "--no-progress", "--full"],
            cwd=repo_path,
            capture_output=True,
            check=False,
            timeout=FSCK_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        findings.append("git CLI not available; fsck could not run (integrity unknown)")
    except subprocess.TimeoutExpired:
        findings.append(
            f"git fsck timed out after {FSCK_TIMEOUT_SECONDS:g}s (integrity unknown)"
        )
    else:
        for line in _fsck_output(result).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if (
                any(marker in lowered for marker in _CORRUPTION_MARKERS)
                and len(findings) < _MAX_FSCK_FINDINGS
            ):
                corrupted = True
                findings.append(stripped)
        if result.returncode != 0:
            corrupted = True
            findings.insert(0, f"git fsck exited {result.returncode}")
    head_finding = _head_resolution_finding(repo_path)
    if head_finding is not None:
        corrupted = True
        findings.append(head_finding)
    lock_finding = _stale_lock_finding(repo_path, lock_stale_seconds, now)
    if lock_finding is not None:
        corrupted = True
        findings.append(lock_finding)
    return CorruptionReport(corrupted=corrupted, findings=findings)


def attempt_repair(
    repo_path: Path | str,
    *,
    lock_stale_seconds: float = LOCK_STALE_SECONDS,
    now: datetime | None = None,
) -> RepairOutcome:
    """Attempt the plan's repair escalation: locks + ``git gc`` + fsck re-check.

    Steps: remove a stale ``.git/index.lock`` (never a fresh one — a live
    writer may own it), run ``git gc --prune=now`` (git has no
    ``fsck --fix``; gc's prune/repack is the repair step), then re-run
    :func:`detect_corruption`. ``repaired=True`` only on a clean re-check;
    object corruption gc cannot fix stays ``repaired=False`` with the fsck
    evidence recorded, and the caller escalates to
    :func:`recover_from_backup`.
    """
    repo_path = Path(repo_path)
    if not _is_git_repo(repo_path):
        return RepairOutcome(
            repaired=False,
            actions=[f"no git repository at {repo_path}; nothing to repair"],
        )
    actions: list[str] = []
    lock_age = _lock_age_seconds(repo_path, now)
    if lock_age is not None:
        if lock_age >= lock_stale_seconds:
            lock = repo_path / ".git" / "index.lock"
            try:
                lock.chmod(0o644)  # Windows: unlink needs the write bit cleared
                lock.unlink()
                actions.append(
                    f"removed stale .git/index.lock (age {lock_age:.0f}s)"
                )
            except OSError as exc:
                actions.append(f"could not remove stale .git/index.lock: {exc}")
        else:
            actions.append(
                f".git/index.lock is fresh (age {lock_age:.0f}s < "
                f"{lock_stale_seconds:g}s); left in place"
            )
    try:
        result = subprocess.run(
            ["git", "gc", "--prune=now"],
            cwd=repo_path,
            capture_output=True,
            check=False,
            timeout=GC_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        actions.append("git CLI not available; gc skipped")
    except subprocess.TimeoutExpired:
        actions.append(f"git gc timed out after {GC_TIMEOUT_SECONDS:g}s")
    else:
        if result.returncode == 0:
            actions.append("git gc --prune=now (exit 0)")
        else:
            detail_lines = _fsck_output(result).strip().splitlines()
            tail = detail_lines[-1] if detail_lines else ""
            actions.append(
                f"git gc --prune=now failed (exit {result.returncode}): {tail}"
            )
    recheck = detect_corruption(
        repo_path, lock_stale_seconds=lock_stale_seconds, now=now
    )
    if recheck.corrupted:
        actions.append("corruption remains after repair: " + "; ".join(recheck.findings[:3]))
        return RepairOutcome(repaired=False, actions=actions)
    actions.append("git fsck clean after repair")
    return RepairOutcome(repaired=True, actions=actions)


def find_latest_backup(backups_dir: Path | str) -> Path:
    """Newest ``YYYY-MM-DD.tar.gz`` archive in ``backups_dir`` (name = date).

    Raises ``FileNotFoundError`` when the directory is missing or holds no
    dated archives (non-dated files are ignored, matching the P18.T1
    layout the BackupManager writes).
    """
    backups_dir = Path(backups_dir)
    if not backups_dir.is_dir():
        raise FileNotFoundError(f"no backups directory at {backups_dir}")
    archives = [
        path for path in backups_dir.glob("*.tar.gz") if _ARCHIVE_NAME_RE.fullmatch(path.name)
    ]
    if not archives:
        raise FileNotFoundError(f"no dated (*.tar.gz) backups found in {backups_dir}")
    # YYYY-MM-DD names sort chronologically, so the max name is the newest.
    return max(archives, key=lambda path: path.name)


def recover_from_backup(
    repo_path: Path | str,
    backups_dir: Path | str,
    backup_manager: BackupManager | None = None,
    *,
    pre_restore_backup: bool = False,
) -> RecoveryResult:
    """Restore the corrupted repository from the latest backup (plan P18.T3).

    Picks the newest dated archive via :func:`find_latest_backup` and
    replaces the live repository with it through
    :meth:`hermes.backup.recovery.Recovery.restore`. ``backup_manager``
    defaults to one mirroring the caller's paths (``repo_path``'s parent as
    hermes root, the path's own name as repo subdir); passing a manager
    whose ``repo_dir`` differs from ``repo_path`` raises ``ValueError`` —
    the restore wipes its target, so it must target the corrupted repo.
    ``pre_restore_backup`` defaults to ``False``: the archive is the safety
    copy, and duplicating a corrupted tree first is pointless at best.
    """
    repo_path = Path(repo_path)
    backups_dir = Path(backups_dir)
    latest = find_latest_backup(backups_dir)
    if backup_manager is None:
        backup_manager = BackupManager(
            hermes_root=repo_path.parent,
            repo_subdir=repo_path.name,
            backups_subdir=backups_dir.name,
        )
    recovery = Recovery(backup_manager)
    if recovery.repo_dir != repo_path:
        raise ValueError(
            f"backup_manager restores {recovery.repo_dir}, but the corrupted "
            f"repository is {repo_path}; pass a manager targeting the right repo"
        )
    return recovery.restore(latest, pre_restore_backup=pre_restore_backup)
