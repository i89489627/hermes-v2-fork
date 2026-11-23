"""hermes-v2 health checks (plan P14.T1, plan Section 5.1).

Six self-diagnostics checks, intended to run every 5 minutes (plan Section
5.1) and each completing well within 2 seconds — no real API calls: the
main-model reachability check uses a cheap TCP socket probe (1.5 s timeout)
of the configured provider endpoint, and every check accepts explicit paths
so tests inject fast fakes.

- memory_store_integrity: pass = all trust cards valid, warn = store dir not
  initialized, fail = broken/missing frontmatter;
- git_state: pass = clean working tree, warn = dirty tree (count) or no
  repo, fail = repo unreadable;
- main_model_reachable: pass = probe succeeds, warn = no provider
  configured, fail = probe fails/raises;
- trace_store_size: pass = under 80% of limit, warn = >= 80% of limit,
  fail = over the limit;
- pareto_front_freshness: pass = updated < max_age_days, warn = stale or
  not yet present*, fail = stat/parse error;
- backup_freshness: pass = newest < max_age_hours, warn = stale or never
  backed up, fail = stat error.

*Missing Pareto front is a **warn**, not a fail: no GEPA run yet is a
legitimate early state of a fresh install (plan Section 5.3 alerts on
``gepa_stale`` after 8 days, also warning severity).

Plan P18.T3 adds a seventh, opt-in check: :func:`check_git_corruption`
(git fsck + HEAD resolution + stale locks via
:mod:`hermes.backup.git_recovery`) is appended to :func:`run_all_checks`
output — at the end, additive to the six plan-table checks — when
:class:`HealthCheckContext` carries ``include_git_corruption=True``.

Every check is exception-safe: an internal error yields ``status="fail"``
with the error in ``details``, never a raised exception. All checks return a
:class:`Check`; :func:`run_all_checks` aggregates the six into the
HealthReport-compatible list (plan Section 5.1), with paths and the probe
assembled from :func:`hermes.config.hermes_home` defaults in
:class:`HealthCheckContext`.
"""

from __future__ import annotations

import contextlib
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

import yaml

from hermes.backup.git_recovery import detect_corruption
from hermes.config import hermes_home
from hermes.memory.git_backend import GitBackend
from hermes.memory.trust_card import parse
from hermes.optimization.pareto import default_pareto_path

__all__ = [
    "Check",
    "HealthCheckContext",
    "check_backup_freshness",
    "check_git_corruption",
    "check_git_state",
    "check_main_model_reachable",
    "check_memory_store_integrity",
    "check_pareto_front_freshness",
    "check_trace_store_size",
    "run_all_checks",
]

#: Valid :class:`Check` statuses (plan P14.T1 acceptance: pass|fail|warn).
CheckStatus = Literal["pass", "fail", "warn"]
_STATUSES: Final[frozenset[str]] = frozenset({"pass", "fail", "warn"})

#: GB as used by the trace-store limit: 2^30 bytes (documented binary gigabytes).
_GB: Final[float] = float(1024**3)

#: Share of the limit at which the trace-store check starts warning.
_TRACE_WARN_SHARE: Final[float] = 0.8

#: Socket-connect timeout of the default main-model probe (kept inside the
#: 2 s per-check budget; DNS resolution is the only unbounded part).
_PROBE_TIMEOUT_SECONDS: Final[float] = 1.5

#: Backup archives are named ``YYYY-MM-DD.tar.gz`` (plan P18.T1 layout).
_ARCHIVE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}\.tar\.gz")

_SECONDS_PER_DAY: Final[float] = 86400.0
_SECONDS_PER_HOUR: Final[float] = 3600.0


@dataclass
class Check:
    """Result of one health check (plan Section 5.1).

    ``status`` is one of ``pass``/``fail``/``warn``; ``details`` is a
    human-readable explanation (counts, ages, the failing file or error).
    ``auto_repair_*`` fields are filled by the P14.T4 auto-repair layer
    after it attempts a fix for a failing check.
    """

    name: str
    status: CheckStatus
    details: str
    auto_repair_attempted: bool = False
    auto_repair_result: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            allowed = sorted(_STATUSES)
            raise ValueError(f"check status must be one of {allowed}, got {self.status!r}")


def _default_store_root() -> Path:
    """Default memory-store root (the Git repository) under ``hermes_home()``."""
    return hermes_home() / "memory-repo"


def _default_traces_root() -> Path:
    """Default trace-store root (plan P11.T1 layout) under ``hermes_home()``."""
    return hermes_home() / "traces"


def _default_backups_dir() -> Path:
    """Default backups directory (plan P18.T1 layout) under ``hermes_home()``."""
    return hermes_home() / "backups"


@dataclass
class HealthCheckContext:
    """Paths + probe for :func:`run_all_checks`, hermes-home by default.

    ``store_root`` and ``repo_path`` both default to ``<hermes_home>/memory-repo``
    because the v2 memory store IS the Git repository; pass explicit values
    when they differ. ``probe=None`` selects the default cheap socket probe
    of the configured provider endpoint (see ``check_main_model_reachable``).
    ``include_git_corruption`` opts the P18.T3 ``git_corruption`` check into
    :func:`run_all_checks` (appended additively at the end; default off so
    the plan-5.1 six-check aggregation stays byte-compatible for existing
    consumers).
    """

    store_root: Path = field(default_factory=_default_store_root)
    repo_path: Path = field(default_factory=_default_store_root)
    probe: Callable[[], bool] | None = None
    traces_root: Path = field(default_factory=_default_traces_root)
    pareto_path: Path = field(default_factory=default_pareto_path)
    backups_dir: Path = field(default_factory=_default_backups_dir)
    include_git_corruption: bool = False


def _as_utc_now(now: datetime | None) -> datetime:
    """``now`` or the wall clock; naive input is taken to be UTC."""
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _scan_fact_files(store_root: Path) -> list[Path]:
    """All fact files in the v2 layout: ``global/facts`` + ``contexts/*/facts``."""
    files: list[Path] = []
    global_facts = store_root / "global" / "facts"
    if global_facts.is_dir():
        files.extend(path for path in sorted(global_facts.glob("*.md")) if path.is_file())
    contexts_root = store_root / "contexts"
    if contexts_root.is_dir():
        for context_dir in sorted(path for path in contexts_root.iterdir() if path.is_dir()):
            facts_dir = context_dir / "facts"
            if facts_dir.is_dir():
                files.extend(path for path in sorted(facts_dir.glob("*.md")) if path.is_file())
    return files


def _configured_endpoint() -> tuple[str, int] | None:
    """``(host, port)`` of the configured main-model API, or ``None``.

    Reads the fork-base ``config.yaml`` ``model.base_url`` (the same key the
    agent's model configuration uses). A missing/invalid/unreadable config
    means "no provider configured" — reported as warn, never probed.
    """
    config_path = hermes_home() / "config.yaml"
    try:
        with open(config_path, encoding="utf-8") as file_handle:
            document = yaml.safe_load(file_handle)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, dict):
        return None
    model = document.get("model")
    if not isinstance(model, dict):
        return None
    base_url = model.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _socket_probe(endpoint: tuple[str, int]) -> bool:
    """Cheap TCP reachability probe: connect, close, report success."""
    try:
        with socket.create_connection(endpoint, timeout=_PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def check_memory_store_integrity(store_root: Path) -> Check:
    """Every ``.md`` fact file parses with a valid trust card (plan 5.1).

    Scans ``global/facts`` + ``contexts/*/facts``; any file with missing or
    invalid frontmatter fails the check (auto-repair re-runs the migration
    for the affected entries, plan Section 5.4). A store that does not exist
    yet warns; a store with zero entries passes (nothing to validate).
    """
    name = "memory_store_integrity"
    try:
        if not store_root.is_dir():
            return Check(
                name=name,
                status="warn",
                details=f"memory store directory not found: {store_root}; nothing to check yet",
            )
        fact_files = _scan_fact_files(store_root)
        if not fact_files:
            return Check(
                name=name,
                status="pass",
                details="memory store present, no entries stored yet; no trust cards to validate",
            )
        broken: list[str] = []
        for path in fact_files:
            try:
                parse(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                broken.append(f"{path.relative_to(store_root).as_posix()} ({exc})")
        if broken:
            shown = "; ".join(broken[:5]) + ("; ..." if len(broken) > 5 else "")
            return Check(
                name=name,
                status="fail",
                details=(
                    f"{len(broken)} of {len(fact_files)} fact files have a "
                    f"missing/invalid trust card: {shown}"
                ),
            )
        return Check(
            name=name,
            status="pass",
            details=f"{len(fact_files)} entries, all trust cards valid",
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"integrity check crashed: {exc}")


def check_git_state(repo_path: Path) -> Check:
    """Clean working tree passes; a dirty tree warns with the file count."""
    name = "git_state"
    try:
        if not repo_path.is_dir() or not (repo_path / ".git").exists():
            return Check(
                name=name,
                status="warn",
                details=f"no git repository at {repo_path}; git state unknown",
            )
        try:
            backend = GitBackend(repo_path)
            # pygit2's status map (path -> flags) via the repository handle;
            # GitBackend itself only exposes the boolean is_clean(), and the
            # dirty-file count needs the per-file map (same accessor the
            # MemoryStore internals use for index surgery).
            status: dict[str, int] = backend._repo.status()
        except Exception as exc:  # noqa: BLE001 - corrupt repo must not raise
            return Check(name=name, status="fail", details=f"git repository unreadable: {exc}")
        dirty = len(status)
        if dirty == 0:
            return Check(name=name, status="pass", details="working tree clean")
        sample = ", ".join(sorted(status)[:5]) + ("..." if dirty > 5 else "")
        return Check(
            name=name,
            status="warn",
            details=f"working tree dirty: {dirty} file(s): {sample}",
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"git state check crashed: {exc}")


def check_main_model_reachable(probe: Callable[[], bool] | None = None) -> Check:
    """Main model API answers a ping (plan 5.1).

    ``probe`` is injectable for tests; the default attempts a cheap TCP
    socket probe (1.5 s timeout) of the ``model.base_url`` endpoint from the
    fork-base ``config.yaml``. With no provider configured the check warns
    (reachability unknown, nothing to probe); a failed or raising probe
    fails the check — plan Section 5.5 degradations kick in downstream.
    """
    name = "main_model_reachable"
    try:
        if probe is None:
            endpoint = _configured_endpoint()
            if endpoint is None:
                return Check(
                    name=name,
                    status="warn",
                    details="no provider configured (no model.base_url in config.yaml); "
                    "main-model reachability unknown",
                )
            probe = lambda: _socket_probe(endpoint)  # noqa: E731 - tiny default adapter
        try:
            reachable = probe()
        except Exception as exc:  # noqa: BLE001 - a crashing probe is a failure
            return Check(
                name=name,
                status="fail",
                details=f"main-model probe raised {type(exc).__name__}: {exc}",
            )
        if reachable:
            return Check(name=name, status="pass", details="main model API reachable")
        return Check(name=name, status="fail", details="main model API unreachable")
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"reachability check crashed: {exc}")


def check_trace_store_size(traces_root: Path, max_gb: float = 10.0) -> Check:
    """Trace store below ``max_gb`` (binary GB) passes; over the limit fails.

    Sums the sizes of every file under ``traces_root`` (JSONL days + the
    SQLite index). At >= 80% of the limit the check warns ("approaching")
    so archiving (plan Section 5.4) can kick in before the limit bites. A
    missing traces root counts as 0 bytes and passes.
    """
    name = "trace_store_size"
    try:
        total_bytes = 0
        if traces_root.is_dir():
            for path in traces_root.rglob("*"):
                if path.is_file():
                    with contextlib.suppress(OSError):
                        total_bytes += path.stat().st_size
        limit_bytes = max_gb * _GB
        used_gb = total_bytes / _GB
        if total_bytes > limit_bytes:
            return Check(
                name=name,
                status="fail",
                details=(
                    f"trace store over limit: {used_gb:.3f} GB used, "
                    f"limit {max_gb:.1f} GB ({total_bytes} bytes)"
                ),
            )
        if limit_bytes > 0 and total_bytes >= _TRACE_WARN_SHARE * limit_bytes:
            return Check(
                name=name,
                status="warn",
                details=(
                    f"trace store approaching limit: {used_gb:.3f} GB used, "
                    f"limit {max_gb:.1f} GB ({total_bytes} bytes)"
                ),
            )
        return Check(
            name=name,
            status="pass",
            details=(
                f"trace store size {used_gb:.3f} GB of {max_gb:.1f} GB limit "
                f"({total_bytes} bytes)"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"trace size check crashed: {exc}")


def check_pareto_front_freshness(
    pareto_path: Path, max_age_days: float = 8, now: datetime | None = None
) -> Check:
    """Pareto front updated within ``max_age_days`` passes; stale warns.

    Freshness comes from the file's mtime (``ParetoFrontStore`` rewrites the
    file atomically on every GEPA run, so mtime == last run). A missing
    front file warns rather than fails: no GEPA run yet is a legitimate
    early state of a fresh install (documented plan P14.T1 choice; the
    ``gepa_stale`` alert in plan Section 5.3 is also warning severity).
    """
    name = "pareto_front_freshness"
    try:
        moment = _as_utc_now(now)
        if not pareto_path.is_file():
            return Check(
                name=name,
                status="warn",
                details=(
                    f"no Pareto front stored yet at {pareto_path} "
                    "(no GEPA run has completed; legitimate early state)"
                ),
            )
        modified = datetime.fromtimestamp(pareto_path.stat().st_mtime, tz=UTC)
        age_days = (moment - modified).total_seconds() / _SECONDS_PER_DAY
        if age_days > max_age_days:
            return Check(
                name=name,
                status="warn",
                details=(
                    f"Pareto front stale: last update {age_days:.1f} days ago "
                    f"(limit {max_age_days:g} days)"
                ),
            )
        return Check(
            name=name,
            status="pass",
            details=(
                f"Pareto front fresh: last update {age_days:.1f} days ago "
                f"(limit {max_age_days:g} days)"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"pareto freshness check crashed: {exc}")


def check_backup_freshness(
    backups_dir: Path, max_age_hours: float = 25, now: datetime | None = None
) -> Check:
    """Newest ``YYYY-MM-DD.tar.gz`` backup within ``max_age_hours`` passes.

    Age is measured from the newest archive's mtime. A missing backups
    directory, or one without dated archives, warns ("never backed up") —
    the plan's daily 02:00 run (P18.T1) makes that a setup problem, not a
    crash; a stale latest backup also warns (``backup_stale`` alert severity
    in plan Section 5.3).
    """
    name = "backup_freshness"
    try:
        moment = _as_utc_now(now)
        if not backups_dir.is_dir():
            return Check(
                name=name,
                status="warn",
                details=f"no backups directory at {backups_dir}: never backed up",
            )
        archives = [
            path
            for path in backups_dir.glob("*.tar.gz")
            if _ARCHIVE_NAME_RE.fullmatch(path.name)
        ]
        if not archives:
            return Check(
                name=name,
                status="warn",
                details=f"no dated (*.tar.gz) backups found in {backups_dir}: never backed up",
            )
        newest = max(archives, key=lambda path: path.stat().st_mtime)
        age_hours = (
            moment - datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
        ).total_seconds() / _SECONDS_PER_HOUR
        if age_hours > max_age_hours:
            return Check(
                name=name,
                status="warn",
                details=(
                    f"latest backup {newest.name} is {age_hours:.1f} h old "
                    f"(limit {max_age_hours:g} h)"
                ),
            )
        return Check(
            name=name,
            status="pass",
            details=(
                f"latest backup {newest.name} is {age_hours:.1f} h old "
                f"(limit {max_age_hours:g} h)"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"backup freshness check crashed: {exc}")


def check_git_corruption(repo_path: Path) -> Check:
    """Git repository integrity via fsck + HEAD resolution (plan P18.T3).

    Delegates to :func:`hermes.backup.git_recovery.detect_corruption`:
    pass = fsck clean, HEAD resolves, no stale locks; fail = positive
    corruption evidence (fsck errors, missing objects, unresolvable HEAD,
    a stale ``.git/index.lock`` older than the 5-minute cadence); warn =
    the check could not establish integrity (no repository at ``repo_path``,
    git binary missing, fsck timeout) or crashed internally.
    """
    name = "git_corruption"
    try:
        if not repo_path.is_dir() or not (repo_path / ".git").exists():
            return Check(
                name=name,
                status="warn",
                details=f"no git repository at {repo_path}; git corruption unknown",
            )
        report = detect_corruption(repo_path)
        if report.corrupted:
            shown = "; ".join(report.findings[:5]) + ("; ..." if len(report.findings) > 5 else "")
            return Check(name=name, status="fail", details=f"git corruption detected: {shown}")
        if report.findings:
            # fsck could not run (no git binary / timeout): integrity unknown
            return Check(name=name, status="warn", details="; ".join(report.findings[:5]))
        return Check(
            name=name,
            status="pass",
            details="git fsck clean, HEAD resolves, no stale locks",
        )
    except Exception as exc:  # noqa: BLE001 - checks must never raise
        return Check(name=name, status="fail", details=f"git corruption check crashed: {exc}")


def run_all_checks(context: HealthCheckContext | None = None) -> list[Check]:
    """Run all six checks (plan Section 5.1); HealthReport-compatible list.

    ``context`` defaults to :class:`HealthCheckContext` (all paths assembled
    from ``hermes_home()``; default socket probe). The returned list is in
    plan-table order and every element is a :class:`Check` — never a raised
    exception, even when a check internally fails. With
    ``context.include_git_corruption=True`` (plan P18.T3) the
    ``git_corruption`` check is appended additively at the end.
    """
    ctx = context if context is not None else HealthCheckContext()
    checks = [
        check_memory_store_integrity(ctx.store_root),
        check_git_state(ctx.repo_path),
        check_main_model_reachable(ctx.probe),
        check_trace_store_size(ctx.traces_root),
        check_pareto_front_freshness(ctx.pareto_path),
        check_backup_freshness(ctx.backups_dir),
    ]
    if ctx.include_git_corruption:
        checks.append(check_git_corruption(ctx.repo_path))
    return checks
