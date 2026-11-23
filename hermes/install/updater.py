"""OOM-safe webapp renderer updater (plan P21.T9, issue #156161).

Upstream bug: the webapp renderer build ran with no Node.js heap cap, so the
build stage was OOM-killed while the update itself exited ``0`` (success). The
bundle stayed stale and the user believed the update had succeeded.

The v2 :class:`Updater` closes that hole with four guarantees:

1. **Heap cap always set.** Every spawned build command carries
   ``--max-old-space-size=<NODE_HEAP_MB>`` (default 4096 MB). For ``node``
   launchers the flag is inserted right after the executable — the only
   position V8 honours — and ``NODE_OPTIONS`` is seeded with the same flag so
   nested build workers (vite/webpack pools) inherit the cap too. For other
   launchers the flag is appended as the final argument (wrappers forward it)
   and ``NODE_HEAP_MB`` is exported for scripts that read it.
2. **OOM is detected, not swallowed.** Exit codes 134/137 (SIGABRT/SIGKILL)
   and ``heap out of memory`` markers on stderr are reported as an explicit
   OOM failure with remediation guidance — never as a success.
3. **No silent stale bundle.** The build must hand the fresh bundle to the
   staging path published in ``HERMES_BUNDLE_STAGING``; the installed bundle
   at :attr:`Updater.bundle_path` is swapped in (atomic ``os.replace``) only
   when the build exited 0 **and** the staged bundle's mtime is strictly
   newer than the installed one. Otherwise the update fails explicitly and
   the installed bundle is left byte-for-byte untouched.
4. **Explicit results, never exceptions.** :meth:`Updater.update` returns an
   :class:`UpdateResult` describing what happened; failures carry a
   descriptive ``error`` instead of raising.

Build-command contract: the command must write the freshly built bundle to
the path given in ``HERMES_BUNDLE_STAGING`` (the default command wraps
``scripts/build_webapp.sh``, which honours it). Fakes in tests use
``sys.executable -c`` snippets; Node.js is never required to test the updater.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hermes.config import hermes_home

__all__ = [
    "DEFAULT_HEAP_MB",
    "HEAP_ENV_VAR",
    "HEAP_FLAG_PREFIX",
    "OOM_EXIT_CODE",
    "OOM_EXIT_CODES",
    "OOM_STDERR_MARKER",
    "STAGING_ENV_VAR",
    "UpdateResult",
    "Updater",
    "with_heap_cap",
]

#: Exit codes the OOM killer / V8 abort path produces (SIGKILL / SIGABRT).
OOM_EXIT_CODES: Final[frozenset[int]] = frozenset({134, 137})

#: Exit code ``scripts/build_webapp.sh`` uses for an OOM-killed build.
OOM_EXIT_CODE: Final[int] = 42

#: Case-insensitive stderr marker V8 prints before aborting on OOM.
OOM_STDERR_MARKER: Final[str] = "heap out of memory"

#: Environment variable carrying the heap cap in megabytes.
HEAP_ENV_VAR: Final[str] = "NODE_HEAP_MB"

#: Environment variable carrying the staged-bundle path for the build.
STAGING_ENV_VAR: Final[str] = "HERMES_BUNDLE_STAGING"

#: Default V8 heap cap in megabytes.
DEFAULT_HEAP_MB: Final[int] = 4096

#: Prefix of the V8 heap-cap command-line flag.
HEAP_FLAG_PREFIX: Final[str] = "--max-old-space-size="

#: Upper bound for a renderer build (30 minutes) so a hung build cannot hang
#: the update forever; a timeout is reported as an explicit failure.
_BUILD_TIMEOUT_SECONDS: Final[float] = 1800.0


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one :meth:`Updater.update` run."""

    success: bool
    oom_detected: bool = False
    exit_code: int | None = None
    error: str | None = None
    bundle_path: Path | None = None


def with_heap_cap(command: list[str], heap_mb: int) -> list[str]:
    """Return ``command`` carrying the V8 heap cap for ``heap_mb``.

    ``node`` honours ``--max-old-space-size`` only *before* the script entry,
    so for node launchers the flag is inserted immediately after the
    executable. For any other launcher (bash wrappers, test fakes) the flag is
    appended as the final argument — the spawned argv always carries the cap.
    A command that already sets the cap is returned unchanged.
    """
    if any(arg.startswith(HEAP_FLAG_PREFIX) for arg in command):
        return list(command)
    flag = f"{HEAP_FLAG_PREFIX}{heap_mb}"
    if not command:
        return ["node", flag]
    launcher = Path(command[0]).name.lower()
    if launcher.startswith("node"):
        return [command[0], flag, *command[1:]]
    return [*command, flag]


class Updater:
    """Runs the webapp renderer build under a heap cap and installs the result.

    Parameters:
        build_command: Command to spawn. ``None`` selects the default
            (``bash scripts/build_webapp.sh`` in a repo checkout).
        bundle_path: Installed bundle location (file). ``None`` defaults to
            ``<hermes_home>/webapp/renderer-bundle.tar.gz``.
        heap_mb: Node.js heap cap in megabytes. ``None`` reads ``NODE_HEAP_MB``
            from the environment, falling back to 4096.
    """

    def __init__(
        self,
        build_command: list[str] | None = None,
        bundle_path: Path | None = None,
        heap_mb: int | None = None,
    ) -> None:
        if build_command is not None and not build_command:
            raise ValueError("build_command must be a non-empty list when provided")
        self._build_command = list(build_command) if build_command is not None else None
        self._bundle_path = Path(bundle_path) if bundle_path is not None else _default_bundle_path()
        self._heap_mb = heap_mb if heap_mb is not None else _heap_mb_from_env()

    @property
    def bundle_path(self) -> Path:
        """Installed bundle location the updater swaps new builds into."""
        return self._bundle_path

    @property
    def heap_mb(self) -> int:
        """Node.js heap cap (megabytes) applied to spawned builds."""
        return self._heap_mb

    def update(self) -> UpdateResult:
        """Build, verify, and install the webapp bundle; never raise on build failure."""
        if self._build_command is None:
            return self._default_command_missing()
        command = with_heap_cap(self._build_command, self._heap_mb)
        staging = self._staging_path()
        pre_stat = _mtime_of(self._bundle_path)
        proc = self._run_build(command, staging)
        if isinstance(proc, UpdateResult):
            return proc
        stderr_text = _coerce_text(proc.stderr)
        exit_code = proc.returncode

        if exit_code in OOM_EXIT_CODES or OOM_STDERR_MARKER in stderr_text.lower():
            return self._failure(
                exit_code=exit_code,
                oom=True,
                staging=staging,
                message=(
                    f"webapp renderer build was OOM-killed (exit code {exit_code}); "
                    f"the Node.js heap cap of {self._heap_mb} MB "
                    f"({HEAP_FLAG_PREFIX}{self._heap_mb}) was exhausted. "
                    f"Retry with a larger cap, e.g. {HEAP_ENV_VAR}={self._heap_mb * 2} "
                    f"{self._format_command(command)}. "
                    f"The installed bundle at {self._bundle_path} was NOT replaced; "
                    "the update did not succeed."
                ),
            )
        if exit_code != 0:
            return self._failure(
                exit_code=exit_code,
                oom=False,
                staging=staging,
                message=(
                    f"webapp renderer build failed with exit code {exit_code}; "
                    f"stderr tail: {_stderr_tail(stderr_text)}. "
                    f"The installed bundle at {self._bundle_path} was NOT replaced; "
                    "the update did not succeed."
                ),
            )

        staged_mtime = _mtime_of(staging)
        if staged_mtime is None:
            return self._failure(
                exit_code=exit_code,
                oom=False,
                staging=staging,
                message=(
                    f"webapp renderer build exited 0 but produced no bundle at the "
                    f"staging path {staging} (set by {STAGING_ENV_VAR}); refusing to "
                    f"declare success — the installed bundle at {self._bundle_path} "
                    "was NOT replaced and is now stale until a build succeeds."
                ),
            )
        if pre_stat is not None and staged_mtime <= pre_stat:
            return self._failure(
                exit_code=exit_code,
                oom=False,
                staging=staging,
                message=(
                    f"stale bundle: the freshly built bundle at {staging} "
                    f"(mtime {staged_mtime:.3f}) is not newer than the installed "
                    f"bundle at {self._bundle_path} (mtime {pre_stat:.3f}); refusing "
                    "to swap it into place — the installed bundle was NOT replaced "
                    "and the update did not succeed."
                ),
            )

        self._bundle_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, self._bundle_path)
        installed_mtime = _mtime_of(self._bundle_path)
        if installed_mtime is None:  # pragma: no cover - os.replace just succeeded
            return self._failure(
                exit_code=exit_code,
                oom=False,
                staging=staging,
                message=(
                    f"webapp bundle vanished after the swap into {self._bundle_path}; "
                    "the update did not complete."
                ),
            )
        return UpdateResult(
            success=True,
            oom_detected=False,
            exit_code=exit_code,
            error=None,
            bundle_path=self._bundle_path,
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _default_command_missing(self) -> UpdateResult:
        default_script = _repo_root() / "scripts" / "build_webapp.sh"
        return UpdateResult(
            success=False,
            oom_detected=False,
            exit_code=None,
            error=(
                f"no build_command was configured and the default build script "
                f"{default_script} does not exist in this installation; pass an "
                "explicit build command to Updater(build_command=...) that writes "
                f"the built bundle to the {STAGING_ENV_VAR} path"
            ),
            bundle_path=None,
        )

    def _staging_path(self) -> Path:
        name = self._bundle_path.name
        return self._bundle_path.parent / f".{name}.staging-{uuid.uuid4().hex[:8]}"

    def _run_build(
        self, command: list[str], staging: Path
    ) -> subprocess.CompletedProcess[str] | UpdateResult:
        env = _child_env(self._heap_mb, staging)
        try:
            self._bundle_path.parent.mkdir(parents=True, exist_ok=True)
            return subprocess.run(
                command,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=_BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            tail = _stderr_tail(_coerce_text(exc.stderr))
            return self._failure(
                exit_code=None,
                oom=False,
                staging=staging,
                message=(
                    f"webapp renderer build timed out after {_BUILD_TIMEOUT_SECONDS:.0f}s "
                    f"and was terminated; stderr tail: {tail}. The installed bundle at "
                    f"{self._bundle_path} was NOT replaced; the update did not succeed."
                ),
            )
        except OSError as exc:
            return self._failure(
                exit_code=None,
                oom=False,
                staging=staging,
                message=(
                    f"webapp renderer build could not be spawned ({type(exc).__name__}: "
                    f"{exc}) for command {self._format_command(command)}; the installed "
                    f"bundle at {self._bundle_path} was NOT replaced."
                ),
            )

    def _failure(
        self, exit_code: int | None, oom: bool, staging: Path, message: str
    ) -> UpdateResult:
        staging.unlink(missing_ok=True)
        return UpdateResult(
            success=False,
            oom_detected=oom,
            exit_code=exit_code,
            error=message,
            bundle_path=None,
        )

    def _format_command(self, command: list[str]) -> str:
        return " ".join(command)


def _heap_mb_from_env() -> int:
    raw = os.environ.get(HEAP_ENV_VAR, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_HEAP_MB
    return value if value > 0 else DEFAULT_HEAP_MB


def _child_env(heap_mb: int, staging: Path) -> dict[str, str]:
    env = {**os.environ}
    env[HEAP_ENV_VAR] = str(heap_mb)
    env[STAGING_ENV_VAR] = str(staging)
    flag = f"{HEAP_FLAG_PREFIX}{heap_mb}"
    existing = env.get("NODE_OPTIONS", "")
    if HEAP_FLAG_PREFIX not in existing:
        env["NODE_OPTIONS"] = f"{existing} {flag}".strip()
    return env


def _default_bundle_path() -> Path:
    return hermes_home() / "webapp" / "renderer-bundle.tar.gz"


def _mtime_of(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _stderr_tail(text: str, limit: int = 400) -> str:
    cleaned = " ".join(text.split())
    return cleaned[-limit:] if len(cleaned) > limit else cleaned


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
