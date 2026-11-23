"""P16.T4 acceptance: PyPI packaging — build, clean install, migration CLI.

Two end-to-end checks (both subprocess-based, both slow-but-real):

- ``test_clean_install_works``: build the wheel with ``pip wheel``, install it
  into a FRESH venv, and import ``hermes``/``hermes.config`` from that venv.
  The fork-base ``setup.py`` guards ``bdist_wheel``/``sdist`` behind the
  ``HERMES_NIX_BUILD=1`` escape hatch (the one documented legitimate wheel
  builder — see setup.py); the build below opts in through that env var
  rather than touching setup.py. The installed venv proves the wheel is
  well-formed: package data ships (the diagnostics dashboard template) and
  the v2 package imports. Only ``pyyaml`` is added alongside the wheel
  (``--no-deps`` install): it is the minimal import-time dependency of
  ``hermes.config`` — resolving the full dependency tree is the production
  installer's job, and a clean install proves the package itself works.
- ``test_migration_script_runs``: run ``scripts/migrate_fork_to_v2_0.py``
  with ``--dry-run`` on a fixture fork-base memories directory: exit 0,
  report on stdout, and ZERO writes to the target (the always-kept
  pre-migration source backup is the only artifact). Data preservation
  through an APPLIED migration is covered by the dedicated suite in
  ``tests/integration/test_migration.py`` (``test_fork_to_v2_0_preserves_entries``
  and ``test_migration_rollback_script``) and is not duplicated here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_SCRIPT = REPO_ROOT / "scripts" / "migrate_fork_to_v2_0.py"


def _report_field(report_stdout: str, field: str) -> int:
    """Parse ``field: <int>`` out of the migration CLI's report."""
    for line in report_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return int(stripped.split(":", 1)[1].strip())
    raise AssertionError(f"report field {field!r} missing from CLI stdout:\n{report_stdout}")


@pytest.mark.integration
def test_clean_install_works(tmp_path: Path) -> None:
    """Build the wheel, clean-install it into a fresh venv, import hermes."""
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    # 1) Build the wheel from the repo root (opt into the documented Nix
    #    wheel-build escape hatch; see setup.py's bdist_wheel guard).
    build_env = dict(os.environ, HERMES_NIX_BUILD="1")
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPO_ROOT),
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheelhouse),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=build_env,
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stderr[-4000:]}"
    wheels = list(wheelhouse.glob("hermes_agent-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    # 2) Package data ships: the diagnostics dashboard template (declared in
    #    pyproject [tool.setuptools.package-data] for P16.T4) is in the wheel.
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    assert "hermes/diagnostics/templates/dashboard.html" in names
    assert "hermes/memory/gating.py" in names  # the v2 pipeline is packaged

    # 3) Fresh venv (no shared state with the dev environment).
    venv_dir = tmp_path / "fresh-venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert created.returncode == 0, f"venv creation failed:\n{created.stderr[-2000:]}"
    python_dir = "Scripts" if os.name == "nt" else "bin"
    python_name = "python.exe" if os.name == "nt" else "python"
    venv_python = venv_dir / python_dir / python_name
    assert venv_python.exists(), f"venv python missing at {venv_python}"

    # 4) Clean install: the wheel alone (--no-deps, --no-index: offline).
    installed = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheels[0]),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert installed.returncode == 0, f"wheel install failed:\n{installed.stderr[-4000:]}"

    # 5) pyyaml: the minimal import-time dependency of hermes.config.
    dep = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--quiet", "pyyaml==6.0.3"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert dep.returncode == 0, f"pyyaml install failed:\n{dep.stderr[-2000:]}"

    # 6) The clean venv imports the v2 package. cwd is NOT the repo (a repo
    #    cwd would shadow site-packages with the source tree and make the
    #    check vacuous; HERMES_HOME from the hermetic test env is inherited,
    #    so hermes_home() stays inside the per-test sandbox).
    probe = (
        "import hermes, hermes.config\n"
        "from hermes.config import hermes_home\n"
        "import importlib.metadata as md\n"
        "print('ok', hermes.__version__, md.version('hermes-agent'))\n"
        "assert str(hermes_home())\n"
    )
    verified = subprocess.run(
        [str(venv_python), "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=str(tmp_path),
    )
    assert verified.returncode == 0, f"clean-venv import failed:\n{verified.stderr[-4000:]}"
    # "ok <hermes module version> <installed dist version>": both metadata
    # surfaces resolve inside the clean venv
    assert verified.stdout.startswith("ok ")
    assert len(verified.stdout.split()) >= 3


@pytest.mark.integration
def test_migration_script_runs(tmp_path: Path) -> None:
    """The fork->v2.0 migration CLI dry-runs cleanly and writes no target."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(
        "- Always run the test suite before pushing.\n"
        "- Prefers concise answers over long explanations.\n"
        "\n"
        "Uses Windows for daily work.\n",
        encoding="utf-8",
    )
    (source / "USER.md").write_text(
        "# User profile\n\nName: Test User\nTimezone: UTC+1\n",
        encoding="utf-8",
    )
    target = tmp_path / "memory-repo"

    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATION_SCRIPT),
            str(source),
            str(target),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    # exit 0 with the human-readable report on stdout
    assert result.returncode == 0, f"migration CLI failed:\n{result.stderr[-2000:]}"
    assert "dry run" in result.stdout and "yes" in result.stdout
    # 3 MEMORY.md lines (2 bullets + 1 plain) + 1 whole-file USER.md unit
    found = _report_field(result.stdout, "entries found")
    migrated = _report_field(result.stdout, "entries migrated")
    assert found == 4
    assert migrated == 0  # dry run writes nothing
    assert "memory-repo" in result.stdout  # the target is named in the report

    # dry run: the target repository was NOT created (zero writes there)
    assert not target.exists()
    # the source is untouched, and the always-kept pre-migration backup exists
    assert (source / "MEMORY.md").read_text(encoding="utf-8").startswith(
        "- Always run the test suite"
    )
    backup = Path(f"{source}-premigration-backup")
    assert backup.is_dir()
    assert sorted(path.name for path in backup.iterdir()) == ["MEMORY.md", "USER.md"]
