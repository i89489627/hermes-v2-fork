"""Integration tests: OOM-safe webapp renderer updater (plan P21.T9, issue #156161).

Reproduces the upstream bug: the webapp renderer build had no Node.js heap
cap, so an OOM-killed build stage still let the update exit 0 and the user
was left with a silently stale bundle. These tests run the v2
:class:`hermes.install.updater.Updater` against real subprocesses on Windows
(fake builds are ``sys.executable -c`` snippets — Node.js is not required):

* an OOM-killed build (exit 134/137 or the V8 "heap out of memory" stderr
  marker) is detected and reported as an explicit failure;
* a build that exits 0 without producing a *newer* bundle never swaps the
  installed bundle and never reports success (no silent stale bundle);
* every spawned build command carries the ``--max-old-space-size`` heap cap.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from hermes.install.updater import STAGING_ENV_VAR, Updater


def _fake_build(snippet: str) -> list[str]:
    """A fake renderer build: a ``python -c`` snippet (no Node.js needed)."""
    return [sys.executable, "-c", snippet]


def _staging_writer(content: str) -> str:
    """Snippet for a build that honours the updater's staging contract."""
    return (
        "import os, pathlib\n"
        f"pathlib.Path(os.environ[{STAGING_ENV_VAR!r}]).write_text({content!r}, encoding='utf-8')\n"
    )


@pytest.mark.integration
def test_oom_detected_and_reported(tmp_path: Path) -> None:
    """OOM exit codes and the V8 stderr marker must fail the update explicitly."""
    bundle = tmp_path / "webapp" / "renderer-bundle.tar.gz"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("INSTALLED BUNDLE", encoding="utf-8")
    old = time.time() - 3600
    os.utime(bundle, (old, old))

    cases = [
        ("sigkill-exit-137", "import sys; sys.exit(137)", 137),
        ("sigabort-exit-134", "import sys; sys.exit(134)", 134),
        (
            # Exit code lies about success; the V8 stderr marker catches it.
            "heap-marker-on-stderr",
            'import sys; sys.stderr.write("FATAL ERROR: JavaScript heap out of memory"); sys.exit(0)',
            0,
        ),
    ]
    for name, snippet, expected_exit in cases:
        result = Updater(build_command=_fake_build(snippet), bundle_path=bundle).update()
        assert not result.success, f"{name} must not be reported as a successful update"
        assert result.oom_detected, f"{name} must be detected as OOM"
        assert result.exit_code == expected_exit, f"{name} must record the build exit code"
        assert result.error is not None, f"{name} must carry a descriptive error"
        assert "OOM" in result.error, f"{name} error must name the OOM cause: {result.error!r}"
        assert "heap cap" in result.error, f"{name} error must point at the heap cap: {result.error!r}"
        assert result.bundle_path is None, f"{name} must not present a bundle as installed"
        # The OOM must never be masked: the installed bundle is untouched and
        # the update explicitly failed instead of exiting 0.
        assert bundle.read_text(encoding="utf-8") == "INSTALLED BUNDLE", name


@pytest.mark.integration
def test_no_silent_stale_bundle(tmp_path: Path) -> None:
    """A build that exits 0 without a newer bundle fails loudly and swaps nothing."""
    bundle = tmp_path / "webapp" / "renderer-bundle.tar.gz"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("OLD BUNDLE", encoding="utf-8")
    old = time.time() - 3600
    os.utime(bundle, (old, old))

    # Control: a healthy build that honours the staging contract DOES swap in
    # the fresh bundle, proving the guards below fail for real reasons.
    healthy = Updater(build_command=_fake_build(_staging_writer("NEW BUNDLE v2")), bundle_path=bundle)
    result = healthy.update()
    assert result.success, f"healthy build must succeed: {result.error!r}"
    assert result.exit_code == 0
    assert not result.oom_detected
    assert result.error is None
    assert result.bundle_path == bundle
    assert bundle.read_text(encoding="utf-8") == "NEW BUNDLE v2"

    installed_time = time.time() - 1800
    os.utime(bundle, (installed_time, installed_time))

    # Bug scenario 1: build exits 0 but writes no bundle at the staging path.
    result = Updater(build_command=_fake_build("import sys; sys.exit(0)"), bundle_path=bundle).update()
    assert not result.success, "exit 0 without a bundle must not report success"
    assert result.exit_code == 0
    assert result.error is not None and "no bundle" in result.error and "staging" in result.error
    assert bundle.read_text(encoding="utf-8") == "NEW BUNDLE v2", "installed bundle must stay untouched"

    # Bug scenario 2: build exits 0 and produces a bundle that is OLDER than
    # the installed one — exactly the silent-stale-bundle regression.
    stale_writer = (
        "import os, pathlib, time\n"
        "staging = pathlib.Path(os.environ['HERMES_BUNDLE_STAGING'])\n"
        "staging.write_text('STALE OUTPUT', encoding='utf-8')\n"
        "older = time.time() - 3600\n"
        "os.utime(staging, (older, older))\n"
    )
    result = Updater(build_command=_fake_build(stale_writer), bundle_path=bundle).update()
    assert not result.success, "a non-newer bundle must not be swapped in as success"
    assert result.error is not None and "stale bundle" in result.error and "not newer" in result.error
    assert result.bundle_path is None
    assert bundle.read_text(encoding="utf-8") == "NEW BUNDLE v2", "installed bundle must stay untouched"

    # Never leave a silently stale bundle behind either: no staging leftovers.
    leftovers = sorted(p.name for p in bundle.parent.iterdir() if ".staging-" in p.name)
    assert leftovers == [], f"failed updates must clean their staging files: {leftovers}"


@pytest.mark.integration
def test_heap_cap_set(tmp_path: Path) -> None:
    """Every spawned build command carries the Node.js --max-old-space-size cap."""
    bundle = tmp_path / "webapp" / "renderer-bundle.tar.gz"
    argv_file = tmp_path / "spawned-argv.json"
    # Fake recorder build: dumps its own argv (the command the Updater spawned)
    # and honours the staging contract so the update completes successfully.
    recorder = (
        "import json, os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv), encoding='utf-8')\n"
        f"pathlib.Path(os.environ[{STAGING_ENV_VAR!r}]).write_text('RECORDED BUNDLE', encoding='utf-8')\n"
    )
    updater = Updater(
        build_command=[sys.executable, "-c", recorder, str(argv_file)],
        bundle_path=bundle,
    )
    result = updater.update()
    assert result.success, f"recorder build must succeed: {result.error!r}"
    assert bundle.read_text(encoding="utf-8") == "RECORDED BUNDLE"

    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    flags = [arg for arg in argv if arg.startswith("--max-old-space-size=")]
    assert flags, f"spawned command {argv!r} must contain the --max-old-space-size heap cap"
    default_cap = int(flags[0].split("=", 1)[1])
    assert default_cap >= 128, f"heap cap must be a real megabyte budget, got {default_cap}"

    # An explicit cap is forwarded verbatim into the spawned command.
    argv_file.write_text("[]", encoding="utf-8")
    explicit = Updater(
        build_command=[sys.executable, "-c", recorder, str(argv_file)],
        bundle_path=bundle,
        heap_mb=8192,
    )
    result = explicit.update()
    assert result.success, f"explicit-cap build must succeed: {result.error!r}"
    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert "--max-old-space-size=8192" in argv, f"spawned command {argv!r} must carry the 8192 MB cap"
