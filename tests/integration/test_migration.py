"""Integration tests: zero-loss config migrations (plan P21.T11, issue #25272).

Reproduces the upstream update bug: the update migration overwrote the
user's ``config.yaml``, so every custom model configuration vanished. These
tests exercise :func:`hermes.install.migrations.run_migration` against real
files in a per-test temp dir: config preserved across an update, backup
taken before any change, user prompted on schema changes, and the exact
issue-#25272 scenario (5 custom models vs an incoming 2-built-in template)
surviving with zero loss.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from hermes.config import CONFIG_SCHEMA_VERSION_KEY, load_raw_config
from hermes.install.migrations import (
    STATUS_MERGED,
    STATUS_USER_DECIDED,
    run_migration,
)
from hermes.memory.git_backend import GitBackend
from hermes.memory.migrations import MigrationReport, migrate, rollback
from hermes.memory.migrations import fork_to_v2_0 as fork_migration_module
from hermes.memory.migrations.fork_to_v2_0 import (
    SoulMigrationResult,
    migrate_soul,
    migrate_to_global,
    verify_global_context,
)
from hermes.memory.soul import SoulManager
from hermes.memory.store import MemoryStore
from hermes.memory.trust_card import compute_trust_score, parse, serialize


def _custom_models(count: int) -> dict[str, Any]:
    """Build ``count`` custom model config entries like a real user's setup."""
    return {
        f"custom-{index}": {
            "provider": "openai-compatible",
            "base_url": f"https://inference.example-{index}.internal/v1",
            "model": f"teamFineTune-{index}",
            "api_key_env": f"TEAM_MODEL_{index}_KEY",
        }
        for index in range(1, count + 1)
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


@pytest.mark.integration
def test_config_yaml_preserved(tmp_path: Path) -> None:
    """A config.yaml update keeps the user's file valid with all settings intact."""
    config_path = tmp_path / "config.yaml"
    current = {
        CONFIG_SCHEMA_VERSION_KEY: 2,
        "custom_models": _custom_models(3),
        "v2": {"memory_pipeline": {"retry": {"max_retries": 7}}},
        "terminal": {"backend": "local"},
    }
    _write_yaml(config_path, current)

    incoming = {
        CONFIG_SCHEMA_VERSION_KEY: 2,
        "v2": {
            "memory_pipeline": {
                "retry": {"max_retries": 3, "backoff_max_seconds": 90.0},
                "gating": {"mode": "batched"},
            }
        },
    }

    result = run_migration(config_path, incoming)

    # The incoming template lacks the user's custom keys — the #25272 drop
    # signature — so the safe default merge runs even with the schema
    # version unchanged (2 -> 2).
    assert result.status == STATUS_MERGED
    on_disk = _read_yaml(config_path)
    # User's file still holds every pre-update setting, unchanged where the
    # user had customized it (current values win scalar conflicts).
    assert on_disk["custom_models"] == current["custom_models"]
    assert on_disk["v2"]["memory_pipeline"]["retry"]["max_retries"] == 7
    assert on_disk["terminal"] == {"backend": "local"}
    # New release keys were added.
    assert on_disk["v2"]["memory_pipeline"]["gating"] == {"mode": "batched"}
    assert on_disk["v2"]["memory_pipeline"]["retry"]["backoff_max_seconds"] == 90.0
    # File is still valid for the normal v2 loader (schema unchanged).
    assert load_raw_config(config_path) == on_disk


@pytest.mark.integration
def test_backup_before_config_change(tmp_path: Path) -> None:
    """A timestamped backup capturing the pre-change file exists before any write."""
    config_path = tmp_path / "config.yaml"
    original = {
        CONFIG_SCHEMA_VERSION_KEY: 2,
        "custom_models": _custom_models(2),
    }
    _write_yaml(config_path, original)

    incoming = {CONFIG_SCHEMA_VERSION_KEY: 3, "models": {"built-in-a": {}, "built-in-b": {}}}

    # Dry run: a backup is still taken (ALWAYS before any change) but the
    # config file itself is untouched.
    dry = run_migration(config_path, incoming, apply=False)
    assert dry.status == STATUS_MERGED
    assert dry.backup_path is not None
    assert dry.backup_path.exists()
    assert dry.backup_path.name.startswith("config.yaml.bak-")
    assert load_raw_config(dry.backup_path) == original
    assert _read_yaml(config_path) == original  # dry run wrote nothing

    # Applied run: a second, separate backup; the file now holds the merge.
    result = run_migration(config_path, incoming)
    assert result.backup_path is not None
    assert result.backup_path != dry.backup_path
    assert load_raw_config(result.backup_path) == original
    merged = _read_yaml(config_path)
    assert merged["custom_models"] == original["custom_models"]
    assert merged[CONFIG_SCHEMA_VERSION_KEY] == 3
    assert set(merged["models"]) == {"built-in-a", "built-in-b"}


@pytest.mark.integration
def test_user_prompted_on_schema_change(tmp_path: Path) -> None:
    """The schema-change callback fires with a description naming the dropped keys."""
    config_path = tmp_path / "config.yaml"
    current = {
        CONFIG_SCHEMA_VERSION_KEY: 2,
        "custom_models": _custom_models(3),
        "legacy_endpoints": {"old-gateway": "https://old.example.internal"},
    }
    _write_yaml(config_path, current)

    incoming = {
        CONFIG_SCHEMA_VERSION_KEY: 3,
        "models": {"built-in-a": {}, "built-in-b": {}},
    }

    prompts: list[str] = []

    def prompt(description: str) -> str:
        prompts.append(description)
        return "merge"

    result = run_migration(config_path, incoming, on_schema_change=prompt)

    assert result.status == STATUS_USER_DECIDED
    assert len(prompts) == 1
    description = prompts[0]
    # The description must tell the user WHAT was at risk: the dropped keys
    # and the version drift.
    assert "custom_models" in description
    assert "legacy_endpoints" in description
    assert "drop" in description
    assert "2" in description and "3" in description

    # The user's "merge" decision preserved the custom models.
    merged = _read_yaml(config_path)
    assert merged["custom_models"] == current["custom_models"]
    assert set(merged["models"]) == {"built-in-a", "built-in-b"}


@pytest.mark.integration
def test_no_config_loss_on_update(tmp_path: Path) -> None:
    """Issue #25272 repro: 5 custom models survive an update shipping 2 built-ins.

    No callback is configured (the update tooling has no UI available), so
    the safe default merge must preserve every custom model and add the
    release's built-ins.
    """
    config_path = tmp_path / "config.yaml"
    custom_models = _custom_models(5)
    current = {
        CONFIG_SCHEMA_VERSION_KEY: 2,
        "custom_models": custom_models,
        "v2": {"memory_pipeline": {"retry": {"max_retries": 9}}},
    }
    _write_yaml(config_path, current)

    incoming = {
        CONFIG_SCHEMA_VERSION_KEY: 3,
        "models": {"built-in-a": {"provider": "nous"}, "built-in-b": {"provider": "openai"}},
        "v2": {"memory_pipeline": {"gating": {"mode": "batched"}}},
    }

    result = run_migration(config_path, incoming)  # no callback -> default merge

    assert result.status == STATUS_MERGED
    assert result.backup_path is not None and result.backup_path.exists()
    after = load_raw_config(config_path)
    assert after is not None
    # All 5 custom models survive, byte-equal to the user's definitions.
    assert after["custom_models"] == custom_models
    # The 2 built-ins from the release were added.
    assert set(after["models"]) == {"built-in-a", "built-in-b"}
    # User's other settings untouched; new release keys added; schema stamped.
    assert after["v2"]["memory_pipeline"]["retry"] == {"max_retries": 9}
    assert after["v2"]["memory_pipeline"]["gating"] == {"mode": "batched"}
    assert after[CONFIG_SCHEMA_VERSION_KEY] == 3
    # The merged result is exactly what run_migration reported.
    assert result.merged_config == after


# ── P1.T6: fork-base -> v2.0 memory migration ────────────────────────────────
#
# The fork-base format is plain markdown without frontmatter under
# ~/.hermes/memories: MEMORY.md personal notes and USER.md profile style.
# migrate() converts them into v2.0 trust-card entries in a git-backed
# memory repo, always keeping a pre-migration backup; the CLI script wraps
# migrate + rollback (rollback restores the fork-base format byte-exact).

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fork_source(tmp_path: Path) -> Path:
    """A fake fork-base memories dir: MEMORY.md notes + USER.md profile."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(
        "# Memory\n"
        "\n"
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
    return source


def _tree_snapshot(root: Path) -> dict[str, str]:
    """``relative posix path -> text`` for every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.integration
def test_fork_to_v2_0_preserves_entries(tmp_path: Path) -> None:
    """Every fork-base entry lands in the v2 repo with its text verbatim."""
    source = _fork_source(tmp_path)
    before = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    report = migrate(source, target, apply=True)

    assert report.dry_run is False
    assert report.entries_found == 5  # 4 MEMORY.md lines + the USER.md profile
    assert report.entries_migrated == 5
    assert report.skipped == 0
    assert report.errors == []
    assert report.backup_path is not None and report.backup_path.is_dir()

    store = MemoryStore(root=target)
    entries = store.list()
    assert len(entries) == 5
    assert sum(entry.type == "Feedback" for entry in entries) == 4
    assert sum(entry.type == "User" for entry in entries) == 1
    assert all(entry.context == "global" for entry in entries)
    bodies = {entry.text for entry in entries}
    # bullet markers stripped, text otherwise verbatim; USER.md body byte-exact
    for expected in (
        "# Memory",
        "Always run the test suite before pushing.",
        "Prefers concise answers over long explanations.",
        "Uses Windows for daily work.",
        "# User profile\n\nName: Test User\nTimezone: UTC+1\n",
    ):
        assert expected in bodies

    # one commit carries the whole migration; nothing left dangling
    history = store.git.log(limit=10)
    assert len(history) == 1
    assert "migrate fork-base" in history[0].message
    assert store.git.is_clean()
    index = (target / "MEMORY.md").read_text(encoding="utf-8")
    assert all(entry.id in index for entry in entries)

    # the fork-base source is left untouched by a successful migration
    assert _tree_snapshot(source) == before


@pytest.mark.integration
def test_fork_to_v2_0_sets_default_trust(tmp_path: Path) -> None:
    """Entries without explicit verification get trust_score 0.5 + one event."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(
        "- Always run the test suite before pushing.\n"
        "- Trust nothing that is not verified.\n",
        encoding="utf-8",
    )

    report = migrate(source, tmp_path / "memory-repo", apply=True)
    assert report.entries_migrated == 2

    store = MemoryStore(root=tmp_path / "memory-repo")
    entries = store.list()
    assert len(entries) == 2
    for entry in entries:
        assert entry.trust_score == 0.5
        assert len(entry.verification_events) == 1
        event = entry.verification_events[0]
        assert event.source == "agent-inferred"
        # the recorded event re-derives the 0.5 default under the plan's
        # weighted-average rule instead of drifting to 1.0 on recomputation
        assert compute_trust_score([event]) == 0.5
        # the trust card on disk agrees with the in-memory entry
        card, body = parse(
            (store.root / "global" / "facts" / f"{entry.id}.md").read_text(encoding="utf-8")
        )
        assert card.trust_score == 0.5
        assert card.context == "global"
        assert body == entry.text


@pytest.mark.integration
def test_migration_dry_run_mode(tmp_path: Path) -> None:
    """Dry run reports the full migration but writes nothing to the target."""
    source = _fork_source(tmp_path)
    before = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    report = migrate(source, target, apply=False)

    assert report.dry_run is True
    assert report.entries_found == 5  # full report of what WOULD be migrated
    assert report.entries_migrated == 0
    assert report.skipped == 0
    assert report.errors == []
    assert not target.exists()  # ZERO writes to the target repository
    # the pre-migration backup is still created (always-backup rule) and kept
    assert report.backup_path == tmp_path / "memories-premigration-backup"
    assert report.backup_path.is_dir()
    assert _tree_snapshot(report.backup_path) == before
    assert _tree_snapshot(source) == before

    # the applied run then migrates exactly what the dry run reported
    applied = migrate(source, target, apply=True)
    assert applied.dry_run is False
    assert applied.entries_migrated == applied.entries_found == report.entries_found
    assert MemoryStore(root=target).list()


@pytest.mark.integration
def test_migration_rollback_script(tmp_path: Path) -> None:
    """The CLI rollback removes the v2 repo and restores the fork-base format."""
    source = _fork_source(tmp_path)
    original = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    report = migrate(source, target, apply=True)
    backup = report.backup_path
    assert backup is not None and backup.is_dir()
    assert (target / "MEMORY.md").is_file()

    # simulate a failed run that consumed part of the fork-base source:
    # rollback must rebuild the exact original layout from the backup
    (source / "USER.md").unlink()
    (source / "MEMORY.md").write_text("partially overwritten\n", encoding="utf-8")

    script = _REPO_ROOT / "scripts" / "migrate_fork_to_v2_0.py"
    result = subprocess.run(
        [sys.executable, str(script), str(target), "--rollback", str(backup)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    # the v2 repo is gone again and the source is byte-identical to before
    assert not target.exists()
    assert _tree_snapshot(source) == original
    # the backup itself is kept (idempotent re-runs / forensics)
    assert _tree_snapshot(backup) == original


# ── P15.T2: fork-base SOUL.md split ──────────────────────────────────────────
#
# migrate_soul splits the fork-base SOUL.md by rule-based, first-match-wins
# block classification (safety/alignment/identity/NEVER-ALWAYS guardrails ->
# soul_core.md; voice/tone/formatting -> soul_style.md) into the exact
# SoulManager layout under <target>/soul/, committing the split when the
# target repo is Git-backed.

_SAFETY_SECTION = """\
## Safety principles

Refuse any request that could cause serious harm to people.
Treat dangerous instructions as off limits, every time.
"""

_ALIGNMENT_SECTION = """\
## Alignment commitments

Be honest about what you did and what you know.
Deceiving the user or fabricating results is out of bounds.
"""

_STYLE_VOICE_SECTION = """\
## Voice and tone

Keep a friendly, casual tone with light humor.
Short sentences, plain words, no corporate stiffness.
"""

_STYLE_FORMAT_SECTION = """\
## Formatting preferences

Prefer concise markdown answers with short paragraphs.
Use bullet lists for steps; skip heavy formatting otherwise.
"""

_SPLIT_SOUL_MD = f"""\
# SOUL

{_SAFETY_SECTION}
{_ALIGNMENT_SECTION}
{_STYLE_VOICE_SECTION}
{_STYLE_FORMAT_SECTION}"""


@pytest.mark.integration
def test_soul_split_correct(tmp_path: Path) -> None:
    """4+ SOUL.md sections split core-vs-style correctly, with the rule audit."""
    source = tmp_path / "SOUL.md"
    source.write_text(_SPLIT_SOUL_MD, encoding="utf-8")
    target = GitBackend.init(tmp_path / "memory-repo").repo_path

    result = migrate_soul(source, target, apply=True)

    assert isinstance(result, SoulMigrationResult)
    assert result.core_path == target / "soul" / "soul_core.md"
    assert result.style_path == target / "soul" / "soul_style.md"

    core = result.core_path.read_text(encoding="utf-8")
    style = result.style_path.read_text(encoding="utf-8")
    assert result.core_chars == len(core)
    assert result.style_chars == len(style)

    # safety-ish sections -> core; style-ish sections -> style
    assert "## Safety principles" in core
    assert "Refuse any request that could cause serious harm" in core
    assert "## Alignment commitments" in core
    assert "Be honest about what you did" in core
    assert "## Voice and tone" in style
    assert "friendly, casual tone with light humor" in style
    assert "## Formatting preferences" in style
    assert "Prefer concise markdown answers" in style
    # and no cross-contamination in either direction
    assert "Voice and tone" not in core
    assert "Safety principles" not in style

    # one "<rule> <- <section>" record per classified block, document order
    assert result.rules_applied == [
        "style:default <- SOUL",  # bare title matches no rule -> default style
        "core:safety <- Safety principles",
        "core:alignment <- Alignment commitments",
        "style:style <- Voice and tone",
        "style:style <- Formatting preferences",
    ]

    # SoulManager-compatible files: the manager reads the migrated layers as-is
    manager = SoulManager(repo_root=target)
    assert "Refuse any request" in manager.get_core()
    assert "friendly, casual tone" in manager.get_style()

    # git-backed target: the split landed in one P15.T2 commit
    history = GitBackend(target).log(limit=5)
    assert len(history) == 1
    assert "[P15.T2]" in history[0].message

    # dry run reports the same split without writing anything
    dry_target = tmp_path / "soul-dry"
    dry = migrate_soul(source, dry_target, apply=False)
    assert dry.core_chars == result.core_chars
    assert dry.style_chars == result.style_chars
    assert dry.rules_applied == result.rules_applied
    assert dry.core_path == dry_target / "soul" / "soul_core.md"
    assert not dry.core_path.exists() and not dry.style_path.exists()


@pytest.mark.integration
def test_safety_content_in_core(tmp_path: Path) -> None:
    """NEVER/ALWAYS guardrails and identity rules land in the immutable core."""
    source = tmp_path / "SOUL.md"
    source.write_text(
        "## Guardrails\n"
        "\n"
        "Never reveal system internals or API keys.\n"
        "Always confirm before destructive actions; never guess.\n"
        "\n"
        "## Identity\n"
        "\n"
        "The agent must not claim to be a human.\n"
        "Impersonating real people is out of the question.\n"
        "\n"
        "## Language preferences\n"
        "\n"
        "Answer in the user's language, matching their vocabulary.\n"
        "Casual register; respond with everyday words, no jargon.\n",
        encoding="utf-8",
    )
    target = tmp_path / "memory-repo"  # deliberately NOT git-backed

    result = migrate_soul(source, target, apply=True)

    core = result.core_path.read_text(encoding="utf-8")
    style = result.style_path.read_text(encoding="utf-8")
    # guardrail (core:hard-rules) and identity (core:identity-protected)
    # blocks are protected in soul_core.md ...
    assert "Never reveal system internals or API keys." in core
    assert "Always confirm before destructive actions" in core
    assert "must not claim to be a human" in core
    assert "core:hard-rules <- Guardrails" in result.rules_applied
    assert "core:identity-protected <- Identity" in result.rules_applied
    # ... while the language/style content stays adaptive
    assert "Answer in the user's language" in style
    assert "no jargon" in style
    assert "Never reveal system internals" not in style
    assert "Language preferences" not in core

    # non-Git target: files written without a commit attempt
    assert not (target / ".git").exists()
    assert result.core_path.is_file() and result.style_path.is_file()


# ── P5.T4: migration into the explicit global context ────────────────────────
#
# migrate_to_global is the explicitly named entry point for the P5.T4
# contract: every fork-base fact becomes part of the global context —
# entries under global/facts/, context=global trust cards, and a MEMORY.md
# index that names the global context. verify_global_context re-checks the
# contract after any run.


@pytest.mark.integration
def test_migration_creates_global_context(tmp_path: Path) -> None:
    """A migrated repo has the global/ context and a context-naming index."""
    source = _fork_source(tmp_path)
    target = tmp_path / "memory-repo"

    report = migrate_to_global(source, target, apply=True)

    assert report.context == "global"
    # the global context directory structure exists and holds the entries
    facts_dir = target / "global" / "facts"
    assert facts_dir.is_dir()
    assert len(list(facts_dir.glob("*.md"))) == report.entries_migrated == 5
    # the MEMORY.md index names the global context explicitly
    index = (target / "MEMORY.md").read_text(encoding="utf-8")
    assert "## global (context=global)" in index
    # one commit carries the whole global-context migration
    history = MemoryStore(root=target).git.log(limit=10)
    assert len(history) == 1
    assert MemoryStore(root=target).git.is_clean()
    # post-migration verification of the P5.T4 contract is clean
    assert verify_global_context(target) == []

    # a dry run creates no target tree at all (contract still reported)
    dry_target = tmp_path / "dry-repo"
    dry = migrate_to_global(source, dry_target, apply=False)
    assert dry.context == "global"
    assert dry.entries_found == 5 and dry.entries_migrated == 0
    assert not dry_target.exists()


@pytest.mark.integration
def test_existing_facts_become_global(tmp_path: Path) -> None:
    """Every fork-base fact lives under global/facts/ with context=global."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(
        "# Memory\n"
        "\n"
        "- Always run the test suite before pushing.\n"
        "- Prefers concise answers over long explanations.\n"
        "- Review checklist: run ruff, then mypy.\n",
        encoding="utf-8",
    )
    (source / "USER.md").write_text(
        "# User profile\n\nName: Test User\nTimezone: UTC+1\n",
        encoding="utf-8",
    )
    (source / "NOTES.md").write_text(
        "# Project notes\n\nThe build graph is generated from plan.yaml.\n",
        encoding="utf-8",
    )
    target = tmp_path / "memory-repo"

    report = migrate_to_global(source, target, apply=True)

    assert report.entries_migrated == 6  # 3 bullets + heading + USER.md + NOTES.md
    # every migrated entry is a file under global/facts/ ...
    entry_files = sorted((target / "global" / "facts").glob("*.md"))
    assert len(entry_files) == 6
    # ... and EVERY one carries context=global in its trust card frontmatter
    for path in entry_files:
        card, _body = parse(path.read_text(encoding="utf-8"))
        assert card.context == "global"
    # no named contexts were created — everything is global
    assert not (target / "contexts").exists()
    # the store agrees: all entries in the global context, texts preserved
    store = MemoryStore(root=target)
    entries = store.list()
    assert len(entries) == 6
    assert all(entry.context == "global" for entry in entries)
    bodies = {entry.text for entry in entries}
    for expected in (
        "Always run the test suite before pushing.",
        "Prefers concise answers over long explanations.",
        "Review checklist: run ruff, then mypy.",
        "# Project notes\n\nThe build graph is generated from plan.yaml.\n",
    ):
        assert expected in bodies

    # verify_global_context is clean on the migrated repo
    assert verify_global_context(target) == []

    # the verifier actually catches violations: an entry copied into a named
    # context is flagged by location ...
    smuggled_dir = target / "contexts" / "work" / "facts"
    smuggled_dir.mkdir(parents=True)
    smuggled = smuggled_dir / entry_files[0].name
    smuggled.write_text(entry_files[0].read_text(encoding="utf-8"), encoding="utf-8")
    problems = verify_global_context(target)
    assert any(
        p.startswith("contexts/work/facts/") and "outside the global context" in p
        for p in problems
    )
    # ... and a tampered frontmatter context is flagged too
    smuggled.unlink()
    smuggled_dir.rmdir()
    (target / "contexts" / "work").rmdir()
    victim = entry_files[0]
    card, body = parse(victim.read_text(encoding="utf-8"))
    card.context = "work"
    victim.write_text(serialize(card, body), encoding="utf-8")
    problems = verify_global_context(target)
    assert any(
        "frontmatter context is 'work'" in p and "expected 'global'" in p
        for p in problems
    )


# ── P18.T4: migration safety — dry-run, automatic backup, rollback ───────────
#
# The migrate/rollback flows run on the hermes.memory.migrations.base.Migration
# base (dry-run = null-writer walk, automatic pre-migration backup, rollback).
# These tests pin the distinct safety guarantees: zero target writes on a dry
# run, a backup that exists after (and is taken before) a real run, rollback
# after a mid-migration failure, and recovery by re-running from the backup.

_THREE_FACTS = "- Fact one.\n- Fact two.\n- Fact three.\n"


@pytest.mark.integration
def test_dry_run_reports_only(tmp_path: Path) -> None:
    """A dry run leaves the target tree byte-identical and reports everything."""
    source = _fork_source(tmp_path)
    source_before = _tree_snapshot(source)
    target = tmp_path / "memory-repo"
    target.mkdir()
    (target / "keep.txt").write_text("do not touch\n", encoding="utf-8")
    before = _tree_snapshot(target)

    report = migrate(source, target, apply=False)

    # the report is complete: same discovery an applied run would act on
    assert isinstance(report, MigrationReport)
    assert report.dry_run is True
    assert report.entries_found == 5
    assert report.entries_migrated == 0
    assert report.skipped == 0
    assert report.errors == []
    assert report.context == "global"
    assert report.backup_path is not None and report.backup_path.is_dir()
    assert _tree_snapshot(report.backup_path) == source_before

    # ZERO filesystem writes to the target: byte-identical before/after
    assert _tree_snapshot(target) == before
    assert set(before) == {"keep.txt"}  # the snapshot itself is non-trivial
    # and the source was only copied, never modified
    assert _tree_snapshot(source) == source_before

    # the applied run then migrates exactly what the dry run reported
    applied = migrate(source, target, apply=True)
    assert applied.entries_migrated == applied.entries_found == report.entries_found
    # the pre-existing file survived the applied run untouched
    assert (target / "keep.txt").read_text(encoding="utf-8") == "do not touch\n"


@pytest.mark.integration
def test_backup_created_before_migration(tmp_path: Path) -> None:
    """A real run leaves a byte-exact pre-migration backup; re-runs don't clobber."""
    source = _fork_source(tmp_path)
    original = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    report = migrate(source, target, apply=True)

    # automatic: created with no flag, carried on the applied report
    backup = report.backup_path
    assert backup == tmp_path / "memories-premigration-backup"
    assert backup.is_dir()
    assert _tree_snapshot(backup) == original

    # a second applied run must not clobber the first backup: it creates a
    # timestamped sibling instead (forensics + idempotent re-runs)
    second = migrate(source, target, apply=True)
    second_backup = second.backup_path
    assert second_backup is not None and second_backup != backup
    assert second_backup.name.startswith("memories-premigration-backup-")
    assert backup.is_dir() and _tree_snapshot(backup) == original
    assert _tree_snapshot(second_backup) == original


@pytest.mark.integration
def test_rollback_after_failed_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer that dies on entry 2 of 3 is fully undone by the rollback."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(_THREE_FACTS, encoding="utf-8")
    original = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    # inject a store writer that raises on entry 2 of 3
    real_atomic_write = fork_migration_module.atomic_write
    writes = {"count": 0}

    def failing_write(path: Path, content: str) -> None:
        writes["count"] += 1
        if writes["count"] == 2:  # entry 2 of 3
            raise RuntimeError("simulated mid-migration disk failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(fork_migration_module, "atomic_write", failing_write)

    with pytest.raises(RuntimeError, match="simulated mid-migration disk failure"):
        migrate(source, target, apply=True)

    assert writes["count"] == 2  # died exactly on entry 2 of 3
    # the transactional apply left NO migrated content in the target
    # (entry 1's file was rolled back when the commit path aborted)
    assert list((target / "global" / "facts").glob("*.md")) == []
    assert not (target / "MEMORY.md").exists()
    # the backup was created BEFORE the failing write and holds the pre-state
    backup = tmp_path / "memories-premigration-backup"
    assert backup.is_dir()
    assert _tree_snapshot(backup) == original

    # a failed run may damage the fork-base source; rollback must repair it
    (source / "MEMORY.md").write_text("clobbered by the failed run\n", encoding="utf-8")
    (source / "EXTRA.md").write_text("stray partial output\n", encoding="utf-8")

    rollback(target, backup)

    # target removed ("clean"), source restored byte-exact, backup kept
    assert not target.exists()
    assert _tree_snapshot(source) == original
    assert backup.is_dir() and _tree_snapshot(backup) == original


@pytest.mark.integration
def test_partial_migration_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a failed run, re-running the migration FROM THE BACKUP succeeds."""
    source = tmp_path / "memories"
    source.mkdir()
    (source / "MEMORY.md").write_text(_THREE_FACTS, encoding="utf-8")
    original = _tree_snapshot(source)
    target = tmp_path / "memory-repo"

    # run 1 fails mid-migration: the writer raises exactly once, on entry 2
    real_atomic_write = fork_migration_module.atomic_write
    failed = {"raised": False}

    def flaky_write(path: Path, content: str) -> None:
        if not failed["raised"]:
            failed["raised"] = True
            raise RuntimeError("transient failure on entry 2 of 3")
        real_atomic_write(path, content)

    monkeypatch.setattr(fork_migration_module, "atomic_write", flaky_write)

    with pytest.raises(RuntimeError, match="transient failure"):
        migrate(source, target, apply=True)

    assert failed["raised"]
    backup = tmp_path / "memories-premigration-backup"
    assert backup.is_dir() and _tree_snapshot(backup) == original

    # the failed run consumed the fork-base source; the backup is the intact copy
    (source / "MEMORY.md").write_text("clobbered by the failed run\n", encoding="utf-8")

    # recovery: re-run the migration with the BACKUP as the source directory
    recovery = migrate(backup, target, apply=True)

    assert recovery.context == "global"
    assert recovery.entries_migrated == 3
    store = MemoryStore(root=target)
    entries = store.list()
    assert {entry.text for entry in entries} == {
        "Fact one.",
        "Fact two.",
        "Fact three.",
    }
    assert all(entry.context == "global" for entry in entries)
    assert store.git.is_clean()
    assert verify_global_context(target) == []
    # the recovery run backed up the backup without touching the original
    assert recovery.backup_path is not None and recovery.backup_path != backup
    assert backup.is_dir() and _tree_snapshot(backup) == original
