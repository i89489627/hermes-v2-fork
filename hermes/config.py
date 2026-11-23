"""hermes-v2 configuration loading (plan P0.T3).

Loads v2 pipeline settings from ``config.yaml`` in the Hermes home directory.
Upstream Hermes configuration (``hermes_cli``) is untouched: v2 settings live
under the ``v2`` key, so an existing fork-base ``config.yaml`` remains valid
and only v2 keys are added.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml


class ConfigError(ValueError):
    """Raised when config.yaml contains values the v2 pipeline cannot use."""


@dataclass(frozen=True)
class GatingConfig:
    """CBDC gate settings (plan Section 2.5)."""

    budget_claims: int = 3
    budget_chars: int = 1200
    bm25_prefilter_top_k: int = 20
    mode: str = "batched"
    timeout_seconds: float = 3.5


@dataclass(frozen=True)
class RetryConfig:
    """Exponential-backoff retry settings for main-model API calls."""

    max_retries: int = 3
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    on_persistent_failure: str = "skip"  # skip | fail


@dataclass(frozen=True)
class MemoryPipelineConfig:
    """Memory pipeline settings (extraction, gating, retry)."""

    extraction_max_drafts_per_turn: int = 5
    async_after_stop: bool = True
    gating: GatingConfig = field(default_factory=GatingConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass(frozen=True)
class V2Config:
    """Top-level hermes-v2 configuration."""

    hermes_home: Path
    memory_pipeline: MemoryPipelineConfig = field(default_factory=MemoryPipelineConfig)


def hermes_home() -> Path:
    """Hermes home directory — resolved exactly like the fork-base runtime does.

    Delegates to ``hermes_constants.get_hermes_home()`` (context-local override →
    ``HERMES_HOME`` env → platform default), so v2 modules and the fork-base agent
    always agree on a single home. The fork-base default is ``%LOCALAPPDATA%\\hermes``
    on Windows and ``~/.hermes`` elsewhere.

    The earlier v2-only fallback (``~/.hermes`` on every platform) opened a second,
    empty home on Windows whenever ``HERMES_HOME`` was not set: a v2 diagnostics call
    run that way created ``~/.hermes`` next to the real home (found 2026-11-23).
    """
    try:
        from hermes_constants import get_hermes_home
    except ImportError:  # fork-base not importable: keep v2 usable standalone
        env = os.environ.get("HERMES_HOME")
        return Path(env) if env else Path.home() / ".hermes"
    return Path(str(get_hermes_home()))


def _int_value(section: dict[str, Any], key: str, default: int, where: str) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{where}.{key} must be an int, got {raw!r}")
    return raw


def _float_value(section: dict[str, Any], key: str, default: float, where: str) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, got {raw!r}")
    return float(raw)


def _str_value(
    section: dict[str, Any],
    key: str,
    default: str,
    where: str,
    allowed: set[str] | None = None,
) -> str:
    raw = section.get(key, default)
    if not isinstance(raw, str):
        raise ConfigError(f"{where}.{key} must be a string, got {raw!r}")
    if allowed is not None and raw not in allowed:
        raise ConfigError(f"{where}.{key} must be one of {sorted(allowed)}, got {raw!r}")
    return raw


def _bool_value(section: dict[str, Any], key: str, default: bool, where: str) -> bool:
    raw = section.get(key, default)
    if not isinstance(raw, bool):
        raise ConfigError(f"{where}.{key} must be a bool, got {raw!r}")
    return raw


def _gating_from(raw: Any) -> GatingConfig:
    if raw is None:
        return GatingConfig()
    if not isinstance(raw, dict):
        raise ConfigError("v2.memory_pipeline.gating must be a mapping")
    where = "v2.memory_pipeline.gating"
    return GatingConfig(
        budget_claims=_int_value(raw, "budget_claims", 3, where),
        budget_chars=_int_value(raw, "budget_chars", 1200, where),
        bm25_prefilter_top_k=_int_value(raw, "bm25_prefilter_top_k", 20, where),
        mode=_str_value(raw, "mode", "batched", where, allowed={"batched"}),
        timeout_seconds=_float_value(raw, "timeout_seconds", 3.5, where),
    )


def _retry_from(raw: Any) -> RetryConfig:
    if raw is None:
        return RetryConfig()
    if not isinstance(raw, dict):
        raise ConfigError("v2.memory_pipeline.retry must be a mapping")
    where = "v2.memory_pipeline.retry"
    return RetryConfig(
        max_retries=_int_value(raw, "max_retries", 3, where),
        backoff_base_seconds=_float_value(raw, "backoff_base_seconds", 2.0, where),
        backoff_max_seconds=_float_value(raw, "backoff_max_seconds", 60.0, where),
        on_persistent_failure=_str_value(
            raw, "on_persistent_failure", "skip", where, allowed={"skip", "fail"}
        ),
    )


def _pipeline_from(raw: Any) -> MemoryPipelineConfig:
    if raw is None:
        return MemoryPipelineConfig()
    if not isinstance(raw, dict):
        raise ConfigError("v2.memory_pipeline must be a mapping")
    where = "v2.memory_pipeline"
    return MemoryPipelineConfig(
        extraction_max_drafts_per_turn=_int_value(raw, "max_drafts_per_turn", 5, where),
        async_after_stop=_bool_value(raw, "async_after_stop", True, where),
        gating=_gating_from(raw.get("gating")),
        retry=_retry_from(raw.get("retry")),
    )


def load_config(path: Path | None = None) -> V2Config:
    """Load and validate the v2 configuration from ``path`` (default: ``<home>/config.yaml``).

    Missing file -> all defaults. Unknown ``v2`` keys are ignored (forward
    compatibility); invalid values raise :class:`ConfigError`.
    """
    home = hermes_home()
    config_path = path if path is not None else home / "config.yaml"
    v2_section: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        if not isinstance(doc, dict):
            raise ConfigError(f"{config_path}: top level must be a mapping")
        raw = doc.get("v2", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError("config.yaml: 'v2' section must be a mapping")
        v2_section = raw
    return V2Config(
        hermes_home=home,
        memory_pipeline=_pipeline_from(v2_section.get("memory_pipeline")),
    )


# ── Raw config / backup / merge helpers (plan P21.T11, issue #25272) ────────
#
# The upstream update migration overwrote the user's ``config.yaml`` wholesale,
# so every custom model configuration vanished after an update. These helpers
# give the v2 migration layer (``hermes.install.migrations``) the primitives it
# needs to make updates zero-loss: read the file WITHOUT validating it against
# the v2 schema, snapshot it to a timestamped backup before any change, and
# merge two config trees with documented, data-preserving precedence.
#
# ``config_schema_version`` key handling: configs stamped by v2 releases carry
# this top-level key. A difference between the current file's version and the
# incoming release's version is the schema-change signal that triggers the
# user merge prompt (see ``hermes.install.migrations.run_migration``).

#: Top-level config.yaml key carrying the config schema version.
CONFIG_SCHEMA_VERSION_KEY: Final[str] = "config_schema_version"

#: Schema version stamped by the current hermes-v2 release tooling.
CURRENT_CONFIG_SCHEMA_VERSION: Final[int] = 2


def load_raw_config(path: Path) -> dict[str, Any] | None:
    """Load ``path`` as an unvalidated YAML mapping.

    Returns ``None`` when the file does not exist, ``{}`` for an empty file,
    and the parsed mapping otherwise. Unlike :func:`load_config` this performs
    NO v2 schema validation — the migration layer must see the user's config
    exactly as it is on disk. Malformed YAML raises :class:`ConfigError` so
    callers can abort a migration before touching the file.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(doc).__name__}")
    return doc


def backup_config(path: Path) -> Path:
    """Snapshot ``path`` to ``<name>.bak-<UTC timestamp>`` next to it.

    The migration layer calls this before ANY config change, so the user's
    pre-update config is always recoverable even if the update crashes
    halfway (issue #25272). Metadata (mtime) is preserved for forensics.
    Raises :class:`FileNotFoundError` when ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"cannot back up missing config file: {path}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 0
    while target.exists():  # pragma: no cover - microsecond-stamp collision
        counter += 1
        target = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
    shutil.copy2(path, target)
    return target


def deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two config trees into a NEW dict (inputs are not mutated).

    Precedence, documented for the migration layer:

    * ``override`` wins on conflicts — for a key present in both trees the
      override's value replaces the base's, EXCEPT when both values are
      mappings, which merge recursively under the same rule;
    * keys only in ``base`` survive untouched;
    * keys only in ``override`` are added;
    * non-mapping values (scalars, lists, ``None``) are atomic: the winning
      side replaces the other wholesale — a list is never element-merged.

    The migration layer preserves user data by calling it as
    ``deep_merge_config(incoming, current)``: the user's current values win
    scalar conflicts while the incoming release's NEW keys are added — the
    safe default that keeps custom model configurations alive across updates
    (issue #25272).
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge_config(current, value)
        else:
            merged[key] = value
    return merged
