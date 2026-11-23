"""Zero-loss config migrations for hermes-v2 updates (plan P21.T11, issue #25272).

Upstream bug: the update migration overwrote the user's ``config.yaml`` with
the new release's template, so every custom model configuration vanished.
The v2 :func:`run_migration` closes that hole with four guarantees:

1. the current file is ALWAYS backed up first (``config.yaml.bak-<UTC ts>``),
   before any possible change, even on a dry run;
2. the ``config_schema_version`` drift and the issue-#25272 signature — the
   incoming config would DROP keys present in the current file — trigger a
   user prompt through the injected ``on_schema_change`` callback (no hard
   dependency on an interactive UI);
3. the default decision (and the ``"merge"`` decision) merges with
   **current values winning** scalar conflicts and incoming NEW keys added,
   so custom models survive every path that does not explicitly overwrite;
4. results are written atomically, and a dry run (``apply=False``) writes
   nothing.

``run_migration`` never raises on user-config problems: an unreadable current
config aborts with status ``"aborted"`` and nothing on disk is touched.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from hermes.config import (
    CONFIG_SCHEMA_VERSION_KEY,
    ConfigError,
    backup_config,
    deep_merge_config,
    load_raw_config,
)

__all__ = [
    "DECISION_KEEP_CURRENT",
    "DECISION_MERGE",
    "DECISION_OVERWRITE",
    "STATUS_ABORTED",
    "STATUS_CLEAN",
    "STATUS_MERGED",
    "STATUS_USER_DECIDED",
    "MigrationResult",
    "run_migration",
]

_LOGGER = logging.getLogger("hermes.install.migrations")

#: Migration statuses (see :class:`MigrationResult`).
STATUS_CLEAN: Final[str] = "clean"
STATUS_MERGED: Final[str] = "merged"
STATUS_USER_DECIDED: Final[str] = "user_decided"
STATUS_ABORTED: Final[str] = "aborted"

#: Decisions the ``on_schema_change`` callback may return.
DECISION_MERGE: Final[str] = "merge"
DECISION_KEEP_CURRENT: Final[str] = "keep-current"
DECISION_OVERWRITE: Final[str] = "overwrite"

#: Cap on dropped-key paths listed in the prompt description.
_MAX_DROPPED_SHOWN: Final[int] = 10


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of one :func:`run_migration` call.

    ``status`` is one of:

    * ``"clean"`` — no schema change detected; the incoming config was
      applied with current values winning scalar conflicts (never loses
      user data);
    * ``"merged"`` — schema change detected, no callback configured, the
      safe default merge was applied;
    * ``"user_decided"`` — schema change detected and the callback's
      decision (``"merge"``/``"keep-current"``/``"overwrite"``) applied;
    * ``"aborted"`` — the current config is unreadable; nothing was written.

    ``backup_path`` is the timestamped backup of the pre-migration file
    (``None`` only when there was no file to back up or the migration
    aborted before any change). ``merged_config`` is what the config file
    contains after the migration (``{}`` when aborted; the current config
    when the user chose ``"keep-current"``).
    """

    status: str
    backup_path: Path | None
    merged_config: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def run_migration(
    current_path: Path,
    incoming: dict[str, Any],
    *,
    on_schema_change: Callable[[str], str] | None = None,
    apply: bool = True,
) -> MigrationResult:
    """Migrate ``current_path`` to the ``incoming`` release config, zero-loss.

    Steps: read the current file (abort if unreadable), back it up ALWAYS,
    detect schema changes, resolve the decision (callback or safe default),
    then atomically write the result when ``apply=True``. ``apply=False``
    is a dry run: everything except the final write happens, including the
    backup. ``on_schema_change`` receives a description of the change and
    returns one of :data:`DECISION_MERGE`, :data:`DECISION_KEEP_CURRENT`,
    :data:`DECISION_OVERWRITE`; without a callback the safe default is
    ``"merge"``. The incoming ``config_schema_version`` is stamped onto the
    merged result so the next update compares against the new baseline.
    """
    if not isinstance(incoming, dict):
        raise TypeError("incoming config must be a dict")

    # 1) Read the current file BEFORE anything on disk is touched.
    current: dict[str, Any] = {}
    if current_path.exists():
        try:
            loaded = load_raw_config(current_path)
        except ConfigError as exc:
            _LOGGER.error("config migration aborted: %s", exc)
            return MigrationResult(
                STATUS_ABORTED,
                None,
                {},
                [f"aborted: current config unreadable, nothing written ({exc})"],
            )
        if loaded:
            current = loaded

    # 2) Backup ALWAYS, before any config change (issue #25272).
    backup_path: Path | None = None
    if current_path.exists():
        backup_path = backup_config(current_path)
        _LOGGER.info("config backup written to %s", backup_path)

    # 3) Detect schema change: version drift or the #25272 drop signature.
    current_version = current.get(CONFIG_SCHEMA_VERSION_KEY)
    incoming_version = incoming.get(CONFIG_SCHEMA_VERSION_KEY)
    dropped = _dropped_key_paths(current, incoming)
    schema_change = bool(current) and (
        current_version != incoming_version or bool(dropped)
    )

    decision: str | None = None
    if schema_change:
        description = _describe_schema_change(
            current_version, incoming_version, dropped
        )
        _LOGGER.warning("%s", description)
        if on_schema_change is None:
            decision = DECISION_MERGE
            status = STATUS_MERGED
            notes = [f"schema change detected; safe default merge applied ({description})"]
        else:
            decision = on_schema_change(description)
            if decision not in (DECISION_MERGE, DECISION_KEEP_CURRENT, DECISION_OVERWRITE):
                raise ValueError(
                    f"on_schema_change must return one of "
                    f"('{DECISION_MERGE}', '{DECISION_KEEP_CURRENT}', "
                    f"'{DECISION_OVERWRITE}'), got {decision!r}"
                )
            status = STATUS_USER_DECIDED
            notes = [f"schema change detected; user decided: {decision}"]
    else:
        status = STATUS_CLEAN
        notes = ["no schema change detected"]
        if not current:
            notes.append("no existing user config to preserve")

    # 4) Build the merged config. Every non-overwrite path keeps the user's
    #    scalar values (current wins) and adds the release's new keys.
    if decision == DECISION_OVERWRITE:
        merged = dict(incoming)
        notes.append(
            "user chose overwrite: incoming replaces current entirely; "
            f"pre-migration copy at {backup_path}"
        )
    elif decision == DECISION_KEEP_CURRENT:
        merged = dict(current)
    else:
        merged = deep_merge_config(incoming, current)

    # 5) Stamp the incoming schema version so the next update compares
    #    against the new baseline (never against the user's old value).
    if decision != DECISION_KEEP_CURRENT and CONFIG_SCHEMA_VERSION_KEY in incoming:
        merged[CONFIG_SCHEMA_VERSION_KEY] = incoming[CONFIG_SCHEMA_VERSION_KEY]

    # 6) Write atomically — except a dry run writes nothing, and
    #    keep-current leaves the user's file byte-for-byte alone.
    if not apply:
        notes.append("dry run: nothing written")
    elif decision == DECISION_KEEP_CURRENT:
        notes.append("keep-current: file left untouched")
    else:
        _atomic_write_yaml(current_path, merged)
        notes.append(f"config written to {current_path}")

    _LOGGER.info("config migration status=%s path=%s", status, current_path)
    return MigrationResult(status, backup_path, merged, notes)


def _dropped_key_paths(
    current: dict[str, Any], incoming: dict[str, Any], prefix: str = ""
) -> list[str]:
    """Dotted paths of keys present in ``current`` that ``incoming`` lacks.

    Walks recursively while both sides agree the node is a mapping; the
    first non-mapping divergence is treated as a replace, not a drop (the
    current-wins merge keeps the user's subtree there anyway). The schema
    version key is exempt — it is expected to drift between releases.
    """
    dropped: list[str] = []
    for key, value in current.items():
        if key == CONFIG_SCHEMA_VERSION_KEY:
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in incoming:
            dropped.append(path)
        elif isinstance(value, dict) and isinstance(incoming[key], dict):
            dropped.extend(_dropped_key_paths(value, incoming[key], path))
    return dropped


def _describe_schema_change(
    current_version: Any, incoming_version: Any, dropped: list[str]
) -> str:
    """Human-readable prompt description naming the version drift and keys."""
    parts: list[str] = []
    if current_version != incoming_version:
        parts.append(
            f"config schema version change {current_version!r} -> {incoming_version!r}"
        )
    if dropped:
        shown = ", ".join(dropped[:_MAX_DROPPED_SHOWN])
        hidden = len(dropped) - _MAX_DROPPED_SHOWN
        extra = f" (+{hidden} more)" if hidden > 0 else ""
        parts.append(
            f"the incoming update would drop {len(dropped)} existing key(s): {shown}{extra}"
        )
    return "config schema change detected: " + "; ".join(parts)


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as YAML via temp-file + ``os.replace`` (atomic on path)."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
