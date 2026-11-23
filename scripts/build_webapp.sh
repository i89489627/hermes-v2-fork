#!/usr/bin/env bash
# Build the webapp renderer bundle under an explicit Node.js heap cap
# (plan P21.T9, issue #156161).
#
# Upstream bug: the renderer build ran with no heap cap, so the build stage
# was OOM-killed while the update exited 0 (success) and the bundle stayed
# silently stale. This script makes every failure explicit:
#
#   * runs the renderer build as
#       node --max-old-space-size=$NODE_HEAP_MB <build-entry> ...
#     (NODE_HEAP_MB, default 4096);
#   * treats exit code 134 (SIGABRT/OOM), 137 (SIGKILL/OOM) and a
#     "JavaScript heap out of memory" marker on stderr as OOM: prints an
#     explicit error and exits 42 (OOM_EXIT_CODE);
#   * before declaring success, validates that the output bundle exists AND
#     has an mtime newer than the previous build (stale-bundle guard);
#   * when invoked by hermes.install.updater.Updater, packages the built
#     bundle into the path published in $HERMES_BUNDLE_STAGING.
#
# Exit codes: 0 success | 42 OOM | 43 bundle missing | 44 stale bundle |
#             45 setup error (node/entry/tar unavailable).
# Requires bash (process substitution); positional arguments are ignored —
# the heap cap comes from $NODE_HEAP_MB (exported by the v2 Updater).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WEB_DIR="${WEBAPP_WEB_DIR:-$REPO_ROOT/web}"
BUILD_ENTRY="${WEBAPP_BUILD_ENTRY:-$WEB_DIR/node_modules/vite/bin/vite.js}"
OUT_DIR="${WEBAPP_OUT_DIR:-$REPO_ROOT/hermes_cli/web_dist}"
NODE_HEAP_MB="${NODE_HEAP_MB:-4096}"
NODE_BIN="${NODE_BIN:-node}"
STAGING="${HERMES_BUNDLE_STAGING:-}"

OOM_EXIT_CODE=42
MISSING_BUNDLE_EXIT_CODE=43
STALE_BUNDLE_EXIT_CODE=44
SETUP_EXIT_CODE=45

log_error() {
  echo "ERROR(build_webapp): $*" >&2
}

# mtime as a decimal number (nanosecond precision where coreutils supports
# it, seconds otherwise; BSD stat fallback). 0 when the path is missing.
mtime_of() {
  local value=""
  value="$(stat -c %.9Y "$1" 2>/dev/null)" && [ -n "$value" ] && { echo "$value"; return 0; }
  value="$(stat -c %Y "$1" 2>/dev/null)" && [ -n "$value" ] && { echo "$value"; return 0; }
  value="$(stat -f %m "$1" 2>/dev/null)" && [ -n "$value" ] && { echo "$value"; return 0; }
  echo 0
}

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  log_error "Node.js executable '$NODE_BIN' not found on PATH; install Node.js or set NODE_BIN."
  exit "$SETUP_EXIT_CODE"
fi
if [ ! -f "$BUILD_ENTRY" ]; then
  log_error "renderer build entry $BUILD_ENTRY does not exist; run 'npm install' in $WEB_DIR or set WEBAPP_BUILD_ENTRY."
  exit "$SETUP_EXIT_CODE"
fi
if ! cd "$WEB_DIR"; then
  log_error "cannot enter webapp source directory $WEB_DIR."
  exit "$SETUP_EXIT_CODE"
fi

PRE_MTIME="$(mtime_of "$OUT_DIR")"
STDERR_FILE="$(mktemp "${TMPDIR:-/tmp}/webapp-build-stderr.XXXXXX")"

echo "Building webapp renderer (heap cap: ${NODE_HEAP_MB} MB, entry: $BUILD_ENTRY, out: $OUT_DIR) ..."
"$NODE_BIN" --max-old-space-size="$NODE_HEAP_MB" "$BUILD_ENTRY" build \
  --outDir "$OUT_DIR" --emptyOutDir 2> >(tee "$STDERR_FILE" >&2)
BUILD_RC=$?

if [ "$BUILD_RC" -eq 134 ] || [ "$BUILD_RC" -eq 137 ] || grep -qi "heap out of memory" "$STDERR_FILE" 2>/dev/null; then
  log_error "renderer build was OOM-killed (exit code $BUILD_RC)."
  log_error "The Node.js heap cap --max-old-space-size=${NODE_HEAP_MB} (NODE_HEAP_MB) was exhausted."
  log_error "Retry with a larger cap, e.g.: NODE_HEAP_MB=$((NODE_HEAP_MB * 2)) $SCRIPT_DIR/build_webapp.sh"
  log_error "No bundle was installed; the previously built bundle is untouched."
  rm -f "$STDERR_FILE"
  exit "$OOM_EXIT_CODE"
fi
if [ "$BUILD_RC" -ne 0 ]; then
  log_error "renderer build failed with exit code $BUILD_RC; captured stderr follows:"
  cat "$STDERR_FILE" >&2 || true
  rm -f "$STDERR_FILE"
  exit "$BUILD_RC"
fi
rm -f "$STDERR_FILE"

# ── stale-bundle guard: success requires a fresh, existing bundle ──────────
if [ ! -d "$OUT_DIR" ] || [ -z "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  log_error "build exited 0 but produced no bundle at $OUT_DIR; refusing to declare success."
  exit "$MISSING_BUNDLE_EXIT_CODE"
fi
POST_MTIME="$(mtime_of "$OUT_DIR")"
if ! awk -v post="$POST_MTIME" -v pre="$PRE_MTIME" 'BEGIN { exit !(post > pre) }'; then
  log_error "stale bundle: $OUT_DIR was not modified by this build (mtime $POST_MTIME <= previous $PRE_MTIME); refusing to declare success."
  exit "$STALE_BUNDLE_EXIT_CODE"
fi

# ── Updater contract: package the built bundle for atomic swap-in ──────────
if [ -n "$STAGING" ]; then
  if ! tar -czf "$STAGING" -C "$OUT_DIR" .; then
    log_error "failed to package bundle $OUT_DIR into staging path $STAGING."
    exit "$SETUP_EXIT_CODE"
  fi
  if [ ! -s "$STAGING" ]; then
    log_error "packaged bundle at $STAGING is empty; refusing to declare success."
    exit "$MISSING_BUNDLE_EXIT_CODE"
  fi
fi

echo "webapp renderer bundle OK: $OUT_DIR (mtime $POST_MTIME, heap cap ${NODE_HEAP_MB} MB)"
exit 0
