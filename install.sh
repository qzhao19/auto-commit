#!/usr/bin/env bash
#
# auto-commit installer:
#   curl -fsSL https://github.com/qzhao19/auto-commit/raw/main/install | bash
#
# Idempotent — re-running upgrades the binary while preserving user config.
#   1. resolve the install dir exactly like the tool resolves its config:
#      $XDG_CONFIG_HOME/autocommit (absolute XDG) else $HOME/.config/autocommit
#   2. create config.toml from config.example.toml (never overwrites)
#   3. download the latest release binary for this platform
#   4. bind `git auto-commit` via a global git alias
#
# Release asset contract (CI must publish, same names every release, each
# tarball containing a single `auto-commit` executable):
#   auto-commit-<target>.tar.gz, target ∈
#     aarch64-apple-darwin | x86_64-apple-darwin |
#     aarch64-unknown-linux-musl | x86_64-unknown-linux-musl
#
# Overrides: AUTOCOMMIT_VERSION=vX.Y.Z pins a specific release tag.

set -euo pipefail

REPO="qzhao19/auto-commit"
BRANCH="main" # branch providing config.example.toml

die() { echo "error: $*" >&2; exit 1; }

# --- required tools 
for tool in curl tar git; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required but not found in PATH"
done
[ -n "${HOME:-}" ] || die "HOME is not set; cannot resolve the install directory"

# --- install dir: mirror the tool's config resolution ---
# The tool reads $XDG_CONFIG_HOME/autocommit/config.toml when XDG_CONFIG_HOME
# is set AND absolute, otherwise $HOME/.config/autocommit/config.toml
# (src/shared/util/loader.rs, resolve_default_config_path).
CONFIG_HOME="${HOME}/.config"
case "${XDG_CONFIG_HOME:-}" in
  /*) CONFIG_HOME="${XDG_CONFIG_HOME%/}" ;;
esac
CONFIG_DIR="${CONFIG_HOME%/}/autocommit"
CONFIG_PATH="${CONFIG_DIR}/config.toml"
BIN_PATH="${CONFIG_DIR}/auto-commit"

# --- platform ---
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}:${ARCH}" in
  Darwin:arm64)  TARGET="aarch64-apple-darwin" ;;
  Darwin:x86_64) TARGET="x86_64-apple-darwin" ;;
  Linux:aarch64) TARGET="aarch64-unknown-linux-musl" ;;
  Linux:x86_64)  TARGET="x86_64-unknown-linux-musl" ;;
  *) die "unsupported platform: ${OS} ${ARCH} (covered: macOS/Linux on arm64/x86_64)" ;;
esac

if [ -n "${AUTOCOMMIT_VERSION:-}" ]; then
  RELEASE_BASE="https://github.com/${REPO}/releases/download/${AUTOCOMMIT_VERSION}"
else
  RELEASE_BASE="https://github.com/${REPO}/releases/latest/download"
fi

mkdir -p "$CONFIG_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- 1. config.toml from template (never clobber an existing config) ---
if [ -f "$CONFIG_PATH" ]; then
  echo "config: ${CONFIG_PATH} exists — keeping it as-is"
else
  echo "config: creating ${CONFIG_PATH} from config.example.toml"
  curl -fsSL "https://raw.githubusercontent.com/${REPO}/${BRANCH}/config.example.toml" \
    -o "$CONFIG_PATH" || die "failed to download config.example.toml"
fi

# --- 2. release binary ---
ARCHIVE="auto-commit-${TARGET}.tar.gz"
echo "binary: downloading ${ARCHIVE}"
curl -fSL "${RELEASE_BASE}/${ARCHIVE}" -o "${TMP}/${ARCHIVE}" \
  || die "failed to download ${ARCHIVE} (is it published under this release?)"
tar -xzf "${TMP}/${ARCHIVE}" -C "$TMP" || die "failed to extract ${ARCHIVE}"
[ -f "${TMP}/auto-commit" ] \
  || die "archive does not contain an 'auto-commit' executable at its root"
install -m 0755 "${TMP}/auto-commit" "$BIN_PATH" \
  || die "failed to install the binary to ${BIN_PATH}"
echo "binary: installed ${BIN_PATH}"

# --- 3. git alias ---
# Absolute path (not ~) keeps the alias deterministic. Git runs '!' aliases
# with cwd = repo top-level, which the tool handles by design.
git config --global alias.auto-commit "!${BIN_PATH}" \
  || die "failed to set the global git alias"
echo "alias:   git auto-commit -> ${BIN_PATH}"

# --- summary ---
echo
echo "auto-commit installed."
echo "  config : ${CONFIG_PATH}"
echo "  binary : ${BIN_PATH}"
echo
echo "Next steps:"
echo "  1. export AUTOCOMMIT_LLM_API_KEY=sk-...   (or edit the config file)"
echo "  2. git add <file> && git auto-commit"
echo
echo "Upgrade:   re-run this installer (config is preserved)"
echo "Uninstall: rm -rf '${CONFIG_DIR}' && git config --global --unset alias.auto-commit"
