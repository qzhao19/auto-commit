#!/usr/bin/env bash
#
# auto-commit release builder — produces exactly the assets install.sh
# downloads (asset contract in install.sh):
#
#   dist/auto-commit-<target>.tar.gz   (tarball root: a single `auto-commit`)
#
# usage:
#   script/build.sh              # host target only
#   script/build.sh all          # every target this host can build
#   script/build.sh <target>...  # specific targets
#
# targets: aarch64-apple-darwin x86_64-apple-darwin
#          aarch64-unknown-linux-musl x86_64-unknown-linux-musl
#
# toolchain rules:
#   - darwin targets need a macOS host (Apple SDK); from Linux: die, use CI
#   - musl targets always build with `cross` (docker): aws-lc-sys via rustls
#     compiles C for the target and needs a musl gcc, which stock CI runners
#     and most dev machines do not have — not even for the host's own arch
#   - a full 4-asset release comes from a CI matrix (macOS + Linux runners)

set -euo pipefail

ALL="aarch64-apple-darwin x86_64-apple-darwin aarch64-unknown-linux-musl x86_64-unknown-linux-musl"

die() { echo "error: $*" >&2; exit 1; }

# run from the repo root, wherever the script is invoked from
cd "$(dirname "$0")/.."
command -v cargo >/dev/null 2>&1 || die "cargo not found in PATH"

# --- host platform (mirrors install.sh's matrix) ---
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}:${ARCH}" in
  Darwin:arm64)  HOST_TARGET="aarch64-apple-darwin" ;;
  Darwin:x86_64) HOST_TARGET="x86_64-apple-darwin" ;;
  Linux:aarch64) HOST_TARGET="aarch64-unknown-linux-musl" ;;
  Linux:x86_64)  HOST_TARGET="x86_64-unknown-linux-musl" ;;
  *) die "unsupported host: ${OS} ${ARCH} (supported: macOS/Linux on arm64/x86_64)" ;;
esac

# --- target selection ---
WANT=()
for arg in "$@"; do
  case "$arg" in
    all) WANT=($ALL) ;;
    *) case " $ALL " in
         *" $arg "*) WANT+=("$arg") ;;
         *) die "unknown target '$arg' (valid: $ALL)" ;;
       esac ;;
  esac
done
[ ${#WANT[@]} -gt 0 ] || WANT=("$HOST_TARGET")

# --- build & package ---
rm -rf dist && mkdir -p dist
for target in "${WANT[@]}"; do
  rustup target add "$target"

  runner="cargo"
  case "$target" in
    *-apple-darwin)
      [[ "$OS" == Darwin ]] || die "$target requires macOS — build it in a CI matrix" ;;
    *-unknown-linux-musl)
      # always via cross, on any host: its docker image bundles the target
      # musl gcc that aws-lc-sys needs.
      command -v cross >/dev/null 2>&1 \
        || die "$target needs 'cross' (cargo install cross)"
      docker info >/dev/null 2>&1 || die "cross needs a running docker daemon"
      runner="cross"
      ;;
  esac

  echo "build:  ${runner} build --release --target ${target}"
  "$runner" build --release --target "$target"

  bin="target/${target}/release/auto-commit"
  [ -s "$bin" ] || die "missing or empty binary: ${bin}"
  tar -czf "dist/auto-commit-${target}.tar.gz" -C "target/${target}/release" auto-commit
  echo "asset:  dist/auto-commit-${target}.tar.gz"
done

# --- summary ---
ls -lh dist/
echo
echo "publish: gh release create vX.Y.Z dist/auto-commit-*.tar.gz"
echo "verify:  shasum -a 256 dist/*.tar.gz   (Linux: sha256sum)"
