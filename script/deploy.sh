#!/usr/bin/env bash
#
# auto-commit release publisher — uploads the assets built by script/build.sh
# to the GitHub release for the current tag (vX.Y.Z).
#
#   script/deploy.sh
#
# Requires: dist/ populated (run script/build.sh first), HEAD on a v* tag
# (GITHUB_REF_NAME in CI, git describe locally), gh CLI authenticated
# (GH_TOKEN on CI, `gh auth login` locally).
#
# Idempotent: if the release already exists its assets are re-uploaded
# (--clobber), so a failed deploy can simply be re-run.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

cd "$(dirname "$0")/.."
command -v gh >/dev/null 2>&1 || die "gh CLI not found in PATH (brew install gh)"

# --- assets must exist ---
ls dist/auto-commit-*.tar.gz >/dev/null 2>&1 \
  || die "no assets in dist/ — run script/build.sh first"

# --- resolve the release tag ---
TAG="${GITHUB_REF_NAME:-}"
[ -n "$TAG" ] || TAG="$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
case "$TAG" in
  v*) ;;
  "") die "no release tag: checkout a vX.Y.Z tag or set GITHUB_REF_NAME" ;;
  *)  die "tag must look like vX.Y.Z, got '$TAG'" ;;
esac

# --- checksums, generated at deploy time ---
(
  cd dist
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum auto-commit-*.tar.gz > checksums.sha256
  else
    shasum -a 256 auto-commit-*.tar.gz > checksums.sha256
  fi
)

# --- publish (create, or clobber assets on re-run) ---
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "deploy: release ${TAG} exists — re-uploading assets"
  gh release upload "$TAG" dist/auto-commit-*.tar.gz dist/checksums.sha256 --clobber
else
  echo "deploy: creating release ${TAG}"
  gh release create "$TAG" dist/auto-commit-*.tar.gz dist/checksums.sha256 \
    --title "auto-commit ${TAG}" --generate-notes
fi

echo "deploy: ${TAG} published:"
gh release view "$TAG" --json assets --jq '.assets[].name'
