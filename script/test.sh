#!/usr/bin/env bash
#
# auto-commit test runner — the single entry point shared by CI and local dev.
#
#   script/test.sh
#
# Runs unit + integration + e2e in one shot. The four live-LLM e2e cases are
# #[ignore]d and only run when AUTOCOMMIT_E2E_API_KEY is set (CI secret or a
# local export) — otherwise they silently skip and the suite still passes.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

cd "$(dirname "$0")/.."
command -v cargo >/dev/null 2>&1 || die "cargo not found in PATH"

echo "test:   cargo test (unit + integration + e2e, free scenarios)"
cargo test

if [ -n "${AUTOCOMMIT_E2E_API_KEY:-}" ]; then
  echo "test:   cargo test --test e2e -- --ignored (live LLM)"
  # cargo test --test e2e -- --ignored
  cargo test --test e2e -- --ignored
else
  echo "skip:   live-LLM e2e (AUTOCOMMIT_E2E_API_KEY not set)"
fi
