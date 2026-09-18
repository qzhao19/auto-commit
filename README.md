# auto-commit

**AI-powered Git commit message generator** written in Rust.

[![test](https://github.com/qzhao19/auto-commit/actions/workflows/release.yml/badge.svg?job=test)](https://github.com/qzhao19/auto-commit/actions/workflows/release.yml)
[![release](https://github.com/qzhao19/auto-commit/actions/workflows/release.yml/badge.svg)](https://github.com/qzhao19/auto-commit/actions/workflows/release.yml)


## Overview

`auto-commit` inspects your **staged** changes, builds a structured prompt from the repository state, calls an LLM (OpenAI-compatible API or local Ollama), validates the result against Conventional Commits rules, and lets you accept or regenerate the message interactively before committing.

## Features

- **Automatic analysis** — Scans staged changes through a safe Git pipeline and produces a Conventional Commits message: `<type>[optional scope]: <description>` (optional body).
- **Seamless Git integration** — `git auto-commit` fits directly into your existing workflow: stage, run, review, commit. No plugins required.
- **Interactive review** — Generate and refine commit messages in the terminal:
  - `Tab` — accept and commit
  - `Enter` — reject and regenerate
  - `←` / `→` — switch among previous candidates
- **Effortless configuration** — One TOML file plus a few environment variables, with sensible defaults. Override generation parameters on the fly, e.g. `git auto-commit --temperature 0.5`.
- **Multi-provider LLM** — Works with any OpenAI-compatible API (OpenAI, DeepSeek, …) and a local [Ollama](https://ollama.com/) instance. Claude and other proprietary APIs are not supported yet.

## Requirements

- [Rust](https://rustup.rs/) (edition 2024 toolchain)
- Git available on `PATH`
- An LLM backend:
  - An OpenAI-compatible API key, **or**
  - A local [Ollama](https://ollama.com/) service

## Installation

> **System support**: Currently macOS and Linux only. Windows contributions are welcome.

### Automated Installation

Install the latest release:

```bash
curl -fsSL https://github.com/qzhao19/auto-commit/raw/main/install | bash
```

To pin a specific version, set `AUTOCOMMIT_VERSION`:

```bash
curl -fsSL https://github.com/qzhao19/auto-commit/raw/main/install | AUTOCOMMIT_VERSION=v0.1.0 bash
```

### Manual Installation

Build from source for full control. A Rust toolchain is required.

```bash
git clone https://github.com/qzhao19/auto-commit.git
cd auto-commit

# 1. Build the release binary
cargo build --release

# 2. Install binary and example config
#    ($XDG_CONFIG_HOME/autocommit when set to an absolute path,
#     otherwise ~/.config/autocommit)
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
mkdir -p "$CONFIG_HOME/autocommit"
cp target/release/auto-commit "$CONFIG_HOME/autocommit/"
cp config.example.toml "$CONFIG_HOME/autocommit/config.toml"   # optional

# 3. Register the Git alias
git config --global alias.auto-commit "!$CONFIG_HOME/autocommit/auto-commit"
```

Notes:

- The config file is optional. When no `config.toml` is present, configuration is taken entirely from environment variables (and CLI flags). For OpenAI-compatible providers you must at least supply `AUTOCOMMIT_LLM_MODEL`, `AUTOCOMMIT_LLM_API_KEY`, and `AUTOCOMMIT_LLM_BASE_URL`.

- `git auto-commit` is the recommended entry point; the binary itself is not added to `PATH`.

- Uninstall:

  ```bash
  rm -rf "$CONFIG_HOME/autocommit"
  git config --global --unset alias.auto-commit
  ```

### Set the API key

For any OpenAI-compatible provider, `auto-commit` always reads the API key from the `AUTOCOMMIT_LLM_API_KEY` environment variable — even if a `config.toml` is present. Putting a real key in the TOML file will trigger a warning on every run, so the shell startup file is the proper place for it:

```bash
echo 'export AUTOCOMMIT_LLM_API_KEY="sk-xxx"' >> ~/.zshrc
source ~/.zshrc          # zsh (macOS default); use ~/.bashrc on bash
```

Ollama does not require a key — skip this step when using a local Ollama instance.

## Quick Start

```bash
# 1. Stage the changes you want to commit
git add path/to/file

# 2. Generate a commit message and review it
git auto-commit

# Interactive keys:
#   Tab     — accept and run git commit
#   Enter   — reject and regenerate
#   ← / →   — switch among previously generated candidates
#   Other   — exit without committing
```

Minimal environment-only setup (no `config.toml` needed):

```bash
export AUTOCOMMIT_LLM_PROVIDER=openai
export AUTOCOMMIT_LLM_API_KEY=xxx
export AUTOCOMMIT_LLM_BASE_URL=https://api.deepseek.com
export AUTOCOMMIT_LLM_MODEL=deepseek-v4-flash
```

## Configuration Reference

Full example: [`config.example.toml`](config.example.toml).

**Configuration precedence (low → high):**  
built-in defaults → TOML config file → `AUTOCOMMIT_*` environment variables → CLI flags.

No layer is required. Each layer only overrides the keys it actually sets, so you can run entirely on defaults (or entirely on environment variables).

> Generation parameters (`temperature`, `maxTokens`, …) are **intentionally not** readable from environment variables. They can only be overridden via CLI flags.

### Provider & Model

| Key        | Description                                          | Env Override              |
| ---------- | ---------------------------------------------------- | ------------------------- |
| `provider` | `"openai"` (OpenAI-compatible) or `"ollama"` (local) | `AUTOCOMMIT_LLM_PROVIDER` |
| `model`    | Model identifier sent to the provider (required)     | `AUTOCOMMIT_LLM_MODEL`    |
| `baseUrl`  | Custom API base URL (omit for provider default)      | `AUTOCOMMIT_LLM_BASE_URL` |
| `apiKey`   | Required for `openai`; leave unset for `ollama`      | `AUTOCOMMIT_LLM_API_KEY`  |

**Notes**


- The API key is read from the `AUTOCOMMIT_LLM_API_KEY` env var — required for `OpenAI-compatible API` with or without a config file (see [Set the API key](#set-the-api-key)). A real key in the TOML file will trigger a warning on every run.
- `baseUrl` behaviour differs by provider:
  - `openai` — the client appends `/chat/completions`. For any endpoint other than the official OpenAI API this value is effectively required.
  - `ollama` — use the plain origin only (e.g. `http://localhost:11434`)

### Generation Parameters (optional)

These are defined under the same `[llm]` table (camelCase in TOML).

| Key                | Range / Default             | Description                  |
| ------------------ | --------------------------- | ---------------------------- |
| `temperature`      | `[0.0, 2.0]`, default `0.8` | Sampling temperature         |
| `maxTokens`        | `> 0`, default `4096`       | Hard cap on generated tokens |
| `topP`             | `(0.0, 1.0]`, default `0.9` | Nucleus sampling             |
| `frequencyPenalty` | `[0.0, 2.0]`, default `0`   | Repetition suppression       |
| `presencePenalty`  | `[0.0, 2.0]`, default `0`   | Topic diversity              |

**CLI equivalents** (kebab-case):

```bash
--temperature <f32>
--max-tokens <i32>
--top-p <f32>
--frequency-penalty <f32>
--presence-penalty <f32>
```

### Resilience (optional)

See [`config.example.toml`](config.example.toml) for the full `[resilience.retry]` and `[resilience.timeout]` sections. Only transient failures (network errors, HTTP 429/5xx, timeouts) consume retries; deterministic errors fail immediately.

## Architecture

### Layout

The binary is a thin orchestrator over three layers:

- `core/` — domain pipeline (Git state → diff budget → prompt → LLM → validation → interactive loop)
- `infra/` — side effects (Git subprocess runner, config loading, terminal UI)
- `shared/` — cross-layer contracts (config types, errors, UI traits)

```text
src/
  main.rs                 # entry: config → pipeline → LLM → UI/commit
  core/
    git/                  # preflight, operation state, staged, diff, types
    pipe/                 # orchestrator, assembler, validator, context
    llm/                  # client + openai / ollama providers
    inter/                # interactive candidate loop
  infra/                  # GitRunner, config loader, terminal UI
  shared/                 # errors, shared UI traits, config types
docs/                     # pipeline & stage design notes
tests/                    # unit, integration, e2e
config.example.toml
```

Design notes are defined in [`docs/`](docs/):

- [`pipeline.md`](docs/pipeline.md) — end-to-end pipeline
- [`stage0-repository-preflight.md`](docs/stage0-repository-preflight.md) — Stage 0 (preflight)
- [`stage1-git-operation-state.md`](docs/stage1-git-operation-state.md) — Stage 1 (operation-state detection)

### Message Rules

Every candidate must pass validation before it can be committed:

- **Header** — `<type>[optional scope]: <description>`  
  `type` ∈ `feat | fix | docs | style | refactor | perf | test | chore | build | ci`  
  scope is short lowercase alphanumerics
- **Description** — non-empty, starts with a lowercase letter, ≤ 72 characters, no trailing period
- **Body** (optional) — separated from the header by exactly one blank line
- Markdown fences, quote wrappers and extraneous chatter are soft-stripped when safe. Invalid output is rejected so the interactive loop can regenerate.

## Testing

```bash
cargo build

# Run all tests
cargo test

# Run only integration tests
cargo test -- integration

# Run only unit tests
cargo test -- unittest

# End-to-end tests (free scenarios only — no real LLM calls).
# The four cases that require a real provider are marked #[ignore]
# and are skipped by default.
cargo test --test e2e

# End-to-end tests including the four cases that call a real provider.
# Arguments after `--` are forwarded to the test binary.
# Without a key the cases print "skipped" and still pass.
AUTOCOMMIT_E2E_API_KEY='xxx' cargo test --test e2e -- --ignored
# Optional:
#   AUTOCOMMIT_E2E_MODEL     (default: deepseek-v4-flash)
#   AUTOCOMMIT_E2E_BASE_URL  (default: provider default endpoint)

# Optimized release build (LTO, strip, opt-level = "z")
cargo build --release
```

Test layout:

- `tests/e2e/` — black-box end-to-end tests of the CLI binary
- `tests/integration/` — config loading, Git pipeline, LLM client, prompt assembler
- `tests/unittest/` — unit tests for core modules

## License

See the repository for license information (add a `LICENSE` file if you publish one).

## Acknowledgments

Built with Tokio, clap, crossterm, async-openai, ollama-rs, and the Conventional Commits specification.
