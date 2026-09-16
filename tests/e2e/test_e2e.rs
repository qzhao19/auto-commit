//! Black-box end-to-end tests for the `auto-commit` CLI binary.
//!
//! The binary is spawned as a real subprocess inside a temp git repository.
//! Assertions use ONLY the process exit code, stdout, stderr, and the
//! resulting git state — no production code is imported, no in-process
//! driver is used. Every scenario mirrors a real usage situation.
//!
//! # Running
//!
//! Zero-cost scenarios (no network, no credentials, always on):
//!
//! ```sh
//! cargo test --test e2e
//! ```
//!
//! Live-LLM scenarios (real provider, real commit messages). They are
//! `#[ignore]`d and additionally early-return when the key is absent,
//! so developers without credentials never trigger them:
//!
//! ```sh
//! AUTOCOMMIT_E2E_API_KEY=... cargo test --test e2e -- --ignored
//! # optional: AUTOCOMMIT_E2E_MODEL (default gpt-4o-mini)
//! #          AUTOCOMMIT_E2E_BASE_URL (default: provider default endpoint)
//! ```

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// const BIN: &str = env!("CARGO_BIN_EXE_auto-commit");
fn binary_path() -> &'static str {
    option_env!("CARGO_BIN_EXE_auto-commit")
        .expect("CARGO_BIN_EXE_auto-commit not set; run via `cargo test --test e2e`")
}

const SHORT_DEADLINE: Duration = Duration::from_secs(10);
const LIVE_DEADLINE: Duration = Duration::from_secs(120);

/// Proxy vars that must not intercept loopback traffic in mock scenarios.
const PROXY_ENV_VARS: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
];

// ---------------------------------------------------------------------------
// Sandbox: temp repo + isolated HOME, cleaned up on drop
// ---------------------------------------------------------------------------

struct Sandbox {
    /// Unique root; removed on drop.
    root: PathBuf,
    /// Git worktree under test.
    repo: PathBuf,
    /// Isolated HOME/XDG_CONFIG_HOME — outside the worktree so it never
    /// shows up as untracked content, and so neither the tool's config
    /// layer nor git's user-level config (~/.gitconfig, e.g.
    /// commit.gpgsign) leaks in.
    home: PathBuf,
}

impl Sandbox {
    fn new(name: &'static str) -> Self {
        let root = std::env::temp_dir().join(format!("autocommit-e2e-{name}-{}", unique_id()));
        let repo = root.join("repo");
        let home = root.join("home");
        std::fs::create_dir_all(&repo).expect("create repo dir");
        std::fs::create_dir_all(&home).expect("create home dir");

        let sandbox = Self { root, repo, home };
        sandbox.git(&["init"]);
        sandbox.git(&["config", "user.name", "e2e-bot"]);
        sandbox.git(&["config", "user.email", "e2e-bot@example.com"]);
        sandbox
    }

    /// Runs git inside the sandbox; asserts success, returns trimmed stdout.
    fn git(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.repo)
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.home)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap_or_else(|err| panic!("failed to run git {args:?}: {err}"));
        assert!(
            output.status.success(),
            "git {args:?} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    }

    fn write(&self, rel_path: &str, content: &str) {
        let path = self.repo.join(rel_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent dir");
        }
        std::fs::write(&path, content).expect("write file");
    }

    fn head(&self) -> String {
        self.git(&["rev-parse", "HEAD"])
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn unique_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos()
}

// ---------------------------------------------------------------------------
// Subprocess harness
// ---------------------------------------------------------------------------

struct Outcome {
    code: i32,
    stdout: String,
    stderr: String,
}

/// Spawns the binary in the sandbox with scenario env applied.
///
/// `strip_proxy` removes proxy env vars (used by loopback scenarios E-06/E-07
/// so a host proxy cannot intercept 127.0.0.1 traffic; live scenarios keep
/// the inherited proxy because the developer may need it to reach the API).
fn run_tool(
    sandbox: &Sandbox,
    env: &[(String, String)],
    deadline: Duration,
    strip_proxy: bool,
) -> Outcome {
    // let mut cmd = Command::new(BIN);
    let mut cmd = Command::new(binary_path());
    cmd.current_dir(&sandbox.repo)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Strip host AUTOCOMMIT_* so a developer's real config cannot leak into
    // a scenario — E-08 in particular depends on the API key being absent.
    for (key, _) in std::env::vars() {
        if key.starts_with("AUTOCOMMIT_") {
            cmd.env_remove(&key);
        }
    }
    if strip_proxy {
        for var in PROXY_ENV_VARS {
            cmd.env_remove(var);
        }
    }
    cmd.env("HOME", &sandbox.home);
    cmd.env("XDG_CONFIG_HOME", &sandbox.home);
    cmd.env("GIT_CONFIG_NOSYSTEM", "1");
    for (key, value) in env {
        cmd.env(key, value);
    }

    let mut child = cmd.spawn().expect("spawn auto-commit binary");
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if started.elapsed() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("auto-commit exceeded {deadline:?} deadline and was killed");
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(err) => panic!("try_wait failed: {err}"),
        }
    }
    // Outputs are tiny (< 4 KiB), safe to drain after exit without
    // risking a full-pipe deadlock.
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut stdout).expect("read stdout");
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_string(&mut stderr).expect("read stderr");
    }
    let status = child.wait().expect("reap child");
    Outcome {
        code: status.code().expect("terminated by signal"),
        stdout,
        stderr,
    }
}

fn assert_exit(outcome: &Outcome, expected: i32) {
    assert_eq!(
        outcome.code, expected,
        "expected exit {expected}, got {}:\nstdout:\n{}\nstderr:\n{}",
        outcome.code, outcome.stdout, outcome.stderr
    );
}

fn env_vec(pairs: &[(&str, String)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Inline Conventional Commits contract check
// ---------------------------------------------------------------------------

/// Keep in sync with `ALLOWED_TYPES` in `src/shared/exception/message.rs`.
const ALLOWED_TYPES: &[&str] = &[
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "chore", "build", "ci",
];

/// The prompt template promises 72; the validator currently tolerates up to
/// 150 (MAX_DESCRIPTION_CHARS). This check pins the TEMPLATE contract on
/// purpose: a live model exceeding it is a contract-mismatch signal to fix
/// in the prompt/validator, never a reason to loosen this check.
const MAX_DESCRIPTION_CHARS: usize = 72;

/// Minimal contract check: header shape, whitelist, lowercase start, length,
/// no trailing period, exactly one blank line before an optional body.
/// Micro-rules (scope length etc.) are the validator's authority and are
/// already unit-tested; this is deliberately only the outward contract.
fn assert_conventional_commit(message: &str) {
    let trimmed = message.trim();
    assert!(!trimmed.is_empty(), "commit message must not be empty");

    let mut lines = trimmed.split('\n');
    let header = lines.next().expect("non-empty").trim();
    let (left, desc) = header.split_once(": ").unwrap_or_else(|| {
        panic!("header must be `<type>[optional scope]: <description>`, got: {header:?}")
    });

    let ty = match (left.find('('), left.ends_with(')')) {
        (None, false) => left,
        (Some(open), true) => {
            let ty = &left[..open];
            let scope = &left[open + 1..left.len() - 1];
            assert!(!ty.is_empty(), "type must not be empty: {header:?}");
            assert!(!scope.is_empty(), "scope must not be empty: {header:?}");
            assert!(
                scope
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
                "scope must be lowercase alphanumeric, got: {scope:?}"
            );
            ty
        }
        _ => panic!("malformed type/scope region: {left:?}"),
    };
    assert!(
        ALLOWED_TYPES.contains(&ty),
        "type must be in the whitelist, got: {ty:?}"
    );
    assert!(!desc.is_empty(), "description must not be empty");
    assert!(
        desc.starts_with(|c: char| c.is_ascii_lowercase()),
        "description must start with a lowercase letter, got: {desc:?}"
    );
    assert!(
        desc.chars().count() <= MAX_DESCRIPTION_CHARS,
        "description must be at most {MAX_DESCRIPTION_CHARS} chars, got: {}",
        desc.chars().count()
    );
    assert!(!desc.ends_with('.'), "no trailing period: {desc:?}");

    if let Some(separator) = lines.next() {
        assert!(
            separator.trim().is_empty(),
            "header and body must be separated by exactly one blank line, got: {separator:?}"
        );
        assert!(
            lines.next().is_some(),
            "a blank separator must be followed by a body"
        );
    }
}

/// Success-path core: exit 0, stdout carries the validated candidate, the
/// candidate conforms to the contract, and `git log -1 %B` is byte-identical
/// — proving "validated message == committed message".
fn assert_committed_message(sandbox: &Sandbox, outcome: &Outcome) -> String {
    assert_exit(outcome, 0);
    let printed = outcome.stdout.trim();
    assert!(
        !printed.is_empty(),
        "headless mode must print the commit candidate on stdout"
    );
    assert_conventional_commit(printed);
    let committed = sandbox.git(&["log", "-1", "--format=%B"]);
    assert_eq!(
        printed,
        committed.trim(),
        "committed message must equal the validated candidate printed on stdout"
    );
    printed.to_owned()
}

/// Proves committed content == staged content (`git diff-tree` is empty for
/// merge commits, so this is only used on non-merge scenarios).
fn assert_committed_files(sandbox: &Sandbox, expected: &[&str]) {
    let out = sandbox.git(&["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]);
    let mut actual: Vec<String> = out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    actual.sort();
    let mut want: Vec<String> = expected.iter().map(|path| path.to_string()).collect();
    want.sort();
    assert_eq!(actual, want, "commit must contain exactly the staged files");
}

// ---------------------------------------------------------------------------
// Live-LLM gating (hard requirement: no key, no test)
// ---------------------------------------------------------------------------

fn live_credentials() -> Option<(String, String, Option<String>)> {
    let key = std::env::var("AUTOCOMMIT_E2E_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())?;

    let model = std::env::var("AUTOCOMMIT_E2E_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| "deepseek-v4-flash".to_owned());
    let base_url = std::env::var("AUTOCOMMIT_E2E_BASE_URL")
        .ok()
        .filter(|url| !url.trim().is_empty())
        .or_else(|| Some("https://api.deepseek.com".to_owned()));
    Some((key, model, base_url))
}

fn live_env(key: &str, model: &str, base_url: Option<&str>) -> Vec<(String, String)> {
    let mut env = env_vec(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "openai".into()),
        ("AUTOCOMMIT_LLM_MODEL", model.into()),
        ("AUTOCOMMIT_LLM_API_KEY", key.into()),
        ("AUTOCOMMIT_ASSUME_YES", "1".into()),
        // Bound each attempt + retry count so worst case (45s + backoff +
        // 45s ≈ 91s) stays under the 120s harness deadline: a hung
        // provider then exits 4 instead of being killed here.
        ("AUTOCOMMIT_TIMEOUT_MS", "45000".into()),
        ("AUTOCOMMIT_RETRY_MAX_RETRIES", "1".into()),
    ]);
    if let Some(url) = base_url {
        env.push(("AUTOCOMMIT_LLM_BASE_URL".into(), url.into()));
    }
    env
}

// ---------------------------------------------------------------------------
// E-01..E-04: real-LLM scenarios (ignored + runtime early-exit)
// ---------------------------------------------------------------------------

const README_BASELINE: &str = "# demo\n\nA demo project.\n";
const README_UPDATED: &str =
    "# demo\n\nA demo project.\n\n## Usage\n\nRun the parser over any input file.\n";
const LIB_RS: &str = r#"//! Demo library root.

pub mod parser;

/// Returns the crate version reported to callers.
pub fn version() -> &'static str {
    "0.1.0"
}
"#;
const PARSER_RS: &str = r#"//! Tiny line-oriented parser used by the demo CLI.

pub struct Line {
    pub indent: usize,
    pub text: String,
}

/// Splits input into trimmed, non-empty lines with their indent level.
pub fn parse_lines(input: &str) -> Vec<Line> {
    input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let indent = line.len() - line.trim_start().len();
            Line { indent, text: line.trim().to_owned() }
        })
        .collect()
}
"#;

#[test]
#[ignore = "requires live LLM credentials (set AUTOCOMMIT_E2E_API_KEY)"]
fn e01_live_functional_development() {
    let Some((key, model, base_url)) = live_credentials() else {
        eprintln!("skipped: AUTOCOMMIT_E2E_API_KEY not set");
        return;
    };
    let sandbox = Sandbox::new("e01");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);

    sandbox.write("src/lib.rs", LIB_RS);
    sandbox.write("src/parser.rs", PARSER_RS);
    sandbox.write("README.md", README_UPDATED);
    sandbox.git(&["add", "src/lib.rs", "src/parser.rs", "README.md"]);

    let env = live_env(&key, &model, base_url.as_deref());
    let outcome = run_tool(&sandbox, &env, LIVE_DEADLINE, false);
    assert_committed_message(&sandbox, &outcome);
    assert_committed_files(&sandbox, &["README.md", "src/lib.rs", "src/parser.rs"]);
}

const CARGO_TOML_BASELINE: &str = r#"[package]
name = "demo-app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0.200"
"#;
const CARGO_TOML_UPDATED: &str = r#"[package]
name = "demo-app"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8.5"
serde = "1.0.200"
"#;
const CARGO_LOCK_BASELINE: &str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
version = 3

[[package]]
name = "demo-app"
version = "0.1.0"
dependencies = ["serde"]

[[package]]
name = "serde"
version = "1.0.200"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc900def0000000000000000000000000000000000000000000000000000000"
"#;
const CARGO_LOCK_UPDATED: &str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
version = 3

[[package]]
name = "demo-app"
version = "0.1.0"
dependencies = ["rand", "serde"]

[[package]]
name = "rand"
version = "0.8.5"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc900def0000000000000000000000000000000000000000000000000000000001"

[[package]]
name = "serde"
version = "1.0.200"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc900def0000000000000000000000000000000000000000000000000000000"
"#;

#[test]
#[ignore = "requires live LLM credentials (set AUTOCOMMIT_E2E_API_KEY)"]
fn e02_live_dependency_update() {
    let Some((key, model, base_url)) = live_credentials() else {
        eprintln!("skipped: AUTOCOMMIT_E2E_API_KEY not set");
        return;
    };
    let sandbox = Sandbox::new("e02");
    sandbox.write("Cargo.toml", CARGO_TOML_BASELINE);
    sandbox.write("Cargo.lock", CARGO_LOCK_BASELINE);
    sandbox.git(&["add", "Cargo.toml", "Cargo.lock"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);

    sandbox.write("Cargo.toml", CARGO_TOML_UPDATED);
    sandbox.write("Cargo.lock", CARGO_LOCK_UPDATED);
    sandbox.git(&["add", "Cargo.toml", "Cargo.lock"]);

    let env = live_env(&key, &model, base_url.as_deref());
    let outcome = run_tool(&sandbox, &env, LIVE_DEADLINE, false);
    assert_committed_message(&sandbox, &outcome);
    assert_committed_files(&sandbox, &["Cargo.toml", "Cargo.lock"]);
}

const E03_LIB_BASELINE: &str = r#"//! Generated-schema host crate.

pub fn version() -> u32 {
    1
}
"#;
const E03_LIB_UPDATED: &str = r#"//! Generated-schema host crate.

pub fn version() -> u32 {
    1
}

pub fn schema_revision() -> u32 {
    2
}
"#;

/// ~420 lines of protobuf-codegen-style Rust: the real-world "big diff"
/// that exercises the default budget policy end to end.
fn generated_schema(messages: usize) -> String {
    let mut out =
        String::from("// Generated by protoc-gen-demo. DO NOT EDIT.\n\npub mod schema {\n");
    for i in 0..messages {
        out.push_str(&format!(
            "    /// Auto-generated message struct.\n    #[derive(Debug, Clone)]\n    pub struct Message{i} {{\n        pub id: u64,\n        pub name: String,\n        pub active: bool,\n    }}\n\n    impl Message{i} {{\n        pub fn new(id: u64, name: String, active: bool) -> Self {{\n            Self {{ id, name, active }}\n        }}\n\n        pub fn id(&self) -> u64 {{\n            self.id\n        }}\n\n        pub fn name(&self) -> &str {{\n            &self.name\n        }}\n\n        pub fn is_active(&self) -> bool {{\n            self.active\n        }}\n    }}\n\n"
        ));
    }
    out.push_str("}\n");
    out
}

#[test]
#[ignore = "requires live LLM credentials (set AUTOCOMMIT_E2E_API_KEY)"]
fn e03_live_large_generated_diff() {
    let Some((key, model, base_url)) = live_credentials() else {
        eprintln!("skipped: AUTOCOMMIT_E2E_API_KEY not set");
        return;
    };
    let sandbox = Sandbox::new("e03");
    sandbox.write("src/lib.rs", E03_LIB_BASELINE);
    sandbox.git(&["add", "src/lib.rs"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);

    sandbox.write("src/gen/schema.rs", &generated_schema(30));
    sandbox.write("src/lib.rs", E03_LIB_UPDATED);
    sandbox.git(&["add", "src/gen/schema.rs", "src/lib.rs"]);

    let env = live_env(&key, &model, base_url.as_deref());
    let outcome = run_tool(&sandbox, &env, LIVE_DEADLINE, false);
    assert_committed_message(&sandbox, &outcome);
    assert_committed_files(&sandbox, &["src/gen/schema.rs", "src/lib.rs"]);
}

#[test]
#[ignore = "requires live LLM credentials (set AUTOCOMMIT_E2E_API_KEY)"]
fn e04_live_merge_no_commit() {
    let Some((key, model, base_url)) = live_credentials() else {
        eprintln!("skipped: AUTOCOMMIT_E2E_API_KEY not set");
        return;
    };
    let sandbox = Sandbox::new("e04");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);

    // Realistic flow: finish work on a feature branch, then stop the merge
    // before committing so auto-commit drives the conclusion.
    let base_branch = sandbox.git(&["branch", "--show-current"]);
    sandbox.git(&["checkout", "-b", "feature"]);
    sandbox.write("feature.rs", "pub fn feature() -> bool {\n    true\n}\n");
    sandbox.git(&["add", "feature.rs"]);
    sandbox.git(&["commit", "-m", "feat: add feature module"]);
    sandbox.git(&["checkout", base_branch.as_str()]);
    sandbox.git(&["merge", "--no-ff", "--no-commit", "feature"]);
    sandbox.git(&["add", "-A"]);

    // Merge state must be present (MERGE_HEAD is the detector's anchor).
    sandbox.git(&["rev-parse", "--verify", "MERGE_HEAD"]);

    let env = live_env(&key, &model, base_url.as_deref());
    let outcome = run_tool(&sandbox, &env, LIVE_DEADLINE, false);
    assert_committed_message(&sandbox, &outcome);

    // `git diff-tree` prints nothing for merge commits, so conclude via
    // merge ground truth instead: two parents, clean worktree, and the
    // feature file present in HEAD's tree.
    let parents = sandbox.git(&["log", "-1", "--format=%P"]);
    assert_eq!(
        parents.split_whitespace().count(),
        2,
        "must conclude as a merge commit, parents: {parents}"
    );
    assert!(
        sandbox.git(&["status", "--porcelain"]).is_empty(),
        "worktree must be clean after the merge commit"
    );
    sandbox.git(&["cat-file", "-e", "HEAD:feature.rs"]); // panics if absent
}

// ---------------------------------------------------------------------------
// E-05..E-08: zero-cost scenarios (always on, no credentials)
// ---------------------------------------------------------------------------

#[test]
fn e05_nothing_staged_fails_before_llm() {
    let sandbox = Sandbox::new("e05");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);
    let head_before = sandbox.head();

    // A minimal VALID config is required: config validation (step 1 in
    // main) runs BEFORE the git pipeline, so with no config at all the tool
    // would exit 2 (missing llm.model) instead of reaching the
    // nothing-staged check. Ollama needs no key and no network is touched —
    // the pipeline rejects the repo before any LLM call.
    let env = env_vec(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "ollama".into()),
        ("AUTOCOMMIT_LLM_MODEL", "unused".into()),
    ]);
    let outcome = run_tool(&sandbox, &env, SHORT_DEADLINE, false);

    assert_exit(&outcome, 3);
    assert!(
        outcome.stderr.starts_with("auto-commit: pipeline error:"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("nothing_staged"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert_eq!(sandbox.head(), head_before, "HEAD must be unchanged");
}

#[test]
fn e06_non_tty_refused_even_when_llm_succeeds() {
    let sandbox = Sandbox::new("e06");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);
    sandbox.write("src/lib.rs", LIB_RS);
    sandbox.git(&["add", "src/lib.rs"]);
    let head_before = sandbox.head();

    // The TTY wall sits AFTER a successful LLM call, so this scenario needs
    // the mock endpoint to reach it — proving that even a fully valid
    // pipeline is refused without an interactive terminal.
    // AUTOCOMMIT_ASSUME_YES is deliberately NOT set.
    let (mock_port, _mock) = spawn_mock_openai();
    let env = env_vec(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "openai".into()),
        ("AUTOCOMMIT_LLM_MODEL", "mock".into()),
        ("AUTOCOMMIT_LLM_API_KEY", "test-key".into()),
        (
            "AUTOCOMMIT_LLM_BASE_URL",
            format!("http://127.0.0.1:{mock_port}/v1"),
        ),
    ]);
    let outcome = run_tool(&sandbox, &env, SHORT_DEADLINE, true);

    assert_exit(&outcome, 5);
    assert!(
        outcome.stderr.contains("requires an interactive terminal"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert_eq!(sandbox.head(), head_before, "no commit may be created");
}

#[test]
fn e07_provider_unreachable_fails_fast() {
    let sandbox = Sandbox::new("e07");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);
    sandbox.write("src/lib.rs", LIB_RS);
    sandbox.git(&["add", "src/lib.rs"]);
    let head_before = sandbox.head();

    let dead_port = reserve_dead_port();
    let env = env_vec(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "openai".into()),
        ("AUTOCOMMIT_LLM_MODEL", "mock".into()),
        ("AUTOCOMMIT_LLM_API_KEY", "test-key".into()),
        (
            "AUTOCOMMIT_LLM_BASE_URL",
            format!("http://127.0.0.1:{dead_port}/v1"),
        ),
        // Single attempt, no backoff sleep: connection refused is transient
        // and must be exhausted within milliseconds.
        ("AUTOCOMMIT_RETRY_MAX_RETRIES", "0".into()),
        ("AUTOCOMMIT_ASSUME_YES", "1".into()),
    ]);
    let outcome = run_tool(&sandbox, &env, SHORT_DEADLINE, true);

    assert_exit(&outcome, 4);
    assert!(
        outcome.stderr.starts_with("auto-commit: LLM error:"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("failed after 1 attempt(s)"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert_eq!(sandbox.head(), head_before, "HEAD must be unchanged");
}

#[test]
fn e08_missing_api_key_is_a_config_error() {
    let sandbox = Sandbox::new("e08");
    sandbox.write("README.md", README_BASELINE);
    sandbox.git(&["add", "README.md"]);
    sandbox.git(&["commit", "-m", "chore: baseline"]);
    let head_before = sandbox.head();

    // Realistic mistake: provider and model configured, API key forgotten.
    // Config validation runs before the git pipeline, so staging is
    // irrelevant here.
    let env = env_vec(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "openai".into()),
        ("AUTOCOMMIT_LLM_MODEL", "gpt-4o-mini".into()),
        // AUTOCOMMIT_LLM_API_KEY deliberately absent.
    ]);
    let outcome = run_tool(&sandbox, &env, SHORT_DEADLINE, false);

    assert_exit(&outcome, 2);
    assert!(
        outcome
            .stderr
            .starts_with("auto-commit: configuration error:"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("llm.api_key"),
        "stderr:\n{}",
        outcome.stderr
    );
    assert_eq!(sandbox.head(), head_before, "HEAD must be unchanged");
}

// ---------------------------------------------------------------------------
// Mock OpenAI endpoint (E-06 only)
// ---------------------------------------------------------------------------

/// Fixed, validator-clean completion body. async-openai POSTs
/// `{base_url}/chat/completions` and deserializes exactly these fields.
const MOCK_COMPLETION_BODY: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","created":0,"model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"feat: add user authentication"},"finish_reason":"stop"}]}"#;

fn spawn_mock_openai() -> (u16, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock listener");
    let port = listener.local_addr().expect("mock local addr").port();
    let handle = thread::spawn(move || {
        // The tool makes exactly one call in the E-06 flow.
        if let Ok((stream, _)) = listener.accept() {
            serve_chat_completion(stream);
        }
    });
    (port, handle)
}

fn serve_chat_completion(mut stream: TcpStream) {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];

    // Read until end of headers.
    let header_end = loop {
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            return; // client vanished; nothing to answer
        }
        buf.extend_from_slice(&chunk[..read]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return; // abusive request; drop it
        }
    };

    // Drain the request body per Content-Length so the client observes a
    // clean response instead of a reset connection.
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_uppercase();
    let content_length = headers
        .lines()
        .find_map(|line| line.strip_prefix("CONTENT-LENGTH:"))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..read]);
    }

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        MOCK_COMPLETION_BODY.len(),
        MOCK_COMPLETION_BODY
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Bind an ephemeral port, then drop the listener: connections are refused
/// immediately (the tiny reuse race window is acceptable for E-07).
fn reserve_dead_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
