//! Integration tests for stage 4: Prompt assembly (`PromptAssembler::assemble`).
//!
//! Every scenario mirrors the C-series (stage 2–3) in `test_git_pipeline.rs`
//! one-to-one — same fixture, same git commands, same test names — and pins
//! the resulting `LlmMessage` instead of the intermediate metadata:
//!
//! - c1..c9 : regular template (`FromStaging`) — change summary, staged diff,
//!            budget degradation notices, C/D metadata-only sections.
//! - d1..d3 : dedicated template (`FromOperation`) — native git message plus
//!            the oid trailers; never the empty-diff main template.
//!
//! Stage 2/3 facts stay owned by the C-series; here each test only pins the
//! stage-2/3 discriminator (strategy / body / counts) needed to prove the
//! prompt was assembled from the intended context, then asserts the prompt.
//! Squash/rebase prompts share the merge code path (only the op name differs)
//! and are deliberately not repeated.
//!
//! Mounting: declared as `mod test_prompt_assembler;` in
//! `test/integration/mod.rs`, which `src/main.rs` mounts via
//! `#[path = "../test/integration/mod.rs"] mod integration;` (in-crate tests,
//! `crate::` imports).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::git::types::{
    BudgetDecision, BudgetPolicy, ClassifiedSnapshot, DiffPayload, DiffStrategy, Operation,
};
use crate::core::pipe::assembler::PromptAssembler;
use crate::core::pipe::context::AssemblyContext;
use crate::core::pipe::orchestrator::PipeOrchestrator;
use crate::core::pipe::template::SYSTEM_PROMPT;
use crate::infra::git::GitRunner;
use crate::shared::config::LlmMessage;
use crate::shared::exception::GitError;

// --- Pipeline + prompt adapters ---

fn runner_at(dir: &Path) -> GitRunner {
    GitRunner::new(Some(dir.to_path_buf()))
}

/// Full pipeline (stage 0 → 3) through the public orchestrator.
async fn run_pipeline(dir: &Path, policy: BudgetPolicy) -> Result<AssemblyContext, GitError> {
    let runner = runner_at(dir);
    PipeOrchestrator::new(&runner, policy).run().await
}

/// Stage 4: assemble the `LlmMessage` from a pipeline context.
fn prompt(ctx: &AssemblyContext) -> LlmMessage {
    PromptAssembler::assemble(ctx)
}

/// Unpack `FromStaging`, panicking on the operation branch.
fn staging(ctx: AssemblyContext) -> (ClassifiedSnapshot, BudgetDecision, DiffPayload) {
    match ctx {
        AssemblyContext::FromStaging {
            snapshot,
            decision,
            payload,
            ..
        } => (snapshot, decision, payload),
        other => panic!("expected staged outcome, got {other:?}"),
    }
}

// --- Fixture helpers (mirror of test_git_pipeline.rs) ---

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Fresh empty temp directory (std only), removed on drop.
struct TempRepo {
    path: PathBuf,
}

impl TempRepo {
    fn new(label: &str) -> Self {
        let n = DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("auto-commit-it-{label}-{n}"));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn run_git(dir: &Path, args: &[&str]) -> Output {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("LC_ALL", "C")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn git: {err}"))
}

/// Runs git, panics on non-zero exit.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = run_git(dir, args);
    assert!(
        out.status.success(),
        "git {args:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("git stdout not UTF-8")
}

/// Runs git asserting a non-zero exit (conflicting cherry-pick…).
fn git_fails(dir: &Path, args: &[&str]) -> Output {
    let out = run_git(dir, args);
    assert!(
        !out.status.success(),
        "git {args:?} unexpectedly succeeded:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, bytes).unwrap();
}

fn git_add(dir: &Path, paths: &[&str]) {
    let mut args: Vec<&str> = vec!["add"];
    args.extend_from_slice(paths);
    git(dir, &args);
}

// --- Rust project fixture ---

const SEED_LIB_RS: &str = "pub fn seed() -> u32 {\n    40 + 2\n}\n";
const SEED_MAIN_RS: &str = "fn main() {\n    println!(\"seed\");\n}\n";
const SEED_CARGO_TOML: &str =
    "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n";

/// C1 edit of src/lib.rs: delete 1 line, add 2 lines carrying a unique marker.
const C1_LIB_RS: &str = "// probe-C1: integration marker\npub fn seed() -> u32 {\n    42\n}\n";

fn init_git(dir: &Path) {
    git(dir, &["init", "-b", "trunk"]);
    git(dir, &["config", "user.name", "Auto Commit Test"]);
    git(dir, &["config", "user.email", "test@example.com"]);
}

fn seed_files(dir: &Path) {
    write(&dir.join("Cargo.toml"), SEED_CARGO_TOML);
    write(&dir.join("Cargo.lock"), &seed_lock_text());
    write(&dir.join(".gitignore"), "/target\n");
    write(&dir.join("README.md"), "demo readme\n");
    write(&dir.join("src/lib.rs"), SEED_LIB_RS);
    write(&dir.join("src/main.rs"), SEED_MAIN_RS);
}

/// Standard fixture: initialized repo with one committed Rust scaffold.
fn init_rust_repo(dir: &Path) {
    init_git(dir);
    seed_files(dir);
    git_add(dir, &["."]);
    git(dir, &["commit", "-m", "chore: seed repository"]);
}

// --- Cargo.lock fixture ---

fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 64 hex chars — pure [0-9a-f], therefore never a lock-signal line.
fn fake_checksum(seed: &str) -> String {
    format!("{:016x}", fnv1a(seed)).repeat(4)
}

#[derive(Clone)]
struct LockEntry {
    name: String,
    version: String,
    checksum: String,
}

impl LockEntry {
    fn new(name: &str, version: &str) -> Self {
        let seed = format!("{name}@{version}");
        Self {
            name: name.to_string(),
            version: version.to_string(),
            checksum: fake_checksum(&seed),
        }
    }
}

fn lock_text(entries: &[LockEntry]) -> String {
    let mut text = String::from(
        "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n",
    );
    for entry in entries {
        text.push_str(&format!(
            "[[package]]\nname = \"{name}\"\nversion = \"{version}\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"{checksum}\"\n\n",
            name = entry.name,
            version = entry.version,
            checksum = entry.checksum,
        ));
    }
    text
}

fn seed_lock_text() -> String {
    lock_text(&[
        LockEntry::new("demo-core", "1.2.3"),
        LockEntry::new("demo-dep", "0.9.0"),
    ])
}

// --- Binary / generated fixtures ---

/// PNG signature + explicit NUL bytes so git's numstat reports `-\t-` (Binary).
fn png_bytes() -> Vec<u8> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    bytes.extend([0u8; 32]);
    bytes
}

/// Path/name carry no rule signal — classification must come from the
/// Phase B header probe (`@generated` marker inside the first 512 bytes).
const GENERATED_RS: &str = "\
// @generated by prost-build. DO NOT EDIT.
// source: demo.proto

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Demo {
    #[prost(string, tag = \"1\")]
    pub id: String,
}
";

fn big_lines(prefix: &str, n: usize) -> String {
    (1..=n)
        .map(|i| format!("{prefix} probe line {i:03}\n"))
        .collect()
}

// C. Stage 4 on the regular template — mirrors the C-series one-to-one.

#[tokio::test]
async fn c1_semantic_full_plus_lock_digest() {
    // Mirror of c1: A+B under Full → summary + full A diff + lock digest in the prompt.
    let tmp = TempRepo::new("p-c1");
    let repo = tmp.path();
    init_rust_repo(repo);

    write(&repo.join("src/lib.rs"), C1_LIB_RS);
    let mut entries = vec![
        LockEntry::new("demo-core", "1.2.3"),
        LockEntry::new("demo-dep", "0.9.0"),
    ];
    entries[0].version = "1.2.4".to_string();
    entries[0].checksum =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string();
    write(&repo.join("Cargo.lock"), &lock_text(&entries));
    git_add(repo, &["src/lib.rs", "Cargo.lock"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    // Stage 2/3 discriminator: this context is the Full A+B case.
    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert!(
        payload.body.contains("+// probe-C1: integration marker"),
        "body: {}",
        payload.body
    );

    let user = &msg.user_message;
    assert_eq!(msg.system_message.as_deref(), Some(SYSTEM_PROMPT));
    assert!(user.contains("## Repository\n"), "user: {user}");
    assert!(user.contains("branch: trunk\n"), "user: {user}");
    assert!(user.contains("## Change summary\n"), "user: {user}");
    assert!(
        user.contains("- source files: 1 changed, +2 insertions/-1 deletions\n"),
        "user: {user}"
    );
    assert!(user.contains("- lock files: 1 changed\n"), "user: {user}");
    assert!(user.contains("## Staged changes\n"), "user: {user}");
    assert!(
        user.contains("+// probe-C1: integration marker"),
        "user: {user}"
    );
    assert!(user.contains("+version = \"1.2.4\""), "user: {user}");
    assert!(!user.contains("checksum"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
    assert!(!user.contains("## Non-text files"), "user: {user}");
}

#[tokio::test]
async fn c2_deleted_semantic() {
    // Mirror of c2: deletion → `+0/-N` summary and the removed lines in the prompt.
    let tmp = TempRepo::new("p-c2");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["rm", "src/lib.rs"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert!(
        payload.body.contains("-    40 + 2"),
        "body: {}",
        payload.body
    );

    let user = &msg.user_message;
    assert!(
        user.contains("- source files: 1 changed, +0 insertions/-3 deletions\n"),
        "user: {user}"
    );
    assert!(user.contains("## Staged changes\n"), "user: {user}");
    assert!(user.contains("-    40 + 2"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
}

#[tokio::test]
async fn c3_pure_rename() {
    // Mirror of c3: rename-only → rename summary line, no diff budget section.
    let tmp = TempRepo::new("p-c3");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["mv", "src/lib.rs", "src/lib2.rs"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert!(
        payload
            .body
            .contains("diff --git a/src/lib.rs b/src/lib2.rs"),
        "body: {}",
        payload.body
    );

    let user = &msg.user_message;
    assert!(
        user.contains("- source files: 1 renamed only (no content change)\n"),
        "user: {user}"
    );
    assert!(user.contains("similarity index 100%"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
    assert!(!user.contains("## Non-text files"), "user: {user}");
}

#[tokio::test]
async fn c4_generated_header_probe() {
    // Mirror of c4: generated-only → prompt survives on metadata; the generated
    // file content itself never leaks into the message.
    let tmp = TempRepo::new("p-c4");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/pb/messages.rs"), GENERATED_RS);
    git_add(repo, &["src/pb/messages.rs"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(payload.body, "");

    let user = &msg.user_message;
    assert!(
        !user.is_empty(),
        "prompt must be generated for C/D-only changes"
    );
    assert!(!user.contains("## Change summary"), "user: {user}");
    assert!(
        user.contains("## Non-text files (metadata only)\n"),
        "user: {user}"
    );
    assert!(
        user.contains("- [generated] Added src/pb/messages.rs\n"),
        "user: {user}"
    );
    assert!(
        user.contains("(no diff content; rely on the change summary and file list above)"),
        "user: {user}"
    );
    // Generated file CONTENT must not appear — only its metadata line.
    assert!(!user.contains("prost-build"), "user: {user}");
    assert!(!user.contains("::prost::Message"), "user: {user}");
    assert!(
        !user.contains('\0'),
        "binary bytes must not leak into the prompt"
    );
}

#[tokio::test]
async fn c5_budget_zero_respects_categories() {
    // Mirror of c5: zero budget → PathSummaryOnly; prompt discloses the
    // reduction/guidance, lists C/D as metadata only, hides all raw diff.
    let tmp = TempRepo::new("p-c5");
    let repo = tmp.path();
    init_rust_repo(repo);

    write(&repo.join("src/lib.rs"), C1_LIB_RS);
    let mut entries = vec![
        LockEntry::new("demo-core", "1.2.3"),
        LockEntry::new("demo-dep", "0.9.0"),
    ];
    entries[0].version = "1.2.4".to_string();
    write(&repo.join("Cargo.lock"), &lock_text(&entries));
    write_bytes(&repo.join("assets/icon.png"), &png_bytes());
    write(&repo.join("src/pb/messages.rs"), GENERATED_RS);
    git_add(
        repo,
        &[
            "src/lib.rs",
            "Cargo.lock",
            "assets/icon.png",
            "src/pb/messages.rs",
        ],
    );

    let policy = BudgetPolicy {
        context_token_limit: 1_000,
        reserved_tokens: 1_000,
        ..BudgetPolicy::default()
    };
    let ctx = run_pipeline(repo, policy).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
    assert_eq!(payload.body, "M  Cargo.lock\nM  src/lib.rs\n");

    let user = &msg.user_message;
    assert!(user.contains("## Diff budget\n"), "user: {user}");
    assert!(user.contains("available: 0"), "user: {user}");
    assert!(
        user.contains("- reduction: diff content replaced by a path listing\n"),
        "user: {user}"
    );
    assert!(
        user.contains("- guidance: write a high-level message from the paths alone\n"),
        "user: {user}"
    );
    assert!(
        user.contains("- [binary] Added assets/icon.png\n"),
        "user: {user}"
    );
    assert!(
        user.contains("- [generated] Added src/pb/messages.rs\n"),
        "user: {user}"
    );
    assert!(user.contains("M  Cargo.lock\n"), "user: {user}");
    assert!(user.contains("M  src/lib.rs\n"), "user: {user}");
    assert!(!user.contains("diff --git"), "user: {user}");
    assert!(!user.contains("prost-build"), "user: {user}");
    assert!(
        !user.contains('\0'),
        "binary bytes must not leak into the prompt"
    );
}

#[tokio::test]
async fn c6_over_budget_truncate_lines() {
    // Mirror of c6: TruncateLines → per-file cap disclosed + truncation notice.
    let tmp = TempRepo::new("p-c6");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/big.rs"), &big_lines("// old", 250));
    git_add(repo, &["src/big.rs"]);
    git(repo, &["commit", "-m", "feat: seed big file"]);
    write(&repo.join("src/big.rs"), &big_lines("// new", 250));
    git_add(repo, &["src/big.rs"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(
        decision.strategy,
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400
        }
    );
    assert_eq!(payload.truncated_file_count, 1);

    let user = &msg.user_message;
    assert!(user.contains("## Diff budget\n"), "user: {user}");
    assert!(
        user.contains("- reduction: each file capped at 400 changed lines\n"),
        "user: {user}"
    );
    assert!(user.contains("diff --git a/src/big.rs"), "user: {user}");
    assert!(
        user.contains("... [100 more changed lines truncated]"),
        "user: {user}"
    );
    assert!(!user.contains("+// new probe line 151"), "user: {user}");
    assert!(
        user.contains("1 of 1 file diff(s) were truncated to fit the context budget"),
        "user: {user}"
    );
}

#[tokio::test]
async fn c7_over_budget_sample_hunks() {
    // Mirror of c7: SampleHunks → hunk+line cap wording reaches the prompt.
    let tmp = TempRepo::new("p-c7");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/big2.rs"), &big_lines("// added", 500));
    git_add(repo, &["src/big2.rs"]);

    let policy = BudgetPolicy {
        context_token_limit: 6_000,
        reserved_tokens: 2_000,
        ..BudgetPolicy::default()
    };
    let ctx = run_pipeline(repo, policy).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(
        decision.strategy,
        DiffStrategy::SampleHunks {
            max_hunks_per_file: 8,
            max_changed_lines_per_file: 400
        }
    );
    assert_eq!(payload.truncated_file_count, 1);

    let user = &msg.user_message;
    assert!(user.contains("## Diff budget\n"), "user: {user}");
    assert!(
        user.contains("- reduction: each file capped at 8 hunk(s) and 400 changed lines\n"),
        "user: {user}"
    );
    assert!(user.contains("@@ -0,0 +1,500 @@"), "user: {user}");
    assert!(
        user.contains("... [100 more changed lines truncated]"),
        "user: {user}"
    );
}

#[tokio::test]
async fn c8_lock_digest_capacity() {
    // Mirror of c8: lock digest truncation stays inside the body; Full means no
    // Diff-budget section and digest truncation is NOT a file-truncation notice.
    let tmp = TempRepo::new("p-c8");
    let repo = tmp.path();
    init_rust_repo(repo);

    let v1 = (0..100)
        .map(|i| LockEntry::new(&format!("pkg{i:03}"), &format!("1.{i}.0")))
        .collect::<Vec<_>>();
    write(&repo.join("Cargo.lock"), &lock_text(&v1));
    git_add(repo, &["Cargo.lock"]);
    git(repo, &["commit", "-m", "feat: seed 100-entry lock"]);

    let mut v2 = v1.clone();
    for i in 0..70 {
        v2[i].version = format!("2.{i}.0");
    }
    for entry in v2.iter_mut().take(80).skip(70) {
        entry.checksum = fake_checksum(&format!("{}-changed", entry.name));
    }
    write(&repo.join("Cargo.lock"), &lock_text(&v2));
    git_add(repo, &["Cargo.lock"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(payload.truncated_file_count, 0);

    let user = &msg.user_message;
    assert!(user.contains("- lock files: 1 changed\n"), "user: {user}");
    assert!(user.contains("diff --git a/Cargo.lock"), "user: {user}");
    assert!(user.contains("+version = \"2.0.0\""), "user: {user}");
    assert!(
        user.contains("... [76 more dependency lines truncated]"),
        "user: {user}"
    );
    assert!(!user.contains("checksum"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
    assert!(
        !user.contains("were truncated to fit the context budget"),
        "digest truncation must not trigger the file-truncation notice"
    );
}

#[tokio::test]
async fn c9_lock_digest_empty_when_no_signal() {
    // Mirror of c9: zero-signal lock edit → summary only, placeholder diff;
    // distinct from c8's truncation and from c4's missing Change-summary.
    let tmp = TempRepo::new("p-c9");
    let repo = tmp.path();
    init_rust_repo(repo);

    let mut entries = vec![
        LockEntry::new("demo-core", "1.2.3"),
        LockEntry::new("demo-dep", "0.9.0"),
    ];
    entries[1].checksum = fake_checksum("demo-dep@0.9.0-changed");
    write(&repo.join("Cargo.lock"), &lock_text(&entries));
    git_add(repo, &["Cargo.lock"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);
    let (_snapshot, decision, payload) = staging(ctx);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(payload.body, "");

    let user = &msg.user_message;
    assert!(user.contains("## Change summary\n"), "user: {user}");
    assert!(user.contains("- lock files: 1 changed\n"), "user: {user}");
    assert!(
        user.contains("(no diff content; rely on the change summary and file list above)"),
        "user: {user}"
    );
    assert!(!user.contains("## Non-text files"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
}

// D. Stage 4 on the dedicated template — mirrors B3 / B6 / B7.

#[tokio::test]
async fn d1_merge_native_message() {
    // Mirror of b3: merge reuse → native message template, no main-template sections.
    let tmp = TempRepo::new("p-d1");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(&repo.join("side.rs"), "pub fn side() {}\n");
    git_add(repo, &["side.rs"]);
    git(repo, &["commit", "-m", "feat: side work"]);
    git(repo, &["switch", "trunk"]);
    git(
        repo,
        &[
            "merge",
            "--no-ff",
            "--no-commit",
            "-m",
            "chore: merge feature",
            "side",
        ],
    );

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);

    let operation = match &ctx {
        AssemblyContext::FromOperation { operation, .. } => *operation,
        other => panic!("expected operation outcome, got {other:?}"),
    };
    assert_eq!(operation, Operation::Merge);

    let user = &msg.user_message;
    assert_eq!(msg.system_message.as_deref(), Some(SYSTEM_PROMPT));
    assert!(user.contains("## Repository\n"), "user: {user}");
    assert!(user.contains("## Git operation: merge\n"), "user: {user}");
    assert!(user.contains("## Git-native message\n"), "user: {user}");
    assert!(user.contains("chore: merge feature"), "user: {user}");
    // Dedicated template never falls into the empty-diff main template.
    assert!(!user.contains("## Staged changes"), "user: {user}");
    assert!(!user.contains("## Change summary"), "user: {user}");
    assert!(!user.contains("## Diff budget"), "user: {user}");
    assert!(!user.contains("no diff content"), "user: {user}");
}

#[tokio::test]
async fn d2_cherry_pick_oid_trailer() {
    // Mirror of b6: cherry-pick template carries subject + `cherry picked from` trailer.
    let tmp = TempRepo::new("p-d2");
    let repo = tmp.path();
    init_rust_repo(repo);

    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: cherry target"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();

    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);

    // Conflicting pick: writes CHERRY_PICK_HEAD, stops with unmerged entries;
    // resolve in the index only, keeping the sequencer state untouched.
    git_fails(repo, &["cherry-pick", &oid]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);

    let (operation, commit_oid) = match &ctx {
        AssemblyContext::FromOperation {
            operation,
            commit_oid,
            ..
        } => (*operation, commit_oid.clone()),
        other => panic!("expected operation outcome, got {other:?}"),
    };
    assert_eq!(operation, Operation::CherryPick);
    assert_eq!(commit_oid.as_deref(), Some(oid.as_str()));

    let user = &msg.user_message;
    assert!(
        user.contains("## Git operation: cherry-pick\n"),
        "user: {user}"
    );
    assert!(user.contains("feat: cherry target"), "user: {user}");
    assert!(
        user.contains(&format!("(cherry picked from commit {oid})")),
        "user: {user}"
    );
    assert!(!user.contains("## Staged changes"), "user: {user}");
}

#[tokio::test]
async fn d3_revert_oid_trailer() {
    // Mirror of b7: revert template carries `Revert "…"` + `This reverts commit`.
    let tmp = TempRepo::new("p-d3");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("second.rs"), "pub fn second() {}\n");
    git_add(repo, &["second.rs"]);
    git(repo, &["commit", "-m", "chore: add second"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["revert", "--no-commit", &oid]);

    let ctx = run_pipeline(repo, BudgetPolicy::default()).await.unwrap();
    let msg = prompt(&ctx);

    let operation = match &ctx {
        AssemblyContext::FromOperation { operation, .. } => *operation,
        other => panic!("expected operation outcome, got {other:?}"),
    };
    assert_eq!(operation, Operation::Revert);

    let user = &msg.user_message;
    assert!(user.contains("## Git operation: revert\n"), "user: {user}");
    assert!(
        user.contains("Revert \"chore: add second\""),
        "user: {user}"
    );
    assert!(
        user.contains(&format!("This reverts commit {oid}.")),
        "user: {user}"
    );
    assert!(!user.contains("## Staged changes"), "user: {user}");
}
