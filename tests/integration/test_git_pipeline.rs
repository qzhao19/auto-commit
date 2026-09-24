//! Integration tests for the Git information-flow pipeline (stage 0–3).
//!
//! Every test runs against a fresh, real git repository in a temp directory:
//! - A: stage 0 preflight
//! - B: stage 1 operation-state detection
//! - C: stage 2–3 staged metadata → classification → budget → extraction
//!
//! Mounting: declared as `mod test_git_pipeline;` in `test/integration/mod.rs`,
//! which `src/main.rs` mounts via `#[path = "../test/integration/mod.rs"]`
//! (in-crate tests, `crate::` imports). For a top-level `tests/` dir instead,
//! replace `crate::` with the package name (no `pub(crate)` test-only exports
//! are used).
//!
//! VERIFY markers: `GitRunner::new` signature (see `runner_at`) and
//! `GitError` Display / `GitErrorCode` PartialEq (see `assert_err`).
//! Prompt assembly has no dedicated test IDs in the spec tables; its inputs
//! are pinned via B (seed/native message) and C (diff body).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::git::operation::OperationStateDetector;
use crate::core::git::preflight::RepoPreflightCollector;
use crate::core::git::types::{
    BudgetDecision, BudgetPolicy, ChangeType, ClassifiedSnapshot, DiffPayload, DiffStrategy,
    FileCategory, Operation, OperationAction, OperationState, StagedFile,
};
use crate::core::pipe::context::AssemblyContext;
use crate::core::pipe::orchestrator::PipeOrchestrator;
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

// Pipeline adapter — the only place that touches pipe + runner construction.

/// VERIFY: assumed constructor `GitRunner::new(path: PathBuf)`.
fn runner_at(dir: &Path) -> GitRunner {
    GitRunner::new(Some(dir.to_path_buf()))
}

/// Full pipeline (stage 0 → 3) through the public orchestrator.
async fn run_pipeline(dir: &Path, policy: BudgetPolicy) -> Result<AssemblyContext, GitError> {
    let runner = runner_at(dir);
    PipeOrchestrator::new(&runner, policy).run().await
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

/// Stage-1 detector over a preflighted repository (for B-group pins).
async fn detect(dir: &Path) -> OperationState {
    let runner = runner_at(dir);
    let ctx = RepoPreflightCollector::new(&runner).run().await.unwrap();
    OperationStateDetector::new(&runner)
        .run(&ctx)
        .await
        .unwrap()
}

// Fixture helpers

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

/// Runs git asserting a non-zero exit (rebase --exec false, conflicts…).
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

// --- Assertion helpers ---

fn file_by_path<'a>(snapshot: &'a ClassifiedSnapshot, path: &str) -> &'a StagedFile {
    snapshot
        .files()
        .iter()
        .find(|file| file.path == Path::new(path))
        .unwrap_or_else(|| {
            let all = snapshot
                .files()
                .iter()
                .map(|f| f.path.display().to_string())
                .collect::<Vec<_>>();
            panic!("expected staged file {path}, staged: {all:?}")
        })
}

/// VERIFY: assumes `GitError` Display = message text and `code` is PartialEq.
fn assert_err(err: &GitError, code: GitErrorCode, needle: &str) {
    assert_eq!(err.code, code, "unexpected error: {err}");
    let text = err.to_string();
    assert!(
        text.contains(needle),
        "expected message containing {needle:?}, got: {text}"
    );
}

// A. Stage 0 — repository preflight

#[tokio::test]
async fn a1_not_a_repository() {
    // Empty temp dir is not a git repository → NotARepository.
    let tmp = TempRepo::new("a1");
    let err = run_pipeline(tmp.path(), BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(&err, GitErrorCode::NotARepository, "not a git repository");
}

#[tokio::test]
async fn a2_bare_repository() {
    // Bare repos have no working tree → rejected.
    let tmp = TempRepo::new("a2");
    let bare = tmp.path().join("bare.git");
    std::fs::create_dir_all(&bare).unwrap();
    git(&bare, &["init", "--bare"]);
    let err = run_pipeline(&bare, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(
        &err,
        GitErrorCode::Other,
        "bare repository is not supported",
    );
}

#[tokio::test]
async fn a3_index_lock() {
    // `index.lock` present → rejected before any diff work.
    let tmp = TempRepo::new("a3");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join(".git/index.lock"), "");
    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(&err, GitErrorCode::Other, "git index is locked");
}

#[tokio::test]
async fn a4_dirty_tracked_not_staged() {
    // Tracked modification without `git add` → NothingStaged (dirty branch).
    let tmp = TempRepo::new("a4");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    41\n}\n",
    );
    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(
        &err,
        GitErrorCode::NothingStaged,
        "no staged changes; run `git add` or `git rm` first",
    );
}

#[tokio::test]
async fn a5_fully_clean() {
    // Nothing staged, tracked, or untracked → clean-branch message.
    let tmp = TempRepo::new("a5");
    let repo = tmp.path();
    init_rust_repo(repo);
    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(
        &err,
        GitErrorCode::NothingStaged,
        "nothing to commit: staging area and working tree are clean",
    );
}

#[tokio::test]
async fn a6_untracked_not_staged() {
    // Untracked file without `git add` → NothingStaged (untracked branch).
    let tmp = TempRepo::new("a6");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/new.rs"), "pub fn new() {}\n");
    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_err(
        &err,
        GitErrorCode::NothingStaged,
        "no staged changes; run `git add` or `git rm` first",
    );
}

#[tokio::test]
async fn a7_standard_ok() {
    // Normal repo with a staged change → FromStaging with full repo context.
    let tmp = TempRepo::new("a7");
    let repo_path = tmp.path();
    init_rust_repo(repo_path);
    write(&repo_path.join("src/lib.rs"), C1_LIB_RS);
    git_add(repo_path, &["src/lib.rs"]);

    let (snapshot, _decision, payload) = match run_pipeline(repo_path, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            repo,
            snapshot,
            decision,
            payload,
        } => {
            assert_eq!(
                repo.worktree_root.canonicalize().unwrap(),
                repo_path.canonicalize().unwrap()
            );
            assert_eq!(
                repo.git_dir().canonicalize().unwrap(),
                repo_path.join(".git").canonicalize().unwrap()
            );
            assert_eq!(repo.branch.as_deref(), Some("trunk"));
            assert!(repo.head_oid.is_some());
            assert!(!repo.is_initial_commit());
            assert!(!repo.is_detached_head());
            (snapshot, decision, payload)
        }
        other => panic!("expected staging outcome, got {other:?}"),
    };

    assert_eq!(snapshot.files().len(), 1);
    assert_eq!(payload.file_count, 1);
}

#[tokio::test]
async fn a8_unborn_head() {
    // Staged files but no commits yet → initial commit detected.
    let tmp = TempRepo::new("a8");
    let repo = tmp.path();
    init_git(repo);
    seed_files(repo);
    git_add(repo, &["."]);

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromStaging { repo, .. } => {
            assert_eq!(repo.head_oid, None);
            assert!(repo.is_initial_commit());
            assert_eq!(repo.branch.as_deref(), Some("trunk"));
        }
        other => panic!("expected staging outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn a9_detached_head() {
    // Detached HEAD → branch is None but the pipeline still succeeds.
    let tmp = TempRepo::new("a9");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "--detach"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    41\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromStaging { repo, .. } => {
            assert_eq!(repo.branch, None);
            assert!(repo.is_detached_head());
            assert!(repo.head_oid.is_some());
        }
        other => panic!("expected staging outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn a10_staged_deletion_ok() {
    // A staged deletion alone passes preflight. Per spec, Deleted metadata
    // details belong to stage 2 (C2) — here we only pin Ok(FromStaging).
    let tmp = TempRepo::new("a10");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["rm", "--cached", "README.md"]);

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromStaging { .. } => {}
        other => panic!("expected staging outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn a11_subdir_cwd() {
    // Runs from a subdirectory: worktree root resolves independently of cwd.
    let tmp = TempRepo::new("a11");
    let repo_path = tmp.path();
    init_rust_repo(repo_path);
    write(
        &repo_path.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    41\n}\n",
    );
    git_add(repo_path, &["src/lib.rs"]);

    let subdir = repo_path.join("src");
    match run_pipeline(&subdir, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging { repo, .. } => {
            assert_eq!(
                repo.worktree_root.canonicalize().unwrap(),
                repo_path.canonicalize().unwrap()
            );
            assert_eq!(repo.branch.as_deref(), Some("trunk"));
        }
        other => panic!("expected staging outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn a12_linked_worktree() {
    // Linked worktree: preflight resolves the worktree's own root.
    let tmp = TempRepo::new("a12");
    let main = tmp.path().join("main");
    let wt = tmp.path().join("wt");
    std::fs::create_dir_all(&main).unwrap();
    init_rust_repo(&main);
    git(
        &main,
        &["worktree", "add", "-b", "wt", wt.to_str().unwrap()],
    );
    write(
        &wt.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    41\n}\n",
    );
    git_add(&wt, &["src/lib.rs"]);

    match run_pipeline(&wt, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromStaging { repo, .. } => {
            assert_eq!(
                repo.worktree_root.canonicalize().unwrap(),
                wt.canonicalize().unwrap()
            );
            assert_eq!(repo.branch.as_deref(), Some("wt"));
        }
        other => panic!("expected staging outcome, got {other:?}"),
    }
}

// B. Stage 1 — operation state detection

#[tokio::test]
async fn b1_bisect_abort() {
    // Active bisect aborts even with a clean index.
    let tmp = TempRepo::new("b1");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["bisect", "start"]);

    let state = detect(repo).await;
    assert_eq!(state, OperationState::Bisect);
    assert_eq!(state.action(), OperationAction::Abort);

    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("middle of a git bisect"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn b2_bisect_wins_over_merge() {
    // BISECT_LOG present wins over MERGE_HEAD (bisect is probed first).
    let tmp = TempRepo::new("b2");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(&repo.join("side.rs"), "pub fn side() {}\n");
    git_add(repo, &["side.rs"]);
    git(repo, &["commit", "-m", "feat: side work"]);
    git(repo, &["switch", "trunk"]);
    git(repo, &["bisect", "start"]);
    git(repo, &["merge", "--no-ff", "--no-commit", "side"]);

    let state = detect(repo).await;
    assert_eq!(state, OperationState::Bisect);
    assert_eq!(state.action(), OperationAction::Abort);
}

#[tokio::test]
async fn b3_merge_reuse() {
    // Stalled `git merge --no-ff --no-commit` reuses the native merge message.
    let tmp = TempRepo::new("b3");
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

    let state = detect(repo).await;
    match &state {
        OperationState::Merge { message } => {
            assert!(
                message.as_deref().unwrap().contains("chore: merge feature"),
                "message: {message:?}"
            );
        }
        other => panic!("expected merge state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Reuse);

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::Merge,
            message,
            commit_oid,
            ..
        } => {
            assert!(message.unwrap().contains("chore: merge feature"));
            assert!(commit_oid.is_none());
        }
        other => panic!("expected merge-reuse outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn b3b_merge_reuse_survives_index_reset() {
    // Merge state survives index-only unstaging: `restore --staged` must not
    // touch MERGE_HEAD (plain `git reset` may clear it on some git versions).
    let tmp = TempRepo::new("b3b");
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

    // Unstage WITHOUT aborting the merge.
    git(repo, &["restore", "--staged", "."]);

    // Fixture self-check: turns a confusing "got Clean" into an explicit
    // setup failure if the state file vanished.
    assert!(
        repo.join(".git/MERGE_HEAD").exists(),
        "fixture setup lost merge state"
    );

    let state = detect(repo).await;
    match &state {
        OperationState::Merge { message } => {
            assert!(
                message.as_deref().unwrap().contains("chore: merge feature"),
                "message: {message:?}"
            );
        }
        other => panic!("expected merge state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Reuse);

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::Merge,
            message,
            commit_oid,
            ..
        } => {
            assert!(message.unwrap().contains("chore: merge feature"));
            assert!(commit_oid.is_none());
        }
        other => panic!("expected merge-reuse outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn b4_squash_reuse() {
    // `git merge --squash --no-commit` → SQUASH_MSG reuse.
    let tmp = TempRepo::new("b4");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(&repo.join("side.rs"), "pub fn side() {}\n");
    git_add(repo, &["side.rs"]);
    git(repo, &["commit", "-m", "feat: side work"]);
    git(repo, &["switch", "trunk"]);
    git(repo, &["merge", "--squash", "--no-commit", "side"]);

    let state = detect(repo).await;
    match &state {
        OperationState::Squash { message } => {
            assert!(
                message
                    .as_deref()
                    .unwrap()
                    .contains("Squashed commit of the following"),
                "message: {message:?}"
            );
        }
        other => panic!("expected squash state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Reuse);

    // Pipeline-level pin (spec B4): Reuse arm → FromOperation, no commit oid.
    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::Squash,
            message,
            commit_oid,
            ..
        } => {
            assert!(
                message
                    .as_deref()
                    .unwrap()
                    .contains("Squashed commit of the following"),
                "message: {message:?}"
            );
            assert!(commit_oid.is_none());
        }
        other => panic!("expected squash-reuse outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn b5_rebase_reuse() {
    // Stopped rebase (`--exec false` exit ≠ 0 is expected) → Rebase + Reuse.
    let tmp = TempRepo::new("b5");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "topic"]);
    write(&repo.join("topic.rs"), "pub fn topic() {}\n");
    git_add(repo, &["topic.rs"]);
    git(repo, &["commit", "-m", "feat: topic work"]);
    git(repo, &["switch", "trunk"]);
    write(&repo.join("trunk.rs"), "pub fn trunk() {}\n");
    git_add(repo, &["trunk.rs"]);
    git(repo, &["commit", "-m", "feat: trunk work"]);
    git(repo, &["switch", "topic"]);
    git_fails(repo, &["rebase", "--exec", "false", "trunk"]);

    let state = detect(repo).await;
    match &state {
        OperationState::Rebase { .. } => {} // message deliberately unpinned
        other => panic!("expected rebase state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Reuse);

    // Pipeline-level pin (spec B5): Reuse → FromOperation{Rebase}; the
    // message stays unpinned, Rebase carries no source commit oid.
    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::Rebase,
            commit_oid,
            ..
        } => {
            assert!(commit_oid.is_none());
        }
        other => panic!("expected rebase-reuse outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn b6_cherry_pick_template() {
    // Cherry-pick in progress with conflicts RESOLVED (staged, not committed)
    // → TEMPLATE carrying source oid + subject.
    // A clean-apply pause cannot be constructed portably (`--no-commit` records
    // no state; the sequencer's internal commit ignores the commit-msg hook),
    // so build the classic "conflict → resolved → git add, no commit yet"
    // state: CHERRY_PICK_HEAD present, index fully merged.
    let tmp = TempRepo::new("b6");
    let repo = tmp.path();
    init_rust_repo(repo);

    // Side commit touches a line trunk will also touch.
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: cherry target"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();

    // Diverge trunk on the same line so the pick conflicts.
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);

    // Conflicting pick: writes CHERRY_PICK_HEAD, stops with unmerged entries.
    git_fails(repo, &["cherry-pick", &oid]);

    // Resolve the conflict in the index ONLY: no unmerged entries remain and
    // the sequencer state (CHERRY_PICK_HEAD) stays untouched.
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);

    // Fixture self-checks: state file present, index fully merged.
    assert!(
        repo.join(".git/CHERRY_PICK_HEAD").exists(),
        "resolved cherry-pick did not leave CHERRY_PICK_HEAD"
    );
    assert_eq!(
        git(repo, &["ls-files", "--unmerged"]),
        "",
        "fixture still has unmerged entries"
    );

    let state = detect(repo).await;
    match &state {
        OperationState::CherryPick { head, subject } => {
            assert_eq!(head, &oid);
            assert_eq!(subject.as_deref(), Some("feat: cherry target"));
        }
        other => panic!("expected cherry-pick state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Template);
    assert_eq!(state.seed_message().as_deref(), Some("feat: cherry target"));
    assert_eq!(state.source_commit_oid().as_deref(), Some(oid.as_str()));

    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::CherryPick,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(message.as_deref(), Some("feat: cherry target"));
            assert_eq!(commit_oid.as_deref(), Some(oid.as_str()));
        }
        other => panic!("expected cherry-pick template outcome, got {other:?}"),
    }
}

/// B6b (deliberate hard-fail): with NO unmerged entries, a present but
/// empty CHERRY_PICK_HEAD aborts stage 1 instead of classifying Clean —
/// an undeterminable mid-operation state must not be committed on top
/// of (git would treat the commit as concluding the pick).
#[tokio::test]
async fn b6b_empty_marker_without_conflicts_fails_loudly() {
    let tmp = TempRepo::new("b6b");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    1\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "chore: seed"]);
    // Hand-write the damaged marker: no real pick, clean index.
    write(&repo.join(".git/CHERRY_PICK_HEAD"), "");

    let runner = runner_at(repo);
    let ctx = RepoPreflightCollector::new(&runner).run().await.unwrap();
    let err = OperationStateDetector::new(&runner)
        .run(&ctx)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("present but empty"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn b7_revert_template() {
    // revert --no-commit → TEMPLATE with `Revert "<subject>"` seed.
    let tmp = TempRepo::new("b7");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("second.rs"), "pub fn second() {}\n");
    git_add(repo, &["second.rs"]);
    git(repo, &["commit", "-m", "chore: add second"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["revert", "--no-commit", &oid]);

    let state = detect(repo).await;
    match &state {
        OperationState::Revert { head, subject } => {
            assert_eq!(head, &oid);
            assert_eq!(subject.as_deref(), Some("chore: add second"));
        }
        other => panic!("expected revert state, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Template);
    assert_eq!(
        state.seed_message().as_deref(),
        Some("Revert \"chore: add second\"")
    );

    // Pipeline-level pin (spec B7): Template arm carries the Revert seed
    // message AND the reverted commit oid.
    match run_pipeline(repo, BudgetPolicy::default()).await.unwrap() {
        AssemblyContext::FromOperation {
            operation: Operation::Revert,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(message.as_deref(), Some("Revert \"chore: add second\""));
            assert_eq!(commit_oid.as_deref(), Some(oid.as_str()));
        }
        other => panic!("expected revert-template outcome, got {other:?}"),
    }
}

#[tokio::test]
async fn b8_merge_conflict_abort() {
    // Conflicting merge: unmerged index wins over the MERGE_HEAD reuse path.
    let tmp = TempRepo::new("b8");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: side change"]);
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);
    git_fails(repo, &["merge", "--no-ff", "side"]);

    let state = detect(repo).await;
    match &state {
        OperationState::Conflicts {
            context: Some(Operation::Merge),
        } => {}
        other => panic!("expected merge conflicts, got {other:?}"),
    }
    assert_eq!(state.action(), OperationAction::Abort);

    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("unresolved conflicts during merge"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn b9_cherry_pick_conflict_owns_context() {
    // Default cherry-pick conflict → Conflicts with the owning operation.
    let tmp = TempRepo::new("b9");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: side change"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);
    git_fails(repo, &["cherry-pick", &oid]);

    let state = detect(repo).await;
    match &state {
        OperationState::Conflicts {
            context: Some(Operation::CherryPick),
        } => {}
        other => panic!("expected cherry-pick conflicts, got {other:?}"),
    }

    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("unresolved conflicts during cherry-pick"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn b9b_conflict_without_owner() {
    // `git cherry-pick --quit` (git ≥ 2.26): owner cleared, unmerged index
    // kept → Conflicts{None}, exact message pin.
    let tmp = TempRepo::new("b9b");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: side change"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);
    git_fails(repo, &["cherry-pick", &oid]);
    git(repo, &["cherry-pick", "--quit"]);

    let state = detect(repo).await;
    match &state {
        OperationState::Conflicts { context: None } => {}
        other => panic!("expected ownerless conflicts, got {other:?}"),
    }

    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "[other] unresolved conflicts; resolve them before committing"
    );
}

/// B9c: a conflicted cherry-pick whose CHERRY_PICK_HEAD is unusable
/// (truncated by a crash mid-write) must still classify as Conflicts —
/// the unmerged index is ground truth; the probe failure only costs the
/// owning-operation context.
#[tokio::test]
async fn b9c_conflicted_pick_with_damaged_marker_reports_conflicts() {
    let tmp = TempRepo::new("b9c");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: side change"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);
    git_fails(repo, &["cherry-pick", &oid]);

    // Damage the marker as a crash mid-write would: present but empty.
    write(&repo.join(".git/CHERRY_PICK_HEAD"), "");

    let state = detect(repo).await; // must NOT unwrap_err anymore
    match &state {
        OperationState::Conflicts { context: None } => {}
        other => panic!("expected conflicts with degraded context, got {other:?}"),
    }

    let err = run_pipeline(repo, BudgetPolicy::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("unresolved conflicts"),
        "unexpected: {err}"
    );
}

/// B9d: garbage in CHERRY_PICK_HEAD (damaged but non-empty) — the
/// marker's presence still identifies the owning operation; the
/// unresolvable OID only costs the seed subject.
#[tokio::test]
async fn b9d_conflicted_pick_with_garbage_oid_keeps_context() {
    let tmp = TempRepo::new("b9c");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["switch", "-c", "side"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    SIDE_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: side change"]);
    let oid = git(repo, &["rev-parse", "HEAD"]).trim().to_string();
    git(repo, &["switch", "trunk"]);
    write(
        &repo.join("src/lib.rs"),
        "pub fn seed() -> u32 {\n    TRUNK_VALUE\n}\n",
    );
    git_add(repo, &["src/lib.rs"]);
    git(repo, &["commit", "-m", "feat: trunk change"]);
    git_fails(repo, &["cherry-pick", &oid]);

    // `git_fails(repo, &["cherry-pick", &oid])`
    write(&repo.join(".git/CHERRY_PICK_HEAD"), "not-an-oid\n");

    let state = detect(repo).await;
    match &state {
        OperationState::Conflicts {
            context: Some(Operation::CherryPick),
        } => {}
        other => panic!("expected cherry-pick conflicts, got {other:?}"),
    }
}

#[tokio::test]
async fn b10_clean_continues_to_staging() {
    // No special state + staged change → Clean/Continue → stage 2/3 runs.
    let tmp = TempRepo::new("b10");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/lib.rs"), C1_LIB_RS);
    git_add(repo, &["src/lib.rs"]);

    let state = detect(repo).await;
    assert_eq!(state, OperationState::Clean);
    assert_eq!(state.action(), OperationAction::Continue);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());
    assert_eq!(snapshot.files().len(), 1);
    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert!(!payload.body.is_empty());
}

// C. Stage 2–3 — staged metadata → classify → budget → extract

#[tokio::test]
async fn c1_semantic_full_plus_lock_digest() {
    // A-layer full diff + B-layer digest in one run; lock stays a digest even
    // under Full (changed checksum never leaks).
    let tmp = TempRepo::new("c1");
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

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 2);
    let lib = file_by_path(&snapshot, "src/lib.rs");
    assert_eq!(lib.change_type, ChangeType::Modified);
    assert_eq!(lib.old_path, None);
    assert_eq!((lib.insertions, lib.deletions), (Some(2), Some(1)));
    assert_eq!(lib.category, FileCategory::SemanticText);

    let lock = file_by_path(&snapshot, "Cargo.lock");
    assert_eq!(lock.change_type, ChangeType::Modified);
    assert_eq!(lock.category, FileCategory::DependencyLock);
    // Cross-layer pin: locks are TEXT at stage 2 (counts Some, unlike Binary).
    assert_eq!((lock.insertions, lock.deletions), (Some(2), Some(2)));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.available_for_diff, 126_000);

    let body = &payload.body;
    assert!(body.contains("diff --git a/src/lib.rs"), "body: {body}");
    assert!(
        body.contains("+// probe-C1: integration marker"),
        "body: {body}"
    );
    assert!(body.contains("diff --git a/Cargo.lock"), "body: {body}");
    assert!(body.contains("+version = \"1.2.4\""), "body: {body}");
    assert!(!body.contains("checksum"), "body: {body}");
    assert_eq!(payload.file_count, 2);
    assert_eq!(payload.truncated_file_count, 0);
}

#[tokio::test]
async fn c2_deleted_semantic() {
    // `git rm` on a semantic file: Deleted via HEAD-blob probe + full diff.
    let tmp = TempRepo::new("c2");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["rm", "src/lib.rs"]);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let file = file_by_path(&snapshot, "src/lib.rs");
    assert_eq!(file.change_type, ChangeType::Deleted);
    assert_eq!(file.category, FileCategory::SemanticText);
    let seed_lines = SEED_LIB_RS.lines().count() as u64;
    assert_eq!(
        (file.insertions, file.deletions),
        (Some(0), Some(seed_lines))
    );

    assert_eq!(decision.strategy, DiffStrategy::Full);
    let body = &payload.body;
    assert!(body.contains("diff --git a/src/lib.rs"), "body: {body}");
    assert!(body.contains("+++ /dev/null"), "body: {body}");
    assert!(body.contains("-    40 + 2"), "body: {body}");
    assert_eq!(payload.file_count, 1);
}

#[tokio::test]
async fn c3_pure_rename() {
    // Pure `git mv`: Renamed 100% and BOTH sides of the pathspec included —
    // regression pin for the rename diff header.
    let tmp = TempRepo::new("c3");
    let repo = tmp.path();
    init_rust_repo(repo);
    git(repo, &["mv", "src/lib.rs", "src/lib2.rs"]);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let file = file_by_path(&snapshot, "src/lib2.rs");
    assert_eq!(file.change_type, ChangeType::Renamed);
    assert_eq!(file.old_path.as_deref(), Some(Path::new("src/lib.rs")));
    assert_eq!(file.similarity, Some(100));
    assert_eq!((file.insertions, file.deletions), (Some(0), Some(0)));
    assert_eq!(file.category, FileCategory::SemanticText);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    let body = &payload.body;
    assert!(
        body.contains("diff --git a/src/lib.rs b/src/lib2.rs"),
        "body: {body}"
    );
    assert!(body.contains("similarity index 100%"), "body: {body}");
    assert_eq!(payload.file_count, 1);
}

#[tokio::test]
async fn c4_generated_header_probe() {
    // Generated classification via Phase B header probe (no path/name hints);
    // Generated is TEXT (counts Some) but excluded from extraction entirely.
    let tmp = TempRepo::new("c4");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/pb/messages.rs"), GENERATED_RS);
    git_add(repo, &["src/pb/messages.rs"]);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let file = file_by_path(&snapshot, "src/pb/messages.rs");
    assert_eq!(file.change_type, ChangeType::Added);
    assert_eq!(file.category, FileCategory::Generated);
    let line_count = GENERATED_RS.lines().count() as u64;
    assert_eq!(
        (file.insertions, file.deletions),
        (Some(line_count), Some(0))
    );

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(payload.body, "");
    assert_eq!(payload.file_count, 0);
    assert_eq!(payload.truncated_file_count, 0);
}

#[tokio::test]
async fn c5_budget_zero_respects_categories() {
    // available == 0 forces PathSummaryOnly; C/D stay invisible in the body
    // (downgrade still respects the stage-3.1 classification).
    let tmp = TempRepo::new("c5");
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
    let (snapshot, decision, payload) = staging(run_pipeline(repo, policy).await.unwrap());

    assert_eq!(snapshot.files().len(), 4);
    assert_eq!(
        file_by_path(&snapshot, "src/lib.rs").category,
        FileCategory::SemanticText
    );
    assert_eq!(
        file_by_path(&snapshot, "Cargo.lock").category,
        FileCategory::DependencyLock
    );
    let png = file_by_path(&snapshot, "assets/icon.png");
    assert_eq!(png.category, FileCategory::Binary);
    assert_eq!((png.insertions, png.deletions), (None, None));
    let generated = file_by_path(&snapshot, "src/pb/messages.rs");
    assert_eq!(generated.category, FileCategory::Generated);
    let line_count = GENERATED_RS.lines().count() as u64;
    assert_eq!(
        (generated.insertions, generated.deletions),
        (Some(line_count), Some(0))
    );

    assert_eq!(decision.available_for_diff, 0);
    assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);

    assert_eq!(payload.body, "M  Cargo.lock\nM  src/lib.rs\n"); // status + TWO spaces
    assert!(!payload.body.contains("diff --git"));
    assert!(!payload.body.contains("icon.png"));
    assert!(!payload.body.contains("messages.rs"));
    assert_eq!(payload.file_count, 2);
    assert_eq!(payload.truncated_file_count, 0);
}

#[tokio::test]
async fn c6_over_budget_truncate_lines() {
    // 500 changed lines > per-file cap 400 → TruncateLines with marker.
    let tmp = TempRepo::new("c6");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/big.rs"), &big_lines("// old", 250));
    git_add(repo, &["src/big.rs"]);
    git(repo, &["commit", "-m", "feat: seed big file"]);
    write(&repo.join("src/big.rs"), &big_lines("// new", 250));
    git_add(repo, &["src/big.rs"]);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let file = file_by_path(&snapshot, "src/big.rs");
    assert_eq!(file.change_type, ChangeType::Modified);
    assert_eq!((file.insertions, file.deletions), (Some(250), Some(250)));
    assert_eq!(file.category, FileCategory::SemanticText);

    assert_eq!(
        decision.strategy,
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400
        }
    );
    let body = &payload.body;
    assert!(body.starts_with("diff --git a/src/big.rs"), "body: {body}");
    assert!(body.contains("-// old probe line 250"), "body: {body}");
    assert!(body.contains("+// new probe line 150"), "body: {body}");
    assert!(!body.contains("+// new probe line 151"), "body: {body}");
    assert!(
        body.contains("... [100 more changed lines truncated]"),
        "body: {body}"
    );
    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 1);
}

#[tokio::test]
async fn c7_over_budget_sample_hunks() {
    // available == 4000 → SampleHunks; caps threaded from policy to decision.
    // Note: the single-hunk fixture cannot trigger the hunk cap — the 8/10
    // boundary belongs to the unit suite (`truncate_section`).
    let tmp = TempRepo::new("c7");
    let repo = tmp.path();
    init_rust_repo(repo);
    write(&repo.join("src/big2.rs"), &big_lines("// added", 500));
    git_add(repo, &["src/big2.rs"]);

    let policy = BudgetPolicy {
        context_token_limit: 6_000,
        reserved_tokens: 2_000,
        ..BudgetPolicy::default()
    };
    let (snapshot, decision, payload) = staging(run_pipeline(repo, policy).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let file = file_by_path(&snapshot, "src/big2.rs");
    assert_eq!(file.change_type, ChangeType::Added);
    assert_eq!((file.insertions, file.deletions), (Some(500), Some(0)));
    assert_eq!(file.category, FileCategory::SemanticText);

    assert_eq!(decision.available_for_diff, 4_000);
    assert_eq!(
        decision.strategy,
        DiffStrategy::SampleHunks {
            max_hunks_per_file: 8,
            max_changed_lines_per_file: 400
        }
    );

    let body = &payload.body;
    let hunk_headers = body.lines().filter(|l| l.starts_with("@@")).count();
    assert_eq!(hunk_headers, 1, "body: {body}");
    assert!(body.contains("+// added probe line 400"), "body: {body}");
    assert!(!body.contains("+// added probe line 401"), "body: {body}");
    assert!(
        body.contains("... [100 more changed lines truncated]"),
        "body: {body}"
    );
    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 1);
}

#[tokio::test]
async fn c8_lock_digest_capacity() {
    // Lock digest: 140 signal lines vs capacity 64 → 76 truncated; changed
    // checksum hex (non-signal atoms) never enters the digest.
    let tmp = TempRepo::new("c8");
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

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let lock = file_by_path(&snapshot, "Cargo.lock");
    assert_eq!(lock.category, FileCategory::DependencyLock);
    assert_eq!((lock.insertions, lock.deletions), (Some(80), Some(80)));

    // Source-fact pin: B layer never joins budget math (zero A layer → Full).
    assert_eq!(decision.strategy, DiffStrategy::Full);
    let body = &payload.body;
    assert!(body.contains("diff --git a/Cargo.lock"), "body: {body}");
    assert!(body.contains("+version = \"2.0.0\""), "body: {body}");
    assert!(!body.contains("+version = \"2.32.0\""), "body: {body}");
    assert!(
        body.contains("... [76 more dependency lines truncated]"),
        "body: {body}"
    );
    assert!(!body.contains("checksum"), "body: {body}");
    assert_eq!(payload.file_count, 1);
}

#[tokio::test]
async fn c9_lock_digest_empty_when_no_signal() {
    // Only a checksum line changed: zero signal atoms → digest omitted
    // ENTIRELY (distinct branch from C8's truncation: file_count stays 0).
    let tmp = TempRepo::new("c9");
    let repo = tmp.path();
    init_rust_repo(repo);

    let mut entries = vec![
        LockEntry::new("demo-core", "1.2.3"),
        LockEntry::new("demo-dep", "0.9.0"),
    ];
    entries[1].checksum = fake_checksum("demo-dep@0.9.0-changed");
    write(&repo.join("Cargo.lock"), &lock_text(&entries));
    git_add(repo, &["Cargo.lock"]);

    let (snapshot, decision, payload) =
        staging(run_pipeline(repo, BudgetPolicy::default()).await.unwrap());

    assert_eq!(snapshot.files().len(), 1);
    let lock = file_by_path(&snapshot, "Cargo.lock");
    assert_eq!(lock.category, FileCategory::DependencyLock);
    assert_eq!((lock.insertions, lock.deletions), (Some(1), Some(1)));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(payload.body, "");
    assert_eq!(payload.file_count, 0);
    assert_eq!(payload.truncated_file_count, 0);
}
