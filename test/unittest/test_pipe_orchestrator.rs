use std::path::{Path, PathBuf};

use crate::core::git::types::{
    BudgetPolicy, ChangeType, ClassifiedSnapshot, DiffStrategy, FileCategory, Operation, StagedFile,
};
use crate::core::pipe::context::AssemblyContext;
use crate::core::pipe::orchestrator::PipeOrchestrator;
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

//  helpers

/// RAII empty directory under `std::env::temp_dir()`, removed on drop.
/// Filename unique per test to allow parallel execution.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("autocommit-test-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run a fixture git command; panics on failure so test failures point at
/// the broken fixture step, not the assertion.
async fn git(runner: &GitRunner, args: &[&str]) {
    runner
        .run(args, None)
        .await
        .unwrap_or_else(|e| panic!("fixture `git {}` failed: {e}", args.join(" ")));
}

async fn init_repo(dir: &Path) -> GitRunner {
    let runner = GitRunner::new(Some(dir.to_path_buf()));
    git(&runner, &["init"]).await;
    git(&runner, &["config", "user.email", "test@example.invalid"]).await;
    git(&runner, &["config", "user.name", "AutoCommit Test"]).await;
    runner
}

/// Stage everything and commit with signing disabled.
async fn commit_all(runner: &GitRunner, message: &str) {
    git(&runner, &["add", "-A"]).await;
    git(
        &runner,
        &["-c", "commit.gpgsign=false", "commit", "-m", message],
    )
    .await;
}

fn write_file(dir: &Path, name: &str, content: &[u8]) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

/// Write content into a file inside `.git/`.
fn write_git_marker(dir: &Path, name: &str, content: &str) {
    let path = dir.join(".git").join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

async fn head_oid(runner: &GitRunner) -> String {
    runner
        .run(&["rev-parse", "HEAD"], None)
        .await
        .unwrap()
        .stdout_str()
        .trim()
        .to_owned()
}

async fn run_orchestrator(
    runner: &GitRunner,
    policy: BudgetPolicy,
) -> Result<AssemblyContext, GitError> {
    PipeOrchestrator::new(runner, policy).run().await
}

/// Budget policy with tuned limits, other knobs at defaults.
/// `available_for_diff = limit - reserved`.
fn tuned_policy(limit: u64, reserved: u64, cap: u64) -> BudgetPolicy {
    BudgetPolicy {
        context_token_limit: limit,
        reserved_tokens: reserved,
        max_changed_lines_per_file: cap,
        ..BudgetPolicy::default()
    }
}

fn count_category(files: &[StagedFile], category: FileCategory) -> usize {
    files
        .iter()
        .filter(|file| file.category == category)
        .count()
}

fn sole_file(snapshot: &ClassifiedSnapshot) -> &StagedFile {
    assert_eq!(snapshot.files.len(), 1, "expected exactly one staged file");
    &snapshot.files[0]
}

const PNG_BYTES: &[u8] = &[
    0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0, 0, 0, b'\r', b'I', b'H', b'D', b'R',
];

//  A. stage 0: interception before any stage-1 probing

#[tokio::test]
async fn not_a_repository_is_intercepted() {
    let dir = TempDir::new("orch_not_a_repo").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NotARepository, "got: {err}");
}

#[tokio::test]
async fn bare_repository_is_intercepted() {
    let dir = TempDir::new("orch_bare").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    git(&runner, &["init", "--bare"]).await;

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::Other, "got: {err}");
    assert!(
        err.message.contains("bare repository"),
        "should explain the bare rejection; got: {err}"
    );
}

#[tokio::test]
async fn index_lock_is_intercepted() {
    let dir = TempDir::new("orch_index_lock").unwrap();
    let runner = init_repo(dir.path()).await;
    write_git_marker(dir.path(), "index.lock", "");

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::Other, "got: {err}");
    assert!(
        err.message.contains("index is locked"),
        "should point at the lock file; got: {err}"
    );
}

#[tokio::test]
async fn subdirectory_cwd_resolves_repository_root() {
    // Design pin: stage 0 must resolve the repository root from any cwd
    // inside the work tree (`--show-toplevel` + `--absolute-git-dir`).
    // Everything downstream (stage-1 markers under <root>/.git, index paths
    // relative to the root) depends on this; running auto-commit from a
    // subdirectory must behave identically to running it at the root.
    let dir = TempDir::new("orch_subdir_cwd").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    // Runner rooted at a subdirectory, change staged from there.
    let sub_runner = GitRunner::new(Some(dir.path().join("nested").to_path_buf()));
    write_file(dir.path(), "nested/c.txt", b"two");
    git(&sub_runner, &["add", "c.txt"]).await;

    match run_orchestrator(&sub_runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging { repo, snapshot, .. } => {
            // 0.2 + 0.3: root and git dir resolved from the subdirectory cwd.
            let canonical_root = dir.path().canonicalize().unwrap();
            assert_eq!(repo.worktree_root, canonical_root);
            assert_eq!(repo.git_dir(), &canonical_root.join(".git"));

            // Index paths are relative to the repository root, not the cwd.
            let file = sole_file(&snapshot);
            assert_eq!(file.path, Path::new("nested/c.txt"));
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

//  B. stage 1: abort states never reach the LLM flow

#[tokio::test]
async fn bisect_state_aborts_with_guidance() {
    let dir = TempDir::new("orch_bisect").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_git_marker(dir.path(), "BISECT_LOG", "git bisect start\n");

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert!(err.message.contains("bisect"), "got: {err}");
}

#[tokio::test]
async fn merge_conflict_aborts_naming_the_operation() {
    let dir = TempDir::new("orch_merge_conflict").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "f.txt", b"base");
    commit_all(&runner, "base").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    write_file(dir.path(), "f.txt", b"feature");
    commit_all(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    write_file(dir.path(), "f.txt", b"main");
    commit_all(&runner, "main change").await;

    // Conflict: exit 1 is expected.
    let _ = runner.run(&["merge", "feature"], None).await;

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert!(
        err.message.contains("unresolved conflicts during merge"),
        "should name the owning operation; got: {err}"
    );
}

#[tokio::test]
async fn conflicts_without_operation_marker_abort_generically() {
    // Edge: unmerged index without MERGE_HEAD (e.g. a marker was cleaned up
    // externally) must still abort, with no "during <op>" suffix.
    let dir = TempDir::new("orch_conflicts_orphan").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "f.txt", b"base");
    commit_all(&runner, "base").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    write_file(dir.path(), "f.txt", b"feature");
    commit_all(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    write_file(dir.path(), "f.txt", b"main");
    commit_all(&runner, "main change").await;

    let _ = runner.run(&["merge", "feature"], None).await;
    std::fs::remove_file(dir.path().join(".git").join("MERGE_HEAD")).unwrap();

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert!(err.message.contains("unresolved conflicts"), "got: {err}");
    assert!(
        !err.message.contains(" during "),
        "no owning operation known; got: {err}"
    );
}

//  C. dedicated operation branch: stages 2–3 are bypassed

#[tokio::test]
async fn merge_in_progress_routes_to_operation_seed() {
    let dir = TempDir::new("orch_merge_seed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    write_file(dir.path(), "b.txt", b"feat");
    commit_all(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    git(&runner, &["merge", "--no-commit", "--no-ff", "feature"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(operation, Operation::Merge);
            assert_eq!(message.as_deref(), Some("Merge branch 'feature'"));
            assert_eq!(commit_oid, None, "merge carries no source OID");
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn merge_with_empty_staging_still_routes_to_operation() {
    // Design pin (regression, bug 1): a merge whose result tree equals HEAD
    // has NOTHING staged. The staged-changes check belongs to the regular
    // branch only; the dedicated branch must pass through.
    let dir = TempDir::new("orch_merge_empty_staging").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    let oid = head_oid(&runner).await;

    // Simulate "merge stopped, result tree == HEAD": markers present,
    // index matches HEAD.
    write_git_marker(dir.path(), "MERGE_HEAD", &oid);
    write_git_marker(dir.path(), "MERGE_MSG", "Merge branch 'feature'");

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation, message, ..
        } => {
            assert_eq!(operation, Operation::Merge);
            assert_eq!(message.as_deref(), Some("Merge branch 'feature'"));
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn rebase_routes_to_operation_seed_stripping_comments() {
    let dir = TempDir::new("orch_rebase_seed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    write_git_marker(
        dir.path(),
        "rebase-merge/message",
        "Pick: real message\n# comment line\n",
    );

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(operation, Operation::Rebase);
            assert_eq!(message.as_deref(), Some("Pick: real message"));
            assert_eq!(commit_oid, None);
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn rebase_without_seed_yields_none_message() {
    // Empty-value pin: bare rebase-merge/ directory, no message, no
    // REBASE_HEAD — the seed is legitimately absent, not an error.
    let dir = TempDir::new("orch_rebase_no_seed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_git_marker(dir.path(), "rebase-merge/.keep", "");

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation, message, ..
        } => {
            assert_eq!(operation, Operation::Rebase);
            assert_eq!(message, None);
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn squash_routes_to_operation_seed() {
    let dir = TempDir::new("orch_squash_seed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    // SQUASH_MSG without MERGE_HEAD → squash.
    write_git_marker(
        dir.path(),
        "SQUASH_MSG",
        "Squash: combined changes\n# comment\n",
    );

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(operation, Operation::Squash);
            assert_eq!(message.as_deref(), Some("Squash: combined changes"));
            assert_eq!(commit_oid, None);
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_cherry_pick_routes_to_operation_with_oid() {
    // Design pin (regression, bug 1 + bug 2): cherry-picking an empty commit
    // leaves CHERRY_PICK_HEAD with an UNTOUCHED index. The orchestrator must
    // route to the dedicated branch (not NothingStaged) AND carry the source
    // OID into AssemblyContext.
    let dir = TempDir::new("orch_cherry_pick_empty").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    git(
        &runner,
        &[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "empty source",
        ],
    )
    .await;
    let oid = head_oid(&runner).await;
    git(&runner, &["reset", "--hard", "HEAD~1"]).await;

    // Empty cherry-pick stops with exit 1 and keeps CHERRY_PICK_HEAD.
    let _ = runner.run(&["cherry-pick", &oid], None).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(operation, Operation::CherryPick);
            assert_eq!(message.as_deref(), Some("empty source"));
            assert_eq!(
                commit_oid.as_deref(),
                Some(oid.as_str()),
                "source OID must survive into AssemblyContext"
            );
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

#[tokio::test]
async fn revert_no_commit_routes_to_operation_with_oid() {
    let dir = TempDir::new("orch_revert").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "the reverted commit").await;
    let oid = head_oid(&runner).await;

    git(&runner, &["revert", "--no-commit", "HEAD"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromOperation {
            operation,
            message,
            commit_oid,
            ..
        } => {
            assert_eq!(operation, Operation::Revert);
            // seed_message applies the git "Revert \"...\"" convention.
            assert_eq!(
                message.as_deref(),
                Some("Revert \"the reverted commit\""),
                "got: {message:?}"
            );
            assert_eq!(commit_oid.as_deref(), Some(oid.as_str()));
        }
        other => panic!("expected FromOperation, got {other:?}"),
    }
}

//  D. regular branch: stages 2–3 → FromStaging

#[tokio::test]
async fn staged_modification_produces_full_staging_context() {
    let dir = TempDir::new("orch_regular_modification").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "a.txt", b"two");
    git(&runner, &["add", "a.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            repo,
            snapshot,
            payload,
            decision,
        } => {
            // repo passthrough: worktree + git dir survive into the assembly context
            let canonical_root = dir.path().canonicalize().unwrap();
            assert_eq!(repo.worktree_root, canonical_root);
            assert_eq!(repo.git_dir(), &canonical_root.join(".git"));
            assert!(repo.branch.as_deref().is_some());
            assert!(repo.head_oid.is_some());

            // classification
            let file = sole_file(&snapshot);
            assert_eq!(file.category, FileCategory::SemanticText);
            assert_eq!(file.change_type, ChangeType::Modified);

            // budget passthrough (default policy: 128k − 2k)
            assert_eq!(decision.strategy, DiffStrategy::Full);
            assert_eq!(decision.available_for_diff, 126_000);
            assert!(decision.estimated_diff_tokens > 0);

            // payload
            assert!(payload.body.contains("diff --git a/a.txt"));
            assert!(payload.body.contains("+two"));
            assert_eq!(payload.file_count, 1);
            assert_eq!(payload.truncated_file_count, 0);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn initial_commit_is_supported() {
    // Edge: unborn HEAD with the very first change staged is auto-commit's
    // primary use case; `git diff --cached` vs the empty tree needs git>=2.22.
    let dir = TempDir::new("orch_initial_commit").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    git(&runner, &["add", "a.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            repo,
            snapshot,
            payload,
            ..
        } => {
            assert_eq!(repo.head_oid, None, "unborn HEAD");
            let file = sole_file(&snapshot);
            assert_eq!(file.change_type, ChangeType::Added);
            assert!(payload.body.contains("diff --git a/a.txt"));
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn detached_head_keeps_branch_none() {
    let dir = TempDir::new("orch_detached").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    git(&runner, &["checkout", "--detach"]).await;
    write_file(dir.path(), "b.txt", b"two");
    git(&runner, &["add", "b.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging { repo, .. } => {
            assert_eq!(repo.branch, None);
            assert!(repo.is_detached_head());
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_rename_is_captured_with_zero_line_counts() {
    // Zero-value pin: a 100% rename has Some(0) line counts (not None),
    // stays cheap in the budget, and renders as a rename header.
    let dir = TempDir::new("orch_rename").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    git(&runner, &["mv", "a.txt", "b.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot,
            payload,
            decision,
            ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.change_type, ChangeType::Renamed);
            assert_eq!(file.old_path.as_deref(), Some("a.txt".as_ref()));
            assert_eq!(file.similarity, Some(100));
            assert_eq!(file.insertions, Some(0));
            assert_eq!(file.deletions, Some(0));
            assert_eq!(decision.strategy, DiffStrategy::Full);
            assert!(payload.body.contains("rename from a.txt"));
            assert!(payload.body.contains("rename to b.txt"));
            assert_eq!(payload.file_count, 1);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_binary_is_metadata_only_with_empty_payload() {
    // Empty-value pin: a binary-only staging produces a well-formed but
    // EMPTY diff body — stage 4.2 must handle this, not crash.
    let dir = TempDir::new("orch_binary").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "img.png", PNG_BYTES);
    git(&runner, &["add", "img.png"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.category, FileCategory::Binary);
            assert_eq!(file.insertions, None, "binary carries no line counts");
            assert!(payload.body.is_empty(), "body: {:?}", payload.body);
            assert_eq!(payload.file_count, 0);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_lock_file_is_summarized_not_inlined() {
    // Lock files render as a signal-line digest: version/name atoms are
    // kept, noise lines are dropped.
    let dir = TempDir::new("orch_lock_digest").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "Cargo.lock", b"version = 3\nnoise = aaa\n");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "Cargo.lock", b"version = 4\nnoise = bbb\n");
    git(&runner, &["add", "Cargo.lock"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.category, FileCategory::DependencyLock);
            assert!(
                payload.body.contains("+version = 4"),
                "signal line must survive; body: {}",
                payload.body
            );
            assert!(
                !payload.body.contains("+noise = bbb"),
                "noise line must be dropped; body: {}",
                payload.body
            );
            assert!(payload.body.contains("Cargo.lock"));
            assert_eq!(payload.file_count, 1);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_submodule_pointer_is_metadata_only() {
    // Design pin: a gitlink (mode 160000) must classify as Submodule and
    // stay out of the diff body — otherwise `git diff` emits
    // `-Subproject commit xxx` lines that would leak into the prompt.
    // Local-path submodule fixtures need `-c protocol.file.allow=always`
    // on the command line (git >= 2.38.1): the spawned clone reads the
    // config of the repo it is creating, which does not exist yet, so a
    // repo-local setting in the outer repo is invisible to it.
    let dir = TempDir::new("orch_submodule").unwrap();
    let runner = init_repo(dir.path()).await;

    let inner_dir = TempDir::new("orch_submodule_inner").unwrap();
    let inner_runner = init_repo(inner_dir.path()).await;
    write_file(inner_dir.path(), "f.txt", b"one");
    commit_all(&inner_runner, "inner init").await;

    git(
        &runner,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            inner_dir.path().to_str().unwrap(),
            "dep",
        ],
    )
    .await;
    commit_all(&runner, "add submodule").await;

    // Advance the inner repo, then stage the updated pointer.
    write_file(inner_dir.path(), "f.txt", b"two");
    commit_all(&inner_runner, "inner change").await;
    git(
        &runner,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--remote",
            "dep",
        ],
    )
    .await;
    git(&runner, &["add", "dep"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot,
            payload,
            decision,
            ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.category, FileCategory::Submodule);
            assert_eq!(file.change_type, ChangeType::Modified);
            assert_eq!(file.insertions, None, "gitlinks carry no line counts");
            assert_eq!(file.deletions, None);
            assert!(
                payload.body.is_empty(),
                "submodule pointer must stay out of the body; got: {}",
                payload.body
            );
            assert_eq!(payload.file_count, 0);
            assert_eq!(decision.strategy, DiffStrategy::Full);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn mixed_categories_route_into_their_lanes() {
    let dir = TempDir::new("orch_mixed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    write_file(dir.path(), "Cargo.lock", b"version = 3\n");
    write_file(dir.path(), "generated/x.pb.rs", b"one");
    commit_all(&runner, "base").await;

    write_file(dir.path(), "a.txt", b"two");
    write_file(dir.path(), "Cargo.lock", b"version = 4\n");
    write_file(dir.path(), "generated/x.pb.rs", b"two");
    write_file(dir.path(), "img.png", PNG_BYTES);
    git(&runner, &["add", "-A"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            assert_eq!(snapshot.files.len(), 4);
            assert_eq!(
                count_category(&snapshot.files, FileCategory::SemanticText),
                1
            );
            assert_eq!(
                count_category(&snapshot.files, FileCategory::DependencyLock),
                1
            );
            assert_eq!(count_category(&snapshot.files, FileCategory::Binary), 1);
            assert_eq!(count_category(&snapshot.files, FileCategory::Generated), 1);

            assert!(payload.body.contains("diff --git a/a.txt"));
            assert!(payload.body.contains("+version = 4"));
            assert!(
                !payload.body.contains("img.png"),
                "binary must stay out of the body"
            );
            assert!(
                !payload.body.contains("x.pb.rs"),
                "generated must stay out of the body"
            );
            // semantic diff + lock digest sections only
            assert_eq!(payload.file_count, 2);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

//  E. budget strategies: policy wiring into stages 3.2 / 3.3

/// 1 base line → 100 added lines: 101 changed lines, estimated ≈ 1375
/// tokens (48 overhead + 1010 line cost, ×1.3 safety).
async fn stage_large_change(runner: &GitRunner, dir: &Path) {
    let mut content = String::new();
    for i in 0..100 {
        content.push_str(&format!("l{i}\n"));
    }
    write_file(dir, "a.txt", content.as_bytes());
    git(&runner, &["add", "a.txt"]).await;
}

#[tokio::test]
async fn over_budget_diff_is_truncated_per_file() {
    // available = 500 < estimated 1375; capped estimate (48 + 30×10) × 1.3
    // = 452 ≤ 500 → TruncateLines with the configured cap.
    let dir = TempDir::new("orch_truncate").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"base\n");
    commit_all(&runner, "init").await;
    stage_large_change(&runner, dir.path()).await;

    match run_orchestrator(&runner, tuned_policy(1_000, 500, 30))
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            payload, decision, ..
        } => {
            assert_eq!(
                decision.strategy,
                DiffStrategy::TruncateLines {
                    max_changed_lines_per_file: 30
                }
            );
            assert_eq!(payload.truncated_file_count, 1);
            assert!(payload.body.contains("more changed lines truncated"));
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn starved_budget_falls_back_to_path_summary() {
    // available = 1; capped estimate 452 > 1 × 3 → PathSummaryOnly.
    let dir = TempDir::new("orch_starved").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"base\n");
    commit_all(&runner, "init").await;
    stage_large_change(&runner, dir.path()).await;

    match run_orchestrator(&runner, tuned_policy(100, 99, 30))
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            payload, decision, ..
        } => {
            assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
            assert!(payload.body.contains("M  a.txt"));
            assert!(!payload.body.contains("@@"), "no hunks in path summary");
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn zero_available_budget_short_circuits_to_path_summary() {
    // Zero-value pin: limit == reserved → available == 0 is an explicit
    // short-circuit in the planner.
    let dir = TempDir::new("orch_zero_budget").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"base\n");
    commit_all(&runner, "init").await;
    stage_large_change(&runner, dir.path()).await;

    match run_orchestrator(&runner, tuned_policy(50, 50, 30))
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging { decision, .. } => {
            assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn middling_budget_samples_hunks_per_file() {
    // available = 200: estimated 1375 > 200 (not Full); capped 452 > 200
    // (not TruncateLines); 452 ≤ 200 × 3 = 600 (not PathSummaryOnly)
    // → SampleHunks with the configured hunk/line caps.
    let dir = TempDir::new("orch_sample_hunks").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"base\n");
    commit_all(&runner, "init").await;
    stage_large_change(&runner, dir.path()).await;

    match run_orchestrator(&runner, tuned_policy(700, 500, 30))
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            payload, decision, ..
        } => {
            assert_eq!(
                decision.strategy,
                DiffStrategy::SampleHunks {
                    max_hunks_per_file: 8,
                    max_changed_lines_per_file: 30
                }
            );
            assert_eq!(payload.truncated_file_count, 1);
            assert!(payload.body.contains("more changed lines truncated"));
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn multi_file_over_budget_truncates_each_and_counts_both() {
    // Two large files: estimate ≈ 2750 > available 1000; capped estimate
    // ≈ 904 ≤ 1000 → TruncateLines, per-file counts must both be set.
    let dir = TempDir::new("orch_multi_truncate").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"base\n");
    write_file(dir.path(), "b.txt", b"base\n");
    commit_all(&runner, "init").await;

    let mut content = String::new();
    for i in 0..100 {
        content.push_str(&format!("l{i}\n"));
    }
    write_file(dir.path(), "a.txt", content.as_bytes());
    write_file(dir.path(), "b.txt", content.as_bytes());
    git(&runner, &["add", "-A"]).await;

    match run_orchestrator(&runner, tuned_policy(1_500, 500, 30))
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            payload, decision, ..
        } => {
            assert_eq!(
                decision.strategy,
                DiffStrategy::TruncateLines {
                    max_changed_lines_per_file: 30
                }
            );
            assert_eq!(payload.file_count, 2);
            assert_eq!(payload.truncated_file_count, 2);
            assert_eq!(payload.body.matches("diff --git ").count(), 2);
            assert_eq!(
                payload.body.matches("more changed lines truncated").count(),
                2
            );
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

//  F. empty staging on the regular branch (ensure_staged_changes wiring)

#[tokio::test]
async fn clean_tree_is_rejected_as_nothing_to_commit() {
    let dir = TempDir::new("orch_clean_tree").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(err.message.contains("nothing to commit"), "got: {err}");
}

#[tokio::test]
async fn unstaged_modification_is_rejected_with_add_hint() {
    let dir = TempDir::new("orch_unstaged_mod").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "a.txt", b"two");

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(err.message.contains("git add"), "got: {err}");
}

#[tokio::test]
async fn untracked_file_is_rejected_with_add_hint() {
    let dir = TempDir::new("orch_untracked").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "new.txt", b"fresh");

    let err = run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(err.message.contains("git add"), "got: {err}");
}

//  G. path & change-type edges

#[tokio::test]
async fn path_with_spaces_flows_through_all_stages() {
    let dir = TempDir::new("orch_spaces").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "my file.txt", b"one");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "my file.txt", b"two");
    git(&runner, &["add", "my file.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            assert_eq!(snapshot.files[0].path, Path::new("my file.txt"));
            assert!(
                payload.body.contains("my file.txt"),
                "path must survive into the diff body"
            );
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn type_changed_file_flows_through_as_semantic() {
    // file → symlink: T status; the classifier probes the index blob via
    // the TypeChanged `:path` arm, and the diff renders mode lines.
    let dir = TempDir::new("orch_type_changed").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "f.txt", b"one");
    commit_all(&runner, "init").await;

    std::fs::remove_file(dir.path().join("f.txt")).unwrap();
    std::os::unix::fs::symlink("target", dir.path().join("f.txt")).unwrap();
    git(&runner, &["add", "f.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.change_type, ChangeType::TypeChanged);
            assert_eq!(file.category, FileCategory::SemanticText);
            assert!(
                payload.body.contains("new file mode 120000"),
                "body: {}",
                payload.body
            );
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn deleted_file_is_classified_from_head_blob() {
    // Edge: deleted paths are probed via `HEAD:<path>` in the classifier
    // blob_spec branch; the diff body must contain the deletion header.
    let dir = TempDir::new("orch_deleted").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    git(&runner, &["rm", "a.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot, payload, ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.change_type, ChangeType::Deleted);
            assert_eq!(file.category, FileCategory::SemanticText);
            assert!(payload.body.contains("deleted file mode"));
            assert!(payload.body.contains("-one"));
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_empty_file_yields_full_strategy_with_no_hunks() {
    // Zero-value pin: 0 bytes / 0 lines. numstat 0/0, estimate 0 → Full;
    // the diff section carries the file header but no hunks.
    let dir = TempDir::new("orch_empty_file").unwrap();
    let runner = init_repo(dir.path()).await;
    write_file(dir.path(), "a.txt", b"one");
    commit_all(&runner, "init").await;
    write_file(dir.path(), "empty.txt", b"");
    git(&runner, &["add", "empty.txt"]).await;

    match run_orchestrator(&runner, BudgetPolicy::default())
        .await
        .unwrap()
    {
        AssemblyContext::FromStaging {
            snapshot,
            payload,
            decision,
            ..
        } => {
            let file = sole_file(&snapshot);
            assert_eq!(file.change_type, ChangeType::Added);
            assert_eq!(file.category, FileCategory::SemanticText);
            assert_eq!(file.insertions, Some(0));
            assert_eq!(file.deletions, Some(0));
            assert_eq!(decision.strategy, DiffStrategy::Full);
            assert!(payload.body.contains("diff --git a/empty.txt"));
            assert!(!payload.body.contains("@@"), "no hunks for an empty file");
            assert_eq!(payload.file_count, 1);
        }
        other => panic!("expected FromStaging, got {other:?}"),
    }
}
