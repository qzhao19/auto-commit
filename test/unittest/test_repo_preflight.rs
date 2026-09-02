use std::path::PathBuf;

use crate::core::git::preflight::RepoPreflightCollector;
use crate::infra::git::GitRunner;
use crate::shared::exception::GitErrorCode;

//  helpers

/// RAII empty directory under `std::env::temp_dir()`, removed on drop.
/// Filename unique per test to allow parallel execution.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("autocommit-test-{name}"));
        // Best-effort: a previous run may have crashed and left the dir.
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run a fixture git command inside `dir`; panics with the git error on
/// failure so test failures point at the broken fixture step.
async fn git(runner: &GitRunner, args: &[&str]) {
    runner
        .run(args, None)
        .await
        .unwrap_or_else(|e| panic!("fixture `git {}` failed: {e}", args.join(" ")));
}

/// `git init` + repo-local identity; returns a runner rooted at the repo.
async fn init_repo(dir: &std::path::Path) -> GitRunner {
    let runner = GitRunner::new(Some(dir.to_path_buf()));
    git(&runner, &["init"]).await;
    git(&runner, &["config", "user.email", "test@example.invalid"]).await;
    git(&runner, &["config", "user.name", "AutoCommit Test"]).await;
    runner
}

/// Create a commit with gpg signing disabled (CI machines may enable it
/// globally without a usable key).
async fn commit(runner: &GitRunner, message: &str) {
    git(
        runner,
        &["-c", "commit.gpgsign=false", "commit", "-m", message],
    )
    .await;
}

//  0.1 discover repository

#[tokio::test]
async fn not_a_repository_maps_to_not_a_repository_code() {
    // Design pin: the discovery error must be reclassified from the
    // runner's generic CommandFailed to NotARepository, so the CLI layer
    // can route it ("run inside a git repo") instead of parsing messages.
    let dir = TempDir::new("preflight_not_a_repository").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));

    let err = RepoPreflightCollector::new(&runner)
        .run()
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NotARepository, "got: {err}");
}

#[tokio::test]
async fn bare_repository_is_rejected() {
    let dir = TempDir::new("preflight_bare_rejected").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    git(&runner, &["init", "--bare"]).await;

    let err = RepoPreflightCollector::new(&runner)
        .run()
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::Other, "got: {err}");
    assert!(
        err.message.contains("bare repository"),
        "should explain the bare rejection; got: {err}"
    );
}

//  0.4 index.lock

#[tokio::test]
async fn index_lock_blocks_preflight() {
    // The lock check runs before HEAD/branch/staged probes, so a plain
    // repo plus a lock file is enough to trigger it.
    let dir = TempDir::new("preflight_index_lock").unwrap();
    let runner = init_repo(dir.path()).await;

    let lock = dir.path().join(".git").join("index.lock");
    std::fs::write(&lock, b"").unwrap();

    let err = RepoPreflightCollector::new(&runner)
        .run()
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::Other, "got: {err}");
    assert!(
        err.message.contains("index is locked"),
        "should point at the lock file; got: {err}"
    );
}

//  0.7 empty staging variants

#[tokio::test]
async fn clean_worktree_reports_nothing_to_commit() {
    let dir = TempDir::new("preflight_clean_worktree").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    let preflight = RepoPreflightCollector::new(&runner);

    // Design pin: 0.7 moved out of run() — a clean worktree no longer
    // blocks preflight; the dedicated operation branch depends on this.
    preflight
        .run()
        .await
        .unwrap_or_else(|e| panic!("run() must succeed on a clean worktree, got: {e}"));

    let err = preflight.ensure_staged_changes().await.unwrap_err();
    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(err.message.contains("nothing to commit"), "got: {err}");
}

#[tokio::test]
async fn unstaged_modification_prompts_git_add() {
    let dir = TempDir::new("preflight_unstaged_modification").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    // Tracked file modified but not staged.
    std::fs::write(dir.path().join("a.txt"), b"two").unwrap();

    let err = RepoPreflightCollector::new(&runner)
        .ensure_staged_changes()
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(err.message.contains("git add"), "got: {err}");
}

#[tokio::test]
async fn untracked_file_prompts_git_add() {
    // Design pin: `git diff --quiet` cannot see untracked files, so the
    // preflight runs a dedicated `ls-files --others` probe. Without it,
    // this scenario would misleadingly report "nothing to commit" on a
    // clean tree.
    let dir = TempDir::new("preflight_untracked_file").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    // Untracked only: no tracked modifications.
    std::fs::write(dir.path().join("new.txt"), b"fresh").unwrap();

    let err = RepoPreflightCollector::new(&runner)
        .ensure_staged_changes()
        .await
        .unwrap_err();

    assert_eq!(err.code, GitErrorCode::NothingStaged, "got: {err}");
    assert!(
        err.message.contains("git add"),
        "untracked file must route to the `git add` hint; got: {err}"
    );
}

//  0.5–0.8 context assembly

#[tokio::test]
async fn success_returns_full_context() {
    let dir = TempDir::new("preflight_success_context").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    std::fs::write(dir.path().join("b.txt"), b"two").unwrap();
    git(&runner, &["add", "b.txt"]).await;

    let expected_head = runner.run(&["rev-parse", "HEAD"], None).await.unwrap();

    let ctx = RepoPreflightCollector::new(&runner).run().await.unwrap();

    // git resolves symlinks; compare against the canonicalized fixture.
    assert_eq!(ctx.worktree_root, dir.path().canonicalize().unwrap());
    assert_eq!(
        ctx.git_dir(),
        &dir.path().canonicalize().unwrap().join(".git")
    );
    assert_eq!(
        ctx.head_oid.as_deref(),
        Some(expected_head.stdout_str().trim())
    );
    assert!(ctx.branch.as_deref().is_some_and(|b| !b.is_empty()));
    assert!(!ctx.is_initial_commit());
    assert!(!ctx.is_detached_head());
}

#[tokio::test]
async fn unborn_head_reports_initial_commit() {
    // Fresh repo, first change staged, no commit yet: the preflight must
    // succeed (auto-commit's primary use case) and report an unborn HEAD
    // while still resolving the branch name (symref exists pre-commit).
    // `git diff --cached` vs the empty tree requires git >= 2.22.
    let dir = TempDir::new("preflight_unborn_head").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;

    let ctx = RepoPreflightCollector::new(&runner).run().await.unwrap();

    assert_eq!(ctx.head_oid, None);
    assert!(ctx.is_initial_commit());
    // `branch --show-current` reads the HEAD symref, which exists even
    // before the first commit — so an unborn repo is NOT detached.
    assert!(
        ctx.branch.as_deref().is_some_and(|b| !b.is_empty()),
        "unborn branch must still be reported; got: {:?}",
        ctx.branch
    );
    assert!(!ctx.is_detached_head());
}

#[tokio::test]
async fn detached_head_context_has_no_branch() {
    let dir = TempDir::new("preflight_detached_head").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    git(&runner, &["checkout", "--detach"]).await;

    let expected_head = runner.run(&["rev-parse", "HEAD"], None).await.unwrap();

    std::fs::write(dir.path().join("b.txt"), b"two").unwrap();
    git(&runner, &["add", "b.txt"]).await;

    let ctx = RepoPreflightCollector::new(&runner).run().await.unwrap();

    assert_eq!(ctx.branch, None);
    assert!(ctx.is_detached_head());
    assert_eq!(
        ctx.head_oid.as_deref(),
        Some(expected_head.stdout_str().trim())
    );
}
