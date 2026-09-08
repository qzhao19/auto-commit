use std::path::{Path, PathBuf};

use crate::core::git::operation::OperationStateDetector;
use crate::core::git::types::{GitPaths, Operation, OperationState, RepositoryContext};
use crate::infra::git::GitRunner;

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

async fn commit(runner: &GitRunner, message: &str) {
    git(
        runner,
        &["-c", "commit.gpgsign=false", "commit", "-am", message],
    )
    .await;
}

/// Build a `RepositoryContext` rooted at `dir/.git`.
/// Only `git_paths` is consumed by the detector; other fields are placeholder.
fn make_ctx(dir: &Path) -> RepositoryContext {
    let git_dir = dir.join(".git");
    RepositoryContext {
        worktree_root: dir.to_path_buf(),
        git_paths: GitPaths::from_git_dir(git_dir),
        head_oid: None,
        branch: None,
    }
}

/// Write content into a file inside `.git/`.
fn write_marker(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(".git").join(name), content).unwrap();
}

/// Create a directory inside `.git/`.
fn create_git_subdir(dir: &Path, name: &str) {
    std::fs::create_dir(dir.join(".git").join(name)).unwrap();
}

/// OID of HEAD via the runner.
async fn head_oid(runner: &GitRunner) -> String {
    runner
        .run(&["rev-parse", "HEAD"], None)
        .await
        .unwrap()
        .stdout_str()
        .trim()
        .to_owned()
}

//  1.8 clean

#[tokio::test]
async fn clean_repo_yields_Clean() {
    let dir = TempDir::new("ops_clean").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(state, OperationState::Clean);
}

//  1.1 conflicts

#[tokio::test]
async fn merge_conflict_yields_Conflicts() {
    let dir = TempDir::new("ops_merge_conflict").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("f.txt"), b"base").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "base").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    std::fs::write(dir.path().join("f.txt"), b"feature").unwrap();
    commit(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    std::fs::write(dir.path().join("f.txt"), b"main").unwrap();
    commit(&runner, "main change").await;

    // merge → conflict (non-zero exit is expected here).
    let _ = runner.run(&["merge", "feature"], None).await;

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    // merge_conflict_yields_Conflicts 内
    assert_eq!(
        state,
        OperationState::Conflicts {
            context: Some(Operation::Merge),
        }
    );
}

//  1.2 bisect

#[tokio::test]
async fn bisect_log_yields_Bisect() {
    let dir = TempDir::new("ops_bisect").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    write_marker(dir.path(), "BISECT_LOG", "git bisect start\n");

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(state, OperationState::Bisect);
}

//  1.3 rebase

#[tokio::test]
async fn rebase_merge_dir_yields_Rebase_with_none_message() {
    let dir = TempDir::new("ops_rebase_merge_dir").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    create_git_subdir(dir.path(), "rebase-merge");

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(state, OperationState::Rebase { message: None });
}

#[tokio::test]
async fn rebase_apply_dir_yields_Rebase_with_none_message() {
    let dir = TempDir::new("ops_rebase_apply_dir").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    create_git_subdir(dir.path(), "rebase-apply");

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(state, OperationState::Rebase { message: None });
}

#[tokio::test]
async fn rebase_seed_from_message_file_strips_comments() {
    let dir = TempDir::new("ops_rebase_seed_msg").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    create_git_subdir(dir.path(), "rebase-merge");
    write_marker(
        dir.path(),
        "rebase-merge/message",
        "Pick: real message\n# comment line\n# another comment\n",
    );

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Rebase {
            message: Some("Pick: real message".to_owned())
        }
    );
}

#[tokio::test]
async fn rebase_seed_falls_back_to_rebase_head_subject() {
    let dir = TempDir::new("ops_rebase_seed_fallback").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "the rebase commit").await;
    let oid = head_oid(&runner).await;

    create_git_subdir(dir.path(), "rebase-merge");
    // No message file, but REBASE_HEAD points at a real commit.
    write_marker(dir.path(), "REBASE_HEAD", &oid);

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Rebase {
            message: Some("the rebase commit".to_owned())
        }
    );
}

//  1.6 merge

#[tokio::test]
async fn merge_head_yields_Merge_with_stripped_message() {
    let dir = TempDir::new("ops_merge").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    let oid = head_oid(&runner).await;

    write_marker(dir.path(), "MERGE_HEAD", &oid);
    write_marker(
        dir.path(),
        "MERGE_MSG",
        "Merge branch 'feature'\n# Conflicts:\n#   f.txt\n",
    );

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Merge {
            message: Some("Merge branch 'feature'".to_owned())
        }
    );
}

#[tokio::test]
async fn merge_msg_only_comments_yields_none_message() {
    let dir = TempDir::new("ops_merge_comments_only").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    let oid = head_oid(&runner).await;

    write_marker(dir.path(), "MERGE_HEAD", &oid);
    write_marker(
        dir.path(),
        "MERGE_MSG",
        "# Conflicts:\n#   f.txt\n# Please enter the commit message\n",
    );

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(state, OperationState::Merge { message: None });
}

//  1.7 squash

#[tokio::test]
async fn squash_msg_yields_Squash_with_message() {
    let dir = TempDir::new("ops_squash").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    // SQUASH_MSG without MERGE_HEAD → squash state.
    write_marker(
        dir.path(),
        "SQUASH_MSG",
        "Squash: combined changes\n# comment\n",
    );

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Squash {
            message: Some("Squash: combined changes".to_owned())
        }
    );
}

//  1.4 cherry-pick

#[tokio::test]
async fn cherry_pick_head_yields_CherryPick_with_subject() {
    let dir = TempDir::new("ops_cherry_pick").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "the cherry-picked commit").await;
    let oid = head_oid(&runner).await;

    write_marker(dir.path(), "CHERRY_PICK_HEAD", &oid);

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    match state {
        OperationState::CherryPick { head, subject } => {
            assert_eq!(head, oid);
            assert_eq!(subject.as_deref(), Some("the cherry-picked commit"));
        }
        other => panic!("expected CherryPick, got {other:?}"),
    }
}

#[tokio::test]
async fn cherry_pick_empty_marker_errors() {
    let dir = TempDir::new("ops_cherry_pick_empty").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;

    write_marker(dir.path(), "CHERRY_PICK_HEAD", "");

    let err = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap_err();

    assert!(err.message.contains("present but empty"), "got: {err}");
}

//  1.5 revert

#[tokio::test]
async fn revert_head_yields_Revert_with_subject() {
    let dir = TempDir::new("ops_revert").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "the reverted commit").await;
    let oid = head_oid(&runner).await;

    write_marker(dir.path(), "REVERT_HEAD", &oid);

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    match state {
        OperationState::Revert { head, subject } => {
            assert_eq!(head, oid);
            assert_eq!(subject.as_deref(), Some("the reverted commit"));
        }
        other => panic!("expected Revert, got {other:?}"),
    }
}

//  precedence pins

#[tokio::test]
async fn conflicted_rebase_yields_Conflicts_not_Rebase() {
    // Design pin: a conflicted rebase leaves both unmerged entries AND
    // rebase-merge/ behind. Conflicts must be detected first, otherwise
    // the state is misclassified as Rebase and the abort is lost.
    let dir = TempDir::new("ops_pin_conflicted_rebase").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("f.txt"), b"base").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "base").await;

    // main moves forward
    std::fs::write(dir.path().join("f.txt"), b"main").unwrap();
    commit(&runner, "main change").await;
    let onto = head_oid(&runner).await;

    // feature diverges from base
    git(&runner, &["checkout", "-b", "feature", "HEAD~1"]).await;
    std::fs::write(dir.path().join("f.txt"), b"feature").unwrap();
    commit(&runner, "feature change").await;

    // rebase onto main → conflict (non-zero exit expected).
    let _ = runner.run(&["rebase", &onto], None).await;

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Conflicts {
            context: Some(Operation::Rebase),
        }
    );
}

#[tokio::test]
async fn conflicted_merge_yields_Conflicts_not_Merge() {
    // Design pin: a conflicted merge leaves both unmerged entries AND
    // MERGE_HEAD. Conflicts must be detected first, otherwise the state
    // is misclassified as Merge.
    let dir = TempDir::new("ops_pin_conflicted_merge").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("f.txt"), b"base").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "base").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    std::fs::write(dir.path().join("f.txt"), b"feature").unwrap();
    commit(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    std::fs::write(dir.path().join("f.txt"), b"main").unwrap();
    commit(&runner, "main change").await;

    let _ = runner.run(&["merge", "feature"], None).await;

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Conflicts {
            context: Some(Operation::Merge),
        }
    );
}

#[tokio::test]
async fn merge_takes_precedence_over_squash() {
    // Design pin: merge writes MERGE_MSG while MERGE_HEAD is present.
    // Squash is defined as SQUASH_MSG *without* MERGE_HEAD. When both
    // markers exist, merge must win so the message source is correct.
    let dir = TempDir::new("ops_pin_merge_over_squash").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.txt"), b"one").unwrap();
    git(&runner, &["add", "a.txt"]).await;
    commit(&runner, "init").await;
    let oid = head_oid(&runner).await;

    write_marker(dir.path(), "MERGE_HEAD", &oid);
    write_marker(dir.path(), "MERGE_MSG", "Merge branch 'feature'");
    write_marker(dir.path(), "SQUASH_MSG", "Squash: should be ignored");

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Merge {
            message: Some("Merge branch 'feature'".to_owned())
        }
    );
}

#[tokio::test]
async fn rebase_conflict_yields_Conflicts_during_rebase() {
    let dir = TempDir::new("ops_rebase_conflict").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("f.txt"), b"base").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "base").await;

    git(&runner, &["checkout", "-b", "feature"]).await;
    std::fs::write(dir.path().join("f.txt"), b"feature").unwrap();
    commit(&runner, "feature change").await;

    git(&runner, &["checkout", "-"]).await;
    std::fs::write(dir.path().join("f.txt"), b"main").unwrap();
    commit(&runner, "main change").await;

    let _ = runner.run(&["rebase", "feature"], None).await;

    let state = OperationStateDetector::new(&runner)
        .run(&make_ctx(dir.path()))
        .await
        .unwrap();

    assert_eq!(
        state,
        OperationState::Conflicts {
            context: Some(Operation::Rebase),
        }
    );
}
