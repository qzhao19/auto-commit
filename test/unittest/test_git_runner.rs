use std::collections::HashMap;
use std::path::PathBuf;

use crate::core::git::runner::{GitRunOptions, GitRunner};
use crate::shared::exception::GitCode;

//  helpers

/// RAII empty directory under `std::env::temp_dir()`, removed on drop.
/// Filename unique per test (the test name) to allow parallel execution.
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

//  GitRunner::new

#[tokio::test]
async fn new_with_explicit_cwd_uses_it() {
    let target = std::env::temp_dir();
    let runner = GitRunner::new(Some(target.clone()));

    let result = runner.run(&["--version"], None).await.unwrap();
    assert_eq!(result.cwd, target, "default_cwd must come from new(Some)");
}

#[tokio::test]
async fn new_with_none_falls_back_to_current_dir() {
    // `GitRunner::new(None)` calls `std::env::current_dir()`; we can't
    // force its failure in a parallel test, so we only assert the
    // happy path here. The `PathBuf::from(".")` final fallback is an
    // acknowledged gap (see file header).
    let expected = std::env::current_dir().unwrap();
    let runner = GitRunner::new(None);

    let result = runner.run(&["--version"], None).await.unwrap();
    assert_eq!(result.cwd, expected);
}

//  run: success path

#[tokio::test]
async fn run_succeeds_on_zero_exit_code() {
    let runner = GitRunner::new(None);
    let result = runner.run(&["--version"], None).await.unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(
        result.stdout.contains("git version"),
        "expected git version banner, got: {}",
        result.stdout
    );
}

//  run: exit-code gate

#[tokio::test]
async fn run_nonzero_exit_returns_command_failed() {
    let runner = GitRunner::new(None);

    let err = runner
        .run(&["--invalid-flag-autocommit"], None)
        .await
        .unwrap_err();

    assert_eq!(err.code, GitCode::CommandFailed);
    // Message shape: "git command failed with exit code N: git cmd\n<reason>"
    assert!(
        err.message.contains("git command failed with exit code"),
        "got: {err}"
    );
}

#[tokio::test]
async fn run_allowed_exit_codes_accepts_listed_code() {
    // `git status` outside a repo exits with 128, a value documented
    // in `git help git` and stable across git versions.
    let dir = TempDir::new("run_allowed_exit_codes_accepts_listed_code").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let allow = GitRunOptions {
        cwd: None,
        env: HashMap::new(),
        allowed_exit_codes: Some(vec![128]),
    };

    let result = runner.run(&["status"], Some(&allow)).await.unwrap();
    assert_eq!(result.exit_code, 128);
}

#[tokio::test]
async fn run_allowed_exit_codes_rejects_unlisted_code() {
    // `git --version` exits 0; an allow-list of [1] must reject it.
    let runner = GitRunner::new(None);
    let allow = GitRunOptions {
        cwd: None,
        env: HashMap::new(),
        allowed_exit_codes: Some(vec![1]),
    };

    let err = runner.run(&["--version"], Some(&allow)).await.unwrap_err();
    assert_eq!(err.code, GitCode::CommandFailed);
}

//  run: options layer

#[tokio::test]
async fn run_options_cwd_overrides_runner_default() {
    let default = TempDir::new("cwd-default").unwrap();
    let override_dir = TempDir::new("cwd-override").unwrap();
    let runner = GitRunner::new(Some(default.path().to_path_buf()));
    let opts = GitRunOptions {
        cwd: Some(override_dir.path().to_path_buf()),
        env: HashMap::new(),
        allowed_exit_codes: None,
    };

    let result = runner.run(&["--version"], Some(&opts)).await.unwrap();
    assert_eq!(result.cwd, override_dir.path());
}

#[tokio::test]
async fn run_options_env_is_injected_to_subprocess() {
    // `git var GIT_AUTHOR_IDENT` reads GIT_AUTHOR_NAME /
    // GIT_AUTHOR_EMAIL env vars and prints "<Name> <email> ts +tz".
    // No repo required; this exercises the `child.envs(&options.env)`
    // injection path.
    let runner = GitRunner::new(None);
    let mut env = HashMap::new();
    env.insert(
        "GIT_AUTHOR_NAME".to_string(),
        "AutoCommitTester".to_string(),
    );
    env.insert(
        "GIT_AUTHOR_EMAIL".to_string(),
        "tester@example.invalid".to_string(),
    );
    let opts = GitRunOptions {
        cwd: None,
        env,
        allowed_exit_codes: None,
    };

    let result = runner
        .run(&["var", "GIT_AUTHOR_IDENT"], Some(&opts))
        .await
        .unwrap();
    assert!(
        result.stdout.contains("AutoCommitTester"),
        "GIT_AUTHOR_NAME didn't propagate; got: {}",
        result.stdout
    );
    assert!(
        result.stdout.contains("tester@example.invalid"),
        "GIT_AUTHOR_EMAIL didn't propagate; got: {}",
        result.stdout
    );
}

// run: CommandFailed reason

#[tokio::test]
async fn run_command_failed_message_includes_stderr_reason() {
    // `git status` outside a repo emits on stderr a "fatal: not a git
    // repository (or any of the parent directories): .git" message
    // (cross-version stable keyword). Because `run`'s reason
    // selection prefers stderr when non-empty, that text must appear in
    // the resulting `GitError.message`.
    let dir = TempDir::new("run_command_failed_message_includes_stderr_reason").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));

    let err = runner.run(&["status"], None).await.unwrap_err();
    assert_eq!(err.code, GitCode::CommandFailed);
    assert!(
        err.message.contains("not a git repository"),
        "stderr should propagate into message; got: {err}"
    );
}
