use std::collections::HashMap;
use std::path::PathBuf;

use crate::infra::git::GitRunner;
use crate::shared::config::GitRunOptions;
use crate::shared::exception::GitErrorCode;

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

/// Run git via `runner` and return trimmed stdout, panicking on failure.
async fn git_in(runner: &GitRunner, args: &[&str]) -> String {
    runner
        .run(args, None)
        .await
        .unwrap_or_else(|err| panic!("git {:?} failed: {err}", args))
        .stdout_str()
        .trim()
        .to_owned()
}

/// Init a bare repo in `dir` (no commits needed) and return a runner for it.
async fn init_repo(dir: &std::path::Path) -> GitRunner {
    let runner = GitRunner::new(Some(dir.to_path_buf()));
    git_in(&runner, &["init"]).await;
    runner
}

/// Write `content` to `<dir>/<name>` and hash it into the object db.
/// Returns the blob OID — enough to address the object, no index required.
async fn hash_blob(
    runner: &GitRunner,
    dir: &std::path::Path,
    name: &str,
    content: &[u8],
) -> String {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    git_in(runner, &["hash-object", "-w", path.to_str().unwrap()]).await
}

/// A full-length OID that matches no object (object db only holds
/// what this test hashed).
fn missing_oid() -> String {
    "f".repeat(40)
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
        result.stdout_str().contains("git version"),
        "expected git version banner, got: {}",
        result.stdout_str()
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

    assert_eq!(err.code, GitErrorCode::CommandFailed);
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
    assert_eq!(err.code, GitErrorCode::CommandFailed);
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
        result.stdout_str().contains("AutoCommitTester"),
        "GIT_AUTHOR_NAME didn't propagate; got: {}",
        result.stdout_str()
    );
    assert!(
        result.stdout_str().contains("tester@example.invalid"),
        "GIT_AUTHOR_EMAIL didn't propagate; got: {}",
        result.stdout_str()
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
    assert_eq!(err.code, GitErrorCode::CommandFailed);
    assert!(
        err.message.contains("not a git repository"),
        "stderr should propagate into message; got: {err}"
    );
}

#[tokio::test]
async fn cat_file_header_small_blob_returns_full_content() {
    let dir = TempDir::new("cat_file_header_small_blob_returns_full_content").unwrap();
    let runner = init_repo(dir.path()).await;
    let content = b"// Code generated by example. DO NOT EDIT.\n".to_vec();
    let oid = hash_blob(&runner, dir.path(), "small.txt", &content).await;

    let headers = runner.cat_file_header(&[&oid], 512, None).await.unwrap();

    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].as_deref(), Some(content.as_slice()));
}

#[tokio::test]
async fn cat_file_header_multiple_specs_preserve_order() {
    // Request [b, a] so a pass cannot come from accidental ordering.
    let dir = TempDir::new("cat_file_header_multiple_specs_preserve_order").unwrap();
    let runner = init_repo(dir.path()).await;
    let oid_a = hash_blob(&runner, dir.path(), "a.txt", b"content of a").await;
    let oid_b = hash_blob(&runner, dir.path(), "b.txt", b"content of b").await;

    let headers = runner
        .cat_file_header(&[&oid_b, &oid_a], 512, None)
        .await
        .unwrap();

    assert_eq!(headers.len(), 2);
    assert_eq!(headers[0].as_deref(), Some(&b"content of b"[..]));
    assert_eq!(headers[1].as_deref(), Some(&b"content of a"[..]));
}

#[tokio::test]
async fn cat_file_header_truncates_to_max_bytes() {
    let dir = TempDir::new("cat_file_header_truncates_to_max_bytes").unwrap();
    let runner = init_repo(dir.path()).await;
    let content = b"0123456789abcdef".repeat(128); // 2048 bytes
    let oid = hash_blob(&runner, dir.path(), "big.bin", &content).await;

    let headers = runner.cat_file_header(&[&oid], 512, None).await.unwrap();

    let header = headers[0].as_deref().unwrap();
    assert_eq!(header.len(), 512, "must keep exactly max_bytes");
    assert_eq!(header, &content[..512], "must be the leading prefix");
}

#[tokio::test]
async fn cat_file_header_zero_max_bytes_returns_empty() {
    // take = 0 with a non-empty object: everything incl. the trailing LF
    // must be discarded, and the record must still count as present.
    let dir = TempDir::new("cat_file_header_zero_max_bytes_returns_empty").unwrap();
    let runner = init_repo(dir.path()).await;
    let oid = hash_blob(&runner, dir.path(), "a.txt", b"some content").await;

    let headers = runner.cat_file_header(&[&oid], 0, None).await.unwrap();

    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].as_deref(), Some("".as_bytes()));
}

#[tokio::test]
async fn cat_file_header_empty_blob_returns_some_empty_and_stays_aligned() {
    // size = 0 is a present object with no payload: `Some(vec![])`, and
    // the next record must still parse from its true start.
    let dir = TempDir::new("cat_file_header_empty_blob_aligned").unwrap();
    let runner = init_repo(dir.path()).await;
    let empty_oid = hash_blob(&runner, dir.path(), "empty.txt", b"").await;
    let real_oid = hash_blob(&runner, dir.path(), "real.txt", b"real content").await;

    let headers = runner
        .cat_file_header(&[&empty_oid, &real_oid], 512, None)
        .await
        .unwrap();

    assert_eq!(headers[0].as_deref(), Some("".as_bytes()));
    assert_eq!(headers[1].as_deref(), Some(&b"real content"[..]));
}

//  cat_file_header: missing objects

#[tokio::test]
async fn cat_file_header_missing_spec_returns_none() {
    let dir = TempDir::new("cat_file_header_missing_spec_returns_none").unwrap();
    let runner = init_repo(dir.path()).await;
    let missing = missing_oid();

    // `missing` is a normal protocol response, exit 0, not an error.
    let headers = runner
        .cat_file_header(&[&missing], 512, None)
        .await
        .unwrap();

    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0], None);
}

#[tokio::test]
async fn cat_file_header_missing_between_present_keeps_alignment() {
    // Regression: a missing entry has no payload. Treating it as a
    // 0-byte object shifts every later record by one and silently
    // returns wrong bytes with exit 0.
    let dir = TempDir::new("cat_file_header_missing_between_present").unwrap();
    let runner = init_repo(dir.path()).await;
    let oid_a = hash_blob(&runner, dir.path(), "a.txt", b"AAAA").await;
    let oid_b = hash_blob(&runner, dir.path(), "b.txt", b"BBBB").await;
    let missing = missing_oid();

    let headers = runner
        .cat_file_header(&[&oid_a, &missing, &oid_b], 512, None)
        .await
        .unwrap();

    assert_eq!(headers.len(), 3);
    assert_eq!(headers[0].as_deref(), Some(&b"AAAA"[..]));
    assert_eq!(headers[1], None);
    assert_eq!(headers[2].as_deref(), Some(&b"BBBB"[..]));
}

#[tokio::test]
async fn cat_file_header_empty_specs_returns_empty_without_spawn() {
    let dir = TempDir::new("cat_file_header_empty_specs").unwrap();
    let runner = init_repo(dir.path()).await;

    let headers = runner.cat_file_header(&[], 512, None).await.unwrap();
    assert!(headers.is_empty());
}

#[tokio::test]
async fn cat_file_header_index_spec_reads_staged_blob() {
    // Classifier talks to us with `:path`, not a raw OID.
    let dir = TempDir::new("cat_file_header_index_spec").unwrap();
    let runner = init_repo(dir.path()).await;
    let content = b"staged blob body";
    std::fs::write(dir.path().join("staged.txt"), content).unwrap();
    git_in(&runner, &["add", "staged.txt"]).await;

    let headers = runner
        .cat_file_header(&[":staged.txt"], 512, None)
        .await
        .unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].as_deref(), Some(&content[..]));
}

#[tokio::test]
async fn cat_file_header_large_blob_discards_remainder_and_stays_aligned() {
    let dir = TempDir::new("cat_file_header_large_blob_drain").unwrap();
    let runner = init_repo(dir.path()).await;
    let big = vec![b'A'; 256 * 1024]; // 256 KiB ≫ 64 KiB pipe buffer
    let oid_big = hash_blob(&runner, dir.path(), "big.bin", &big).await;
    let oid_tail = hash_blob(&runner, dir.path(), "tail.txt", b"TAIL").await;

    let headers = runner
        .cat_file_header(&[&oid_big, &oid_tail], 512, None)
        .await
        .unwrap();
    assert_eq!(headers[0].as_deref().unwrap(), &big[..512]);
    assert_eq!(headers[1].as_deref(), Some(&b"TAIL"[..]));
}

#[tokio::test]
async fn cat_file_header_many_specs_do_not_deadlock() {
    let dir = TempDir::new("cat_file_header_many_specs").unwrap();
    let runner = init_repo(dir.path()).await;

    const COUNT: usize = 4000;
    let mut bodies = Vec::with_capacity(COUNT);
    let mut args: Vec<String> = vec!["hash-object".into(), "-w".into()];
    for i in 0..COUNT {
        let name = format!("f{i}.txt");
        let body = format!("body-{i}");
        std::fs::write(dir.path().join(&name), &body).unwrap();
        args.push(name);
        bodies.push(body);
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = runner.run(&arg_refs, None).await.unwrap();
    let oids: Vec<String> = output
        .stdout_str()
        .lines()
        .map(str::trim)
        .map(str::to_owned)
        .collect();
    assert_eq!(oids.len(), COUNT, "one OID per file, in order");

    let specs: Vec<&str> = oids.iter().map(String::as_str).collect();
    // let headers = headers_within(&runner, &specs, 512).await.unwrap();
    let headers = runner.cat_file_header(&specs, 512, None).await.unwrap();

    assert_eq!(headers.len(), COUNT);
    for (header, body) in headers.iter().zip(bodies.iter()) {
        assert_eq!(header.as_deref(), Some(body.as_bytes()));
    }
}

#[tokio::test]
async fn cat_file_header_newline_in_spec_is_rejected() {
    let dir = TempDir::new("cat_file_header_newline_spec").unwrap();
    let runner = init_repo(dir.path()).await;

    let err = runner
        .cat_file_header(&["HEAD:foo\nbar"], 512, None)
        .await
        .unwrap_err();
    assert!(
        err.message.contains("newline"),
        "expected newline guard, got: {err}"
    );
}
