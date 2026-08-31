use std::path::{Path, PathBuf};

use crate::core::git::diff::DiffExtractor;
use crate::core::git::diff::{
    is_lock_signal_line, path_summary, split_sections, summarize_lock_diff, truncate_section,
};
use crate::core::git::types::{
    BudgetDecision, ChangeType, ClassifiedSnapshot, DiffStrategy, FileCategory, StagedFile,
};
use crate::infra::git::GitRunner;

//  helpers

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

fn decision(strategy: DiffStrategy) -> BudgetDecision {
    BudgetDecision {
        strategy,
        estimated_diff_tokens: 0,
        available_for_diff: 9_000,
    }
}

fn file(
    path: &str,
    change_type: ChangeType,
    old_path: Option<&str>,
    category: FileCategory,
) -> StagedFile {
    StagedFile {
        path: PathBuf::from(path),
        old_path: old_path.map(PathBuf::from),
        change_type,
        similarity: None,
        insertions: Some(1),
        deletions: Some(0),
        category,
    }
}

fn snapshot(files: Vec<StagedFile>) -> ClassifiedSnapshot {
    ClassifiedSnapshot::from_files(files)
}

fn two_file_diff() -> &'static str {
    concat!(
        "diff --git a/a.rs b/a.rs\n",
        "--- a/a.rs\n",
        "+++ b/a.rs\n",
        "@@ -1 +1 @@\n",
        "-old-a\n",
        "+new-a\n",
        "diff --git a/b.rs b/b.rs\n",
        "--- a/b.rs\n",
        "+++ b/b.rs\n",
        "@@ -1 +1 @@\n",
        "-old-b\n",
        "+new-b\n",
    )
}

fn two_hunk_section() -> &'static str {
    concat!(
        "diff --git a/foo.rs b/foo.rs\n",
        "--- a/foo.rs\n",
        "+++ b/foo.rs\n",
        "@@ -1,2 +1,2 @@\n",
        " context1\n",
        "-old1\n",
        "+new1\n",
        "@@ -10,2 +10,2 @@\n",
        " context2\n",
        "-old2\n",
        "+new2\n",
    )
}

//  split_sections

#[test]
fn split_sections_empty() {
    assert!(split_sections("").is_empty());
}

#[test]
fn split_sections_single_and_multi() {
    let sections = split_sections(two_file_diff());
    assert_eq!(sections.len(), 2);
    assert!(sections[0].starts_with("diff --git a/a.rs"));
    assert!(sections[1].starts_with("diff --git a/b.rs"));
    assert_eq!(split_sections(two_hunk_section()).len(), 1);
}

#[test]
fn added_diff_git_line_is_not_a_section_boundary() {
    let diff = concat!(
        "diff --git a/note.rs b/note.rs\n",
        "--- a/note.rs\n",
        "+++ b/note.rs\n",
        "@@ -0,0 +1 @@\n",
        "+diff --git a/fake b/fake\n",
    );
    assert_eq!(split_sections(diff).len(), 1);
}

//  truncate_section

#[test]
fn truncate_lines_stops_at_cap_and_counts_remainder() {
    // 4 changed lines: -old1 +new1 -old2 +new2
    let (text, truncated) = truncate_section(two_hunk_section(), 2, None);

    assert!(truncated);
    assert!(text.contains("-old1\n+new1\n"));
    assert!(!text.contains("-old2"));
    assert!(text.contains("... [2 more changed lines truncated]\n"));
}

#[test]
fn exact_line_cap_is_not_truncated() {
    let (text, truncated) = truncate_section(two_hunk_section(), 4, None);

    assert!(!truncated);
    assert_eq!(text, two_hunk_section());
}

#[test]
fn sample_hunks_stops_at_hunk_cap() {
    let (text, truncated) = truncate_section(two_hunk_section(), 400, Some(1));

    assert!(truncated);
    assert!(text.contains("@@ -1,2 +1,2 @@"));
    assert!(!text.contains("@@ -10,2 +10,2 @@"));
    assert!(text.contains("... [2 more changed lines truncated]\n"));
}

#[test]
fn sample_hunks_line_cap_can_fire_first() {
    let (text, truncated) = truncate_section(two_hunk_section(), 2, Some(8));

    assert!(truncated);
    assert!(text.contains("+new1\n"));
    assert!(!text.contains("@@ -10,2 +10,2 @@"));
    assert!(text.contains("... [2 more changed lines truncated]\n"));
}

#[test]
fn rename_only_section_is_never_capped() {
    let section = concat!(
        "diff --git a/old.rs b/new.rs\n",
        "similarity index 100%\n",
        "rename from old.rs\n",
        "rename to new.rs\n",
    );
    let (text, truncated) = truncate_section(section, 0, Some(0));

    assert!(!truncated);
    assert_eq!(text, section);
}

#[test]
fn no_newline_marker_stays_with_its_changed_line() {
    let section = concat!(
        "diff --git a/a b/a\n",
        "--- a/a\n",
        "+++ b/a\n",
        "@@ -1 +1 @@\n",
        "-old\n",
        "\\ No newline at end of file\n",
        "+new\n",
        "\\ No newline at end of file\n",
    );
    let (text, truncated) = truncate_section(section, 1, None);

    assert!(truncated);
    assert!(text.contains("-old\n\\ No newline at end of file\n"));
    assert!(!text.contains("+new"));
    assert!(text.contains("... [1 more changed lines truncated]\n"));
}

//  path_summary

#[test]
fn path_summary_renders_semantic_files_only() {
    let payload = path_summary(&snapshot(vec![
        file(
            "src/a.rs",
            ChangeType::Modified,
            None,
            FileCategory::SemanticText,
        ),
        file("logo.png", ChangeType::Modified, None, FileCategory::Binary),
        file(
            "new.rs",
            ChangeType::Renamed,
            Some("old.rs"),
            FileCategory::SemanticText,
        ),
        file(
            "copy.rs",
            ChangeType::Copied,
            Some("src.rs"),
            FileCategory::SemanticText,
        ),
    ]));

    assert_eq!(payload.file_count, 3);
    assert_eq!(payload.truncated_file_count, 0);
    assert_eq!(
        payload.body,
        "M  src/a.rs\nR  old.rs -> new.rs\nC  src.rs -> copy.rs\n"
    );
}

#[test]
fn path_summary_includes_lock_files() {
    let payload = path_summary(&snapshot(vec![
        file(
            "src/a.rs",
            ChangeType::Modified,
            None,
            FileCategory::SemanticText,
        ),
        file(
            "Cargo.lock",
            ChangeType::Modified,
            None,
            FileCategory::DependencyLock,
        ),
        file("logo.png", ChangeType::Modified, None, FileCategory::Binary),
    ]));

    assert_eq!(payload.file_count, 2);
    assert_eq!(payload.body, "M  src/a.rs\nM  Cargo.lock\n");
}

//  summarize_lock_diff

#[test]
fn summarize_lock_diff_keeps_name_version_pairs() {
    let diff = concat!(
        "diff --git a/Cargo.lock b/Cargo.lock\n",
        "--- a/Cargo.lock\n",
        "+++ b/Cargo.lock\n",
        "@@ -1,4 +1,4 @@\n",
        " name = \"anyhow\"\n",
        "-version = \"1.0.80\"\n",
        "+version = \"1.0.86\"\n",
        " checksum = \"aaaa\"\n",
    );

    let out = summarize_lock_diff(diff, 64);

    assert!(out.contains("diff --git a/Cargo.lock b/Cargo.lock\n"));
    assert!(out.contains(" name = \"anyhow\"\n"));
    assert!(out.contains("-version = \"1.0.80\"\n"));
    assert!(out.contains("+version = \"1.0.86\"\n"));
    assert!(!out.contains("checksum"));
    assert!(!out.contains("@@"));
}

#[test]
fn summarize_lock_diff_keeps_dotted_version_without_keyword() {
    let diff = concat!(
        "diff --git a/go.sum b/go.sum\n",
        "--- a/go.sum\n",
        "+++ b/go.sum\n",
        "@@ -1 +1 @@\n",
        "-github.com/foo/bar v1.2.3 h1:abc\n",
        "+github.com/foo/bar v1.2.4 h1:def\n",
    );

    let out = summarize_lock_diff(diff, 64);

    assert!(out.contains("-github.com/foo/bar v1.2.3 h1:abc\n"));
    assert!(out.contains("+github.com/foo/bar v1.2.4 h1:def\n"));
}

#[test]
fn summarize_lock_diff_caps_and_reports_remainder() {
    let diff = concat!(
        "diff --git a/Cargo.lock b/Cargo.lock\n",
        "@@ -1,6 +1,6 @@\n",
        "-version = \"1\"\n",
        "+version = \"2\"\n",
        "-version = \"3\"\n",
        "+version = \"4\"\n",
        "-version = \"5\"\n",
        "+version = \"6\"\n",
    );

    let out = summarize_lock_diff(diff, 2);

    assert!(out.contains("-version = \"1\"\n"));
    assert!(out.contains("+version = \"2\"\n"));
    assert!(!out.contains("version = \"3\""));
    assert!(out.contains("... [4 more dependency lines truncated]\n"));
}

#[tokio::test]
async fn lock_only_commit_produces_digest_in_body() {
    let dir = TempDir::new("extractor_lock_only").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(
        dir.path().join("Cargo.lock"),
        "[[package]]\nname = \"anyhow\"\nversion = \"1.0.80\"\n",
    )
    .unwrap();
    git(&runner, &["add", "Cargo.lock"]).await;
    git(
        &runner,
        &["-c", "commit.gpgsign=false", "commit", "-m", "init"],
    )
    .await;
    std::fs::write(
        dir.path().join("Cargo.lock"),
        "[[package]]\nname = \"anyhow\"\nversion = \"1.0.86\"\n",
    )
    .unwrap();
    git(&runner, &["add", "Cargo.lock"]).await;

    let extractor = DiffExtractor::new(&runner);
    let payload = extractor
        .extract(
            &snapshot(vec![file(
                "Cargo.lock",
                ChangeType::Modified,
                None,
                FileCategory::DependencyLock,
            )]),
            &decision(DiffStrategy::Full),
        )
        .await
        .unwrap();

    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 0);
    assert!(payload.body.contains("diff --git"));
    assert!(payload.body.contains("1.0.80"));
    assert!(payload.body.contains("1.0.86"));
    assert!(!payload.body.contains("[[package]]"));
}

#[test]
fn lock_signal_covers_registry_pin_lines() {
    let hits = [
        r#"+  "version": "1.2.3","#,
        r#"+name = "serde""#,
        r#"+version = "5.0.0""#,
        "+github.com/foo/bar v1.2.3 h1:abc",
        "+    rails (7.0.4)",
        r#"+        "rev": "a1b2c3d4e5f6789012345678901234567890abcd","#,
        r#"+        "ref": "nixos-unstable","#,
        r#"+          "revision" : "abc123","#,
        r#"+      "identity" : "alamofire","#,
        "+    specifier: 4.17.21",
        r#"+github "Alamofire/Alamofire" "4b1af0318ce26f2a42385b50e0349d6d112ea92c""#,
        r#"+  "foo": {:git, "https://github.com/x/foo.git", "abcdef", []},"#,
        r#"+{<<"foo">>, {git, "https://example.com/foo.git", {ref, "abcdef"}}}"#,
        r#"+        "rrev": "abc123","#,
        r#"+      "RemoteSha": "abcdef123","#,
        r#"+  "phoenix": {:hex, :phoenix, "1.7.0", "deadbeef","#,
    ];
    for line in hits {
        assert!(is_lock_signal_line(line), "expected hit: {line}");
    }
}

#[test]
fn lock_signal_rejects_checksum_noise_and_short_rev_substring() {
    let misses = [
        r#"+checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""#,
        r#"+  integrity "sha512-aaaa""#,
        r#"+      "narHash": "sha256-xxxx=""#,
        r#"+            "contentHash": "bbbb""#,
        r#"+# DO NOT EDIT - copy to .env.local"#,
        r#"+    "revalidate": true,"#,
        r#"+    "preview": false,"#,
    ];
    for line in misses {
        assert!(!is_lock_signal_line(line), "expected miss: {line}");
    }
}

//  extract

#[tokio::test]
async fn path_summary_only_does_not_need_a_repo() {
    let dir = TempDir::new("extractor_path_only").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let extractor = DiffExtractor::new(&runner);

    let payload = extractor
        .extract(
            &snapshot(vec![file(
                "src/a.rs",
                ChangeType::Added,
                None,
                FileCategory::SemanticText,
            )]),
            &decision(DiffStrategy::PathSummaryOnly),
        )
        .await
        .unwrap();

    assert_eq!(payload.body, "A  src/a.rs\n");
    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 0);
}

#[tokio::test]
async fn empty_snapshot_is_empty_payload() {
    let dir = TempDir::new("extractor_empty").unwrap();
    let runner = GitRunner::new(Some(dir.path().to_path_buf()));
    let extractor = DiffExtractor::new(&runner);

    let payload = extractor
        .extract(&snapshot(vec![]), &decision(DiffStrategy::Full))
        .await
        .unwrap();

    assert_eq!(payload, Default::default());
}

#[tokio::test]
async fn full_matches_git_diff_cached() {
    let dir = TempDir::new("extractor_full").unwrap();
    let runner = init_repo(dir.path()).await;
    std::fs::write(dir.path().join("a.rs"), "hello\n").unwrap();
    git(&runner, &["add", "a.rs"]).await;
    git(
        &runner,
        &["-c", "commit.gpgsign=false", "commit", "-m", "init"],
    )
    .await;
    std::fs::write(dir.path().join("a.rs"), "hello\nworld\n").unwrap();
    git(&runner, &["add", "a.rs"]).await;

    let extractor = DiffExtractor::new(&runner);
    let payload = extractor
        .extract(
            &snapshot(vec![file(
                "a.rs",
                ChangeType::Modified,
                None,
                FileCategory::SemanticText,
            )]),
            &decision(DiffStrategy::Full),
        )
        .await
        .unwrap();

    let expected = runner
        .run(
            &[
                "diff",
                "--cached",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "-U1",
                "-M",
                "-C",
                "--",
                "a.rs",
            ],
            None,
        )
        .await
        .unwrap();

    assert_eq!(payload.body, expected.stdout_str().as_ref());
    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 0);
    assert!(payload.body.contains("+world"));
}

#[tokio::test]
async fn truncate_lines_inserts_marker_on_large_file() {
    let dir = TempDir::new("extractor_truncate").unwrap();
    let runner = init_repo(dir.path()).await;

    let original: String = (0..20).map(|i| format!("line-{i}\n")).collect();
    let updated: String = (0..20).map(|i| format!("changed-{i}\n")).collect();
    std::fs::write(dir.path().join("big.rs"), &original).unwrap();
    git(&runner, &["add", "big.rs"]).await;
    git(
        &runner,
        &["-c", "commit.gpgsign=false", "commit", "-m", "init"],
    )
    .await;
    std::fs::write(dir.path().join("big.rs"), &updated).unwrap();
    git(&runner, &["add", "big.rs"]).await;

    let extractor = DiffExtractor::new(&runner);
    let payload = extractor
        .extract(
            &snapshot(vec![file(
                "big.rs",
                ChangeType::Modified,
                None,
                FileCategory::SemanticText,
            )]),
            &decision(DiffStrategy::TruncateLines {
                max_changed_lines_per_file: 4,
            }),
        )
        .await
        .unwrap();

    assert_eq!(payload.file_count, 1);
    assert_eq!(payload.truncated_file_count, 1);
    assert!(payload.body.contains("more changed lines truncated"));
    assert!(!payload.body.contains("changed-19"));
}
