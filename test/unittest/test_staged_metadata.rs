use std::path::{Path, PathBuf};

use crate::core::git::staged::{
    NameStatusEntry, StagedMetadataCollector, parse_name_status, parse_numstat,
    parse_submodule_flags,
};
use crate::core::git::types::{ChangeType, FileCategory, StagedFile, StagedSnapshot};
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

/// Build a name-status entry for lockstep parser tests.
fn entry(
    change_type: ChangeType,
    path: &str,
    old_path: Option<&str>,
    similarity: Option<u8>,
) -> NameStatusEntry {
    NameStatusEntry {
        change_type,
        path: PathBuf::from(path),
        old_path: old_path.map(PathBuf::from),
        similarity,
    }
}

/// Find a staged file by path; panics with the full snapshot on miss.
fn find<'a>(snapshot: &'a StagedSnapshot, path: &str) -> &'a StagedFile {
    snapshot
        .files
        .iter()
        .find(|f| f.path == Path::new(path))
        .unwrap_or_else(|| panic!("path {path} not in snapshot: {:?}", snapshot.files))
}

//  parser: name-status

#[test]
fn name_status_plain_records() {
    let data = b"A\0new.txt\0M\0mod.txt\0D\0gone.txt\0T\0type.txt\0";
    let entries = parse_name_status(data).unwrap();

    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0].change_type, ChangeType::Added);
    assert_eq!(entries[0].path, Path::new("new.txt"));
    assert_eq!(entries[0].old_path, None);
    assert_eq!(entries[0].similarity, None);

    assert_eq!(entries[1].change_type, ChangeType::Modified);
    assert_eq!(entries[2].change_type, ChangeType::Deleted);
    assert_eq!(entries[3].change_type, ChangeType::TypeChanged);
}

#[test]
fn name_status_rename_record() {
    let data = b"R100\0old.rs\0new.rs\0";
    let entries = parse_name_status(data).unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].change_type, ChangeType::Renamed);
    assert_eq!(entries[0].path, Path::new("new.rs"));
    assert_eq!(entries[0].old_path.as_deref(), Some(Path::new("old.rs")));
    assert_eq!(entries[0].similarity, Some(100));
}

#[test]
fn name_status_empty_input() {
    assert!(parse_name_status(b"").unwrap().is_empty());
}

#[test]
fn name_status_truncated_plain_errors() {
    let err = parse_name_status(b"M\0").unwrap_err();
    assert!(err.message.contains("truncated"), "got: {err}");
}

#[test]
fn name_status_truncated_rename_errors() {
    // R/C records need two path fields; only one is present.
    let err = parse_name_status(b"R90\0old.rs\0").unwrap_err();
    assert!(err.message.contains("rename/copy"), "got: {err}");
}

#[test]
fn name_status_unexpected_status_errors() {
    // `U` (unmerged) must have been filtered by stage 1; hitting it here
    // means the pipeline contract broke — hard error, not a guess.
    let err = parse_name_status(b"U\0file.txt\0").unwrap_err();
    assert!(err.message.contains("unexpected change type"), "got: {err}");
}

#[test]
fn name_status_invalid_similarity_errors() {
    let err = parse_name_status(b"Rabc\0old.rs\0new.rs\0").unwrap_err();
    assert!(err.message.contains("similarity"), "got: {err}");
}

#[cfg(unix)]
#[test]
fn name_status_non_utf8_path_preserved() {
    use std::os::unix::ffi::OsStrExt;

    let data = b"M\0\xff\xfe.txt\0";
    let entries = parse_name_status(data).unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path.as_os_str().as_bytes(), b"\xff\xfe.txt");
}

//  parser: numstat

#[test]
fn numstat_consecutive_plain_records() {
    // Regression pin: A/M/D/T records embed the path inside the counts
    // field (`ins\tdel\tpath`), so NOTHING is skipped between records.
    // Skipping one field here (the old bug) eats the next counts field.
    let entries = vec![
        entry(ChangeType::Modified, "a.txt", None, None),
        entry(ChangeType::Modified, "b.txt", None, None),
    ];
    let data = b"1\t2\ta.txt\03\t4\tb.txt\0";

    let rows = parse_numstat(data, &entries).unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].insertions, Some(1));
    assert_eq!(rows[0].deletions, Some(2));
    assert!(!rows[0].is_binary);
    assert_eq!(rows[1].insertions, Some(3));
    assert_eq!(rows[1].deletions, Some(4));
}

#[test]
fn numstat_binary_record() {
    let entries = vec![entry(ChangeType::Added, "bin.dat", None, None)];
    let data = b"-\t-\tbin.dat\0";

    let rows = parse_numstat(data, &entries).unwrap();

    assert_eq!(rows.len(), 1);
    assert!(rows[0].is_binary);
    assert_eq!(rows[0].insertions, None);
    assert_eq!(rows[0].deletions, None);
}

#[test]
fn numstat_rename_record_skips_two_paths() {
    // R/C records: counts field ends with a bare tab, then TWO path
    // fields follow as separate NUL fields.
    let entries = vec![entry(ChangeType::Renamed, "new.rs", Some("old.rs"), None)];
    let data = b"5\t1\t\0old.rs\0new.rs\0";

    let rows = parse_numstat(data, &entries).unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].insertions, Some(5));
    assert_eq!(rows[0].deletions, Some(1));
}

#[test]
fn numstat_mixed_rename_and_plain_alignment() {
    // The critical lockstep case: a rename (2 path fields) followed by a
    // plain record (0 path fields) must not shift the second counts.
    let entries = vec![
        entry(ChangeType::Renamed, "new.rs", Some("old.rs"), None),
        entry(ChangeType::Modified, "a.txt", None, None),
    ];
    let data = b"5\t1\t\0old.rs\0new.rs\02\t3\ta.txt\0";

    let rows = parse_numstat(data, &entries).unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].insertions, Some(5));
    assert_eq!(rows[1].insertions, Some(2));
    assert_eq!(rows[1].deletions, Some(3));
}

#[test]
fn numstat_short_stream_errors() {
    // name-status has 2 records but numstat only carries 1.
    let entries = vec![
        entry(ChangeType::Modified, "a.txt", None, None),
        entry(ChangeType::Modified, "b.txt", None, None),
    ];

    let err = parse_numstat(b"1\t2\ta.txt\0", &entries).unwrap_err();
    assert!(err.message.contains("mismatch"), "got: {err}");
}

#[test]
fn numstat_extra_records_error() {
    // name-status has 1 record but numstat carries 2 — the shared-order
    // invariant is violated and must be a hard error, not a silent drop.
    let entries = vec![entry(ChangeType::Modified, "a.txt", None, None)];
    let data = b"1\t2\ta.txt\03\t4\tb.txt\0";

    let err = parse_numstat(data, &entries).unwrap_err();
    assert!(err.message.contains("more records"), "got: {err}");
}

#[test]
fn numstat_empty_streams_ok() {
    // Empty staging produces empty stdout (or a lone NUL); the trailing
    // empty field from the NUL terminator must not be treated as data.
    let entries: Vec<NameStatusEntry> = vec![];

    assert!(parse_numstat(b"", &entries).unwrap().is_empty());
    assert!(parse_numstat(b"\0", &entries).unwrap().is_empty());
}

//  parser: raw / submodule

#[test]
fn raw_regular_file_not_submodule() {
    let entries = vec![entry(ChangeType::Modified, "a.txt", None, None)];
    let data = b":100644 100644 abc123 def456 M\0a.txt\0";

    let flags = parse_submodule_flags(data, &entries).unwrap();

    assert_eq!(flags, vec![false]);
}

#[test]
fn raw_gitlink_mode_is_submodule() {
    // Added submodule: dst mode 160000.
    let entries = vec![entry(ChangeType::Added, "sub", None, None)];
    let data = b":000000 160000 0000000 1234567 A\0sub\0";

    let flags = parse_submodule_flags(data, &entries).unwrap();

    assert_eq!(flags, vec![true]);
}

#[test]
fn raw_sha_substring_no_false_positive() {
    // Design pin: the SHA is hex and may legitimately CONTAIN "160000"
    // as a substring. Detection must match the mode tokens exactly, not
    // substring-scan the whole meta field (the old bug).
    let entries = vec![entry(ChangeType::Modified, "a.txt", None, None)];
    let data = b":100644 100644 160000abc def456 M\0a.txt\0";

    let flags = parse_submodule_flags(data, &entries).unwrap();

    assert_eq!(flags, vec![false]);
}

#[test]
fn raw_rename_alignment() {
    // Raw records always carry standalone path fields: 2 for R/C, 1 for
    // everything else (unlike numstat, where plain paths are embedded).
    let entries = vec![
        entry(ChangeType::Renamed, "new.rs", Some("old.rs"), Some(90)),
        entry(ChangeType::Modified, "a.txt", None, None),
    ];
    let data = b":100644 100644 aa bb R90\0old.rs\0new.rs\0:100644 100644 aa bb M\0a.txt\0";

    let flags = parse_submodule_flags(data, &entries).unwrap();

    assert_eq!(flags, vec![false, false]);
}

#[test]
fn raw_short_stream_errors() {
    let entries = vec![
        entry(ChangeType::Modified, "a.txt", None, None),
        entry(ChangeType::Modified, "b.txt", None, None),
    ];

    let err = parse_submodule_flags(b":100644 100644 a b M\0a.txt\0", &entries).unwrap_err();
    assert!(err.message.contains("mismatch"), "got: {err}");
}

//  collector: integration with real git

#[tokio::test]
async fn two_modified_files_collect_counts() {
    // End-to-end regression pin for the numstat 0-skip bug: two plain
    // records in a row must both keep their own line counts.
    let dir = TempDir::new("staged_two_modified").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("a.txt"), b"one\ntwo\nthree\n").unwrap();
    std::fs::write(dir.path().join("b.txt"), b"x\ny\n").unwrap();
    git(&runner, &["add", "a.txt", "b.txt"]).await;
    commit(&runner, "init").await;

    // a.txt: -two +TWO +four → 2/1;  b.txt: -y → 0/1.
    std::fs::write(dir.path().join("a.txt"), b"one\nTWO\nthree\nfour\n").unwrap();
    std::fs::write(dir.path().join("b.txt"), b"x\n").unwrap();
    git(&runner, &["add", "-A"]).await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    assert_eq!(snapshot.total_files(), 2);

    let a = find(&snapshot, "a.txt");
    assert_eq!(a.change_type, ChangeType::Modified);
    assert_eq!(a.category, FileCategory::Unknown);
    assert_eq!(a.insertions, Some(2));
    assert_eq!(a.deletions, Some(1));

    let b = find(&snapshot, "b.txt");
    assert_eq!(b.insertions, Some(0));
    assert_eq!(b.deletions, Some(1));

    assert_eq!(snapshot.total_insertions(), 2);
    assert_eq!(snapshot.total_deletions(), 2);
    assert_eq!(snapshot.text_file_count(), 0);
}

#[tokio::test]
async fn added_and_deleted_files() {
    let dir = TempDir::new("staged_add_delete").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("c.txt"), b"keep\n").unwrap();
    std::fs::write(dir.path().join("d.txt"), b"m1\nm2\n").unwrap();
    git(&runner, &["add", "c.txt", "d.txt"]).await;
    commit(&runner, "init").await;

    git(&runner, &["rm", "d.txt"]).await;
    std::fs::write(dir.path().join("e.txt"), b"n1\nn2\n").unwrap();
    git(&runner, &["add", "e.txt"]).await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    let d = find(&snapshot, "d.txt");
    assert_eq!(d.change_type, ChangeType::Deleted);
    assert_eq!(d.insertions, Some(0));
    assert_eq!(d.deletions, Some(2));

    let e = find(&snapshot, "e.txt");
    assert_eq!(e.change_type, ChangeType::Added);
    assert_eq!(e.insertions, Some(2));
    assert_eq!(e.deletions, Some(0));
}

#[tokio::test]
async fn renamed_file_collects_old_path_and_similarity() {
    let dir = TempDir::new("staged_rename").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("r1.txt"), b"alpha\nbeta\n").unwrap();
    git(&runner, &["add", "r1.txt"]).await;
    commit(&runner, "init").await;

    git(&runner, &["mv", "r1.txt", "r2.txt"]).await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    assert!(snapshot.has_rename());
    assert_eq!(snapshot.total_files(), 1);

    let r = find(&snapshot, "r2.txt");
    assert_eq!(r.change_type, ChangeType::Renamed);
    assert_eq!(r.old_path.as_deref(), Some(Path::new("r1.txt")));
    // Unmodified content → R100, 0/0 lines.
    assert_eq!(r.similarity, Some(100));
    assert_eq!(r.insertions, Some(0));
    assert_eq!(r.deletions, Some(0));
    assert_eq!(r.category, FileCategory::Unknown);
}

#[tokio::test]
async fn binary_file_flagged() {
    let dir = TempDir::new("staged_binary").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("t.txt"), b"text\n").unwrap();
    git(&runner, &["add", "t.txt"]).await;
    commit(&runner, "init").await;

    std::fs::write(dir.path().join("t.txt"), b"\x00\x01\x02\xff").unwrap();
    git(&runner, &["add", "t.txt"]).await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    assert_eq!(snapshot.binary_file_count(), 1);

    let t = find(&snapshot, "t.txt");
    assert_eq!(t.category, FileCategory::Binary);
    assert_eq!(t.insertions, None);
    assert_eq!(t.deletions, None);
}

#[tokio::test]
async fn gitlink_staged_as_submodule() {
    // A gitlink entry is staged via plumbing, avoiding the network /
    // protocol restrictions of `git submodule add` in test environments.
    let dir = TempDir::new("staged_submodule").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("f.txt"), b"x\n").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "init").await;

    let sha = runner
        .run(&["rev-parse", "HEAD"], None)
        .await
        .unwrap()
        .stdout_str()
        .trim()
        .to_owned();

    // Stage a regular file next to the gitlink to test mixed assembly.
    std::fs::write(dir.path().join("e.txt"), b"y\n").unwrap();
    git(&runner, &["add", "e.txt"]).await;

    let cacheinfo = format!("160000,{sha},sub");
    git(
        &runner,
        &["update-index", "--add", "--cacheinfo", &cacheinfo],
    )
    .await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    assert_eq!(snapshot.submodule_file_count(), 1);

    let sub = find(&snapshot, "sub");
    assert_eq!(sub.change_type, ChangeType::Added);
    assert_eq!(sub.category, FileCategory::Submodule);
    assert_eq!(sub.insertions, None);
    assert_eq!(sub.deletions, None);

    let e = find(&snapshot, "e.txt");
    assert_eq!(e.category, FileCategory::Unknown);
    assert_eq!(e.insertions, Some(1));
}

#[tokio::test]
async fn empty_staging_yields_empty_snapshot() {
    // Stage 0 guarantees non-empty staging, but the collector itself
    // must degrade gracefully on empty output, not error out.
    let dir = TempDir::new("staged_empty").unwrap();
    let runner = init_repo(dir.path()).await;

    std::fs::write(dir.path().join("f.txt"), b"x\n").unwrap();
    git(&runner, &["add", "f.txt"]).await;
    commit(&runner, "init").await;

    let snapshot = StagedMetadataCollector::new(&runner)
        .collect()
        .await
        .unwrap();

    assert_eq!(snapshot.total_files(), 0);
    assert_eq!(snapshot.binary_file_count(), 0);
    assert!(!snapshot.has_rename());
}
