use std::path::PathBuf;

use crate::core::git::types::{
    BudgetDecision, ChangeType, ClassifiedSnapshot, DiffPayload, DiffStrategy, FileCategory,
    GitPaths, Operation, RepositoryContext, StagedFile,
};
use crate::core::pipe::assembler::PromptAssembler;
use crate::core::pipe::context::AssemblyContext;
use crate::shared::config::LlmMessage;

// ---- Fixture builders ----

fn repo(branch: Option<&str>, head: Option<&str>) -> RepositoryContext {
    RepositoryContext {
        worktree_root: PathBuf::from("/repo"),
        git_paths: GitPaths::from_git_dir(PathBuf::from("/repo/.git")),
        head_oid: head.map(str::to_owned),
        branch: branch.map(str::to_owned),
    }
}

fn text_file(path: &str, change_type: ChangeType, insertions: u64, deletions: u64) -> StagedFile {
    StagedFile {
        path: PathBuf::from(path),
        old_path: None,
        change_type,
        similarity: None,
        insertions: Some(insertions),
        deletions: Some(deletions),
        category: FileCategory::SemanticText,
    }
}

fn meta_file(path: &str, change_type: ChangeType, category: FileCategory) -> StagedFile {
    StagedFile {
        path: PathBuf::from(path),
        old_path: None,
        change_type,
        similarity: None,
        insertions: None,
        deletions: None,
        category,
    }
}

fn renamed_meta_file(
    old: &str,
    new: &str,
    category: FileCategory,
    similarity: Option<u8>,
) -> StagedFile {
    StagedFile {
        old_path: Some(PathBuf::from(old)),
        similarity,
        ..meta_file(new, ChangeType::Renamed, category)
    }
}

fn decision(strategy: DiffStrategy) -> BudgetDecision {
    BudgetDecision {
        strategy,
        estimated_diff_tokens: 5_000,
        available_for_diff: 126_000,
    }
}

fn payload(body: &str, file_count: usize, truncated: usize) -> DiffPayload {
    DiffPayload {
        body: body.to_owned(),
        file_count,
        truncated_file_count: truncated,
    }
}

fn staging_ctx(
    files: Vec<StagedFile>,
    payload: DiffPayload,
    strategy: DiffStrategy,
) -> AssemblyContext {
    AssemblyContext::FromStaging {
        repo: repo(Some("main"), Some("abc123")),
        snapshot: ClassifiedSnapshot::from_files(files),
        payload,
        decision: decision(strategy),
    }
}

fn operation_ctx(
    operation: Operation,
    message: Option<&str>,
    oid: Option<&str>,
) -> AssemblyContext {
    AssemblyContext::FromOperation {
        repo: repo(Some("main"), Some("abc123")),
        operation,
        message: message.map(str::to_owned),
        commit_oid: oid.map(str::to_owned),
    }
}

fn user_of(ctx: &AssemblyContext) -> String {
    PromptAssembler::assemble(ctx).user_message
}

// ---- Assembly plumbing ----

#[test]
fn assemble_attaches_same_system_prompt_to_both_branches() {
    let op = PromptAssembler::assemble(&operation_ctx(Operation::Merge, Some("msg"), None));
    let st = PromptAssembler::assemble(&staging_ctx(vec![], payload("", 0, 0), DiffStrategy::Full));

    let assert_message = |m: &LlmMessage| {
        assert!(m.system_message.as_deref().is_some_and(|s| !s.is_empty()));
        assert!(!m.user_message.is_empty());
    };
    assert_message(&op);
    assert_message(&st);
    assert_eq!(op.system_message, st.system_message);
}

// ---- Repository context (stage 0 inputs) ----

#[test]
fn repo_context_renders_branch_name() {
    let user = user_of(&operation_ctx(Operation::Merge, Some("msg"), None));
    assert!(user.contains("## Repository\n\nbranch: main\n"));
}

#[test]
fn repo_context_renders_detached_head_placeholder() {
    let ctx = AssemblyContext::FromOperation {
        repo: repo(None, Some("abc123")),
        operation: Operation::Merge,
        message: Some("msg".to_owned()),
        commit_oid: None,
    };
    assert!(user_of(&ctx).contains("branch: detached HEAD\n"));
}

#[test]
fn repo_context_flags_initial_commit() {
    let ctx = |head: Option<&str>| AssemblyContext::FromOperation {
        repo: repo(Some("main"), head),
        operation: Operation::Merge,
        message: Some("msg".to_owned()),
        commit_oid: None,
    };
    assert!(user_of(&ctx(None)).contains("Initial commit: yes\n"));
    assert!(!user_of(&ctx(Some("abc123"))).contains("Initial commit"));
}

// ---- Operation seed branch (stage 1 inputs) ----

#[test]
fn operation_headers_render_operation_kind() {
    let user = user_of(&operation_ctx(Operation::Rebase, Some("msg"), None));
    assert!(user.contains("\n## Git operation: rebase\n\n"));
    assert!(user.contains("## Git-native message\n\n"));
}

#[test]
fn operation_merge_reuses_seed_and_ignores_commit_oid() {
    let user = user_of(&operation_ctx(
        Operation::Merge,
        Some("Merge branch 'feature/x'"),
        Some("9fceb02"),
    ));
    assert!(user.contains("Merge branch 'feature/x'"));
    // Provenance must never be composed for merge/rebase/squash,
    // and the oid must not leak anywhere.
    assert!(!user.contains("9fceb02"));
    assert!(!user.contains("cherry picked"));
    assert!(!user.contains("This reverts"));
}

#[test]
fn operation_cherry_pick_appends_provenance_after_subject() {
    let user = user_of(&operation_ctx(
        Operation::CherryPick,
        Some("feat: add cache"),
        Some("9fceb02"),
    ));
    assert!(user.contains("feat: add cache\n\n(cherry picked from commit 9fceb02)\n"));

    let subject = user.find("feat: add cache").unwrap();
    let provenance = user.find("(cherry picked from commit").unwrap();
    assert!(subject < provenance);
}

#[test]
fn operation_cherry_pick_without_oid_emits_subject_only() {
    let user = user_of(&operation_ctx(
        Operation::CherryPick,
        Some("feat: add cache"),
        None,
    ));
    assert!(user.contains("feat: add cache\n"));
    assert!(!user.contains("cherry picked from commit"));
}

#[test]
fn operation_revert_composes_this_reverts_trailer() {
    let user = user_of(&operation_ctx(
        Operation::Revert,
        Some("Revert \"feat: add cache\""),
        Some("9fceb02"),
    ));
    assert!(user.contains("Revert \"feat: add cache\"\n\nThis reverts commit 9fceb02.\n"));
}

#[test]
fn operation_missing_seed_falls_back_to_op_specific_guidance() {
    let user = user_of(&operation_ctx(Operation::Rebase, None, None));
    assert!(user.contains("(none was found; write a commit message that completes this rebase)\n"));
}

#[test]
fn operation_blank_seed_is_treated_as_missing() {
    for blank in ["", "   ", "\n \t\n"] {
        let user = user_of(&operation_ctx(Operation::Squash, Some(blank), None));
        assert!(
            user.contains("(none was found; write a commit message that completes this squash)"),
            "blank seed {blank:?} must hit the fallback"
        );
    }
}

#[test]
fn operation_seed_is_trimmed_before_rendering() {
    let user = user_of(&operation_ctx(
        Operation::Merge,
        Some("  fix: typo\n\n"),
        None,
    ));
    assert!(user.contains("fix: typo\n"));
    assert!(!user.contains("  fix: typo"));
}

// ---- Change summary (stage 2 / 3.1 inputs) ----

#[test]
fn change_summary_aggregates_source_line_totals() {
    let files = vec![
        text_file("src/a.rs", ChangeType::Modified, 10, 3),
        text_file("src/b.rs", ChangeType::Added, 0, 2),
    ];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 2, 0),
        DiffStrategy::Full,
    ));
    assert!(user.contains("\n## Change summary\n\n"));
    assert!(user.contains("- source files: 2 changed, +10 insertions/-5 deletions\n"));
}

#[test] // [补丁C]
fn change_summary_reports_rename_only_files_separately() {
    let renamed = StagedFile {
        old_path: Some(PathBuf::from("src/old.rs")),
        similarity: Some(100),
        ..text_file("src/new.rs", ChangeType::Renamed, 0, 0)
    };
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 5, 1), renamed];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 2, 0),
        DiffStrategy::Full,
    ));
    assert!(user.contains("- source files: 1 changed, +5 insertions/-1 deletions\n"));
    assert!(user.contains("- source files: 1 renamed only (no content change)\n"));
}

#[test]
fn change_summary_lock_only_commit_omits_source_line() {
    let files = vec![meta_file(
        "Cargo.lock",
        ChangeType::Modified,
        FileCategory::DependencyLock,
    )];
    let user = user_of(&staging_ctx(
        files,
        payload("lock digest", 1, 0),
        DiffStrategy::Full,
    ));
    assert!(user.contains("- lock files: 1 changed\n"));
    assert!(!user.contains("source files:"));
}

#[test]
fn change_summary_binary_only_staging_omits_whole_section() {
    let files = vec![meta_file(
        "assets/logo.png",
        ChangeType::Added,
        FileCategory::Binary,
    )];
    let user = user_of(&staging_ctx(files, payload("", 0, 0), DiffStrategy::Full));
    assert!(!user.contains("## Change summary"));
}

// ---- Non-text file listing (stage 3 path-one inputs) ----

#[test]
fn non_text_files_are_listed_in_input_order_with_kind_and_status() {
    let files = vec![
        meta_file(
            "dist/bundle.js",
            ChangeType::Modified,
            FileCategory::Generated,
        ),
        meta_file("assets/logo.png", ChangeType::Added, FileCategory::Binary),
        meta_file("dep", ChangeType::Modified, FileCategory::Submodule),
        text_file("src/lib.rs", ChangeType::Modified, 3, 1), // must NOT appear below
    ];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::Full,
    ));

    assert_eq!(user.matches("## Non-text files (metadata only)").count(), 1);
    assert!(user.contains("- [generated] Modified dist/bundle.js\n"));
    assert!(user.contains("- [binary] Added assets/logo.png\n"));
    assert!(user.contains("- [submodule] Modified dep\n"));
    assert!(!user.contains("[submodule] Modified src/lib.rs"));

    let generated = user.find("dist/bundle.js").unwrap();
    let binary = user.find("assets/logo.png").unwrap();
    let submodule = user.find("dep\n").unwrap();
    assert!(generated < binary && binary < submodule);
}

#[test]
fn non_text_rename_renders_arrow_and_similarity() {
    let files = vec![renamed_meta_file(
        "assets/old.png",
        "assets/new.png",
        FileCategory::Binary,
        Some(75),
    )];
    let user = user_of(&staging_ctx(files, payload("", 0, 0), DiffStrategy::Full));
    assert!(
        user.contains("- [binary] Renamed assets/old.png -> assets/new.png (similarity 75%)\n")
    );
}

#[test] // [补丁A]
fn non_text_rename_without_similarity_has_no_trailing_space() {
    let files = vec![renamed_meta_file(
        "a.png",
        "b.png",
        FileCategory::Binary,
        None,
    )];
    let user = user_of(&staging_ctx(files, payload("", 0, 0), DiffStrategy::Full));
    assert!(user.contains("- [binary] Renamed a.png -> b.png\n"));
    assert!(!user.contains("-> b.png \n"));
}

#[test]
fn non_text_section_is_omitted_when_all_files_are_text_or_lock() {
    let files = vec![
        text_file("src/a.rs", ChangeType::Modified, 1, 1),
        meta_file(
            "Cargo.lock",
            ChangeType::Modified,
            FileCategory::DependencyLock,
        ),
    ];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::Full,
    ));
    assert!(!user.contains("## Non-text files"));
}

// ---- Diff budget note (stage 3.2 inputs) ----

#[test]
fn full_strategy_omits_diff_budget_section() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::Full,
    ));
    assert!(!user.contains("## Diff budget"));
}

#[test] // [决策D]
fn reduced_strategies_report_estimated_and_available_tokens() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400,
        },
    ));
    assert!(user.contains("- estimated: 5000 tokens; available: 126000\n"));
}

#[test]
fn truncate_lines_strategy_reports_line_cap() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400,
        },
    ));
    assert!(user.contains("\n## Diff budget\n"));
    assert!(user.contains("- reduction: each file capped at 400 changed lines\n"));
    assert!(user.contains("- guidance: prefer high-level wording where detail is missing\n"));
}

#[test]
fn sample_hunks_strategy_reports_hunk_and_line_caps() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::SampleHunks {
            max_hunks_per_file: 8,
            max_changed_lines_per_file: 400,
        },
    ));
    assert!(user.contains("- reduction: each file capped at 8 hunk(s) and 400 changed lines\n"));
}

#[test]
fn path_summary_only_strategy_instructs_high_level_message() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("M  src/a.rs", 1, 0),
        DiffStrategy::PathSummaryOnly,
    ));
    assert!(user.contains("- reduction: diff content replaced by a path listing\n"));
    assert!(user.contains("- guidance: write a high-level message from the paths alone\n"));
}

// ---- Staged changes payload (stage 3.3 inputs) ----

#[test]
fn staged_changes_body_is_embedded_verbatim_with_trailing_blank_lines_trimmed() {
    let body = "diff --git a/src/lib.rs b/src/lib.rs\n@@ -1 +1 @@\n-fn old()\n+fn new()\n\n\n";
    let files = vec![text_file("src/lib.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(files, payload(body, 1, 0), DiffStrategy::Full));
    assert!(user.contains("@@ -1 +1 @@\n-fn old()\n+fn new()\n"));
    assert!(user.ends_with("+fn new()\n"));
}

#[test] // [补丁B]
fn staged_changes_empty_body_renders_placeholder() {
    let files = vec![meta_file(
        "assets/logo.png",
        ChangeType::Added,
        FileCategory::Binary,
    )];
    let user = user_of(&staging_ctx(files, payload("", 0, 0), DiffStrategy::Full));
    assert!(user.contains("(no diff content; rely on the change summary and file list above)\n"));
}

#[test]
fn staged_changes_truncation_note_reports_counts() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 900, 100)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 5, 2),
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400,
        },
    ));
    assert!(user.contains(
        "\n2 of 5 file diff(s) were truncated to fit the context budget. Do not assume omitted hunks are unchanged.\n"
    ));
}

#[test]
fn staged_changes_without_truncation_omits_note() {
    let files = vec![text_file("src/a.rs", ChangeType::Modified, 1, 1)];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::Full,
    ));
    assert!(!user.contains("truncated"));
}

// ---- Section ordering & extreme inputs ----

#[test]
fn staging_sections_appear_in_stable_order() {
    let files = vec![
        text_file("src/a.rs", ChangeType::Modified, 1, 1),
        meta_file("assets/logo.png", ChangeType::Added, FileCategory::Binary),
    ];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 400,
        },
    ));

    let repo = user.find("## Repository").unwrap();
    let summary = user.find("## Change summary").unwrap();
    let non_text = user.find("## Non-text files").unwrap();
    let budget = user.find("## Diff budget").unwrap();
    let staged = user.find("## Staged changes").unwrap();
    assert!(repo < summary && summary < non_text && non_text < budget && budget < staged);
}

#[test]
fn unicode_and_spaced_paths_are_rendered_verbatim() {
    let files = vec![
        meta_file("assets/图片 1.png", ChangeType::Added, FileCategory::Binary),
        text_file("src/データ/ファイル名.rs", ChangeType::Added, 42, 0),
    ];
    let user = user_of(&staging_ctx(
        files,
        payload("diff", 1, 0),
        DiffStrategy::Full,
    ));
    assert!(user.contains("assets/图片 1.png"));
}

#[test]
fn extremely_long_body_is_embedded_without_truncation() {
    let body = "+".repeat(100_000);
    let files = vec![text_file(
        "src/huge.rs",
        ChangeType::Modified,
        50_000,
        50_000,
    )];
    let user = user_of(&staging_ctx(
        files,
        payload(&body, 1, 0),
        DiffStrategy::Full,
    ));
    assert!(user.contains(&body));
}
