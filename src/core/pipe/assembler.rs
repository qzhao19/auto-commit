use std::fmt::Write as _;

use crate::core::git::types::{
    BudgetDecision, ChangeType, ClassifiedSnapshot, DiffPayload, DiffStrategy, FileCategory,
    Operation, RepositoryContext, SemanticTextStats, StagedFile,
};
use crate::shared::config::LlmMessage;

use super::context::AssemblyContext;
use super::template::SYSTEM_PROMPT;

pub struct PromptAssembler;

impl PromptAssembler {
    pub fn assemble(ctx: &AssemblyContext) -> LlmMessage {
        match ctx {
            AssemblyContext::FromOperation {
                repo,
                operation,
                message,
                commit_oid,
            } => {
                let user_msg = build_operation_message(
                    repo,
                    *operation,
                    message.as_deref(),
                    commit_oid.as_deref(),
                );

                LlmMessage {
                    system_message: Some(SYSTEM_PROMPT.to_owned()),
                    user_message: user_msg,
                }
            }
            AssemblyContext::FromStaging {
                repo,
                snapshot,
                payload,
                decision,
            } => {
                let user_msg = build_staging_message(repo, snapshot, payload, decision);

                LlmMessage {
                    system_message: Some(SYSTEM_PROMPT.to_owned()),
                    user_message: user_msg,
                }
            }
        }
    }
}

fn build_operation_message(
    repo: &RepositoryContext,
    operation: Operation,
    git_message: Option<&str>,
    commit_oid: Option<&str>,
) -> String {
    let mut message = String::with_capacity(1_024);
    append_seed_message(&mut message, operation, git_message, commit_oid);
    append_repo_context(&mut message, repo);
    message
}

fn build_staging_message(
    repo: &RepositoryContext,
    snapshot: &ClassifiedSnapshot,
    payload: &DiffPayload,
    decision: &BudgetDecision,
) -> String {
    let mut message = String::with_capacity(payload.body.len() + 2_048);
    append_repo_context(&mut message, repo);
    append_change_summary(&mut message, snapshot);
    append_staged_changes(&mut message, payload);
    message
}

fn append_repo_context(out: &mut String, repo: &RepositoryContext) {
    out.push_str("## Repository\n\n");
    match repo.branch.as_deref() {
        Some(branch) => {
            out.push_str("branch: ");
            out.push_str(branch);
            out.push('\n');
        }
        None => out.push_str("branch: detached HEAD\n"),
    }
    if repo.is_initial_commit() {
        out.push_str("Initial commit: yes\n");
    }
}

/// Build operation seed message draft
///
/// merge / rebase / squash → git-native message as-is
/// cherry-pick             → subject + "(cherry picked from commit <oid>)"
/// revert                  → `Revert "subject"` + "This reverts commit <oid>."
fn append_seed_message(
    out: &mut String,
    operation: Operation,
    git_message: Option<&str>,
    commit_oid: Option<&str>,
) {
    let op = operation.as_str();
    out.push_str(&format!("\n## Git operation: {op}\n\n"));
    out.push_str("## Git-native message\n\n");

    let mut recorded = false;

    if let Some(message) = git_message.map(str::trim).filter(|m| !m.is_empty()) {
        out.push_str(message);
        recorded = true;
    }

    let source = match (operation, commit_oid) {
        (Operation::CherryPick, Some(oid)) => Some(format!("(cherry picked from commit {oid})")),
        (Operation::Revert, Some(oid)) => Some(format!("This reverts commit {oid}.")),
        (Operation::Merge | Operation::Rebase | Operation::Squash, _) => None,
        (_, None) => None,
    };

    if let Some(source) = source {
        if recorded {
            out.push_str("\n\n");
        }
        out.push_str(&source);
        recorded = true;
    }

    if recorded {
        out.push('\n');
    } else {
        let _ = writeln!(
            out,
            "(none was found; write a commit message that completes this {op})"
        );
    }
}

fn append_change_summary(out: &mut String, snapshot: &ClassifiedSnapshot) {
    let stats = SemanticTextStats::from_snapshot(&snapshot);
    let lock_count = snapshot
        .files
        .iter()
        .filter(|f| f.category == FileCategory::DependencyLock)
        .count();

    // Change summary section (source + lock files)
    if stats.semantic_file_count > 0 || lock_count > 0 {
        out.push_str("\n## Change summary\n\n");
        if stats.semantic_file_count > 0 {
            let _ = writeln!(
                out,
                "- source files: {} changed, +{} insertions/-{} deletions",
                stats.semantic_file_count, stats.total_insertions, stats.total_deletions
            );
        }

        if lock_count > 0 {
            let _ = writeln!(out, "- lock files: {lock_count} changed");
        }
    }

    // Non-text files section: submodule / binary / generated
    const KINDS: &[(FileCategory, &str)] = &[
        (FileCategory::Submodule, "submodule"),
        (FileCategory::Binary, "binary"),
        (FileCategory::Generated, "generated"),
    ];

    let mut any = false;
    for file in &snapshot.files {
        let Some(kind) = KINDS
            .iter()
            .find_map(|(category, kind)| (*category == file.category).then_some(*kind))
        else {
            continue;
        };
        if !any {
            out.push_str("\n## Non-text files (metadata only)\n\n");
            any = true;
        }
        out.push_str(&format_file_line(kind, file));
    }
}

fn append_staged_changes(out: &mut String, payload: &DiffPayload) {
    out.push_str("\n## Staged changes\n\n");
    let body = payload.body.trim_end();
    out.push_str(body);
    out.push('\n');

    if payload.truncated_file_count > 0 {
        let _ = writeln!(
            out,
            "\n{} of {} file diff(s) were truncated to fit the context budget. Do not assume omitted hunks are unchanged.",
            payload.truncated_file_count, payload.file_count
        );
    }
}

// Helper function

fn format_file_line(kind: &str, file: &StagedFile) -> String {
    let status = match file.change_type {
        ChangeType::Added => "Added",
        ChangeType::Modified => "Modified",
        ChangeType::Deleted => "Deleted",
        ChangeType::Renamed => "Renamed",
        ChangeType::Copied => "Copied",
        ChangeType::TypeChanged => "TypeChanged",
    };
    match (file.change_type, file.old_path.as_ref()) {
        (ChangeType::Renamed | ChangeType::Copied, Some(old)) => {
            format!(
                "- [{kind}] {status} {} -> {}\n",
                old.display(),
                file.path.display()
            )
        }
        _ => format!("- [{kind}] {status} {}\n", file.path.display()),
    }
}
