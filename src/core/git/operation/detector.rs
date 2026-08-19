use std::path::Path;

use crate::core::git::types::{GitPaths, OperationState, RepositoryContext};
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

/// Stage 1: Operation State detection.
///
/// Responsibilities:
///
/// 1. unresolved conflicts (`git ls-files --unmerged`)   → ABORT
/// 2. bisect (`BISECT_LOG`)                              → ABORT
/// 3. rebase (`rebase-merge/` or `rebase-apply/`)        → REUSE
/// 4. merge (`MERGE_HEAD`)                               → REUSE / TEMPLATE
/// 5. squash (`SQUASH_MSG` without `MERGE_HEAD`)         → REUSE / TEMPLATE
/// 6. cherry-pick (`CHERRY_PICK_HEAD`)                   → TEMPLATE
/// 7. revert (`REVERT_HEAD`)                             → TEMPLATE
/// 8. clean                                              → continue to stage 2
///
pub struct OperationStateDetector<'a> {
    runner: &'a GitRunner,
}

impl<'a> OperationStateDetector<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn run(&self, ctx: &RepositoryContext) -> Result<OperationState, GitError> {
        let paths = &ctx.git_paths;

        // 1.1 Conflict: a conflicted rebase leaves both unmerged
        // entries AND rebase-merge
        if self.has_unmerged_entries().await? {
            return Ok(OperationState::Conflicts);
        }

        // 1.2 Bisect → hard abort.
        if Self::path_exists(&paths.bisect_log)? {
            return Ok(OperationState::Bisect);
        }

        // 1.3 Rebase
        if Self::path_exists(&paths.rebase_merge)? || Self::path_exists(&paths.rebase_apply)? {
            let message = self.read_rebase_seed(paths).await?;
            return Ok(OperationState::Rebase { message });
        }

        // 1.6 Merge
        if Self::path_exists(&paths.merge_head)? {
            let message = Self::read_msg_file(&paths.merge_msg)?;
            return Ok(OperationState::Merge { message });
        }

        // 1.7 Squash: SQUASH_MSG without MERGE_HEAD.
        if Self::path_exists(&paths.squash_msg)? {
            let message = Self::read_msg_file(&paths.squash_msg)?;
            return Ok(OperationState::Squash { message });
        }

        // 1.4 Cherry-pick.
        if Self::path_exists(&paths.cherry_pick_head)? {
            let head = Self::read_head_oid(&paths.cherry_pick_head)?;
            let subject = self.head_subject(&head).await?;
            return Ok(OperationState::CherryPick { head, subject });
        }

        // 1.5 Revert.
        if Self::path_exists(&paths.revert_head)? {
            let head = Self::read_head_oid(&paths.revert_head)?;
            let subject = self.head_subject(&head).await?;
            return Ok(OperationState::Revert { head, subject });
        }

        // 1.8 Clean.
        Ok(OperationState::Clean)
    }

    // `git ls-files --unmerged` output = unresolved conflicts
    async fn has_unmerged_entries(&self) -> Result<bool, GitError> {
        let result = self.runner.run(&["ls-files", "--unmerged"], None).await?;
        Ok(!result.stdout.is_empty())
    }

    async fn read_rebase_seed(&self, paths: &GitPaths) -> Result<Option<String>, GitError> {
        let msg_path = paths.rebase_merge.join("message");

        if let Some(msg) = Self::read_msg_file(&msg_path)? {
            return Ok(Some(msg));
        }

        if Self::path_exists(&paths.rebase_head)? {
            let head = Self::read_head_oid(&paths.rebase_head)?;
            return self.head_subject(&head).await;
        }

        Ok(None)
    }

    async fn head_subject(&self, head: &str) -> Result<Option<String>, GitError> {
        let result = self
            .runner
            .run(&["log", "-1", "--format=%s", head], None)
            .await?;
        let subject = result.stdout_str().trim().to_owned();

        Ok(if subject.is_empty() {
            None
        } else {
            Some(subject)
        })
    }

    // Helper function

    fn read_msg_file(path: &Path) -> Result<Option<String>, GitError> {
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(GitError::new(
                    GitErrorCode::Other,
                    format!("failed to read {}: {err}", path.display()),
                ));
            }
        };

        let msg = content
            .lines()
            .filter(|line| !line.trim_start().starts_with("#"))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_owned();

        Ok(if msg.is_empty() { None } else { Some(msg) })
    }

    fn read_head_oid(path: &Path) -> Result<String, GitError> {
        let oid = std::fs::read_to_string(path)
            .map_err(|err| {
                GitError::new(
                    GitErrorCode::Other,
                    format!("failed to read {}: {err}", path.display()),
                )
            })?
            .trim()
            .to_owned();

        if oid.is_empty() {
            return Err(GitError::new(
                GitErrorCode::Other,
                format!("state file {} is present but empty", path.display()),
            ));
        }

        Ok(oid)
    }

    fn path_exists(path: &Path) -> Result<bool, GitError> {
        path.try_exists().map_err(|err| {
            GitError::new(
                GitErrorCode::Other,
                format!("failed to inspect {}: {err}", path.display()),
            )
        })
    }
}
