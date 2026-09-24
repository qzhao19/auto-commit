use std::path::Path;

use crate::core::git::types::{GitPaths, OperationState, RepositoryContext};
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

/// Stage 1: Operation State detection.
///
/// Responsibilities:
///
/// 1. bisect (`BISECT_LOG`)                              → ABORT
/// 2. rebase (`rebase-merge/` or `rebase-apply/`)        → REUSE
/// 3. merge (`MERGE_HEAD`)                               → REUSE
/// 4. squash (`SQUASH_MSG`, no `MERGE_HEAD`)             → REUSE
/// 5. cherry-pick (`CHERRY_PICK_HEAD`)                   → TEMPLATE
/// 6. revert (`REVERT_HEAD`)                             → TEMPLATE
/// 7. unmerged index → CONFLICTS, carrying the owning operation
/// 8. clean
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

        // 1.1 Bisect → hard abort.
        if Self::path_exists(&paths.bisect_log).await? {
            return Ok(OperationState::Bisect);
        }

        if self.has_unmerged_entries().await? {
            let context = self
                .probe_active_operation(paths)
                .await
                .ok()
                .flatten()
                .as_ref()
                .and_then(OperationState::kind);
            return Ok(OperationState::Conflicts { context });
        }

        let in_progress = self.probe_active_operation(paths).await?;
        Ok(in_progress.unwrap_or(OperationState::Clean))
    }

    // async fn probe_active_operation(
    //     &self,
    //     paths: &GitPaths,
    // ) -> Result<Option<OperationState>, GitError> {
    //     // 1.2 Rebase
    //     if Self::path_exists(&paths.rebase_merge).await?
    //         || Self::path_exists(&paths.rebase_apply).await?
    //     {
    //         let message = self.read_rebase_seed(paths).await?;
    //         return Ok(Some(OperationState::Rebase { message }));
    //     }
    //     // 1.3 Merge HEAD
    //     if Self::path_exists(&paths.merge_head).await? {
    //         let message = Self::read_msg_file(&paths.merge_msg).await?;
    //         return Ok(Some(OperationState::Merge { message }));
    //     }
    //     // 1.4 Squash: SQUASH_MSG without MERGE_HEAD
    //     if Self::path_exists(&paths.squash_msg).await? {
    //         let message = Self::read_msg_file(&paths.squash_msg).await?;
    //         return Ok(Some(OperationState::Squash { message }));
    //     }
    //     // 1.5 Cherry-pick
    //     if Self::path_exists(&paths.cherry_pick_head).await? {
    //         let head = Self::read_head_oid(&paths.cherry_pick_head).await?;
    //         let subject = self.head_subject(&head).await?;
    //         return Ok(Some(OperationState::CherryPick { head, subject }));
    //     }
    //     // Revert
    //     if Self::path_exists(&paths.revert_head).await? {
    //         let head = Self::read_head_oid(&paths.revert_head).await?;
    //         let subject = self.head_subject(&head).await?;
    //         return Ok(Some(OperationState::Revert { head, subject }));
    //     }
    //     Ok(None)
    // }

    async fn probe_active_operation(
        &self,
        paths: &GitPaths,
    ) -> Result<Option<OperationState>, GitError> {
        // Marker existence checks are independent and all must succeed 
        // rebase > merge > squash > cherry-pick > revert
        let (rebase_merge, rebase_apply, merge_head, squash_msg, cherry_pick_head, revert_head) = tokio::try_join!(
            Self::path_exists(&paths.rebase_merge),
            Self::path_exists(&paths.rebase_apply),
            Self::path_exists(&paths.merge_head),
            Self::path_exists(&paths.squash_msg),
            Self::path_exists(&paths.cherry_pick_head),
            Self::path_exists(&paths.revert_head),
        )?;

        if rebase_merge || rebase_apply {
            let message = self.read_rebase_seed(paths).await?;
            return Ok(Some(OperationState::Rebase { message }));
        }
        if merge_head {
            let message = Self::read_msg_file(&paths.merge_msg).await?;
            return Ok(Some(OperationState::Merge { message }));
        }
        if squash_msg {
            let message = Self::read_msg_file(&paths.squash_msg).await?;
            return Ok(Some(OperationState::Squash { message }));
        }
        if cherry_pick_head {
            let head = Self::read_head_oid(&paths.cherry_pick_head).await?;
            let subject = self.head_subject(&head).await?;
            return Ok(Some(OperationState::CherryPick { head, subject }));
        }
        if revert_head {
            let head = Self::read_head_oid(&paths.revert_head).await?;
            let subject = self.head_subject(&head).await?;
            return Ok(Some(OperationState::Revert { head, subject }));
        }

        Ok(None)
    }

    // `git ls-files --unmerged` output = unresolved conflicts
    async fn has_unmerged_entries(&self) -> Result<bool, GitError> {
        let result = self.runner.run(&["ls-files", "--unmerged"], None).await?;
        Ok(!result.stdout.is_empty())
    }

    async fn read_rebase_seed(&self, paths: &GitPaths) -> Result<Option<String>, GitError> {
        let msg_path = paths.rebase_merge.join("message");

        if let Some(msg) = Self::read_msg_file(&msg_path).await? {
            return Ok(Some(msg));
        }

        if Self::path_exists(&paths.rebase_head).await? {
            let head = Self::read_head_oid(&paths.rebase_head).await?;
            return self.head_subject(&head).await;
        }

        Ok(None)
    }

    async fn head_subject(&self, head: &str) -> Result<Option<String>, GitError> {
        let subject = match self
            .runner
            .run(&["log", "-1", "--format=%s", head], None)
            .await
        {
            Ok(result) => result.stdout_str().trim().to_owned(),
            Err(_) => String::new(),
        };

        Ok(if subject.is_empty() {
            None
        } else {
            Some(subject)
        })
    }

    // Helper function

    async fn read_msg_file(path: &Path) -> Result<Option<String>, GitError> {
        let content = match tokio::fs::read_to_string(path).await {
            Ok(content) => content,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(GitError::with_source(
                    GitErrorCode::Other,
                    format!("failed to read {}: {err}", path.display()),
                    err,
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

    async fn read_head_oid(path: &Path) -> Result<String, GitError> {
        let oid = tokio::fs::read_to_string(path)
            .await
            .map_err(|err| {
                GitError::with_source(
                    GitErrorCode::Other,
                    format!("failed to read {}: {err}", path.display()),
                    err,
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

    async fn path_exists(path: &Path) -> Result<bool, GitError> {
        tokio::fs::try_exists(path).await.map_err(|err| {
            GitError::with_source(
                GitErrorCode::Other,
                format!("failed to inspect {}: {err}", path.display()),
                err,
            )
        })
    }
}
