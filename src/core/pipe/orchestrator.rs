use crate::core::git::diff::{BudgetPlanner, DiffExtractor, FileClassifier};
use crate::core::git::operation::OperationStateDetector;
use crate::core::git::preflight::RepoPreflightCollector;
use crate::core::git::staged::StagedMetadataCollector;
use crate::core::git::types::{BudgetPolicy, OperationAction, OperationState, RepositoryContext};
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

use super::context::PipeInput;

pub struct PipeOrchestrator<'a> {
    runner: &'a GitRunner,
    policy: BudgetPolicy,
}

impl<'a> PipeOrchestrator<'a> {
    pub fn new(runner: &'a GitRunner, policy: BudgetPolicy) -> Self {
        Self { runner, policy }
    }

    pub async fn run(&self) -> Result<PipeInput, GitError> {
        // Stage 0: preflight
        // let repo = RepoPreflightCollector::new(self.runner).run().await?;
        let preflight = RepoPreflightCollector::new(self.runner);
        let repo = preflight.run().await?;

        // Stage 1: Operation
        let state = OperationStateDetector::new(self.runner).run(&repo).await?;

        match state.action() {
            OperationAction::Abort => Err(abort_error(&state)),
            OperationAction::Reuse | OperationAction::Template => {
                let operation = state
                    .kind()
                    .expect("reuse/template state always carries Operation");
                Ok(PipeInput::FromOperation {
                    repo,
                    operation,
                    message: state.seed_message(),
                    commit_oid: resolve_commit_oid(&state),
                })
            }
            OperationAction::Continue => {
                preflight.ensure_staged_changes().await?;
                self.collect_staged_diff(repo).await
            }
        }
    }

    async fn collect_staged_diff(&self, repo: RepositoryContext) -> Result<PipeInput, GitError> {
        // Stage 2: Satged metadata
        let metadata = StagedMetadataCollector::new(self.runner).collect().await?;

        // Stage 3.1: File classification
        let snapshot = FileClassifier::new(self.runner).classify(&metadata).await?;

        // Stage 3.2 Budget planning
        let decision = BudgetPlanner::new(self.policy).plan(&snapshot);

        // Stage 3.3 Diff extractor
        let payload = DiffExtractor::new(self.runner)
            .extract(&snapshot, &decision)
            .await?;

        Ok(PipeInput::FromStaging {
            repo,
            snapshot,
            payload,
            decision,
        })
    }
}

fn resolve_commit_oid(state: &OperationState) -> Option<String> {
    match state {
        OperationState::CherryPick { head, .. } | OperationState::Revert { head, .. } => {
            Some(head.clone())
        }
        _ => None,
    }
}

fn abort_error(state: &OperationState) -> GitError {
    match state {
        OperationState::Bisect => GitError::new(
            GitErrorCode::Other,
            "repository is in the middle of a git bisect; finish or abort it before committing",
        ),
        OperationState::Conflicts { context } => {
            let detail = match context {
                Some(op) => format!(" during {}", op.as_str()),
                None => String::new(),
            };
            GitError::new(
                GitErrorCode::Other,
                format!("unresolved conflicts{detail}; resolve them before committing"),
            )
        }
        _ => GitError::new(
            GitErrorCode::Other,
            "pipeline attempted to abort from a non-abort operation state",
        ),
    }
}
