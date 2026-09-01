use crate::core::git::types::{
    BudgetDecision, ClassifiedSnapshot, DiffPayload, Operation, RepositoryContext,
};

#[derive(Debug, Clone)]
pub enum PipeInput {
    /// merge / rebase / squash / cherry-pick / revert
    FromOperation {
        repo: RepositoryContext,
        operation: Operation,
        message: Option<String>,
        commit_oid: Option<String>,
    },
    /// Regular branch after stages 2–3
    FromStaging {
        repo: RepositoryContext,
        snapshot: ClassifiedSnapshot,
        payload: DiffPayload,
        decision: BudgetDecision,
    },
}
