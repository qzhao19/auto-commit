use crate::core::git::types::BudgetPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetPlanner {
    policy: BudgetPolicy,
}

