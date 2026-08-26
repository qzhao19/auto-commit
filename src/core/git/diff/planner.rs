use crate::core::git::types::{
    BudgetDecision, BudgetPolicy, ChangeType, ClassifiedSnapshot, DiffStrategy, FileCategory,
    SemanticTextStats,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetPlanner {
    policy: BudgetPolicy,
}

impl BudgetPlanner {
    pub fn new(policy: BudgetPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &BudgetPolicy {
        &self.policy
    }

    pub fn plan(&self, snapshot: &ClassifiedSnapshot) -> BudgetDecision {
        let stats = SemanticTextStats::from_snapshot(snapshot);
        let available = self.policy.available_for_diff();
        let estimated = self.policy.estimate_diff_tokens(&stats);

        BudgetDecision {
            strategy: self.select_strategy(snapshot, &stats, estimated, available),
            estimated_diff_tokens: estimated,
            available_for_diff: available,
        }
    }

    fn select_strategy(
        &self,
        snapshot: &ClassifiedSnapshot,
        stats: &SemanticTextStats,
        estimated: u64,
        available: u64,
    ) -> DiffStrategy {
        let policy = &self.policy;
        let line_capacity = policy.max_changed_lines_per_file;

        if available == 0 {
            return DiffStrategy::PathSummaryOnly;
        }

        if estimated <= available && stats.max_file_changed_lines <= line_capacity {
            return DiffStrategy::Full;
        }

        let limited = Self::estimate_with_line_capacity(policy, snapshot);
        if limited <= available {
            return DiffStrategy::TruncateLines {
                max_changed_lines_per_file: line_capacity,
            };
        }

        let summary_threshold =
            available.saturating_mul(u64::from(policy.summary_only_multiplier.max(1)));
        if limited > summary_threshold {
            return DiffStrategy::PathSummaryOnly;
        }

        DiffStrategy::SampleHunks {
            max_hunks_per_file: policy.max_hunks_per_file,
            max_changed_lines_per_file: line_capacity,
        }
    }

    fn estimate_with_line_capacity(policy: &BudgetPolicy, snapshot: &ClassifiedSnapshot) -> u64 {
        let capacity = policy.max_changed_lines_per_file;
        let mut raw = 0u64;

        for file in &snapshot.files {
            if file.category != FileCategory::SemanticText {
                continue;
            }

            let changed = file.insertions.unwrap_or(0) + file.deletions.unwrap_or(0);
            if changed > 0 {
                raw = raw
                    .saturating_add(policy.tokens_per_file_overhead)
                    .saturating_add(
                        changed
                            .min(capacity)
                            .saturating_mul(policy.tokens_per_changed_line),
                    );
            } else if file.change_type == ChangeType::Renamed {
                raw = raw.saturating_add(policy.tokens_per_rename_only);
            }
        }
        raw.saturating_mul(u64::from(policy.safety_factor_bps)) / 10_000
    }
}

impl Default for BudgetPlanner {
    fn default() -> Self {
        Self::new(BudgetPolicy::default())
    }
}
