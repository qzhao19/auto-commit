use std::path::PathBuf;

use crate::core::git::diff::planner::BudgetPlanner;
use crate::core::git::types::{
    BudgetPolicy, ChangeType, ClassifiedSnapshot, DiffStrategy, FileCategory, StagedFile,
};

//  helpers

/// Clean-number policy: safety_factor 1.0 makes every expectation exact.
/// available = 10_000 - 1_000 = 9_000; line cap 100; summary gate 3x = 27_000.
fn policy() -> BudgetPolicy {
    BudgetPolicy {
        context_token_limit: 10_000,
        reserved_tokens: 1_000,
        tokens_per_file_overhead: 10,
        tokens_per_changed_line: 10,
        tokens_per_rename_only: 20,
        safety_factor_bps: 10_000,
        summary_only_multiplier: 3,
        max_changed_lines_per_file: 100,
        max_hunks_per_file: 4,
    }
}

fn file(
    path: &str,
    change_type: ChangeType,
    insertions: Option<u64>,
    deletions: Option<u64>,
    category: FileCategory,
) -> StagedFile {
    StagedFile {
        path: PathBuf::from(path),
        old_path: None,
        change_type,
        similarity: None,
        insertions,
        deletions,
        category,
    }
}

/// SemanticText Modified file with the given churn.
fn semantic(path: &str, insertions: u64, deletions: u64) -> StagedFile {
    file(
        path,
        ChangeType::Modified,
        Some(insertions),
        Some(deletions),
        FileCategory::SemanticText,
    )
}

fn snapshot(files: Vec<StagedFile>) -> ClassifiedSnapshot {
    ClassifiedSnapshot::from_files(files)
}

//  Full

#[test]
fn empty_snapshot_is_full() {
    let decision = BudgetPlanner::new(policy()).plan(&snapshot(vec![]));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 0);
    assert_eq!(decision.available_for_diff, 9_000);
}

#[test]
fn non_semantic_files_are_ignored() {
    // Filtering must key on category, not on `None` counts: the lock and
    // generated entries carry large numstat values and must still cost 0.
    let snapshot = snapshot(vec![
        file(
            "logo.png",
            ChangeType::Modified,
            None,
            None,
            FileCategory::Binary,
        ),
        file(
            "Cargo.lock",
            ChangeType::Modified,
            Some(50_000),
            Some(40_000),
            FileCategory::DependencyLock,
        ),
        file(
            "gen/api.pb.go",
            ChangeType::Modified,
            Some(30_000),
            Some(30_000),
            FileCategory::Generated,
        ),
        file(
            "vendor/lib",
            ChangeType::Modified,
            None,
            None,
            FileCategory::Submodule,
        ),
    ]);

    let decision = BudgetPlanner::new(policy()).plan(&snapshot);

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 0);
}

#[test]
fn small_diff_within_budget_is_full() {
    // 10 files x 50 lines: 10x10 overhead + 500x10 lines = 5_100.
    let files: Vec<StagedFile> = (0..10)
        .map(|i| semantic(&format!("src/{i}.rs"), 50, 0))
        .collect();

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 5_100);
    assert_eq!(decision.available_for_diff, 9_000);
}

//  TruncateLines

#[test]
fn oversized_single_file_within_budget_truncates() {
    // Budget is fine (1_510 <= 9_000) but the 150-line file breaks the
    // quality cap — truncation is a quality gate, independent of budget.
    let decision = BudgetPlanner::new(policy()).plan(&snapshot(vec![semantic("big.rs", 150, 0)]));

    assert_eq!(
        decision.strategy,
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 100
        }
    );
    assert_eq!(decision.estimated_diff_tokens, 1_510);
}

#[test]
fn mixed_huge_and_small_truncates_instead_of_path_summary() {
    // Raw estimate 55_110 blows past the 27_000 summary gate — but the
    // gate must be applied to the *capped* estimate (6_110), so the ten
    // small files keep their diff. Regression guard for the planner bug
    // where one huge file dragged a mixed snapshot to path-only.
    let mut files = vec![semantic("monster.rs", 5_000, 0)];
    files.extend((0..10).map(|i| semantic(&format!("src/{i}.rs"), 50, 0)));

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(
        decision.strategy,
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 100
        }
    );
    // The decision echoes the UNCAPPED estimate.
    assert_eq!(decision.estimated_diff_tokens, 55_110);
}

#[test]
fn capped_estimate_exactly_equal_to_available_truncates() {
    // capped = 90 overhead + (100+700+91)x10 = exactly 9_000 — "fits"
    // must be `<=`, otherwise this falls through to SampleHunks and
    // loses information for free.
    let mut files = vec![semantic("a.rs", 150, 0)];
    files.extend((0..7).map(|i| semantic(&format!("b{i}.rs"), 100, 0)));
    files.push(semantic("c.rs", 91, 0));

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(
        decision.strategy,
        DiffStrategy::TruncateLines {
            max_changed_lines_per_file: 100
        }
    );
}

//  PathSummaryOnly

#[test]
fn widespread_overflow_goes_path_summary_only() {
    // 100 files x 5_000 lines: even capped (101_000) far above the
    // 27_000 gate — truncation cannot save it.
    let files: Vec<StagedFile> = (0..100)
        .map(|i| semantic(&format!("src/{i}.rs"), 5_000, 0))
        .collect();

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
}

#[test]
fn zero_available_budget_is_path_summary_only() {
    let mut policy = policy();
    policy.reserved_tokens = 10_000; // == context limit

    let decision = BudgetPlanner::new(policy).plan(&snapshot(vec![semantic("a.rs", 10, 10)]));

    assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
    assert_eq!(decision.available_for_diff, 0);
}

#[test]
fn reserved_over_context_saturates_to_path_summary_only() {
    let mut policy = policy();
    policy.reserved_tokens = 20_000; // > context limit

    let decision = BudgetPlanner::new(policy).plan(&snapshot(vec![semantic("a.rs", 10, 10)]));

    assert_eq!(decision.strategy, DiffStrategy::PathSummaryOnly);
    assert_eq!(decision.available_for_diff, 0);
}

//  SampleHunks

#[test]
fn middle_overflow_samples_hunks() {
    // 10 files x 200 lines: capped 10_100 sits between available (9_000)
    // and the summary gate (27_000) — hunk sampling with both caps.
    let files: Vec<StagedFile> = (0..10)
        .map(|i| semantic(&format!("src/{i}.rs"), 200, 0))
        .collect();

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(
        decision.strategy,
        DiffStrategy::SampleHunks {
            max_hunks_per_file: 4,
            max_changed_lines_per_file: 100
        }
    );
}

//  cost model details

#[test]
fn rename_only_files_are_charged_flat() {
    let files = vec![
        file(
            "a.rs",
            ChangeType::Renamed,
            Some(0),
            Some(0),
            FileCategory::SemanticText,
        ),
        file(
            "b.rs",
            ChangeType::Renamed,
            Some(0),
            Some(0),
            FileCategory::SemanticText,
        ),
        file(
            "c.rs",
            ChangeType::Renamed,
            Some(0),
            Some(0),
            FileCategory::SemanticText,
        ),
    ];

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    // 3 x 24? No — policy sets 20: 3 x 20 = 60, no per-file overhead.
    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 60);
}

#[test]
fn renamed_with_content_is_charged_as_content() {
    // churn > 0 puts the rename in the content bucket: overhead + lines.
    let files = vec![file(
        "moved.rs",
        ChangeType::Renamed,
        Some(30),
        Some(20),
        FileCategory::SemanticText,
    )];

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 510); // 10 + 50x10
}

#[test]
fn zero_churn_type_change_costs_nothing() {
    // Current behavior snapshot: zero-churn non-rename entries are free
    // (the known estimation gap — update this test when they get a cost).
    let files = vec![file(
        "mode.sh",
        ChangeType::TypeChanged,
        Some(0),
        Some(0),
        FileCategory::SemanticText,
    )];

    let decision = BudgetPlanner::new(policy()).plan(&snapshot(files));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 0);
}

#[test]
fn safety_factor_inflates_estimate() {
    let mut policy = policy();
    policy.safety_factor_bps = 13_000;

    // 1 file x 99 lines: raw = 10 + 990 = 1_000, inflated x1.3 = 1_300.
    let decision = BudgetPlanner::new(policy).plan(&snapshot(vec![semantic("a.rs", 99, 0)]));

    assert_eq!(decision.strategy, DiffStrategy::Full);
    assert_eq!(decision.estimated_diff_tokens, 1_300);
}

//  planner plumbing

#[test]
fn policy_accessor_returns_constructed_policy() {
    // Regression guard: this accessor once recursed infinitely
    // (`&self.policy()` calling itself) and compiled with only a warning.
    let policy = policy();
    let planner = BudgetPlanner::new(policy);

    assert_eq!(planner.policy(), &policy);
}

#[test]
fn default_planner_uses_default_policy() {
    let planner = BudgetPlanner::default();

    assert_eq!(planner.policy(), &BudgetPolicy::default());
}
