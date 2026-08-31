mod classifier;
mod extractor;
mod planner;
mod rules;

pub use classifier::FileClassifier;
pub use extractor::DiffExtractor;
pub use planner::BudgetPlanner;

#[cfg(test)]
pub(crate) use rules::{
    classify_by_header, classify_by_name, match_generated_header, match_generated_name,
    match_generated_path, match_lock_file,
};

#[cfg(test)]
pub(crate) use extractor::{
    is_lock_signal_line, path_summary, split_sections, summarize_lock_diff, truncate_section,
};
