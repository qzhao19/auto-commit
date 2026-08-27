use std::path::{Path, PathBuf};

use toml::de;

// ---- Repository Preflight ----

/// Paths inside the Git directory that are used by later stages.
#[derive(Debug, Clone)]
pub struct GitPaths {
    pub git_dir: PathBuf,

    // Repository state
    pub index_lock: PathBuf,

    // Merge
    pub merge_head: PathBuf,
    pub merge_msg: PathBuf,

    // Rebase
    pub rebase_head: PathBuf,
    pub rebase_merge: PathBuf,
    pub rebase_apply: PathBuf,

    // Cherry-pick
    pub cherry_pick_head: PathBuf,

    // Revert
    pub revert_head: PathBuf,

    // Squash
    pub squash_msg: PathBuf,

    // Bisect
    pub bisect_log: PathBuf,
}

impl GitPaths {
    pub fn from_git_dir(git_dir: PathBuf) -> Self {
        Self {
            index_lock: git_dir.join("index.lock"),

            merge_head: git_dir.join("MERGE_HEAD"),
            merge_msg: git_dir.join("MERGE_MSG"),

            rebase_head: git_dir.join("REBASE_HEAD"),
            rebase_merge: git_dir.join("rebase-merge"),
            rebase_apply: git_dir.join("rebase-apply"),

            cherry_pick_head: git_dir.join("CHERRY_PICK_HEAD"),

            revert_head: git_dir.join("REVERT_HEAD"),

            squash_msg: git_dir.join("SQUASH_MSG"),

            bisect_log: git_dir.join("BISECT_LOG"),

            git_dir,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RepositoryContext {
    /// Repo working tree root.
    pub worktree_root: PathBuf,

    /// Absolute path to the Git directory.
    pub git_paths: GitPaths,

    /// HEAD commit object ID.
    pub head_oid: Option<String>,

    /// Current branch name.
    pub branch: Option<String>,
}

impl RepositoryContext {
    pub fn git_dir(&self) -> &Path {
        &self.git_paths.git_dir
    }

    pub fn is_initial_commit(&self) -> bool {
        self.head_oid.is_none()
    }

    pub fn is_detached_head(&self) -> bool {
        self.branch.is_none()
    }
}

// ---- Git operation state ----

/// Merge operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Rebase,
    Merge,
    Squash,
    CherryPick,
    Revert,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rebase => "rebase",
            Self::Merge => "merge",
            Self::Squash => "squash",
            Self::CherryPick => "cherry-pick",
            Self::Revert => "revert",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationAction {
    Abort,
    Reuse,
    Template,
    Continue,
}

/// Detection ops state:
/// bisect → rebase → merge → squash → cherry-pick → revert →
/// conflicts → clean.
#[derive(Debug, Clone, PartialEq)]
pub enum OperationState {
    Conflicts {
        context: Option<Operation>,
    },
    Bisect,
    Rebase {
        message: Option<String>,
    },
    Merge {
        message: Option<String>,
    },
    Squash {
        message: Option<String>,
    },
    CherryPick {
        head: String,
        subject: Option<String>,
    },
    Revert {
        head: String,
        subject: Option<String>,
    },
    Clean,
}

impl OperationState {
    pub fn kind(&self) -> Option<Operation> {
        match self {
            Self::Rebase { .. } => Some(Operation::Rebase),
            Self::Merge { .. } => Some(Operation::Merge),
            Self::Squash { .. } => Some(Operation::Squash),
            Self::CherryPick { .. } => Some(Operation::CherryPick),
            Self::Revert { .. } => Some(Operation::Revert),
            Self::Conflicts { .. } | Self::Bisect | Self::Clean => None,
        }
    }

    /// Action the pipeline should take for this detected state
    pub fn action(&self) -> OperationAction {
        match self {
            Self::Conflicts { .. } | Self::Bisect => OperationAction::Abort,
            Self::Rebase { .. } | Self::Merge { .. } | Self::Squash { .. } => {
                OperationAction::Reuse
            }
            Self::CherryPick { .. } | Self::Revert { .. } => OperationAction::Template,
            Self::Clean => OperationAction::Continue,
        }
    }

    pub fn seed_message(&self) -> Option<String> {
        match self {
            Self::Conflicts { .. } | Self::Bisect | Self::Clean => None,
            Self::Rebase { message } | Self::Merge { message } | Self::Squash { message } => {
                message.clone()
            }
            Self::CherryPick { subject, .. } => subject.clone(),
            Self::Revert { subject, .. } => subject.as_ref().map(|s| format!("Revert \"{s}\"")),
        }
    }
}

// ---- Staged Metadata ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
    Renamed,
    Copied,
    TypeChanged,
}

/// Decision-level classification of a staged path
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileCategory {
    Unknown,
    Submodule,
    Binary,
    DependencyLock,
    Generated,
    SemanticText,
}

#[derive(Debug, Clone)]
pub struct StagedFile {
    /// Final path in the index / post-image
    pub path: PathBuf,

    /// Original path for rename/copy
    pub old_path: Option<PathBuf>,

    pub change_type: ChangeType,

    /// Rename/copy similarity score, e.g. 95 means 95%
    pub similarity: Option<u8>,

    /// Number of added lines, `None` for Binary / Submodule
    pub insertions: Option<u64>,

    /// Number of deleted lines, `None` for Binary / Submodule
    pub deletions: Option<u64>,

    /// File classification
    pub category: FileCategory,
}

#[derive(Debug, Clone, Default)]
pub struct StagedSnapshot {
    pub files: Vec<StagedFile>,
}

// ---- File Classification ----

#[derive(Debug, Clone, Default)]
pub struct ClassifiedSnapshot {
    pub files: Vec<StagedFile>,
}

impl ClassifiedSnapshot {
    pub fn from_files(files: Vec<StagedFile>) -> Self {
        debug_assert!(
            files.iter().all(|f| f.category != FileCategory::Unknown),
            "ClassifiedSnapshot must not contain Unknown"
        );
        Self { files }
    }
}

// ---- Budget Planner ----

/// SemanticText layer
#[derive(Debug, Clone, Copy, Default)]
pub struct SemanticTextStats {
    pub semantic_file_count: usize,

    pub content_changed_file_count: usize,

    pub rename_only_file_count: usize,

    pub total_insertions: u64,
    pub total_deletions: u64,

    pub max_file_changed_lines: u64,
}

impl SemanticTextStats {
    pub fn from_snapshot(snapshot: &ClassifiedSnapshot) -> Self {
        let mut stats = Self::default();

        for file in &snapshot.files {
            if file.category != FileCategory::SemanticText {
                continue;
            }

            stats.semantic_file_count += 1;

            let insertions = file.insertions.unwrap_or(0);
            let deletions = file.deletions.unwrap_or(0);
            let changed_lines = insertions + deletions;

            if changed_lines > 0 {
                stats.content_changed_file_count += 1;
            } else if file.change_type == ChangeType::Renamed {
                stats.rename_only_file_count += 1;
            }

            stats.total_insertions += insertions;
            stats.total_deletions += deletions;
            stats.max_file_changed_lines = stats.max_file_changed_lines.max(changed_lines);
        }

        stats
    }

    #[inline]
    pub const fn total_changed_lines(&self) -> u64 {
        self.total_deletions + self.total_insertions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetPolicy {
    pub context_token_limit: u64,
    pub reserved_tokens: u64,

    /// Path + status header of one content-changing file.
    pub tokens_per_file_overhead: u64,
    /// One `+` or `-` line.
    pub tokens_per_changed_line: u64,
    /// One rename-only entry (`old -> new`).
    pub tokens_per_rename_only: u64,

    /// Inflates the estimate before comparing against the budget.
    /// 10000 = 1.0, 12000 = 1.2.
    pub safety_factor_bps: u32,

    /// If estimate > available × this, skip truncation and go path-only.
    pub summary_only_multiplier: u32,

    /// TruncateLines strategy
    pub max_changed_lines_per_file: u64,

    /// SampleHunks strategy
    pub max_hunks_per_file: u32,
}

impl BudgetPolicy {
    pub fn default() -> Self {
        Self {
            context_token_limit: 128_000,
            reserved_tokens: 2_000,

            tokens_per_file_overhead: 48,
            tokens_per_changed_line: 10,
            tokens_per_rename_only: 24,

            safety_factor_bps: 13_000,
            summary_only_multiplier: 3,

            max_changed_lines_per_file: 400,
            max_hunks_per_file: 8,
        }
    }

    pub fn available_for_diff(&self) -> u64 {
        self.context_token_limit
            .saturating_sub(self.reserved_tokens)
    }

    pub fn estimate_diff_tokens(&self, stats: &SemanticTextStats) -> u64 {
        // estimated = file_header_overhead + rename_entries + line_cost
        let raw = (stats.content_changed_file_count as u64)
            .saturating_mul(self.tokens_per_file_overhead)
            .saturating_add(
                (stats.rename_only_file_count as u64).saturating_mul(self.tokens_per_rename_only),
            )
            .saturating_add(
                stats
                    .total_changed_lines()
                    .saturating_mul(self.tokens_per_changed_line),
            );

        raw.saturating_mul(u64::from(self.safety_factor_bps)) / 10_000
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStrategy {
    /// Whole semanticText diff
    Full,

    /// Max number of lines per file at `max_changed_lines_per_file`
    TruncateLines { max_changed_lines_per_file: u64 },

    /// Max number of hunks per file at `max_hunks_per_file`
    SampleHunks {
        max_hunks_per_file: u32,
        max_changed_lines_per_file: u64,
    },

    /// Paths and change type only
    PathSummaryOnly,
}

/// Planner dict for one classified snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetDecision {
    pub strategy: DiffStrategy,

    /// Estimate after `safety_factor_bps`
    pub estimated_diff_tokens: u64,

    /// `context_token_limit - reserved_tokens`
    pub available_for_diff: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffPayload {
    pub body: String,
    pub file_count: usize,
    pub truncated_file_count: usize,
}
