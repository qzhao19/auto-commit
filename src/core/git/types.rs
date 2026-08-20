use std::path::{Path, PathBuf};

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
    /// `None` means the repository has no commits yet.
    pub head_oid: Option<String>,

    /// Current branch name.
    /// `None` means detached HEAD.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationAction {
    Abort,
    Reuse,
    Template,
    Continue,
}

/// Detection ops state:
/// conflicts → bisect → rebase → merge → squash → cherry-pick → revert → clean.
#[derive(Debug, Clone, PartialEq)]
pub enum OperationState {
    Conflicts,
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
    /// Action the pipeline should take for this detected state
    pub fn action(&self) -> OperationAction {
        match self {
            Self::Conflicts | Self::Bisect => OperationAction::Abort,
            Self::Rebase { .. } | Self::Merge { .. } | Self::Squash { .. } => {
                OperationAction::Reuse
            }
            Self::CherryPick { .. } | Self::Revert { .. } => OperationAction::Template,
            Self::Clean => OperationAction::Continue,
        }
    }

    /// Seed message for later stages:
    /// - reuse
    /// - hand to the LLM for polishing.
    /// - `None` for abort/clean states or when the source is unavailable
    pub fn seed_message(&self) -> Option<String> {
        match self {
            Self::Conflicts | Self::Bisect | Self::Clean => None,
            Self::Rebase { message } | Self::Merge { message } | Self::Squash { message } => {
                message.clone()
            }
            Self::CherryPick { subject, .. } => subject.clone(),
            Self::Revert { subject, .. } => subject.as_ref().map(|s| format!("Revert \"{s}\"")),
        }
    }
}

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
    SemanticText, // Ordinary text file (line stats available)
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

impl StagedSnapshot {
    pub fn total_files(&self) -> usize {
        self.files.len()
    }

    pub fn total_insertions(&self) -> u64 {
        self.files.iter().filter_map(|file| file.insertions).sum()
    }

    pub fn total_deletions(&self) -> u64 {
        self.files.iter().filter_map(|file| file.deletions).sum()
    }

    pub fn text_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.category == FileCategory::SemanticText)
            .count()
    }

    pub fn binary_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.category == FileCategory::Binary)
            .count()
    }

    pub fn submodule_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.category == FileCategory::Submodule)
            .count()
    }

    pub fn generated_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.category == FileCategory::Generated)
            .count()
    }

    pub fn dependency_lock_file_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.category == FileCategory::DependencyLock)
            .count()
    }

    pub fn has_rename(&self) -> bool {
        self.files
            .iter()
            .any(|file| file.change_type == ChangeType::Renamed)
    }

    pub fn has_copy(&self) -> bool {
        self.files
            .iter()
            .any(|file| file.change_type == ChangeType::Copied)
    }

    pub fn has_type_change(&self) -> bool {
        self.files
            .iter()
            .any(|file| file.change_type == ChangeType::TypeChanged)
    }
}


