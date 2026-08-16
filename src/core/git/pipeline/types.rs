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
    /// Repository working tree root.
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
enum OperationAction {
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

    /// Seed message for later stages: reuse, or hand to the LLM
    /// for polishing. `None` for abort/clean states or when the source is
    /// unavailable
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
