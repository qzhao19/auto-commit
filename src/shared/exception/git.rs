use std::error::Error;
use std::fmt;

/// Classification of git-layer failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GitErrorCode {
    /// Failed to start the `git` process (not found, permission, etc.).
    SpawnFailed,
    /// Process started, but exit code was not in the allowed set.
    CommandFailed,
    /// Current directory is not inside a git work tree.
    NotARepository,
    /// No staged changes when an operation requires them.
    NothingStaged,
    /// Catch-all for git-layer errors that do not fit the above.
    Other,
}

impl GitErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CommandFailed => "command_failed",
            Self::NotARepository => "not_a_repository",
            Self::NothingStaged => "nothing_staged",
            Self::SpawnFailed => "spawn_failed",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for GitErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct GitError {
    pub code: GitErrorCode,
    pub message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl GitError {
    pub fn new(code: GitErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// Primary constructor with a lower-level cause
    pub fn with_source<E>(code: GitErrorCode, message: impl Into<String>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self {
            code,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for GitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl Error for GitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn Error + 'static))
    }
}
