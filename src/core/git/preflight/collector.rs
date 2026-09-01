use std::path::PathBuf;

use tokio::io::Stdout;

use crate::core::git::types::{GitPaths, RepositoryContext};
use crate::infra::git::GitRunner;
use crate::shared::config::GitRunOptions;
use crate::shared::exception::{GitError, GitErrorCode};

/// Result of repository discovery.
#[derive(Debug)]
struct DiscoveredRepository {
    git_dir: PathBuf,
    is_bare: bool,
}

const BINARY_PROBE_EXIT_CODES: &[i32] = &[0, 1];

/// Stage 0: Repository Preflight.
///
/// Responsibilities:
///
/// 1. Discover the Git repository.
/// 2. Detect bare repositories.
/// 3. Resolve the working tree.
/// 4. Resolve Git paths.
/// 5. Detect index.lock.
/// 6. Resolve HEAD.
/// 7. Resolve current branch.
/// 8. Verify staged changes exist.
/// 9. Build RepositoryContext.
///
pub struct RepoPreflightCollector<'a> {
    runner: &'a GitRunner,
}

impl<'a> RepoPreflightCollector<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn run(&self) -> Result<RepositoryContext, GitError> {
        // 0.1  Discover repository + detect bare repository
        let repository = self.discover_repository().await?;

        if repository.is_bare {
            return Err(GitError::new(
                GitErrorCode::Other,
                "bare repository is not supported; auto-commit requires a working tree",
            ));
        }

        // 0.2 Resolve working tree
        let worktree_root = self.resolve_worktree_root().await?;

        // 0.3 Resolve Git paths
        let git_paths = GitPaths::from_git_dir(repository.git_dir);

        // 0.4 Detect index.lock
        self.check_index_lock(&git_paths)?;

        // 0.5 Resolve HEAD
        let head_oid = self.resolve_head().await?;

        // 0.6 Resolve branch
        let branch = self.resolve_branch().await?;

        Ok(RepositoryContext {
            worktree_root,
            git_paths,
            head_oid,
            branch,
        })
    }

    /// 0.7 Detect staged changes, entry condition of stage ops checks
    pub async fn ensure_staged_changes(&self) -> Result<(), GitError> {
        if self.has_staged_changes().await? {
            return Ok(());
        }
        self.handle_empty_staging().await
    }

    ///  0.1 Discover repository
    async fn discover_repository(&self) -> Result<DiscoveredRepository, GitError> {
        let result = self
            .runner
            .run(
                &["rev-parse", "--absolute-git-dir", "--is-bare-repository"],
                None,
            )
            .await
            .map_err(|err| match err.code {
                GitErrorCode::CommandFailed => GitError::new(
                    GitErrorCode::NotARepository,
                    format!("not a git repository: {err}"),
                ),
                _ => err,
            })?;

        let stdout = result.stdout_str();
        let mut lines = stdout.lines().map(str::trim);

        let git_dir_raw = lines.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::CommandFailed,
                "git rev-parse returned no git directory",
            )
        })?;

        let is_bare = match lines.next() {
            Some("true") => true,
            Some("false") => false,
            Some(value) => {
                return Err(GitError::new(
                    GitErrorCode::CommandFailed,
                    format!("unexpected --is-bare-repository output: {value:?}"),
                ));
            }
            None => {
                return Err(GitError::new(
                    GitErrorCode::CommandFailed,
                    "git rev-parse returned no bare-repository status",
                ));
            }
        };

        let git_dir = self.resolve_git_dir(git_dir_raw)?;

        Ok(DiscoveredRepository { git_dir, is_bare })
    }

    /// 0.2 Resolve working tree
    async fn resolve_worktree_root(&self) -> Result<PathBuf, GitError> {
        let result = self
            .runner
            .run(&["rev-parse", "--show-toplevel"], None)
            .await?;

        let root = result.stdout_str().trim().to_owned();

        if root.is_empty() {
            return Err(GitError::new(
                GitErrorCode::CommandFailed,
                "git rev-parse --show-toplevel returned an empty path",
            ));
        }

        Ok(PathBuf::from(root))
    }

    /// 0.4 index.lock
    fn check_index_lock(&self, paths: &GitPaths) -> Result<(), GitError> {
        let index_lock = &paths.index_lock;

        match index_lock.try_exists() {
            Ok(true) => Err(GitError::new(
                GitErrorCode::Other,
                format!("git index is locked: {}", index_lock.display()),
            )),
            Ok(false) => Ok(()),
            Err(error) => Err(GitError::new(
                GitErrorCode::Other,
                format!(
                    "failed to inspect git index lock {}: {error}",
                    index_lock.display()
                ),
            )),
        }
    }

    ///  0.5 HEAD
    async fn resolve_head(&self) -> Result<Option<String>, GitError> {
        let result = self
            .runner
            .run(
                &["rev-parse", "-q", "--verify", "HEAD"],
                Some(&GitRunOptions {
                    allowed_exit_codes: Some(BINARY_PROBE_EXIT_CODES.to_vec()),
                    ..Default::default()
                }),
            )
            .await?;

        // `run` only returns Ok for exit codes in the allow-list {0, 1}.
        match result.exit_code {
            0 => match result.stdout_str().trim() {
                // Exit 0 must print the OID, empty output is an anomaly,
                "" => Err(GitError::new(
                    GitErrorCode::CommandFailed,
                    "git rev-parse --verify HEAD exited 0 but printed no commit id",
                )),
                oid => Ok(Some(oid.to_owned())),
            },
            // `-q --verify HEAD` exits 1: no commits yet
            _ => Ok(None),
        }
    }

    /// 0.6 Branch
    async fn resolve_branch(&self) -> Result<Option<String>, GitError> {
        let result = self.runner.run(&["branch", "--show-current"], None).await?;

        let branch = result.stdout_str().trim().to_owned();

        // Empty output means detached HEAD
        if branch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(branch))
        }
    }

    /// 0.7a Staged changes
    async fn has_staged_changes(&self) -> Result<bool, GitError> {
        let result = self
            .runner
            .run(
                &["diff", "--cached", "--quiet"],
                Some(&GitRunOptions {
                    allowed_exit_codes: Some(BINARY_PROBE_EXIT_CODES.to_vec()),
                    ..Default::default()
                }),
            )
            .await?;

        // `run` only returns Ok for exit codes in {0, 1}:
        // 0 = index matches HEAD (nothing staged)
        // 1 = differences staged
        Ok(result.exit_code == 1)
    }

    /// 0.7b Empty staging
    async fn handle_empty_staging(&self) -> Result<(), GitError> {
        // `git diff --quiet` only sees tracked modifications
        let has_unstaged = self.is_worktree_dirty().await?;
        let has_untracked = self.has_untracked_files().await?;

        if has_unstaged || has_untracked {
            return Err(GitError::new(
                GitErrorCode::NothingStaged,
                "no staged changes; run `git add` or `git rm` first",
            ));
        }

        Err(GitError::new(
            GitErrorCode::NothingStaged,
            "nothing to commit: staging area and working tree are clean",
        ))
    }

    // Helper functions

    fn resolve_git_dir(&self, git_dir_raw: &str) -> Result<PathBuf, GitError> {
        let path = PathBuf::from(git_dir_raw);

        // Make sure an absolute path: '--absolute-git-dir'
        if path.is_absolute() {
            return Ok(path);
        }

        let cwd = self.runner.cwd();

        Ok(cwd.join(path))
    }

    async fn is_worktree_dirty(&self) -> Result<bool, GitError> {
        let result = self
            .runner
            .run(
                &["diff", "--quiet"],
                Some(&GitRunOptions {
                    allowed_exit_codes: Some(BINARY_PROBE_EXIT_CODES.to_vec()),
                    ..Default::default()
                }),
            )
            .await?;

        // Allow-list is {0, 1}:
        // 0 = clean,
        // 1 = tracked modifications exist.
        Ok(result.exit_code == 1)
    }

    async fn has_untracked_files(&self) -> Result<bool, GitError> {
        // Non-ignored untracked files
        let result = self
            .runner
            .run(&["ls-files", "--others", "--exclude-standard"], None)
            .await?;

        Ok(!result.stdout.is_empty())
    }
}
