use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

use crate::shared::config::{GitCommandResult, GitRunOptions};
use crate::shared::exception::{GitError, GitErrorCode};

const GIT_COMMAND: &str = "git";

const NON_INTERACTIVE_ENV: &[(&str, &str)] =
    &[("GIT_TERMINAL_PROMPT", "0"), ("GIT_ASKPASS", "true")];

#[derive(Debug, Clone)]
pub struct GitRunner {
    default_cwd: PathBuf,
}

impl GitRunner {
    pub fn new(cwd: Option<PathBuf>) -> Self {
        let default_cwd =
            cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        Self { default_cwd }
    }

    pub fn cwd(&self) -> &Path {
        self.default_cwd.as_path()
    }

    pub async fn run(
        &self,
        args: &[&str],
        options: Option<&GitRunOptions>,
    ) -> Result<GitCommandResult, GitError> {
        let allowed_exit_codes = options
            .and_then(|option| option.allowed_exit_codes.as_deref())
            .unwrap_or(&[0]);

        let result = self.run_raw(args, options).await?;

        if !allowed_exit_codes.contains(&result.exit_code) {
            let reason = if !result.stderr.is_empty() {
                result.stderr_str()
            } else if !result.stdout.is_empty() {
                result.stdout_str()
            } else {
                std::borrow::Cow::Borrowed("unknown git error")
            };

            return Err(GitError::new(
                GitErrorCode::CommandFailed,
                format!(
                    "git command failed with exit code {}: {}\n{}",
                    result.exit_code, result.command, reason
                ),
            ));
        }

        Ok(result)
    }

    async fn run_raw(
        &self,
        args: &[&str],
        options: Option<&GitRunOptions>,
    ) -> Result<GitCommandResult, GitError> {
        let cwd = options
            .and_then(|opts| opts.cwd.as_ref())
            .unwrap_or(&self.default_cwd);

        let command = format!("{} {}", GIT_COMMAND, args.join(" "));

        let mut child = Command::new(GIT_COMMAND);
        child
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        for &(key, val) in NON_INTERACTIVE_ENV {
            child.env(key, val);
        }

        if let Some(opts) = options {
            child.envs(&opts.env);
        }

        let output = child.output().await.map_err(|err| {
            GitError::with_source(
                GitErrorCode::SpawnFailed,
                format!("failed to spawn git command: {command}"),
                err,
            )
        })?;

        let exit_code = output.status.code().unwrap_or(-1);

        Ok(GitCommandResult {
            args: args.iter().map(|s| (*s).to_owned()).collect(),
            command,
            cwd: cwd.to_path_buf(),
            exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}
