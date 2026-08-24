use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::usize;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
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

    pub async fn cat_file_header(
        &self,
        specs: &[&str],
        max_bytes: usize,
        options: Option<&GitRunOptions>,
    ) -> Result<Vec<Option<Vec<u8>>>, GitError> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(spec) = specs.iter().find(|spec| spec.contains('\n')) {
            return Err(GitError::new(
                GitErrorCode::Other,
                format!("cat-file spec must not contain a newline: {spec:?}"),
            ));
        }

        let cwd = options
            .and_then(|opts| opts.cwd.as_ref())
            .unwrap_or(&self.default_cwd);
        let mut cmd = Command::new(GIT_COMMAND);
        cmd.arg("cat-file")
            .arg("--batch")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        for &(key, val) in NON_INTERACTIVE_ENV {
            cmd.env(key, val);
        }

        if let Some(opts) = options {
            cmd.envs(&opts.env);
        }

        let mut child = cmd.spawn().map_err(|err| {
            GitError::with_source(
                GitErrorCode::Other,
                "failed to spawn git cat-file --batch",
                err,
            )
        })?;

        //
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| GitError::new(GitErrorCode::Other, "cat-file stdin was not piped"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| GitError::new(GitErrorCode::Other, "cat-file stderr was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| GitError::new(GitErrorCode::Other, "cat-file stdout was not piped"))?;

        let mut input = String::with_capacity(specs.iter().map(|s| s.len() + 1).sum());
        for spec in specs {
            input.push_str(spec);
            input.push('\n');
        }

        // Stdin, stdout and stderr streams must make
        // progress concurrently, or the child deadlocks
        let write_stdin = async {
            let result = stdin.write_all(input.as_bytes()).await;
            drop(stdin);
            result
        };

        let read_stdout = async {
            let mut reader = BufReader::new(stdout);
            let mut results = Vec::with_capacity(specs.len());
            for _ in specs {
                results.push(read_batch_entry(&mut reader, max_bytes).await?);
            }
            Ok::<Vec<Option<Vec<u8>>>, GitError>(results)
        };

        let read_stderr = async {
            let mut buf = Vec::new();
            stderr.read_to_end(&mut buf).await?;
            Ok::<Vec<u8>, std::io::Error>(buf)
        };

        // Concurrent
        let (stdin_result, stdout_result, stderr_result) =
            tokio::join!(write_stdin, read_stdout, read_stderr);

        let status = child.wait().await.map_err(|err| {
            GitError::with_source(GitErrorCode::Other, "cat-file wait failed", err)
        })?;

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            let reason = match &stderr_result {
                Ok(bytes) if !bytes.is_empty() => String::from_utf8_lossy(bytes).trim().to_owned(),
                Ok(_) => format!("exit code {code}"),
                Err(err) => format!("exit code {code}, stderr unreadable: {err}"),
            };
            return Err(GitError::new(
                GitErrorCode::CommandFailed,
                format!("git cat-file --batch failed: {reason}"),
            ));
        }

        stdin_result.map_err(|err| {
            GitError::with_source(GitErrorCode::Other, "cat-file stdin write failed", err)
        })?;

        Ok(stdout_result?)
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

/// One `--batch` record.
///
///     <oid> SP <type> SP <size> LF <contents> LF
///     <object> SP missing LF
///     <object> SP ambiguous LF
async fn read_batch_entry<R>(
    reader: &mut BufReader<R>,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>, GitError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    // Define error closure
    let io_err = |ctx| move |err| GitError::with_source(GitErrorCode::Other, ctx, err);

    // 1. header line
    let mut line = Vec::new();
    let n = reader
        .read_until(b'\n', &mut line)
        .await
        .map_err(io_err("cat-file --batch header read failed"))?;
    if n == 0 || !line.ends_with(&[b'\n']) {
        return Err(GitError::new(
            GitErrorCode::Other,
            "cat-file --batch stream ended mid-record",
        ));
    }

    // 2. None = missing / ambiguous
    let Some(size) = parse_batch_header(&line)? else {
        return Ok(None);
    };

    // 3. keep the first max_bytes
    let take = usize::try_from(size).unwrap_or(usize::MAX).min(max_bytes);
    let mut buf = vec![0u8; take];
    if take > 0 {
        reader
            .read_exact(&mut buf)
            .await
            .map_err(io_err("cat-file --batch read failed"))?;
    }

    // 4. Discard the rest of payload plus the tailing LF
    // discard = size - take + 1
    let skip = size.saturating_add(1) - take as u64;
    if skip > 0 {
        let copied = tokio::io::copy(&mut reader.take(skip), &mut tokio::io::sink())
            .await
            .map_err(io_err("cat-file --batch discard failed"))?;
        if copied != skip {
            return Err(GitError::new(
                GitErrorCode::Other,
                "cat-file --batch stream ended mid-record",
            ));
        }
    }

    Ok(Some(buf))
}

fn parse_batch_header(line: &[u8]) -> Result<Option<u64>, GitError> {
    let malformed = |line: &[u8], why: &str| -> GitError {
        GitError::new(
            GitErrorCode::Other,
            format!(
                "malformed cat-file --batch header ({why}): '{}'",
                String::from_utf8_lossy(line).trim_end()
            ),
        )
    };

    let text = std::str::from_utf8(line).map_err(|_| malformed(line, "invalid UTF-8"))?;
    let mut tokens = text.trim_end().split_whitespace();

    let _oid = tokens.next().ok_or_else(|| malformed(line, "empty"))?;
    let kind = tokens.next().ok_or_else(|| malformed(line, "no type"))?;

    match kind {
        "missing" | "ambiguous" => Ok(None),
        "blob" | "tree" | "commit" | "tag" => {
            let size = tokens
                .next()
                .ok_or_else(|| malformed(line, "no size"))?
                .parse::<u64>()
                .map_err(|_| malformed(line, "size is not a number"))?;
            Ok(Some(size))
        }
        other => Err(malformed(line, &format!("unexpected type '{other}'"))),
    }
}
