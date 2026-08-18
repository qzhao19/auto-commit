use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct GitRunOptions {
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
    /// Default: only `0` is accepted.
    pub allowed_exit_codes: Option<Vec<i32>>,
}

impl Default for GitRunOptions {
    fn default() -> Self {
        Self {
            cwd: None,
            env: HashMap::new(),
            allowed_exit_codes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GitCommandResult {
    pub args: Vec<String>,
    pub command: String,
    pub cwd: PathBuf,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl GitCommandResult {
    pub fn stdout_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.stdout)
    }

    pub fn stderr_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.stderr)
    }
}
