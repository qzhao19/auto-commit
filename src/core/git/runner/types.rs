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
    pub stdout: String,
    pub stderr: String,
}
