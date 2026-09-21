use std::fmt;

use super::config::ConfigError;
use super::git::GitError;
use super::llm::LlmError;

#[derive(Debug)]
pub enum AppError {
    Config(ConfigError), // exit 2
    Pipeline(GitError),  // exit 3 (preflight / bisect / conflicts / empty staging)
    Llm(LlmError),       // exit 4
    Io(String),          // exit 5
    Commit(String),      // exit 5
}

impl AppError {
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Config(_) => 2,
            Self::Pipeline(_) => 3,
            Self::Llm(_) => 4,
            Self::Io(_) | Self::Commit(_) => 5,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "configuration error: {err}"),
            Self::Pipeline(err) => write!(f, "pipeline error: {err}"),
            Self::Llm(err) => write!(f, "LLM error: {err}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Commit(msg) => write!(f, "git commit failed: {msg}"),
        }
    }
}
