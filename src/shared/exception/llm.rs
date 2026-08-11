use std::fmt;

/// Errors for retry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorType {
    /// Transient failures at the transport layer/server level
    Transient,
    /// Deterministic failures (authentication, request format, response parsing)
    /// no retry.
    Fatal,
}

#[derive(Debug)]
pub enum LlmError {
    Build(String),
    Timeout(String),
    Provider(ProviderErrorType, String),
    RetryExhausted { attempts: u32, last: Box<LlmError> },
}

impl LlmError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::Provider(ProviderErrorType::Transient, _)
        )
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Build(msg) => write!(f, "LLM setup failed: {msg}"),
            Self::Timeout(msg) => write!(f, "LLM request timed out: {msg}"),
            Self::Provider(err_type, msg) => match err_type {
                ProviderErrorType::Fatal => write!(f, "LLM error: {msg}"),
                ProviderErrorType::Transient => write!(f, "LLM transient error: {msg}"),
            },
            Self::RetryExhausted { attempts, last } => write!(
                f,
                "LLM call failed after {attempts} attempt(s); last error: {last}"
            ),
        }
    }
}
