use crate::infra::retry::{Retry, RetryResult};
use crate::infra::timeout::Timeout;
use crate::shared::config::AppConfig;
use crate::shared::exception::LlmError;

use super::{Provider, build_provider};

/// High-level LLM call entry.
///
/// Holds `Provider` (OpenAI/Ollama) as well as `Retry` and `Timeout`,
/// `retry { timeout { provider.invoke_raw(prompt) } }`
pub struct LlmClient {
    provider: Box<dyn Provider>,
    retry: Retry,
    timeout: Timeout,
}

impl LlmClient {
    /// Build client from merged AppConfig
    pub fn new(config: &AppConfig) -> Result<Self, LlmError> {
        let provider = build_provider(&config.llm)?;
        let retry = Retry::new(config.resilience.retry.clone()).map_err(LlmError::Build)?;
        let timeout = Timeout::new(config.resilience.timeout.clone()).map_err(LlmError::Build)?;
        Ok(Self {
            provider,
            retry,
            timeout,
        })
    }

    /// Generate a commit message, apply retries + single attempt timeout
    pub async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        let result = self
            .retry
            .execute(LlmError::is_retryable, || async {
                self.invoke_once(prompt).await
            })
            .await;

        match result {
            RetryResult::Ok(value) => Ok(value),
            RetryResult::NonRetryable(err) => Err(err),
            RetryResult::Exhausted { attempts, last } => Err(LlmError::RetryExhausted {
                attempts,
                last: Box::new(last),
            }),
        }
    }

    /// Single attempt: wrap provider future with timeout protect
    async fn invoke_once(&self, prompt: &str) -> Result<String, LlmError> {
        self.timeout
            .execute(self.provider.invoke_raw(prompt))
            .await
            .map_err(LlmError::Timeout)?
    }
}
