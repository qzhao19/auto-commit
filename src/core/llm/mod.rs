pub mod client;
pub mod provider;

use std::future::Future;
use std::pin::Pin;

use crate::shared::config::{LlmConfig, LlmMessage, ProviderName};
use crate::shared::exception::LlmError;

use provider::ollama::OllamaProvider;
use provider::openai::OpenAiProvider;

/// Specific LLM backend adapter interface
///
/// Adapter itself does not need to concern about resilience logic
/// Resilience (retry + timeout) wrapper is handled by  [`crate::core::llm::LlmClient`]
pub trait Provider: Send + Sync {
    /// Returned future borrows both `self` and `prompt`, hence the `+ 'a`
    /// bound: the future cannot outlive either.
    fn invoke_raw<'a>(
        &'a self,
        message: LlmMessage<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, LlmError>> + Send + 'a>>;

    fn name(&self) -> &'static str;
}

pub fn build_provider(config: &LlmConfig) -> Result<Box<dyn Provider>, LlmError> {
    match config.provider.provider {
        ProviderName::Ollama => Ok(Box::new(OllamaProvider::new(config)?)),
        ProviderName::Openai => Ok(Box::new(OpenAiProvider::new(config)?)),
    }
}
