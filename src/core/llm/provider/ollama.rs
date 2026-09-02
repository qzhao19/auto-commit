use std::future::Future;
use std::pin::Pin;

use ollama_rs::Ollama;
use ollama_rs::error::OllamaError;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::models::ModelOptions;

use crate::shared::config::{LlmConfig, LlmGenerationConfig, LlmMessage};
use crate::shared::exception::{LlmError, ProviderErrorType};

use super::super::Provider;

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// Adapter for local Ollam services. No API key required.
pub struct OllamaProvider {
    client: Ollama,
    model: String,
    generation: LlmGenerationConfig,
}

impl OllamaProvider {
    pub fn new(config: &LlmConfig) -> Result<Self, LlmError> {
        let provider = &config.provider;
        let generation = &config.generation;
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or(DEFAULT_OLLAMA_BASE_URL);

        let client = Ollama::try_new(base_url.to_string())
            .map_err(|err| LlmError::Build(format!("invalid Ollama base_url: {err}")))?;

        Ok(Self {
            client,
            model: provider.model.clone(),
            generation: generation.clone(),
        })
    }

    /// Map `LlmGenerationConfig` to ollama-rs' `ModelOptions`.
    fn generation_options(&self) -> ModelOptions {
        let generation = &self.generation;
        ModelOptions::default()
            .temperature(generation.temperature)
            .top_p(generation.top_p)
            .num_predict(generation.max_tokens)
            .extra("frequency_penalty", generation.frequency_penalty)
            .extra("presence_penalty", generation.presence_penalty)
    }
}

impl Provider for OllamaProvider {
    fn invoke_raw<'a>(
        &'a self,
        message: LlmMessage<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, LlmError>> + Send + 'a>> {
        Box::pin(async move {
            let mut messages = Vec::with_capacity(2);
            if let Some(system_message) = message.system_message {
                messages.push(ChatMessage::system(system_message.to_string()));
            }
            messages.push(ChatMessage::user(message.user_message.to_string()));

            let request = ChatMessageRequest::new(self.model.clone(), messages)
                .options(self.generation_options());

            let response = self
                .client
                .send_chat_messages(request)
                .await
                .map_err(ollama_err)?;
            Ok(response.message.content)
        })
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

/// 'RequestError' are retryable error
/// `JsonError` / `ToolCallError` / `Other` are deterministic errors
fn ollama_err(err: OllamaError) -> LlmError {
    let err_type = match &err {
        OllamaError::ReqwestError(_) => ProviderErrorType::Transient,
        _ => ProviderErrorType::Fatal,
    };

    LlmError::Provider(err_type, err.to_string())
}
