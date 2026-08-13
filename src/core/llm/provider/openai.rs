use std::future::Future;
use std::pin::Pin;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};

use crate::shared::config::llm::{LlmConfig, LlmGenerationConfig};
use crate::shared::exception::llm::{LlmError, ProviderErrorType};

use super::super::Provider;

/// Adapts to any model compatible with the OpenAI Chat Completions endpoint.
pub struct OpenAiProvider {
    client: Client<OpenAIConfig>,
    model: String,
    generation: LlmGenerationConfig,
}

impl OpenAiProvider {
    pub fn new(config: &LlmConfig) -> Result<Self, LlmError> {
        let provider = &config.provider;
        let generation = &config.generation;

        let cfg = OpenAIConfig::new();
        let cfg = match &provider.api_key {
            Some(key) => cfg.with_api_key(key.as_str()),
            None => cfg,
        };

        let cfg = match &provider.base_url {
            Some(api_base) => cfg.with_api_base(api_base.as_str()),
            None => cfg,
        };

        Ok(Self {
            client: Client::with_config(cfg),
            model: provider.model.clone(),
            generation: generation.clone(),
        })
    }
}

impl Provider for OpenAiProvider {
    fn invoke_raw<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, LlmError>> + Send + 'a>> {
        Box::pin(async move {
            let max_tokens = u32::try_from(self.generation.max_tokens).map_err(|_| {
                LlmError::Build(format!(
                    "max_tokens {} is negative; OpenAI requires a non-negative value",
                    self.generation.max_tokens
                ))
            })?;

            // Create user message
            let user_msg = ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .map_err(openai_err)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.as_str())
                .messages(vec![user_msg.into()])
                .temperature(self.generation.temperature)
                .top_p(self.generation.top_p)
                .max_tokens(max_tokens)
                .frequency_penalty(self.generation.frequency_penalty)
                .presence_penalty(self.generation.presence_penalty)
                .build()
                .map_err(openai_err)?;

            let response = self
                .client
                .chat()
                .create(request)
                .await
                .map_err(openai_err)?;

            response
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| {
                    LlmError::Provider(
                        ProviderErrorType::Fatal,
                        "response had no content".to_string(),
                    )
                })
        })
    }

    fn name(&self) -> &'static str {
        "openai"
    }
}

/// `ApiError` carries the HTTP status code of the API response:
/// - rate limits (429) and server errors (5xx) are transient;
/// - (400/401/404 …) is deterministi
fn openai_err(err: OpenAIError) -> LlmError {
    let err_type = match &err {
        OpenAIError::Reqwest(_) => ProviderErrorType::Transient,
        OpenAIError::ApiError(api_err) => {
            let code = api_err.status_code.as_u16();
            if code == 429 || (500..=599).contains(&code) {
                ProviderErrorType::Transient
            } else {
                ProviderErrorType::Fatal
            }
        }
        _ => ProviderErrorType::Fatal,
    };

    LlmError::Provider(err_type, err.to_string())
}
