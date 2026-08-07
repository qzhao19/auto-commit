#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderName {
    Openai,
    Ollama,
}

#[derive(Debug, Clone)]
pub struct LlmGenerationConfig {
    pub temperature: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: Some(0.8),
            frequency_penalty: Default::default(),
            presence_penalty: Default::default(),
            top_p: Some(0.9),
            max_tokens: Some(4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmProviderConfig {
    pub model: String,
    pub provider: ProviderName,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}
