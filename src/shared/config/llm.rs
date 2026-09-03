use std::fmt;
use std::str::FromStr;

use serde::Deserialize;

/// API Key，Debug/Display are masked
#[derive(Clone, PartialEq, Eq, Deserialize)]
#[serde(from = "String")]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ApiKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl FromStr for ApiKey {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_owned()))
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0.as_str();

        // Show only the first 4 digits
        if s.len() > 4 && s.is_char_boundary(4) {
            write!(f, "ApiKey({}****)", &s[..4])
        } else {
            write!(f, "ApiKey(****)")
        }
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("****")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderName {
    #[default]
    Openai,
    Ollama,
}

impl FromStr for ProviderName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("openai") {
            Ok(Self::Openai)
        } else if s.eq_ignore_ascii_case("ollama") {
            Ok(Self::Ollama)
        } else {
            Err(format!(
                "unknown provider: {s} (expected \"openai\" or \"ollama\")"
            ))
        }
    }
}

impl fmt::Display for ProviderName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Openai => f.write_str("openai"),
            Self::Ollama => f.write_str("ollama"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmGenerationConfig {
    pub temperature: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub top_p: f32,
    pub max_tokens: i32,
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            top_p: 0.9,
            max_tokens: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmProviderConfig {
    pub model: String,
    pub provider: ProviderName,
    pub api_key: Option<ApiKey>,
    pub base_url: Option<String>,
}

impl Default for LlmProviderConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider: ProviderName::default(),
            api_key: None,
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    pub provider: LlmProviderConfig,
    pub generation: LlmGenerationConfig,
}

#[derive(Debug, Clone)]
pub struct LlmMessage {
    pub system_message: Option<String>,
    pub user_message: String,
}

impl LlmMessage {
    pub fn new(system_message: Option<String>, user_message: String) -> Self {
        Self {
            system_message,
            user_message,
        }
    }

    pub fn user_message_only(user_message: String) -> Self {
        Self {
            system_message: None,
            user_message,
        }
    }
}

// --- Partial input intermediate state ---

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialLlmProviderConfig {
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    pub api_key: Option<ApiKey>,
    pub base_url: Option<String>,
}

impl PartialLlmProviderConfig {
    pub fn merge(self, config: LlmProviderConfig) -> LlmProviderConfig {
        LlmProviderConfig {
            provider: self.provider.unwrap_or(config.provider),
            model: self.model.unwrap_or(config.model),
            api_key: self.api_key.or(config.api_key),
            base_url: self.base_url.or(config.base_url),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialLlmGenerationConfig {
    pub temperature: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i32>,
}

impl PartialLlmGenerationConfig {
    pub fn merge(self, config: LlmGenerationConfig) -> LlmGenerationConfig {
        LlmGenerationConfig {
            temperature: self.temperature.unwrap_or(config.temperature),
            frequency_penalty: self.frequency_penalty.unwrap_or(config.frequency_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(config.presence_penalty),
            top_p: self.top_p.unwrap_or(config.top_p),
            max_tokens: self.max_tokens.unwrap_or(config.max_tokens),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialLlmConfig {
    #[serde(flatten)]
    pub provider: PartialLlmProviderConfig,
    #[serde(flatten)]
    pub generation: PartialLlmGenerationConfig,
}

impl PartialLlmConfig {
    pub fn merge(self, config: LlmConfig) -> LlmConfig {
        LlmConfig {
            provider: self.provider.merge(config.provider),
            generation: self.generation.merge(config.generation),
        }
    }
}
