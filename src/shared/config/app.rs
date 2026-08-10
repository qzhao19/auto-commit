use super::llm::{LlmConfig, PartialLlmConfig};
use super::resilience::{PartialResilienceConfig, ResilienceConfig};

use serde::Deserialize;

/// Merged top-level config: all fields are non-`Option`
#[derive(Debug, Clone, PartialEq)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub resilience: ResilienceConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llm: LlmConfig {
                provider: Default::default(),
                generation: Default::default(),
            },
            resilience: ResilienceConfig::default(),
        }
    }
}

/// Partial-input form: direct output of toml parsing, missing fields are `None`
/// Deserialize the file at once via `toml::from_str::<PartialAppConfig>(&content)`
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialAppConfig {
    #[serde(default)]
    pub llm: PartialLlmConfig,
    #[serde(default)]
    pub resilience: PartialResilienceConfig,
}

impl PartialAppConfig {
    /// Overlay partial input on top of `config`, producing the final config
    /// Call chain: `cli.merge(env.merge(toml.merge(base)))`, where `base` is the default layer.
    pub fn merge(self, config: AppConfig) -> AppConfig {
        AppConfig {
            llm: self.llm.merge(config.llm),
            resilience: self.resilience.merge(config.resilience),
        }
    }
}
