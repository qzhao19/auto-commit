use clap::Parser;

use super::app::PartialAppConfig;
use super::llm::{PartialLlmConfig, PartialLlmGenerationConfig, PartialLlmProviderConfig};
use super::resilience::PartialResilienceConfig;

// By design only generation params are exposed here:
// provider / apiKey / resilience must never appear on the command line
#[derive(Parser, Debug, Clone, Default)]
#[command(
    name = "auto-commit",
    about = "Generate descriptive commit messages via LLM"
)]
pub struct CliArgs {
    #[arg(long, help = "Sampling temperature in [0, 2]")]
    pub temperature: Option<f32>,

    #[arg(long, help = "Max tokens to generate (> 0)")]
    pub max_tokens: Option<i32>,

    #[arg(long, help = "Nucleus sampling probability in (0, 1]")]
    pub top_p: Option<f32>,

    #[arg(long, help = "Frequency penalty in [0, 2]")]
    pub frequency_penalty: Option<f32>,

    #[arg(long, help = "Presence penalty in [0, 2]")]
    pub presence_penalty: Option<f32>,
}

impl CliArgs {
    /// Convert parsed CLI into a partial overlay
    pub fn into_partial(self) -> PartialAppConfig {
        PartialAppConfig {
            llm: PartialLlmConfig {
                provider: PartialLlmProviderConfig::default(),
                generation: PartialLlmGenerationConfig {
                    temperature: self.temperature,
                    max_tokens: self.max_tokens,
                    top_p: self.top_p,
                    frequency_penalty: self.frequency_penalty,
                    presence_penalty: self.presence_penalty,
                },
            },
            resilience: PartialResilienceConfig::default(),
        }
    }
}
