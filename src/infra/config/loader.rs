use clap::Parser;
use std::path::PathBuf;

use crate::shared::config::app::{AppConfig, PartialAppConfig};
use crate::shared::config::llm::{
    ApiKey, PartialLlmConfig, PartialLlmGenerationConfig, PartialLlmProviderConfig,
};
use crate::shared::config::loader::CliArgs;
use crate::shared::config::resilience::{
    PartialResilienceConfig, PartialRetryConfig, PartialTimeoutConfig,
};
use crate::shared::exception::config::ConfigError;
use crate::shared::util::loader::{
    env_str, normalize_empty_strings, parse_env, parse_env_bool, resolve_default_config_path,
    validate,
};

pub struct ConfigLoader {
    file_path: Option<PathBuf>,
    env_vars: Vec<(String, String)>,
    cli_args: CliArgs,
}

impl ConfigLoader {
    /// Production constructor: reads from default setup
    pub fn load_from_defaults() -> Self {
        Self {
            cli_args: CliArgs::parse(),
            file_path: resolve_default_config_path(),
            env_vars: std::env::vars().collect(),
        }
    }

    /// Test constructor
    pub fn new(
        file_path: Option<PathBuf>,
        env_vars: Vec<(String, String)>,
        cli_args: CliArgs,
    ) -> Self {
        Self {
            file_path,
            env_vars,
            cli_args,
        }
    }

    pub fn load(self) -> Result<AppConfig, ConfigError> {
        let base = AppConfig::default();
        let toml_partial = self.load_file()?;
        let env_partial = self.build_env_partial()?;
        let cli_partial = self.cli_args.into_partial();

        // Merge
        let merged = cli_partial.merge(env_partial.merge(toml_partial.merge(base)));
        let merged = normalize_empty_strings(merged);

        validate(&merged)?;
        Ok(merged)
    }

    // --- TOML file layer ---

    fn load_file(&self) -> Result<PartialAppConfig, ConfigError> {
        let path = match &self.file_path {
            None => return Ok(PartialAppConfig::default()),
            Some(p) => p.as_path(),
        };

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(PartialAppConfig::default());
            }
            Err(e) => {
                return Err(ConfigError::FileRead {
                    path: path.to_path_buf(),
                    source: e,
                });
            }
        };

        let partial: PartialAppConfig =
            toml::from_str(&content).map_err(|source| ConfigError::TomlParse {
                path: path.to_path_buf(),
                source,
            })?;

        if partial.llm.provider.api_key.is_some() {
            eprintln!(
                "warning: apiKey found in {}; use AUTOCOMMIT_LLM_API_KEY env var instead",
                path.display()
            );
        }

        Ok(partial)
    }

    // --- Env variables layer ---

    fn build_env_partial(&self) -> Result<PartialAppConfig, ConfigError> {
        let env = &self.env_vars;
        Ok(PartialAppConfig {
            llm: PartialLlmConfig {
                provider: PartialLlmProviderConfig {
                    provider: parse_env(env, "AUTOCOMMIT_LLM_PROVIDER", "openai or ollama")?,
                    model: parse_env(env, "AUTOCOMMIT_LLM_MODEL", "a string")?,
                    // ApiKey accepts any non-empty string, no parse failure possible.
                    api_key: env_str(env, "AUTOCOMMIT_LLM_API_KEY").map(ApiKey::new),
                    base_url: parse_env(env, "AUTOCOMMIT_LLM_BASE_URL", "a URL string")?,
                },
                // Env never overrides generation (by design).
                generation: PartialLlmGenerationConfig::default(),
            },
            resilience: PartialResilienceConfig {
                retry: PartialRetryConfig {
                    max_retries: parse_env(
                        env,
                        "AUTOCOMMIT_RETRY_MAX_RETRIES",
                        "a non-negative integer",
                    )?,
                    initial_delay_ms: parse_env(
                        env,
                        "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS",
                        "a non-negative integer",
                    )?,
                    max_delay_ms: parse_env(
                        env,
                        "AUTOCOMMIT_RETRY_MAX_DELAY_MS",
                        "a non-negative integer",
                    )?,
                    factor: parse_env(env, "AUTOCOMMIT_RETRY_FACTOR", "a float >= 1.0")?,
                    jitter: parse_env_bool(env, "AUTOCOMMIT_RETRY_JITTER")?,
                },
                timeout: PartialTimeoutConfig {
                    timeout_ms: parse_env(env, "AUTOCOMMIT_TIMEOUT_MS", "a non-negative integer")?,
                },
            },
        })
    }
}

//
