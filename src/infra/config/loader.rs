use clap::Parser;
use std::path::PathBuf;
use std::str::FromStr;

use crate::shared::config::app::{AppConfig, PartialAppConfig};
use crate::shared::config::llm::{
    ApiKey, PartialLlmConfig, PartialLlmGenerationConfig, PartialLlmProviderConfig, ProviderName,
};
use crate::shared::config::loader::CliArgs;
use crate::shared::config::resilience::{
    PartialResilienceConfig, PartialRetryConfig, PartialTimeoutConfig,
};
use crate::shared::exception::config::ConfigError;

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
                    // ApiKey accepts any non-empty string; no parse failure possible.
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

fn resolve_default_config_path() -> Option<PathBuf> {
    if let Some(raw) = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let path = PathBuf::from(raw);
        if path.is_absolute() {
            return Some(path.join("autocommit").join("config.toml"));
        }
    }

    std::env::var("HOME").ok().map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("autocommit")
            .join("config.toml")
    })
}

// --- Env helpers ---

fn env_str<'a>(env: &'a [(String, String)], var: &str) -> Option<&'a str> {
    env.iter().find(|(k, _)| k == var).map(|(_, v)| v.as_str())
}

fn parse_env<T: FromStr>(
    env: &[(String, String)],
    var: &str,
    expected: &'static str,
) -> Result<Option<T>, ConfigError> {
    match env_str(env, var) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|_| ConfigError::EnvParse {
                var: var.to_string(),
                expected,
            }),
    }
}

fn parse_env_bool(env: &[(String, String)], var: &str) -> Result<Option<bool>, ConfigError> {
    match env_str(env, var) {
        None => Ok(None),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(Some(true)),
            "false" | "0" => Ok(Some(false)),
            _ => Err(ConfigError::EnvParse {
                var: var.to_string(),
                expected: "true/false/1/0",
            }),
        },
    }
}

// --- Validation ---

fn validate(config: &AppConfig) -> Result<(), ConfigError> {
    let p = &config.llm.provider;
    match p.provider {
        ProviderName::Openai => {
            if p.api_key.as_ref().is_none_or(|k| k.as_str().is_empty()) {
                return Err(ConfigError::MissingRequired {
                    field: "llm.api_key",
                    hint: "export AUTOCOMMIT_LLM_API_KEY=<your-key>".into(),
                });
            }
        }
        ProviderName::Ollama => {
            // No auth required
        }
    }
    if p.model.is_empty() {
        return Err(ConfigError::MissingRequired {
            field: "llm.model",
            hint: "set `model` in ~/.config/autocommit/config.toml or export AUTOCOMMIT_LLM_MODEL"
                .into(),
        });
    }

    let g = &config.llm.generation;
    check_f32("llm.temperature", g.temperature, 0.0, 2.0)?;
    check_f32("llm.frequency_penalty", g.frequency_penalty, 0.0, 2.0)?;
    check_f32("llm.presence_penalty", g.presence_penalty, 0.0, 2.0)?;
    if g.top_p.is_nan() || g.top_p <= 0.0 || g.top_p > 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "llm.top_p",
            reason: "must be in (0, 1]".into(),
        });
    }
    if g.max_tokens == 0 {
        return Err(ConfigError::InvalidValue {
            field: "llm.max_tokens",
            reason: "must be > 0".into(),
        });
    }

    let r = &config.resilience.retry;
    if r.max_retries > 10 {
        return Err(ConfigError::InvalidValue {
            field: "resilience.retry.max_retries",
            reason: format!("{} exceeds upper bound 10", r.max_retries),
        });
    }
    if r.initial_delay.is_zero() {
        return Err(ConfigError::InvalidValue {
            field: "resilience.retry.initial_delay",
            reason: "must be > 0".into(),
        });
    }
    if !r.factor.is_finite() || r.factor < 1.0 {
        return Err(ConfigError::InvalidValue {
            field: "resilience.retry.factor",
            reason: "must be finite and >= 1.0".into(),
        });
    }
    if r.max_delay < r.initial_delay {
        return Err(ConfigError::InvalidValue {
            field: "resilience.retry.max_delay",
            reason: "must be >= initial_delay".into(),
        });
    }
    if config.resilience.timeout.timeout.is_zero() {
        return Err(ConfigError::InvalidValue {
            field: "resilience.timeout.timeout_ms",
            reason: "must be > 0".into(),
        });
    }

    Ok(())
}

fn check_f32(field: &'static str, val: f32, lo: f32, hi: f32) -> Result<(), ConfigError> {
    if val.is_nan() || val < lo || val > hi {
        return Err(ConfigError::InvalidValue {
            field,
            reason: format!("must be in [{lo}, {hi}], got {val}"),
        });
    }
    Ok(())
}
