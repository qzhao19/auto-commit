use std::path::PathBuf;
use std::str::FromStr;

use crate::shared::config::app::AppConfig;
use crate::shared::config::llm::ProviderName;
use crate::shared::exception::config::ConfigError;

pub fn env_str<'a>(env: &'a [(String, String)], var: &str) -> Option<&'a str> {
    env.iter().find(|(k, _)| k == var).map(|(_, v)| v.as_str())
}

pub fn parse_env<T: FromStr>(
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

pub fn parse_env_bool(env: &[(String, String)], var: &str) -> Result<Option<bool>, ConfigError> {
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

pub fn resolve_default_config_path() -> Option<PathBuf> {
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

// --- Validation ---

pub fn validate(config: &AppConfig) -> Result<(), ConfigError> {
    let p = &config.llm.provider;
    if p.model.is_empty() {
        return Err(ConfigError::MissingRequired {
            field: "llm.model",
            hint: "set `model` in ~/.config/autocommit/config.toml or export AUTOCOMMIT_LLM_MODEL"
                .into(),
        });
    }
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

pub fn normalize_empty_strings(mut config: AppConfig) -> AppConfig {
    let p = &mut config.llm.provider;
    if p.base_url.as_deref().is_some_and(str::is_empty) {
        p.base_url = None;
    }
    if p.api_key.as_ref().is_some_and(|k| k.as_str().is_empty()) {
        p.api_key = None;
    }
    config
}
