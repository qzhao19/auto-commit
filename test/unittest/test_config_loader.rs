use std::path::PathBuf;

use crate::infra::config::loader::ConfigLoader;
use crate::shared::config::llm::{LlmGenerationConfig, ProviderName};
use crate::shared::config::loader::CliArgs;
use crate::shared::exception::config::ConfigError;

// helpers 

/// Inline env-var pairs: `env(&[("K", "v"), ...])`.
fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// RAII TOML file in `std::env::temp_dir()`, removed on drop. Filename must
/// be unique per test to allow parallel execution; pass the test name.
struct TempToml(PathBuf);

impl TempToml {
    fn new(name: &str, content: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("autocommit-test-{name}.toml"));
        std::fs::write(&path, content)?;
        Ok(Self(path))
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempToml {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

//  layer precedence 

#[test]
fn no_layers_yields_defaults_then_fails_validation_on_model() {
    // Nothing provides a model — defaults flow in, validate() catches it.
    let err = ConfigLoader::new(None, env(&[]), CliArgs::default())
        .load()
        .unwrap_err();

    let ConfigError::MissingRequired { field, .. } = err else {
        panic!("expected MissingRequired, got {err:?}");
    };
    assert_eq!(field, "llm.model");
}

#[test]
fn toml_only_supplies_full_config() {
    let file = TempToml::new(
        "toml_only_supplies_full_config",
        r#"
[llm]
model = "llama-3"
provider = "ollama"
temperature = 0.5
maxTokens = 256
"#,
    )
    .unwrap();

    let config = ConfigLoader::new(
        Some(file.path().to_path_buf()),
        env(&[]),
        CliArgs::default(),
    )
    .load()
    .unwrap();

    assert_eq!(config.llm.provider.model, "llama-3");
    assert_eq!(config.llm.provider.provider, ProviderName::Ollama);
    assert_eq!(config.llm.generation.temperature, 0.5);
    assert_eq!(config.llm.generation.max_tokens, 256);
    // Untouched generation fields fall back to defaults.
    let defaults = LlmGenerationConfig::default();
    assert_eq!(config.llm.generation.top_p, defaults.top_p);
    assert_eq!(
        config.llm.generation.frequency_penalty,
        defaults.frequency_penalty
    );
}

#[test]
fn env_overrides_toml_for_model() {
    let file = TempToml::new(
        "env_overrides_toml_for_model",
        r#"
[llm]
model = "from-toml"
provider = "ollama"
"#,
    )
    .unwrap();

    let config = ConfigLoader::new(
        Some(file.path().to_path_buf()),
        env(&[("AUTOCOMMIT_LLM_MODEL", "from-env")]),
        CliArgs::default(),
    )
    .load()
    .unwrap();

    assert_eq!(config.llm.provider.model, "from-env");
}

#[test]
fn cli_overrides_toml_for_temperature() {
    let file = TempToml::new(
        "cli_overrides_toml_for_temperature",
        r#"
[llm]
model = "m"
provider = "ollama"
temperature = 0.1
"#,
    )
    .unwrap();

    let cli = CliArgs {
        temperature: Some(0.9),
        ..Default::default()
    };

    let config = ConfigLoader::new(Some(file.path().to_path_buf()), env(&[]), cli)
        .load()
        .unwrap();

    assert_eq!(config.llm.generation.temperature, 0.9);
}

#[test]
fn env_does_not_set_generation_params() {
    // Design pin: `build_env_partial` returns an empty
    // `PartialLlmGenerationConfig`; generation defaults must shine through
    // even when env supplies provider / model. If env ever gains a
    // generation override, update this test alongside the implementation.
    let config = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
            ("AUTOCOMMIT_LLM_MODEL", "m"),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap();

    assert_eq!(config.llm.generation, LlmGenerationConfig::default());
    assert_eq!(config.llm.provider.provider, ProviderName::Ollama);
}

//  validation gate 

#[test]
fn openai_without_api_key_returns_missing_required() {
    // provider defaults to Openai; model is set; api_key absent.
    let err = ConfigLoader::new(
        None,
        env(&[("AUTOCOMMIT_LLM_MODEL", "gpt-4")]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    let ConfigError::MissingRequired { field, .. } = err else {
        panic!("expected MissingRequired, got {err:?}");
    };
    assert_eq!(field, "llm.api_key");
}

#[test]
fn ollama_does_not_require_api_key() {
    // Design pin: ollama is local; auth is optional, validate() must not flag it.
    let result = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
            ("AUTOCOMMIT_LLM_MODEL", "llama-3"),
        ]),
        CliArgs::default(),
    )
    .load();

    assert!(
        result.is_ok(),
        "ollama without api_key must pass validation; got: {result:?}"
    );
}

//  error paths in layers 

#[test]
fn malformed_toml_returns_toml_parse_error() {
    // Unterminated string literal — guaranteed TOML syntax error.
    let file = TempToml::new(
        "malformed_toml_returns_toml_parse_error",
        r#"
provider = "unterminated
"#,
    )
    .unwrap();

    let err = ConfigLoader::new(
        Some(file.path().to_path_buf()),
        env(&[]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    assert!(matches!(err, ConfigError::TomlParse { .. }), "got {err:?}");
}

#[test]
fn bad_integer_env_returns_env_parse() {
    let err = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_MODEL", "gpt-4"),
            ("AUTOCOMMIT_RETRY_MAX_RETRIES", "not-an-int"),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    let ConfigError::EnvParse { var, .. } = err else {
        panic!("expected EnvParse, got {err:?}");
    };
    assert_eq!(var, "AUTOCOMMIT_RETRY_MAX_RETRIES");
}

#[test]
fn bad_bool_env_returns_env_parse() {
    let err = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_MODEL", "gpt-4"),
            ("AUTOCOMMIT_LLM_API_KEY", "k"),
            ("AUTOCOMMIT_RETRY_JITTER", "banana"),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    let ConfigError::EnvParse { var, .. } = err else {
        panic!("expected EnvParse, got {err:?}");
    };
    assert_eq!(var, "AUTOCOMMIT_RETRY_JITTER");
}

//  normalize_empty_strings 

#[test]
fn empty_base_url_in_toml_normalizes_to_none() {
    let file = TempToml::new(
        "empty_base_url_in_toml_normalizes_to_none",
        r#"
[llm]
model = "m"
provider = "ollama"
baseUrl = ""
"#,
    )
    .unwrap();

    let config = ConfigLoader::new(
        Some(file.path().to_path_buf()),
        env(&[]),
        CliArgs::default(),
    )
    .load()
    .unwrap();

    assert!(config.llm.provider.base_url.is_none());
}

#[test]
fn empty_api_key_env_caught_by_openai_validation() {
    // Pin the interaction: env `AUTOCOMMIT_LLM_API_KEY=""` becomes
    // `Some(ApiKey(""))` after merge, then `normalize_empty_strings`
    // converts it to `None`, then validate() flags the missing key for
    // the openai provider. If either step changes, this test catches it.
    let err = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_MODEL", "gpt-4"),
            ("AUTOCOMMIT_LLM_API_KEY", ""),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    let ConfigError::MissingRequired { field, .. } = err else {
        panic!("expected MissingRequired, got {err:?}");
    };
    assert_eq!(field, "llm.api_key");
}
