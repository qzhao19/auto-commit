use std::path::PathBuf;

use clap::Parser;

use crate::infra::config::ConfigLoader;
use crate::shared::config::{CliArgs, LlmGenerationConfig, ProviderName};
use crate::shared::exception::ConfigError;

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
fn openai_without_base_url_returns_missing_required() {
    // api_key satisfied via env; baseUrl absent from every layer → the
    // error must name the env var that fixes it.
    let err = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_MODEL", "gpt-4"),
            ("AUTOCOMMIT_LLM_API_KEY", "sk-test"),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap_err();

    let ConfigError::MissingRequired { field, hint } = err else {
        panic!("expected MissingRequired, got: {err:?}");
    };
    assert_eq!(field, "llm.baseUrl");
    assert!(hint.contains("AUTOCOMMIT_LLM_BASE_URL"), "hint: {hint}");
}

#[test]
fn ollama_without_base_url_still_loads() {
    // ollama keeps its in-code default endpoint — baseUrl stays optional.
    let cfg = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
            ("AUTOCOMMIT_LLM_MODEL", "qwen2.5:3b"),
        ]),
        CliArgs::default(),
    )
    .load()
    .unwrap();
    assert!(cfg.llm.provider.base_url.is_none());
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

//  CLI arguments layer
//
//  CLI is the highest layer and — by design — the ONLY layer allowed to
//  set generation params. These tests pin the full flag surface, the
//  merge granularity, the clap parsing contract, and the boundary that
//  provider / model / apiKey / resilience never appear on the CLI.

/// All 5 flags together override a fully-populated TOML — the complete
/// flag→field mapping, driven through real clap parsing.
#[test]
fn cli_all_flags_override_toml_generation_fields() {
    let file = TempToml::new(
        "cli_all_flags_override_toml",
        r#"
[llm]
model = "m"
provider = "ollama"
temperature = 0.1
maxTokens = 64
topP = 0.2
frequencyPenalty = 0.3
presencePenalty = 0.4
"#,
    )
    .unwrap();

    let cli = CliArgs::try_parse_from([
        "auto-commit",
        "--temperature",
        "0.7",
        "--max-tokens",
        "512",
        "--top-p",
        "0.55",
        "--frequency-penalty",
        "0.15",
        "--presence-penalty",
        "0.25",
    ])
    .expect("valid CLI args");

    let config = ConfigLoader::new(Some(file.path().to_path_buf()), env(&[]), cli)
        .load()
        .unwrap();

    let g = &config.llm.generation;
    assert_eq!(g.temperature, 0.7);
    assert_eq!(g.max_tokens, 512);
    assert_eq!(g.top_p, 0.55);
    assert_eq!(g.frequency_penalty, 0.15);
    assert_eq!(g.presence_penalty, 0.25);
}

/// One flag set, four unset — per-field granularity: the CLI field
/// wins, the TOML values survive for everything not passed.
#[test]
fn cli_partial_override_keeps_toml_for_unset_flags() {
    let file = TempToml::new(
        "cli_partial_override_keeps_toml",
        r#"
[llm]
model = "m"
provider = "ollama"
temperature = 0.1
maxTokens = 64
topP = 0.2
frequencyPenalty = 0.3
presencePenalty = 0.4
"#,
    )
    .unwrap();

    let cli = CliArgs::try_parse_from(["auto-commit", "--frequency-penalty", "0.9"])
        .expect("valid CLI args");

    let config = ConfigLoader::new(Some(file.path().to_path_buf()), env(&[]), cli)
        .load()
        .unwrap();

    let g = &config.llm.generation;
    assert_eq!(g.frequency_penalty, 0.9); // CLI wins
    assert_eq!(g.temperature, 0.1); // toml survives
    assert_eq!(g.max_tokens, 64);
    assert_eq!(g.top_p, 0.2);
    assert_eq!(g.presence_penalty, 0.4);
}

/// No TOML, no generation in env (by design) — the one set flag lands,
/// the other four fall back to built-in defaults.
#[test]
fn cli_only_flag_falls_back_to_generation_defaults() {
    let cli = CliArgs::try_parse_from(["auto-commit", "--top-p", "0.42"]).expect("valid CLI args");

    let config = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
            ("AUTOCOMMIT_LLM_MODEL", "m"),
        ]),
        cli,
    )
    .load()
    .unwrap();

    let defaults = LlmGenerationConfig::default();
    let g = &config.llm.generation;
    assert_eq!(g.top_p, 0.42);
    assert_eq!(g.temperature, defaults.temperature);
    assert_eq!(g.max_tokens, defaults.max_tokens);
    assert_eq!(g.frequency_penalty, defaults.frequency_penalty);
    assert_eq!(g.presence_penalty, defaults.presence_penalty);
}

/// Design pin: `into_partial()` only fills generation — a fully
/// populated CLI cannot satisfy the required `llm.model`. Provider /
/// model / apiKey / resilience are simply not expressible here.
#[test]
fn cli_flags_cannot_satisfy_required_model() {
    let cli = CliArgs::try_parse_from([
        "auto-commit",
        "--temperature",
        "0.7",
        "--max-tokens",
        "512",
        "--top-p",
        "0.55",
        "--frequency-penalty",
        "0.15",
        "--presence-penalty",
        "0.25",
    ])
    .expect("valid CLI args");

    let err = ConfigLoader::new(None, env(&[]), cli).load().unwrap_err();

    let ConfigError::MissingRequired { field, .. } = err else {
        panic!("expected MissingRequired, got {err:?}");
    };
    assert_eq!(field, "llm.model");
}

/// Design pin, the other half: those flags must not even PARSE. If a
/// future change ever exposes provider / apiKey / resilience on the
/// command line, this fails before any security review would.
#[test]
fn cli_rejects_flags_outside_generation_surface() {
    for args in [
        ["--provider", "ollama"],
        ["--api-key", "sk-123"],
        ["--model", "gpt-4"],
        ["--timeout-ms", "1000"],
    ] {
        let argv = std::iter::once("auto-commit").chain(args);
        assert!(
            CliArgs::try_parse_from(argv).is_err(),
            "{args:?} must not be a CLI flag"
        );
    }
}

/// The clap contract itself: kebab-case spelling and strict value
/// types. Underscore spellings and non-numeric values are hard errors
/// at parse time, never ConfigError.
#[test]
fn cli_flag_names_and_value_types_are_strict() {
    // kebab-case spelling is the accepted form
    assert!(CliArgs::try_parse_from(["auto-commit", "--max-tokens", "128"]).is_ok());
    // underscore spelling is rejected
    assert!(CliArgs::try_parse_from(["auto-commit", "--max_tokens", "128"]).is_err());
    // value types are enforced by clap, not by merge/validate
    assert!(CliArgs::try_parse_from(["auto-commit", "--temperature", "abc"]).is_err());
    // f32 text does not parse into the i32 flag
    assert!(CliArgs::try_parse_from(["auto-commit", "--max-tokens", "1.5"]).is_err());
}

/// Env never overrides generation — even a generation-SHAPED variable
/// name is silently ignored: build_env_partial reads a fixed variable
/// list and has no generation entry. The only layers above defaults
/// for generation are TOML and CLI.
#[test]
fn generation_shaped_env_var_is_ignored_cli_wins() {
    let env_layers = env(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
        ("AUTOCOMMIT_LLM_MODEL", "m"),
        ("AUTOCOMMIT_LLM_TEMPERATURE", "0.1"),
    ]);

    let cli =
        CliArgs::try_parse_from(["auto-commit", "--temperature", "0.6"]).expect("valid CLI args");
    let config = ConfigLoader::new(None, env_layers.clone(), cli)
        .load()
        .unwrap();
    assert_eq!(config.llm.generation.temperature, 0.6); // CLI, not env

    // Without a CLI flag the var still does not leak in.
    let config = ConfigLoader::new(None, env_layers, CliArgs::default())
        .load()
        .unwrap();
    assert_eq!(
        config.llm.generation.temperature,
        LlmGenerationConfig::default().temperature
    );
}

/// A bad value from the HIGHEST layer is not rescued by a valid value
/// underneath — validate() sees the merged result. The TOML below every
/// case is fully valid, so the asserted field pins exactly one tripped
/// rule. Values avoid a leading '-' so the failure provably comes from
/// our validate(), not from clap's hyphen handling.
#[test]
fn cli_invalid_values_fail_validation_despite_valid_toml() {
    let cases: &[(&[&str], &str)] = &[
        (&["--temperature", "2.5"], "llm.temperature"),
        (&["--top-p", "0"], "llm.top_p"),
        (&["--max-tokens", "0"], "llm.max_tokens"),
        (&["--frequency-penalty", "2.5"], "llm.frequency_penalty"),
        (&["--presence-penalty", "2.5"], "llm.presence_penalty"),
    ];

    for (args, expected_field) in cases {
        let file = TempToml::new(
            &format!("cli_invalid_{expected_field}"),
            "[llm]\nmodel = \"m\"\nprovider = \"ollama\"\ntemperature = 0.5\ntopP = 0.9\nmaxTokens = 256\nfrequencyPenalty = 0.1\npresencePenalty = 0.1\n",
        )
        .unwrap();

        let argv = std::iter::once("auto-commit").chain(args.iter().copied());
        let cli = CliArgs::try_parse_from(argv).expect("valid CLI args");

        let err = ConfigLoader::new(Some(file.path().to_path_buf()), env(&[]), cli)
            .load()
            .unwrap_err();

        let ConfigError::InvalidValue { field, .. } = err else {
            panic!("expected InvalidValue for {expected_field}, got {err:?}");
        };
        assert_eq!(field, *expected_field);
    }
}

/// The inclusive upper bounds and the smallest legal max-tokens all
/// pass validation when they arrive via CLI.
#[test]
fn cli_boundary_generation_values_are_accepted() {
    let cli = CliArgs::try_parse_from([
        "auto-commit",
        "--temperature",
        "2", // inclusive upper bound
        "--top-p",
        "1", // inclusive upper bound
        "--max-tokens",
        "1", // smallest legal value
    ])
    .expect("valid CLI args");

    let config = ConfigLoader::new(
        None,
        env(&[
            ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
            ("AUTOCOMMIT_LLM_MODEL", "m"),
        ]),
        cli,
    )
    .load()
    .unwrap();

    assert_eq!(config.llm.generation.temperature, 2.0);
    assert_eq!(config.llm.generation.top_p, 1.0);
    assert_eq!(config.llm.generation.max_tokens, 1);
}
