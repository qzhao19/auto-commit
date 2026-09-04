//! Integration tests for ConfigLoader: input sources (default / toml file /
//! env vars / CLI) → final `AppConfig` or `ConfigError`, through the public
//! `ConfigLoader::new(..).load()` path only.
//!
//! Layer precedence (field-level): default < toml file < env < CLI.
//!
//! Facts encoded here that differ from the original spec (source wins):
//! - generation is NOT env-overridable: AUTOCOMMIT_GENERATION_* keys are ignored
//! - apiKey in toml is accepted with a stderr warning, not rejected
//! - validation is inline in `validate()` and additionally bounds
//!   max_retries <= 10 and initial_delay > 0
//! - missing model reports `missing required field llm.model: ...`
//! - empty-string api_key/base_url are normalized to None before validation

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;

use crate::infra::config::ConfigLoader;
use crate::shared::config::{AppConfig, CliArgs, ProviderName};
use crate::shared::exception::ConfigError;

// Helpers

/// Minimal self-cleaning temp dir (no new deps). Replace with the project's
/// shared fixture if one already exists in test/unittest/common.
struct TestDir(PathBuf);

impl TestDir {
    fn new(test: &str) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "autocommit-config-{}-{}-{id}",
            test,
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write `autocommit.toml` content into the temp dir, return its path.
fn write_toml(dir: &TestDir, content: &str) -> PathBuf {
    let path = dir.path().join("config.toml");
    std::fs::write(&path, content).expect("write toml");
    path
}

/// Env is injected as a Vec (not process env) → zero cross-test pollution.
fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Real-process-env semantics for the injected vec: setting a var REPLACES
/// any existing entry. A plain `push` would create a duplicate key, and
/// `env_str` resolves the FIRST match — ollama_env()'s "ollama" would
/// silently win over the test value.
fn env_with(var: &str, value: &str) -> Vec<(String, String)> {
    let mut vars: Vec<(String, String)> =
        ollama_env().into_iter().filter(|(k, _)| k != var).collect();
    vars.push((var.to_string(), value.to_string()));
    vars
}

/// Minimal happy-path env: provider + required model, no api key needed.
fn ollama_env() -> Vec<(String, String)> {
    env(&[
        ("AUTOCOMMIT_LLM_PROVIDER", "ollama"),
        ("AUTOCOMMIT_LLM_MODEL", "qwen2.5:3b"),
    ])
}

fn cli() -> CliArgs {
    CliArgs::default()
}

/// The single entry point every test drives.
fn load_from(
    file: Option<PathBuf>,
    env_vars: Vec<(String, String)>,
    cli_args: CliArgs,
) -> Result<AppConfig, ConfigError> {
    ConfigLoader::new(file, env_vars, cli_args).load()
}

fn assert_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

// --- Error matchers: assert the variant + the business-meaningful parts ---

fn invalid_value_field(err: ConfigError) -> (&'static str, String) {
    match err {
        ConfigError::InvalidValue { field, reason } => (field, reason),
        other => panic!("expected InvalidValue, got: {other}"),
    }
}

fn missing_required(err: ConfigError) -> (&'static str, String) {
    match err {
        ConfigError::MissingRequired { field, hint } => (field, hint),
        other => panic!("expected MissingRequired, got: {other}"),
    }
}

fn env_parse(err: ConfigError) -> (String, &'static str) {
    match err {
        ConfigError::EnvParse { var, expected } => (var, expected),
        other => panic!("expected EnvParse, got: {other}"),
    }
}

fn assert_toml_parse(err: &ConfigError, expected_file: &Path) {
    match err {
        ConfigError::TomlParse { path, .. } => {
            assert_eq!(path.as_path(), expected_file);
            let display = err.to_string();
            assert!(
                display.contains("config.toml"),
                "error must name the file, got: {display}"
            );
        }
        other => panic!("expected TomlParse, got: {other}"),
    }
}

const TOML_OLLAMA: &str = r#"
[llm]
provider = "ollama"
model = "qwen2.5:3b"
"#;

// A. Success paths / layer precedence

// A1: no file, no CLI — env alone satisfies `model`; everything else default.
#[test]
fn a_defaults_plus_env_model_loads_with_code_defaults() {
    let cfg = load_from(None, ollama_env(), cli()).expect("must load");

    assert_eq!(cfg.llm.provider.model, "qwen2.5:3b");

    // generation + resilience must equal code defaults
    let d = AppConfig::default();
    assert_f32(cfg.llm.generation.temperature, d.llm.generation.temperature);
    assert_eq!(cfg.llm.generation.max_tokens, d.llm.generation.max_tokens);
    assert_eq!(
        cfg.resilience.retry.max_retries,
        d.resilience.retry.max_retries
    );
    assert_eq!(cfg.resilience.timeout.timeout, d.resilience.timeout.timeout);
}

// A2: flat [llm] carries provider + generation keys together; unset fields default.
#[test]
fn a_toml_provides_provider_and_partial_generation() {
    let dir = TestDir::new("a2");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "ollama"
model = "llama3.1:8b"
baseUrl = "http://127.0.0.1:11434"
temperature = 0.2
"#,
    );

    let cfg = load_from(Some(path), vec![], cli()).expect("must load");

    assert_eq!(cfg.llm.provider.model, "llama3.1:8b");
    // Regression for: string `provider` must survive the serde flatten channel
    assert!(matches!(cfg.llm.provider.provider, ProviderName::Ollama));
    assert_eq!(
        cfg.llm.provider.base_url.as_deref(),
        Some("http://127.0.0.1:11434")
    );
    assert_f32(cfg.llm.generation.temperature, 0.2);
    assert_eq!(
        cfg.llm.generation.max_tokens,
        AppConfig::default().llm.generation.max_tokens
    );
}

// A3: body unchanged — the fix is the flat TOML_OLLAMA const above.
#[test]
fn a_env_overrides_toml_for_provider_fields() {
    let dir = TestDir::new("a3");
    let path = write_toml(&dir, TOML_OLLAMA);

    let cfg = load_from(
        Some(path),
        env(&[
            ("AUTOCOMMIT_LLM_MODEL", "env-model-wins"),
            ("AUTOCOMMIT_LLM_BASE_URL", "http://env-host:11434"),
        ]),
        cli(),
    )
    .expect("must load");

    assert_eq!(cfg.llm.provider.model, "env-model-wins");
    assert_eq!(
        cfg.llm.provider.base_url.as_deref(),
        Some("http://env-host:11434")
    );
}

// A4: CLI --temperature masks toml's flat temperature; toml maxTokens survives.
#[test]
fn a_cli_overrides_toml_generation() {
    let dir = TestDir::new("a4");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "ollama"
model = "qwen2.5:3b"
temperature = 0.2
maxTokens = 512
"#,
    );
    let cli_args =
        CliArgs::try_parse_from(["auto-commit", "--temperature", "0.9"]).expect("valid CLI");

    let cfg = load_from(Some(path), vec![], cli_args).expect("must load");

    assert_f32(cfg.llm.generation.temperature, 0.9); // CLI over toml
    assert_eq!(cfg.llm.generation.max_tokens, 512); // toml untouched field survives
}

// A5: resilience tables stay nested per the sample; llm goes flat.
#[test]
fn a_all_layers_each_field_from_highest_source() {
    let dir = TestDir::new("a5");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "ollama"
model = "toml-model"
temperature = 0.2
maxTokens = 512

[resilience.timeout]
timeoutMs = 11111
"#,
    );
    let env_vars = env(&[
        ("AUTOCOMMIT_LLM_MODEL", "env-model"),
        ("AUTOCOMMIT_TIMEOUT_MS", "22222"),
    ]);
    let cli_args = CliArgs {
        temperature: Some(0.9),
        ..Default::default()
    };

    let cfg = load_from(Some(path), env_vars, cli_args).expect("must load");

    assert_eq!(cfg.llm.provider.model, "env-model"); // env > toml
    assert_f32(cfg.llm.generation.temperature, 0.9); // CLI > toml
    assert_eq!(cfg.llm.generation.max_tokens, 512); // toml only
    assert_eq!(
        cfg.resilience.timeout.timeout,
        std::time::Duration::from_millis(22222) // env > toml
    );
}

// A6 / E-adjacent: configured file path that does not exist → file layer is
// optional, not an error.
#[test]
fn a_missing_toml_file_is_skipped_not_an_error() {
    let dir = TestDir::new("a6");
    let nonexistent = dir.path().join("does-not-exist.toml");

    let cfg = load_from(Some(nonexistent), ollama_env(), cli()).expect("must load");
    assert_eq!(cfg.llm.provider.model, "qwen2.5:3b");
}

// A7: apiKey arrives only via env and lands in AppConfig.
#[test]
fn a_api_key_from_env_only_for_openai() {
    let dir = TestDir::new("a7");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "openai"
model = "gpt-4o-mini"
"#,
    );
    let cfg = load_from(
        Some(path),
        env(&[("AUTOCOMMIT_LLM_API_KEY", "sk-env-key")]),
        cli(),
    )
    .expect("must load");

    assert_eq!(
        cfg.llm.provider.api_key.as_ref().unwrap().as_str(),
        "sk-env-key"
    );
}

// A8: apiKey in toml is accepted (with stderr warning) — flat [llm].apiKey.
#[test]
fn a_api_key_in_toml_is_accepted_with_warning() {
    let dir = TestDir::new("a8");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "openai"
model = "gpt-4o-mini"
apiKey = "sk-from-toml"
"#,
    );

    let cfg = load_from(Some(path), vec![], cli()).expect("must load");
    assert_eq!(
        cfg.llm.provider.api_key.as_ref().unwrap().as_str(),
        "sk-from-toml"
    );
}

// Env resilience layer: every AUTOCOMMIT_RETRY_*/TIMEOUT_MS key parses and
// maps to the final Duration/int/float fields.
#[test]
fn a_env_sets_resilience_fields_with_ms_to_duration_mapping() {
    let mut env_vars = ollama_env();
    env_vars.extend(env(&[
        ("AUTOCOMMIT_RETRY_MAX_RETRIES", "5"),
        ("AUTOCOMMIT_RETRY_INITIAL_DELAY_MS", "250"),
        ("AUTOCOMMIT_RETRY_MAX_DELAY_MS", "4000"),
        ("AUTOCOMMIT_RETRY_FACTOR", "2.5"),
        ("AUTOCOMMIT_RETRY_JITTER", "true"),
        ("AUTOCOMMIT_TIMEOUT_MS", "45000"),
    ]));

    let cfg = load_from(None, env_vars, cli()).expect("must load");

    let r = &cfg.resilience.retry;
    assert_eq!(r.max_retries, 5);
    assert_eq!(r.initial_delay, std::time::Duration::from_millis(250));
    assert_eq!(r.max_delay, std::time::Duration::from_millis(4000));
    assert!((r.factor - 2.5).abs() < 1e-9);
    assert!(r.jitter);
    assert_eq!(
        cfg.resilience.timeout.timeout,
        std::time::Duration::from_millis(45000)
    );
}

// B. Required fields / missing values

// B8: model absent from every layer → explicit "missing required" error.
#[test]
fn b_model_missing_everywhere_reports_required_field() {
    let err = load_from(None, vec![], cli()).expect_err("must fail");

    let (field, hint) = missing_required(err);
    assert_eq!(field, "llm.model");
    assert!(hint.contains("AUTOCOMMIT_LLM_MODEL"), "hint: {hint}");
}

// B9a: provider = openai without any api key → required api_key error.
#[test]
fn b_openai_provider_requires_api_key() {
    let dir = TestDir::new("b9");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "openai"
model = "gpt-4o-mini"
"#,
    );

    let err = load_from(Some(path), vec![], cli()).expect_err("must fail");
    let (field, hint) = missing_required(err);
    assert_eq!(field, "llm.api_key");
    assert!(hint.contains("AUTOCOMMIT_LLM_API_KEY"), "hint: {hint}");
}

// B9b: ollama needs no auth → loads fine without api key and base_url.
#[test]
fn b_ollama_needs_no_api_key_and_base_url_is_optional() {
    let cfg = load_from(None, ollama_env(), cli()).expect("must load");

    assert!(cfg.llm.provider.api_key.is_none());
    assert_eq!(
        cfg.llm.provider.base_url,
        AppConfig::default().llm.provider.base_url
    );
}

// B-extra: empty-string api key is normalized to None before validation, so
#[test]
fn b_empty_api_key_is_normalized_to_missing() {
    let dir = TestDir::new("b10");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "openai"
model = "gpt-4o-mini"
"#,
    );

    let err = load_from(Some(path), env(&[("AUTOCOMMIT_LLM_API_KEY", "")]), cli())
        .expect_err("empty key must fail like missing");
    assert_eq!(missing_required(err).0, "llm.api_key");
}

// C. Parse errors

// C10a: syntactically broken toml → hard error naming the file (with serde
// line info in the message).
#[test]
fn c_malformed_toml_is_hard_error_with_path() {
    let dir = TestDir::new("c10");
    let path = write_toml(&dir, "[llm.provider\nmodel = \"x\"");

    let err = load_from(Some(path.clone()), vec![], cli()).expect_err("must fail");
    assert_toml_parse(&err, &path);
    // serde/toml diagnostics carry a line reference
    let msg = load_from(Some(path), vec![], cli()).expect_err("must fail");
    assert!(msg.to_string().contains("line"), "msg: {msg}");
}

// C10b: valid syntax, wrong type (`temperature = "hot"`) → same TomlParse
// variant, still names the file.
#[test]
fn c_toml_wrong_field_type_is_parse_error() {
    let dir = TestDir::new("c11");
    let path = write_toml(&dir, "[llm]\ntemperature = \"hot\"\n");

    let err = load_from(Some(path.clone()), vec![], cli()).expect_err("must fail");
    assert_toml_parse(&err, &path);
}

// C11: env keys with unparseable values → EnvParse naming var + expected type.
// NOTE: unlike the spec example, generation has no env keys by design; these
// four cover every env parsing path (int / float / enum / bool).
#[test]
fn c_env_parse_errors_name_variable_and_expected_type() {
    let cases: [(&str, &str, &str); 4] = [
        (
            "AUTOCOMMIT_RETRY_MAX_RETRIES",
            "three",
            "a non-negative integer",
        ),
        ("AUTOCOMMIT_RETRY_FACTOR", "fast", "a float >= 1.0"),
        ("AUTOCOMMIT_LLM_PROVIDER", "bogus", "openai or ollama"),
        ("AUTOCOMMIT_RETRY_JITTER", "maybe", "true/false/1/0"),
    ];

    for (var, value, expected) in cases {
        let env_vars = env_with(var, value);

        let err = load_from(None, env_vars, cli()).expect_err(&format!("{var}={value} must fail"));
        let (got_var, got_expected) = env_parse(err);
        assert_eq!(got_var, var);
        assert_eq!(got_expected, expected);
    }
}

// C12: unrelated env vars (including process-level junk like HOME) never
// reach the parser.
#[test]
fn c_unrelated_env_vars_are_ignored() {
    let mut env_vars = ollama_env();
    env_vars.extend(env(&[
        ("AUTOCOMMIT_SOMETHING_ELSE", "baz"),
        ("HOME", "/nonexistent-home-xyz"),
        ("AUTOCOMMIT_GENERATION_TEMPERATURE", "9.9"), // generation: no env keys exist
    ]));

    let cfg = load_from(None, env_vars, cli()).expect("must load");
    // Ignored generation env junk must not leak into the merged config
    let d = AppConfig::default();
    assert_f32(cfg.llm.generation.temperature, d.llm.generation.temperature);
}

// D. Semantic validation (runs once, after merging)

// D13: temperature outside [0, 2]; CLI carries it, so this is also
// "invalid highest-layer value" for generation boundaries.
#[test]
fn d_temperature_out_of_range_rejected() {
    for bad in [-0.1_f32, 2.1] {
        let cli = CliArgs {
            temperature: Some(bad),
            ..Default::default()
        };
        let err = load_from(None, ollama_env(), cli).expect_err("must fail");
        let (field, reason) = invalid_value_field(err);
        assert_eq!(field, "llm.temperature");
        assert!(reason.contains("[0, 2]"), "reason: {reason}");
    }
}

// Boundary values at both ends of [0, 2], top_p's right end 1.0,
// penalties 0/2, max_tokens 1 → all valid.
#[test]
fn d_boundary_values_are_accepted() {
    let cli = CliArgs {
        temperature: Some(0.0),
        top_p: Some(1.0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(2.0),
        max_tokens: Some(1),
        ..Default::default()
    };
    let cfg = load_from(None, ollama_env(), cli).expect("boundaries must load");
    assert_eq!(cfg.llm.generation.max_tokens, 1);

    let cli = CliArgs {
        temperature: Some(2.0),
        ..Default::default()
    };
    load_from(None, ollama_env(), cli).expect("temperature = 2.0 must load");
}

// D14: top_p must be in (0, 1] — both closed ends are violations.
#[test]
fn d_top_p_zero_and_above_one_rejected() {
    for bad in [0.0_f32, 1.1] {
        let cli = CliArgs {
            top_p: Some(bad),
            ..Default::default()
        };
        let err = load_from(None, ollama_env(), cli).expect_err("must fail");
        let (field, reason) = invalid_value_field(err);
        assert_eq!(field, "llm.top_p");
        assert!(reason.contains("(0, 1]"), "reason: {reason}");
    }
}

// D15: max_tokens <= 0 rejected (zero and negative).
#[test]
fn d_max_tokens_zero_or_negative_rejected() {
    for bad in [0_i32, -100] {
        let cli = CliArgs {
            max_tokens: Some(bad),
            ..Default::default()
        };
        let err = load_from(None, ollama_env(), cli).expect_err("must fail");
        let (field, reason) = invalid_value_field(err);
        assert_eq!(field, "llm.max_tokens");
        assert!(reason.contains("> 0"), "reason: {reason}");
    }
}

// D16: penalties outside [0, 2] rejected.
#[test]
fn d_penalties_out_of_range_rejected() {
    let cli = CliArgs {
        frequency_penalty: Some(2.1),
        ..Default::default()
    };
    let err = load_from(None, ollama_env(), cli).expect_err("must fail");
    assert_eq!(invalid_value_field(err).0, "llm.frequency_penalty");

    let cli = CliArgs {
        presence_penalty: Some(-0.1),
        ..Default::default()
    };
    let err = load_from(None, ollama_env(), cli).expect_err("must fail");
    assert_eq!(invalid_value_field(err).0, "llm.presence_penalty");
}

// D17: resilience validation — source adds two rules the spec didn't list
// (max_retries <= 10, initial_delay > 0); all five are exercised via env,
// which is the highest layer for resilience.
#[test]
fn d_resilience_validation_rejects_each_rule_violation() {
    let cases: [(&str, &str, &str); 5] = [
        (
            "AUTOCOMMIT_RETRY_MAX_RETRIES",
            "11",
            "resilience.retry.max_retries",
        ),
        (
            "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS",
            "0",
            "resilience.retry.initial_delay",
        ),
        ("AUTOCOMMIT_RETRY_FACTOR", "0.9", "resilience.retry.factor"),
        (
            "AUTOCOMMIT_TIMEOUT_MS",
            "0",
            "resilience.timeout.timeout_ms",
        ),
        // max_delay < initial_delay needs two keys; handled separately below
        ("AUTOCOMMIT_RETRY_INITIAL_DELAY_MS", "0", ""), // placeholder, skipped
    ];

    for (var, value, field) in &cases[..4] {
        let mut env_vars = ollama_env();
        env_vars.push((var.to_string(), value.to_string()));
        let err = load_from(None, env_vars, cli()).expect_err(&format!("{var}={value} must fail"));
        assert_eq!(&invalid_value_field(err).0, field);
    }
}

// max_delay < initial_delay is a cross-field rule.
#[test]
fn d_retry_max_delay_below_initial_delay_rejected() {
    let mut env_vars = ollama_env();
    env_vars.extend(env(&[
        ("AUTOCOMMIT_RETRY_INITIAL_DELAY_MS", "500"),
        ("AUTOCOMMIT_RETRY_MAX_DELAY_MS", "100"),
    ]));

    let err = load_from(None, env_vars, cli()).expect_err("must fail");
    let (field, reason) = invalid_value_field(err);
    assert_eq!(field, "resilience.retry.max_delay");
    assert!(reason.contains(">= initial_delay"), "reason: {reason}");
}

// D18: invalid low-layer values masked by valid higher-layer values.
#[test]
fn d_invalid_low_priority_value_masked_by_higher_layer() {
    let dir = TestDir::new("d18");
    let path = write_toml(
        &dir,
        r#"
[llm]
provider = "ollama"
model = "qwen2.5:3b"
temperature = 99.0
maxTokens = 0

[resilience.retry]
factor = 0.1
"#,
    );
    let cli_args = CliArgs {
        temperature: Some(0.7),
        max_tokens: Some(256),
        ..Default::default()
    };
    let env_vars = env(&[("AUTOCOMMIT_RETRY_FACTOR", "2.0")]);

    let cfg = load_from(Some(path), env_vars, cli_args).expect("masked values must load");
    assert_f32(cfg.llm.generation.temperature, 0.7); // CLI masks toml 99.0
    assert_eq!(cfg.llm.generation.max_tokens, 256); // CLI masks toml 0
    assert!((cfg.resilience.retry.factor - 2.0).abs() < 1e-9); // env masks toml 0.1
}

// D19: invalid CLI value fails despite a valid flat toml value underneath.
#[test]
fn d_invalid_cli_value_fails_despite_valid_toml() {
    let dir = TestDir::new("d19");
    let path = write_toml(
        &dir,
        "[llm]\nprovider = \"ollama\"\nmodel = \"m\"\ntemperature = 0.5\n",
    );
    let cli_args = CliArgs {
        temperature: Some(3.0),
        ..Default::default()
    };

    let err = load_from(Some(path), vec![], cli_args).expect_err("must fail");
    assert_eq!(invalid_value_field(err).0, "llm.temperature");
}

// D20: invalid env resilience value fails despite valid toml underneath
// (resilience keeps its nested tables per the sample file).
#[test]
fn d_invalid_env_resilience_value_fails_despite_valid_toml() {
    let dir = TestDir::new("d20");
    let path = write_toml(
        &dir,
        "[llm]\nprovider = \"ollama\"\nmodel = \"m\"\n\n[resilience.retry]\nfactor = 2.0\n",
    );

    let err = load_from(
        Some(path),
        env(&[("AUTOCOMMIT_RETRY_FACTOR", "0.5")]),
        cli(),
    )
    .expect_err("must fail");
    assert_eq!(invalid_value_field(err).0, "resilience.retry.factor");
}

// E. Boundaries / degenerate inputs

// E20: an empty toml file parses to an all-None partial; env still supplies
// the required model.
#[test]
fn e_empty_toml_file_loads_fine() {
    let dir = TestDir::new("e20");
    let path = write_toml(&dir, "");

    let cfg = load_from(Some(path), ollama_env(), cli()).expect("must load");
    assert_eq!(cfg.llm.provider.model, "qwen2.5:3b");
}

// E21: renamed — there is no "generation table" in the flat schema; the
// file supplies only generation FIELDS, env still supplies the model.
#[test]
fn e_toml_with_only_generation_fields() {
    let dir = TestDir::new("e21");
    let path = write_toml(&dir, "[llm]\ntopP = 0.5\n");

    let cfg = load_from(Some(path), ollama_env(), cli()).expect("must load");
    assert_f32(cfg.llm.generation.top_p, 0.5);
    assert_eq!(cfg.llm.provider.model, "qwen2.5:3b"); // env layer still worked
}

// E22: CLI all-None must not stomp on flat toml values.
#[test]
fn e_cli_all_none_leaves_toml_values_intact() {
    let dir = TestDir::new("e22");
    let path = write_toml(
        &dir,
        "[llm]\nprovider = \"ollama\"\nmodel = \"m\"\ntemperature = 0.33\n",
    );

    let cfg = load_from(Some(path), vec![], cli()).expect("must load");
    assert_f32(cfg.llm.generation.temperature, 0.33);
}

// E23: `nan` is legal toml syntax; must reach validate() and be rejected.
// model must come from the file, otherwise MissingRequired fires first.
#[test]
fn e_nan_temperature_from_toml_rejected() {
    let dir = TestDir::new("e23");
    let path = write_toml(
        &dir,
        "[llm]\nprovider = \"ollama\"\nmodel = \"m\"\ntemperature = nan\n",
    );

    let err = load_from(Some(path), vec![], cli()).expect_err("nan must fail");
    assert_eq!(invalid_value_field(err).0, "llm.temperature");
}
