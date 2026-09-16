use crate::shared::config::{PartialAppConfig, ProviderName};

#[test]
fn example_config_parses_and_covers_every_documented_parameter() {
    // Path resolves relative to this source file: <repo>/config.example.toml
    const EXAMPLE: &str = include_str!("../../config.example.toml");

    let partial: PartialAppConfig =
        toml::from_str(EXAMPLE).expect("config.example.toml must parse into PartialAppConfig");

    // [llm] provider layer
    let provider = &partial.llm.provider;
    assert_eq!(provider.provider, Some(ProviderName::Openai));
    assert_eq!(provider.model.as_deref(), Some("deepseek-v4-flash"));
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://api.deepseek.com")
    );
    assert!(provider.api_key.is_none(), "apiKey must stay commented out");

    // [llm] generation layer
    let generation = &partial.llm.generation;
    assert_eq!(generation.temperature, Some(0.8));
    assert_eq!(generation.max_tokens, Some(4096));
    assert_eq!(generation.top_p, Some(0.9));
    assert_eq!(generation.frequency_penalty, Some(0.0));
    assert_eq!(generation.presence_penalty, Some(0.0));

    // [resilience.retry]
    let retry = &partial.resilience.retry;
    assert_eq!(retry.max_retries, Some(3));
    assert_eq!(retry.initial_delay_ms, Some(1000));
    assert_eq!(retry.max_delay_ms, Some(10000));
    assert_eq!(retry.factor, Some(2.0));
    assert_eq!(retry.jitter, Some(true));

    // [resilience.timeout]
    assert_eq!(partial.resilience.timeout.timeout_ms, Some(30000));
}
