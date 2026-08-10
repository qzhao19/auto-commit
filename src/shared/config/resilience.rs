use std::time::Duration;

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub struct TimeoutConfig {
    pub timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub factor: f32,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            factor: 2.0,
            jitter: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResilienceConfig {
    pub retry: RetryConfig,
    pub timeout: TimeoutConfig,
}

// --- Partial ---

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialTimeoutConfig {
    pub timeout_ms: Option<u64>,
}

impl PartialTimeoutConfig {
    pub fn merge(self, config: TimeoutConfig) -> TimeoutConfig {
        TimeoutConfig {
            timeout: opt_ms(self.timeout_ms, config.timeout),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialRetryConfig {
    pub max_retries: Option<u32>,
    pub initial_delay_ms: Option<u64>,
    pub max_delay_ms: Option<u64>,
    pub factor: Option<f32>,
    pub jitter: Option<bool>,
}

impl PartialRetryConfig {
    pub fn merge(self, config: RetryConfig) -> RetryConfig {
        RetryConfig {
            max_retries: self.max_retries.unwrap_or(config.max_retries),
            initial_delay: opt_ms(self.initial_delay_ms, config.initial_delay),
            max_delay: opt_ms(self.max_delay_ms, config.max_delay),
            factor: self.factor.unwrap_or(config.factor),
            jitter: self.jitter.unwrap_or(config.jitter),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialResilienceConfig {
    #[serde(default)]
    pub retry: PartialRetryConfig,
    #[serde(default)]
    pub timeout: PartialTimeoutConfig,
}

impl PartialResilienceConfig {
    pub fn merge(self, config: ResilienceConfig) -> ResilienceConfig {
        ResilienceConfig {
            retry: self.retry.merge(config.retry),
            timeout: self.timeout.merge(config.timeout),
        }
    }
}

#[inline]
fn opt_ms(ms: Option<u64>, config: Duration) -> Duration {
    ms.map(Duration::from_millis).unwrap_or(config)
}
