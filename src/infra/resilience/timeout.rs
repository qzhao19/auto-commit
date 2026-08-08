use std::future::Future;

use crate::shared::config::resilience::TimeoutConfig;

pub struct Timeout {
    config: TimeoutConfig,
}

impl Timeout {
    /// Validates config at construction; returns human-readable message on failure.
    pub fn new(config: TimeoutConfig) -> Result<Self, String> {
        if config.timeout.is_zero() {
            return Err("timeout must be greater than zero".into());
        }
        return Ok(Self { config });
    }

    /// Runs `func` bounded by the configured timeout.
    pub async fn execute<F, T>(&self, func: F) -> Result<T, String>
    where
        F: Future<Output = T>,
    {
        match tokio::time::timeout(self.config.timeout, func).await {
            Ok(value) => Ok(value),
            Err(_) => Err(format!("request timed out after {:?}", self.config.timeout)),
        }
    }
}
