use std::future::Future;
use std::time::Duration;

use rand;

use crate::shared::config::RetryConfig;

/// Retry execution result with three states:
/// - `Ok`: Success
/// - `NonRetryable`: Error is not retryable, passed through as-is
/// - `Exhausted`: Retry limit exhausted, carries actual attempt count and last error
#[derive(Debug)]
pub enum RetryResult<T, E> {
    Ok(T),
    NonRetryable(E),
    Exhausted { attempts: u32, last: E },
}

pub struct Retry {
    config: RetryConfig,
}

impl Retry {
    /// Validates config legality; returns human-readable message on failure
    pub fn new(config: RetryConfig) -> Result<Self, String> {
        validate(&config)?;
        Ok(Self { config })
    }

    /// Executes `func` with exponential backoff.
    pub async fn execute<T, E, F>(
        &self,
        is_retryable: impl Fn(&E) -> bool,
        mut func: impl FnMut() -> F,
    ) -> RetryResult<T, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        for attempt in 0..=self.config.max_retries {
            match func().await {
                Ok(value) => return RetryResult::Ok(value),
                Err(error) => {
                    // Last attempt failed: treat as exhausted regardless of error retryability
                    if attempt == self.config.max_retries {
                        return RetryResult::Exhausted {
                            attempts: attempt + 1, // Actual total attempt count
                            last: error,
                        };
                    }
                    // Non-retryable: return immediately, do not consume retry quota
                    if !is_retryable(&error) {
                        return RetryResult::NonRetryable(error);
                    }
                    tokio::time::sleep(self.calculate_delay(attempt)).await;
                }
            }
        }
        unreachable!() // Last attempt in loop must return
    }

    /// Exponential backoff: initial_delay * factor^attempt, optional ±20% jitter, clamped to max_delay.
    fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.config.initial_delay.as_millis() as f64
            * (self.config.factor as f64).powi(attempt as i32);
        let mut delay_ms = base_ms;
        if self.config.jitter {
            let jitter_ms = base_ms * 0.4 * (rand::random::<f64>() - 0.5);
            delay_ms += jitter_ms;
        }
        let max_ms = self.config.max_delay.as_millis() as f64;
        return Duration::from_millis(delay_ms.clamp(0.0, max_ms) as u64);
    }
}

fn validate(config: &RetryConfig) -> Result<(), String> {
    if config.max_retries > 10 {
        return Err(format!(
            "max_retries {} exceeds upper bound 10",
            config.max_retries
        ));
    }
    if config.initial_delay.is_zero() {
        return Err("initial_delay must be greater than zero".into());
    }
    if !config.factor.is_finite() || config.factor < 1.0 {
        return Err("factor must be finite and >= 1.0 for exponential growth".into());
    }
    if config.max_delay < config.initial_delay {
        return Err("max_delay must be >= initial_delay".into());
    }
    return Ok(());
}
