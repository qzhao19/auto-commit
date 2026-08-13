use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use crate::infra::resilience::retry::{Retry, RetryResult};
use crate::shared::config::resilience::RetryConfig;

/// Test error type — distinguishes retryable vs non-retryable errors.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TestErr {
    Retryable,
    NonRetryable,
}

/// A `RetryConfig` with millisecond-scale delays and no jitter so tests
/// run fast and don't depend on randomness.
fn fast_config(max_retries: u32) -> RetryConfig {
    RetryConfig {
        max_retries,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(8),
        factor: 2.0,
        jitter: false,
    }
}

//  Retry::new validation 

#[test]
fn new_rejects_max_retries_above_10() {
    let Err(err) = Retry::new(fast_config(11)) else {
        panic!("expected Err, got Ok");
    };
    assert!(err.contains("max_retries"), "got: {err}");
}

#[test]
fn new_rejects_zero_initial_delay() {
    let mut cfg = fast_config(2);
    cfg.initial_delay = Duration::ZERO;
    let Err(err) = Retry::new(cfg) else {
        panic!("expected Err, got Ok");
    };
    assert!(err.contains("initial_delay"), "got: {err}");
}

#[test]
fn new_rejects_factor_below_one() {
    let mut cfg = fast_config(2);
    cfg.factor = 0.5;
    let Err(err) = Retry::new(cfg) else {
        panic!("expected Err, got Ok");
    };
    assert!(err.contains("factor"), "got: {err}");
}

#[test]
fn new_rejects_non_finite_factor() {
    for factor in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut cfg = fast_config(2);
        cfg.factor = factor;
        let Err(err) = Retry::new(cfg) else {
            panic!("expected Err for factor {factor}, got Ok");
        };
        assert!(err.contains("factor"), "factor {factor} -> {err}");
    }
}

#[test]
fn new_rejects_max_delay_below_initial_delay() {
    let mut cfg = fast_config(2);
    cfg.initial_delay = Duration::from_millis(10);
    cfg.max_delay = Duration::from_millis(1);
    let Err(err) = Retry::new(cfg) else {
        panic!("expected Err, got Ok");
    };
    assert!(err.contains("max_delay"), "got: {err}");
}

//  Retry::execute state machine 

/// Wrap an `Arc<AtomicU32>` into a `FnMut` closure that counts calls and
/// runs `body(n)` (where `n` is the 1-indexed attempt number) inside an
/// owned async block. Mirrors the pattern used in `LlmClient::invoke`.
fn counting_func(
    counter: &Arc<AtomicU32>,
    body: fn(u32) -> Result<&'static str, TestErr>,
) -> impl FnMut() -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<&'static str, TestErr>> + Send>,
> {
    let counter = counter.clone();
    move || {
        let counter = counter.clone();
        Box::pin(async move {
            let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
            body(n)
        })
    }
}

#[tokio::test]
async fn execute_succeeds_on_first_attempt() {
    let retry = Retry::new(fast_config(2)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(|_| true, counting_func(&calls, |_| Ok("ok")))
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    match result {
        RetryResult::Ok(v) => assert_eq!(v, "ok"),
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_retries_until_success() {
    // Succeeds on the 3rd attempt (after 2 retryable failures).
    let retry = Retry::new(fast_config(2)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(
            |_| true,
            counting_func(&calls, |n| {
                if n < 3 {
                    Err(TestErr::Retryable)
                } else {
                    Ok("ok")
                }
            }),
        )
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 3);
    match result {
        RetryResult::Ok(v) => assert_eq!(v, "ok"),
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_returns_non_retryable_immediately() {
    // First attempt yields a non-retryable error: must not retry.
    let retry = Retry::new(fast_config(3)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(
            |e| matches!(e, TestErr::Retryable),
            counting_func(&calls, |_| Err(TestErr::NonRetryable)),
        )
        .await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "non-retryable must not be retried"
    );
    match result {
        RetryResult::NonRetryable(e) => assert_eq!(e, TestErr::NonRetryable),
        other => panic!("expected NonRetryable, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_exhausts_after_max_retries_plus_one_attempts() {
    let max_retries = 2;
    let retry = Retry::new(fast_config(max_retries)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(|_| true, counting_func(&calls, |_| Err(TestErr::Retryable)))
        .await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        max_retries + 1,
        "should attempt exactly max_retries + 1 times"
    );
    match result {
        RetryResult::Exhausted { attempts, last } => {
            assert_eq!(attempts, max_retries + 1);
            assert_eq!(last, TestErr::Retryable);
        }
        other => panic!("expected Exhausted, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_with_max_retries_zero_makes_single_attempt() {
    let retry = Retry::new(fast_config(0)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(|_| true, counting_func(&calls, |_| Err(TestErr::Retryable)))
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    match result {
        RetryResult::Exhausted { attempts, .. } => assert_eq!(attempts, 1),
        other => panic!("expected Exhausted, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_non_retryable_on_last_attempt_reports_exhausted() {
    let max_retries = 1;
    let retry = Retry::new(fast_config(max_retries)).unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    let result = retry
        .execute(
            |e| matches!(e, TestErr::Retryable),
            counting_func(&calls, |n| {
                if n == 1 {
                    Err(TestErr::Retryable)
                } else {
                    Err(TestErr::NonRetryable)
                }
            }),
        )
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    match result {
        RetryResult::Exhausted { attempts, last } => {
            assert_eq!(attempts, 2);
            assert_eq!(
                last,
                TestErr::NonRetryable,
                "non-retryable on last attempt must surface in `last`"
            );
        }
        other => panic!("expected Exhausted, got {other:?}"),
    }
}
