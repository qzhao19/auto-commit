use std::time::Duration;

use crate::infra::timeout::Timeout;
use crate::shared::config::TimeoutConfig;

//  Timeout::new validation

#[test]
fn new_accepts_nonzero_duration() {
    let cfg = TimeoutConfig {
        timeout: Duration::from_millis(1),
    };
    assert!(Timeout::new(cfg).is_ok());
}

#[test]
fn new_rejects_zero_duration() {
    let cfg = TimeoutConfig {
        timeout: Duration::ZERO,
    };
    let Err(err) = Timeout::new(cfg) else {
        panic!("expected Err, got Ok");
    };
    assert!(err.contains("timeout"), "got: {err}");
}

//  Timeout::execute behavior

#[tokio::test]
async fn execute_returns_value_when_future_completes() {
    let timeout = Timeout::new(TimeoutConfig {
        timeout: Duration::from_secs(1),
    })
    .unwrap();

    let result = timeout.execute(async { 42 }).await;
    assert_eq!(result, Ok(42));
}

#[tokio::test]
async fn execute_returns_err_when_future_exceeds_timeout() {
    let configured = Duration::from_millis(50);
    let timeout = Timeout::new(TimeoutConfig {
        timeout: configured,
    })
    .unwrap();

    // Sleep strictly longer than the configured timeout.
    let result = timeout
        .execute(tokio::time::sleep(Duration::from_secs(1)))
        .await;

    let Err(msg) = result else {
        panic!("expected Err, got Ok");
    };
    // Pin the current error message format: `request timed out after {:?}`.
    assert!(
        msg.contains(&format!("{configured:?}")),
        "error should contain configured duration; got: {msg}"
    );
    assert!(msg.contains("timed out"), "got: {msg}");
}
