//! Integration tests for LlmClient: AppConfig → adapter construction →
//! invoke raw → LlmClient retry/timeout wrapper → result or LlmError.
//!
//! HTTP is mocked with a tokio-only scriptable server — no new dependencies,
//! no real network, no real API keys. Every response carries
//! `connection: close`, so each attempt (initial + retries) arrives as a
//! separate recorded request.
//!
//! Known assumption (flagged): retry count = initial attempt + max_retries
//! retries, i.e. N=2 → 3 requests. Exact-count assertions only assert numbers
//! that hold under both possible semantics; exhaustion tests assert floors.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::core::llm::build_provider;
use crate::core::llm::client::LlmClient;
use crate::shared::config::{ApiKey, AppConfig, LlmMessage, ProviderName};
use crate::shared::exception::{LlmError, ProviderErrorType};

// Mock server

#[derive(Debug, Clone)]
struct RecordedRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: String,
}

#[derive(Debug, Clone)]
enum MockBehavior {
    /// Answer with this JSON payload and status.
    Json { status: u16, body: String },
    /// Read the request, then never answer within the test's lifetime — the
    /// client's timeout is what must cut the connection.
    Hang,
    /// Read the request, then drop the connection without any response —
    /// a transport-level failure (reqwest "connection closed").
    Reset,
}

#[derive(Clone)]
struct MockServer {
    addr: SocketAddr,
    received: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockServer {
    async fn start(script: Vec<MockBehavior>, fallback: MockBehavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().expect("mock addr");
        let received = Arc::new(Mutex::new(Vec::new()));
        let script = Arc::new(Mutex::new(VecDeque::from(script)));

        let server = Self {
            addr,
            received: received.clone(),
        };

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let received = received.clone();
                let script = script.clone();
                let fallback = fallback.clone();
                tokio::spawn(async move {
                    if read_request(&mut socket, &received).await.is_none() {
                        return;
                    }
                    let behavior = script
                        .lock()
                        .expect("script poisoned")
                        .pop_front()
                        .unwrap_or(fallback);
                    serve(socket, behavior).await;
                });
            }
        });

        server
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn request_count(&self) -> usize {
        self.received.lock().expect("received poisoned").len()
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.received.lock().expect("received poisoned").clone()
    }
}

/// Read one HTTP/1.1 request (header block + Content-Length body), record it.
/// Returns `None` on EOF before headers complete.
async fn read_request(
    socket: &mut tokio::net::TcpStream,
    received: &Arc<Mutex<Vec<RecordedRequest>>>,
) -> Option<()> {
    let mut buf = Vec::with_capacity(8 * 1024);
    let mut chunk = [0u8; 8 * 1024];

    let header_end = loop {
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > 64 * 1024 {
            return None; // pathological header: bail
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.lines();
    let mut request_line = lines.next().unwrap_or_default().split_whitespace();
    let method = request_line.next().unwrap_or_default().to_owned();
    let path = request_line.next().unwrap_or_default().to_owned();

    let mut content_length = 0usize;
    let mut authorization = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if name.trim().eq_ignore_ascii_case("authorization") {
            authorization = Some(value.trim().to_owned());
        }
    }

    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body_len = (body_start + content_length).min(buf.len());
    let body = String::from_utf8_lossy(&buf[body_start..body_len]).into_owned();

    received
        .lock()
        .expect("received poisoned")
        .push(RecordedRequest {
            method,
            path,
            authorization,
            body,
        });
    Some(())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

async fn serve(mut socket: tokio::net::TcpStream, behavior: MockBehavior) {
    match behavior {
        MockBehavior::Hang => {
            // Far beyond any test timeout; runtime teardown kills the task.
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        MockBehavior::Reset => {
            // Dropping the socket sends FIN without a response.
        }
        MockBehavior::Json { status, body } => {
            let reason = match status {
                200 => "OK",
                400 => "Bad Request",
                401 => "Unauthorized",
                429 => "Too Many Requests",
                500 => "Internal Server Error",
                503 => "Service Unavailable",
                _ => "Unknown",
            };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    }
}

// Response payload builders + config factories

fn openai_chat_ok(content: &str) -> String {
    format!(
        r#"{{"id":"chatcmpl-test","object":"chat.completion","created":1700000000,"model":"mock","choices":[{{"index":0,"message":{{"role":"assistant","content":"{content}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}}"#
    )
}

fn openai_api_error(message: &str) -> String {
    format!(
        r#"{{"error":{{"message":"{message}","type":"invalid_request_error","param":null,"code":"test"}}}}"#
    )
}

/// ollama-rs ChatMessageResponse — timing fields included because some
/// versions keep them at the top level; if yours nests them under
/// `final_data`, move them into a `"final_data": {{ ... }}` object here only.
fn ollama_chat_ok(content: &str) -> String {
    format!(
        r#"{{"model":"mock","created_at":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":"{content}"}},"done":true,"done_reason":"stop","total_duration":1,"load_duration":1,"prompt_eval_count":1,"prompt_eval_duration":1,"eval_count":1,"eval_duration":1}}"#
    )
}

/// Built without ConfigLoader::validate on purpose — tests need edge inputs
/// (no key, negative max_tokens) that the validated pipeline would reject.
fn openai_config(url: &str) -> AppConfig {
    let mut config = AppConfig::default();
    config.llm.provider.provider = ProviderName::Openai;
    config.llm.provider.model = "gpt-test".to_owned();
    config.llm.provider.api_key = Some(ApiKey::new("sk-test-key"));
    config.llm.provider.base_url = Some(url.to_owned());
    cheap_retries(&mut config, 0);
    config.resilience.timeout.timeout = Duration::from_secs(5);
    config
}

fn ollama_config(url: &str) -> AppConfig {
    let mut config = openai_config(url);
    config.llm.provider.provider = ProviderName::Ollama;
    config.llm.provider.model = "qwen-test".to_owned();
    config.llm.provider.api_key = None;
    config
}

/// 1ms backoff / factor 1.0 / no jitter — retries become observable at the
/// wire level without wall-clock cost.
fn cheap_retries(config: &mut AppConfig, max_retries: u32) {
    let retry = &mut config.resilience.retry;
    retry.max_retries = max_retries.into(); // field width inferred on site
    retry.initial_delay = Duration::from_millis(1);
    retry.max_delay = Duration::from_millis(2);
    retry.factor = 1.0;
    retry.jitter = false;
}

fn with_retry(mut config: AppConfig, max_retries: u32) -> AppConfig {
    cheap_retries(&mut config, max_retries);
    config
}

fn msg() -> LlmMessage {
    LlmMessage::user_message_only("Write a commit message for a small edit".to_owned())
}

// A. Configuration injection → actual HTTP request

// A1 + A5 + B6: model/base_url/api_key from AppConfig reach the wire — the
// openai adapter POSTs to {base_url}/chat/completions with a Bearer header,
// the body carries the model, and invoke returns the mock's message.content.
#[tokio::test(flavor = "current_thread")]
async fn a_openai_config_reaches_http_and_content_round_trips() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("feat: add login endpoint"),
        },
    )
    .await;

    let client = LlmClient::new(&openai_config(&server.url())).expect("client builds");
    let text = client.invoke(msg()).await.expect("invoke succeeds");

    assert_eq!(text, "feat: add login endpoint");
    assert_eq!(server.request_count(), 1);
    let request = &server.requests()[0];
    assert_eq!(request.method, "POST");
    assert!(
        request.path.ends_with("/chat/completions"),
        "path: {}",
        request.path
    );
    assert_eq!(request.authorization.as_deref(), Some("Bearer sk-test-key"));
    assert!(
        request.body.contains("\"model\":\"gpt-test\""),
        "body: {}",
        request.body
    );
}

// A2: every generation parameter is serialized into the request body.
#[tokio::test(flavor = "current_thread")]
async fn a_openai_generation_params_appear_in_request_body() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("x"),
        },
    )
    .await;

    let mut config = openai_config(&server.url());
    config.llm.generation.temperature = 0.1;
    config.llm.generation.max_tokens = 123;
    config.llm.generation.top_p = 0.25;
    config.llm.generation.frequency_penalty = 0.5;
    config.llm.generation.presence_penalty = 0.75;

    let client = LlmClient::new(&config).expect("client builds");
    client.invoke(msg()).await.expect("invoke succeeds");

    let body = &server.requests()[0].body;
    for expected in [
        "\"temperature\":0.1",
        "\"top_p\":0.25",
        "\"max_tokens\":123",
        "\"frequency_penalty\":0.5",
        "\"presence_penalty\":0.75",
    ] {
        assert!(body.contains(expected), "missing {expected} in {body}");
    }
}

// A3 OpenAiProvider::new does NOT verify the key — key enforcement
// lives in ConfigLoader::validate. A key-less client sends no Authorization
// header, and a 401 then surfaces as a fatal, non-retried error.
#[tokio::test(flavor = "current_thread")]
async fn a_openai_without_api_key_sends_empty_bearer_credentials() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 401,
            body: openai_api_error("missing bearer"),
        },
    )
    .await;

    let mut config = with_retry(openai_config(&server.url()), 1);
    config.llm.provider.api_key = None;

    let client = LlmClient::new(&config).expect("construction succeeds without a key");
    let err = client
        .invoke(msg())
        .await
        .expect_err("key-less request against a 401 server must fail");

    assert!(
        matches!(err, LlmError::Provider(ProviderErrorType::Fatal, _)),
        "err: {err}"
    );
    // Fatal errors consume no retry budget — exactly one request.
    assert_eq!(server.request_count(), 1);
    // Empty bearer credentials — no real key material ever reaches the wire.
    assert_eq!(
        server.requests()[0].authorization.as_deref(),
        Some("Bearer")
    );
}

// A4 + B6(ollama): the ollama adapter needs no key, POSTs {base}/api/chat,
// and message.content round-trips through invoke.
#[tokio::test(flavor = "current_thread")]
async fn a_ollama_config_reaches_http_and_content_round_trips() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: ollama_chat_ok("fix: null pointer in parser"),
        },
    )
    .await;

    let config = ollama_config(&server.url());
    assert!(config.llm.provider.api_key.is_none());

    let client = LlmClient::new(&config).expect("client builds");
    let text = client.invoke(msg()).await.expect("invoke succeeds");

    assert_eq!(text, "fix: null pointer in parser");
    let request = &server.requests()[0];
    assert!(
        request.path.ends_with("/api/chat"),
        "path: {}",
        request.path
    );
    assert_eq!(request.authorization, None, "ollama never sends an api key");
    assert!(
        request.body.contains("\"model\":\"qwen-test\""),
        "body: {}",
        request.body
    );
}

// B. Unified invoke contract

// B: LlmMessage roles map to chat message roles in both adapters' bodies.
#[tokio::test(flavor = "current_thread")]
async fn b_system_and_user_messages_map_to_chat_roles() {
    let message = LlmMessage::new(Some("SYS-PROMPT".to_owned()), "USER-PROMPT".to_owned());

    let openai_server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("ok"),
        },
    )
    .await;
    LlmClient::new(&openai_config(&openai_server.url()))
        .expect("builds")
        .invoke(message.clone())
        .await
        .expect("succeeds");
    let body = &openai_server.requests()[0].body;
    assert!(body.contains("\"role\":\"system\""), "body: {body}");
    assert!(body.contains("SYS-PROMPT"), "body: {body}");
    assert!(body.contains("USER-PROMPT"), "body: {body}");

    let ollama_server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: ollama_chat_ok("ok"),
        },
    )
    .await;
    LlmClient::new(&ollama_config(&ollama_server.url()))
        .expect("builds")
        .invoke(message)
        .await
        .expect("succeeds");
    let body = &ollama_server.requests()[0].body;
    assert!(body.contains("\"role\":\"system\""), "body: {body}");
    assert!(body.contains("SYS-PROMPT"), "body: {body}");
    assert!(body.contains("USER-PROMPT"), "body: {body}");
}

// B7 there is no prompt validation — an empty user message is
// sent verbatim, and an extremely long one is not truncated.
#[tokio::test(flavor = "current_thread")]
async fn b_prompt_is_sent_verbatim_at_both_size_extremes() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("ok"),
        },
    )
    .await;
    let client = LlmClient::new(&openai_config(&server.url())).expect("client builds");

    client
        .invoke(LlmMessage::user_message_only(String::new()))
        .await
        .expect("empty prompt must not be rejected locally");
    assert!(
        server.requests()[0].body.contains("\"content\":\"\""),
        "empty content must be forwarded"
    );

    let big = "x".repeat(100_000);
    client
        .invoke(LlmMessage::user_message_only(big.clone()))
        .await
        .expect("huge prompt must not be rejected locally");
    let body = &server.requests()[1].body;
    assert!(
        body.len() >= 100_000 && body.contains(&big),
        "payload must be untruncated"
    );
}

// B8: a 200 with an unparseable payload maps to a fatal error — the response
// deserialization failure must not consume retry budget.
#[tokio::test(flavor = "current_thread")]
async fn b_unexpected_response_shape_is_fatal_not_retried() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: r#"{"not":"a chat completion"}"#.to_owned(),
        },
    )
    .await;
    let client =
        LlmClient::new(&with_retry(openai_config(&server.url()), 3)).expect("client builds");

    let err = client.invoke(msg()).await.expect_err("must fail");
    assert!(
        matches!(err, LlmError::Provider(ProviderErrorType::Fatal, _)),
        "err: {err}"
    );
    assert_eq!(server.request_count(), 1);
}

// B/edge: max_tokens below u32 range never reaches the wire — the adapter
// fails inside invoke with a Build error and zero requests are made.
#[tokio::test(flavor = "current_thread")]
async fn b_negative_max_tokens_fails_before_sending_any_request() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("x"),
        },
    )
    .await;

    let mut config = with_retry(openai_config(&server.url()), 1);
    config.llm.generation.max_tokens = -1; // bypass ConfigLoader::validate on purpose

    let client = LlmClient::new(&config).expect("client builds");
    let err = client.invoke(msg()).await.expect_err("must fail");

    assert!(matches!(err, LlmError::Build(_)), "err: {err}");
    assert!(
        err.to_string().contains("max_tokens"),
        "should name the field: {err}"
    );
    assert_eq!(server.request_count(), 0);
}

// C. Resilience — provided by LlmClient, not the adapters

// C10: a slow upstream is cut by the per-attempt timeout. With retries off,
// the raw Timeout variant surfaces — bounded, no infinite waiting.
#[tokio::test(flavor = "current_thread")]
async fn c_timeout_cuts_slow_response_and_surfaces_bounded() {
    let server = MockServer::start(vec![], MockBehavior::Hang).await;

    let mut config = with_retry(openai_config(&server.url()), 0);
    config.resilience.timeout.timeout = Duration::from_millis(300);

    let client = LlmClient::new(&config).expect("client builds");
    let started = Instant::now();
    let err = client.invoke(msg()).await.expect_err("must time out");
    let elapsed = started.elapsed();

    match err {
        LlmError::RetryExhausted { attempts: 1, last } => {
            assert!(matches!(*last, LlmError::Timeout(_)), "inner: {last}");
        }
        other => panic!("expected RetryExhausted{{attempts:1}} wrapping Timeout, got: {other}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "timeout must not wait forever: {elapsed:?}"
    );
}

// C: Timeout is retryable (LlmError::is_retryable includes it) — every
// attempt hits the mock, and the terminal result wraps into RetryExhausted.
#[tokio::test(flavor = "current_thread")]
async fn c_timed_out_attempts_are_retried_then_exhausted() {
    let server = MockServer::start(vec![], MockBehavior::Hang).await;

    let mut config = with_retry(openai_config(&server.url()), 1);
    config.resilience.timeout.timeout = Duration::from_millis(250);

    let client = LlmClient::new(&config).expect("client builds");
    let err = client.invoke(msg()).await.expect_err("must exhaust");

    assert!(matches!(err, LlmError::RetryExhausted { .. }), "err: {err}");
    assert!(err.to_string().contains("attempt(s)"));
    assert!(
        server.request_count() >= 2,
        "the timed-out attempt must have been retried at least once; got {}",
        server.request_count()
    );
}

// C11: 429 is transient — the retry layer re-fires until success, and the
// wire shows exactly (failures + 1) requests.
#[tokio::test(flavor = "current_thread")]
async fn c_transient_429_is_retried_until_success() {
    let server = MockServer::start(
        vec![
            MockBehavior::Json {
                status: 429,
                body: openai_api_error("rate limited"),
            },
            MockBehavior::Json {
                status: 200,
                body: openai_chat_ok("feat: retry survived"),
            },
        ],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("unreachable"),
        },
    )
    .await;

    let client =
        LlmClient::new(&with_retry(openai_config(&server.url()), 2)).expect("client builds");
    let text = client.invoke(msg()).await.expect("second attempt succeeds");

    assert_eq!(text, "feat: retry survived");
    assert_eq!(server.request_count(), 2);
}

// C12: 4xx is deterministic — the Fatal mapping means one request, no retry.
#[tokio::test(flavor = "current_thread")]
async fn c_fatal_4xx_statuses_are_not_retried() {
    for (status, label) in [(400u16, "bad request"), (401u16, "unauthorized")] {
        let server = MockServer::start(
            vec![],
            MockBehavior::Json {
                status,
                body: openai_api_error(label),
            },
        )
        .await;
        let client =
            LlmClient::new(&with_retry(openai_config(&server.url()), 3)).expect("client builds");

        let err = client.invoke(msg()).await.expect_err("must fail");
        assert!(
            matches!(err, LlmError::Provider(ProviderErrorType::Fatal, _)),
            "{label}: {err}"
        );
        assert_eq!(server.request_count(), 1, "{label} must not be retried");
    }
}

// C13: permanent transients exhaust the budget — the result wraps into
// RetryExhausted with the error chain preserved and retried wire traffic.
#[tokio::test(flavor = "current_thread")]
async fn c_persistent_failure_exhausts_retry_budget() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 503,
            body: openai_api_error("overloaded"),
        },
    )
    .await;

    let client =
        LlmClient::new(&with_retry(openai_config(&server.url()), 2)).expect("client builds");
    let err = client
        .invoke(msg())
        .await
        .expect_err("must fail eventually");

    assert!(matches!(err, LlmError::RetryExhausted { .. }), "err: {err}");
    assert!(err.to_string().contains("attempt(s)"));
    // Exact count = 3 only if max_retries means "extra retries after the
    // initial attempt"; floor-safe either way.
    assert!(server.request_count() >= 2);
}

// C14: retry counts come from AppConfig.resilience — nothing is hard-coded;
// the same failing server records different traffic for different budgets.
#[tokio::test(flavor = "current_thread")]
async fn c_retry_attempts_follow_resilience_config() {
    let zero_budget_server = MockServer::start(vec![], MockBehavior::Reset).await;
    let client_a =
        LlmClient::new(&with_retry(ollama_config(&zero_budget_server.url()), 0)).expect("builds");
    let err = client_a.invoke(msg()).await.expect_err("must fail");

    assert!(
        matches!(err, LlmError::RetryExhausted { attempts: 1, .. }),
        "err: {err}"
    );
    assert_eq!(
        zero_budget_server.request_count(),
        1,
        "budget 0 → single attempt"
    );

    let two_budget_server = MockServer::start(vec![], MockBehavior::Reset).await;
    let client_b =
        LlmClient::new(&with_retry(ollama_config(&two_budget_server.url()), 2)).expect("builds");
    let err = client_b.invoke(msg()).await.expect_err("must fail");

    assert!(
        matches!(err, LlmError::RetryExhausted { attempts: 3, .. }),
        "err: {err}"
    );
    assert_eq!(
        two_budget_server.request_count(),
        3,
        "1 initial + 2 retries"
    );
}

// C15: resilience lives in LlmClient — a bare adapter obtained via
// build_provider makes exactly one request in the face of a transient error,
// even under the same retry budget that makes the client retry.
#[tokio::test(flavor = "current_thread")]
async fn c_bare_providers_do_not_retry_by_themselves() {
    // Phase 1: bare adapter, no retry anywhere in the call path.
    let bare_server = MockServer::start(vec![], MockBehavior::Reset).await;
    let provider = build_provider(&ollama_config(&bare_server.url()).llm).expect("builds");

    let message = msg();
    let err = provider
        .invoke_raw(&message)
        .await
        .expect_err("reset must fail");
    assert!(
        matches!(err, LlmError::Provider(ProviderErrorType::Transient, _)),
        "err: {err}"
    );
    assert_eq!(
        bare_server.request_count(),
        1,
        "the adapter itself never retries"
    );

    // Phase 2: identical transport failure through the client — the wire delta
    // is exactly the configured retry budget (1 initial + 2 retries).
    let client_server = MockServer::start(vec![], MockBehavior::Reset).await;
    let client =
        LlmClient::new(&with_retry(ollama_config(&client_server.url()), 2)).expect("builds");
    let err = client.invoke(msg()).await.expect_err("must fail");

    assert!(matches!(err, LlmError::RetryExhausted { .. }), "err: {err}");
    assert_eq!(
        client_server.request_count(),
        3,
        "1 initial + 2 budgeted retries"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn c_openai_adapter_has_built_in_transport_retries() {
    let server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 503,
            body: openai_api_error("overloaded"),
        },
    )
    .await;

    let provider = build_provider(&openai_config(&server.url()).llm).expect("builds");
    let message = msg();
    let err = provider.invoke_raw(&message).await.expect_err("must fail");

    assert!(
        matches!(err, LlmError::Provider(ProviderErrorType::Transient, _)),
        "err: {err}"
    );
    assert!(
        server.request_count() >= 2,
        "async-openai retried the 503 internally; got {}",
        server.request_count()
    );
}

// C: a dropped connection is a transport failure — the ollama adapter must
// surface it as Transient (retryable), classified by ollama_err.
#[tokio::test(flavor = "current_thread")]
async fn c_ollama_connection_reset_surfaces_as_transient() {
    let server = MockServer::start(vec![], MockBehavior::Reset).await;

    let provider = build_provider(&ollama_config(&server.url()).llm).expect("builds");
    let message = msg();
    let err = provider
        .invoke_raw(&message)
        .await
        .expect_err("reset must fail");

    assert!(
        matches!(err, LlmError::Provider(ProviderErrorType::Transient, _)),
        "err: {err}"
    );
}

// D. Provider selection + wire-shape differences

// D16: the factory maps ProviderName to the right adapter.
#[test]
fn d_build_provider_selects_adapter_by_provider_name() {
    let openai = build_provider(&openai_config("http://127.0.0.1:1").llm).expect("openai builds");
    assert_eq!(openai.name(), "openai");

    let ollama = build_provider(&ollama_config("http://127.0.0.1:1").llm).expect("ollama builds");
    assert_eq!(ollama.name(), "ollama");
}

// D18 + E19: the two adapters speak different protocols — paths, auth
// presence and message keying differ; both adapters are non-streaming on
// the wire (no `"stream":true` ever appears).
#[tokio::test(flavor = "current_thread")]
async fn d_openai_and_ollama_use_their_own_wire_shapes() {
    let openai_server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: openai_chat_ok("ok"),
        },
    )
    .await;
    let ollama_server = MockServer::start(
        vec![],
        MockBehavior::Json {
            status: 200,
            body: ollama_chat_ok("ok"),
        },
    )
    .await;

    LlmClient::new(&openai_config(&openai_server.url()))
        .expect("builds")
        .invoke(msg())
        .await
        .expect("succeeds");
    LlmClient::new(&ollama_config(&ollama_server.url()))
        .expect("builds")
        .invoke(msg())
        .await
        .expect("succeeds");

    let openai_req = &openai_server.requests()[0];
    assert!(openai_req.path.ends_with("/chat/completions"));
    assert!(openai_req.authorization.is_some());
    assert!(
        !openai_req.body.contains("\"stream\":true"),
        "openai must be non-streaming: {}",
        openai_req.body
    );

    let ollama_req = &ollama_server.requests()[0];
    assert!(ollama_req.path.ends_with("/api/chat"));
    assert!(ollama_req.authorization.is_none());
    assert!(
        !ollama_req.body.contains("\"stream\":true"),
        "ollama must be non-streaming: {}",
        ollama_req.body
    );
}

// Live smoke test (opt-in; excluded from CI by default)

// Requires a real Ollama instance. Run explicitly with:
//   cargo test integration::test_llm_client::live_ollama -- --ignored
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a live Ollama instance"]
async fn live_ollama_generates_text() {
    let mut config = AppConfig::default();
    config.llm.provider.provider = ProviderName::Ollama;
    config.llm.provider.model =
        std::env::var("AUTOCOMMIT_LLM_MODEL").unwrap_or_else(|_| "llama3.1:8b".to_owned());
    config.resilience.retry.max_retries = 0u32.into();
    config.resilience.timeout.timeout = Duration::from_secs(30);

    let text = LlmClient::new(&config)
        .expect("client builds")
        .invoke(msg())
        .await
        .expect("live invoke succeeds");
    assert!(!text.trim().is_empty());
}
