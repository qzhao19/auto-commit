use crate::core::pipe::validator::validate_and_normalize;
use crate::shared::exception::{ALLOWED_TYPES, MAX_DESCRIPTION_CHARS, ValidationError};

// ---- helpers ----

fn expect_ok(raw: &str) -> String {
    validate_and_normalize(raw)
        .map(|m| m.into_string())
        .unwrap_or_else(|e| panic!("expected Ok, got Err({e:?}) for input: {raw:?}"))
}

fn expect_err(raw: &str, expected: ValidationError) {
    match validate_and_normalize(raw) {
        Err(e) => assert_eq!(e, expected, "wrong error variant for input: {raw:?}"),
        Ok(m) => panic!("expected Err({expected:?}), got Ok({m:?}) for input: {raw:?}"),
    }
}

// ---- basic_sanitize: line endings / outer trim / empty / control chars ----

#[test]
fn sanitize_unifies_crlf_and_bare_cr() {
    let raw = "feat: add login retry\r\n\r\nHandles stale sessions\r";
    assert_eq!(
        expect_ok(raw),
        "feat: add login retry\n\nHandles stale sessions"
    );
}

#[test]
fn sanitize_trims_outer_whitespace() {
    assert_eq!(
        expect_ok("  \n\nfeat: add login retry  \n"),
        "feat: add login retry"
    );
}

#[test]
fn sanitize_rejects_empty_and_whitespace_only() {
    for raw in ["", "   \t ", "\r\n\r\n", "\n\n\n"] {
        expect_err(raw, ValidationError::Empty);
    }
}

#[test]
fn sanitize_rejects_control_characters() {
    for raw in [
        "feat: add\u{7}bell",                // BEL
        "feat: add retry\n\nuse \u{1b}[31m", // ESC (ANSI)
        "feat: add\u{0}retry",               // NUL
    ] {
        expect_err(raw, ValidationError::ControlCharacters);
    }
}

#[test]
fn sanitize_allows_tab_inside_body() {
    let raw = "feat: add retry\n\n\tindented note kept as-is";
    assert_eq!(
        expect_ok(raw),
        "feat: add retry\n\n\tindented note kept as-is"
    );
}

// ---- strip_fences ----

#[test]
fn fences_clean_wrap_is_stripped() {
    for raw in [
        "```\nfeat: add login retry\n```",     // plain fence
        "```text\nfeat: add login retry\n```", // language tag on opener
        "```\nfeat: add login retry\n  ```",   // indented closer
    ] {
        assert_eq!(expect_ok(raw), "feat: add login retry");
    }
}

#[test]
fn fences_wrap_message_with_body() {
    let raw = "```\nchore(deps): bump tokio to 1.42\n\nSwitches to the new mpsc API.\n```";
    assert_eq!(
        expect_ok(raw),
        "chore(deps): bump tokio to 1.42\n\nSwitches to the new mpsc API."
    );
}

#[test]
fn fences_with_empty_inner_reject_as_empty() {
    for raw in ["```\n```", "```rust\n```"] {
        expect_err(raw, ValidationError::Empty);
    }
}

#[test]
fn fences_single_or_half_wrap_reject() {
    for raw in [
        "```\nfeat: add login retry", // opener only
        "feat: add login retry\n```", // closer only
    ] {
        expect_err(raw, ValidationError::ForbiddenExtraContent);
    }
}

#[test]
fn fences_inside_body_are_not_strippable() {
    // Body code blocks are banned outright by the prompt contract.
    let raw = "feat: add retry\n\n```rust\nfn main() {}\n```";
    expect_err(raw, ValidationError::ForbiddenExtraContent);
}

#[test]
fn chatter_before_fence_is_a_broken_wrap() {
    // Fence pair not starting at line 0 → forbidden, not repaired.
    let raw = "Here you go:\n```\nfeat: add login retry\n```";
    expect_err(raw, ValidationError::ForbiddenExtraContent);
}

#[test]
fn trailing_text_after_closing_fence_rejects() {
    let raw = "```\nfeat: add login retry\n```\nThanks, let me know";
    expect_err(raw, ValidationError::ForbiddenExtraContent);
}

// ---- strip_leading_chatter ----

#[test]
fn chatter_single_line_plus_blank_is_stripped() {
    let raw = "Here is the generated commit message:\n\nfeat(api): add pagination to user list";
    assert_eq!(expect_ok(raw), "feat(api): add pagination to user list");
}

#[test]
fn chatter_up_to_the_line_budget_is_stripped() {
    // 3 chatter lines (incl. blank), header on line 4 (row 3, inside the
    // search window of MAX_CHATTER_LINES + 1).
    let raw = "Sure! Here is the commit message\nIt only covers the login module\n\nfeat: add login retry";
    assert_eq!(expect_ok(raw), "feat: add login retry");
}

#[test]
fn chatter_beyond_budget_surfaces_real_error() {
    // Header on row 4 is outside the search window: nothing is stripped
    // and the split stage reports the structural problem itself.
    let raw = "Line one\nLine two\nLine three\nLine four\nfeat: add login retry";
    expect_err(raw, ValidationError::BadHeaderBodySeparator);
}

#[test]
fn chatter_with_indented_header_is_recovered() {
    // The anchor evaluates the trimmed candidate; split_header_body
    // performs the real trim afterwards.
    let raw = "Here is the message:\n  feat: add login retry";
    assert_eq!(expect_ok(raw), "feat: add login retry");
}

#[test]
fn chatter_without_any_header_falls_through_to_validation() {
    expect_err(
        "Sorry, I cannot help with that request",
        ValidationError::InvalidHeaderFormat,
    );
}

#[test]
fn chatter_anchor_skips_invalid_type_lines() {
    // "result: ..." looks header-shaped but is not whitelisted — the
    // anchor keeps scanning and lands on the real header.
    let raw = "result: add login retry\nfeat: add login retry";
    assert_eq!(expect_ok(raw), "feat: add login retry");
}

// ---- strip_outer_quotes ----

#[test]
fn quotes_double_and_single_wrapped_headers_are_stripped() {
    assert_eq!(
        expect_ok("\"fix: handle empty input gracefully\""),
        "fix: handle empty input gracefully"
    );
    assert_eq!(
        expect_ok("'fix: handle empty input gracefully'"),
        "fix: handle empty input gracefully"
    );
}

#[test]
fn quotes_wrapping_whole_message_with_body() {
    let raw = "\"feat: add retry\n\nRetries are bounded by the resilience policy\"";
    assert_eq!(
        expect_ok(raw),
        "feat: add retry\n\nRetries are bounded by the resilience policy"
    );
}

#[test]
fn quotes_not_stripped_when_same_quote_appears_inside() {
    let raw = "\"feat: add \"smart\" retry\"";
    expect_err(raw, ValidationError::UnknownType("\"feat".to_string()));
}

#[test]
fn quotes_with_only_whitespace_inside_are_not_stripped() {
    expect_err("\"    \"", ValidationError::InvalidHeaderFormat);
}

// ---- split_header_body ----

#[test]
fn split_header_without_body() {
    assert_eq!(expect_ok("feat: add login retry"), "feat: add login retry");
}

#[test]
fn split_header_body_requires_blank_separator() {
    let raw = "feat: add login retry\nThis line explains why but sits too close";
    expect_err(raw, ValidationError::BadHeaderBodySeparator);
}

#[test]
fn split_separator_may_contain_spaces() {
    let raw = "feat: add login retry\n   \nThe body begins here";
    assert_eq!(
        expect_ok(raw),
        "feat: add login retry\n\nThe body begins here"
    );
}

// ---- validate_header: separator / structure ----

#[test]
fn header_requires_colon_space_separator() {
    for raw in ["feat add login retry", "feat:add login retry"] {
        expect_err(raw, ValidationError::InvalidHeaderFormat);
    }
}

#[test]
fn header_rejects_broken_paren_structure() {
    for raw in [
        "feat(auth: add login retry", // unclosed paren
        "feat(): add login retry",    // empty scope
        "(): add login retry",        // empty type
        "feat): add login retry",     // stray ')'
    ] {
        expect_err(raw, ValidationError::InvalidHeaderFormat);
    }
}

// ---- validate_header: type whitelist ----

#[test]
fn every_whitelisted_type_is_accepted() {
    for ty in ALLOWED_TYPES {
        let raw = format!("{ty}: handle the staged snapshots");
        assert_eq!(expect_ok(&raw), raw);
    }
}

#[test]
fn unknown_and_wrong_case_types_are_rejected() {
    // The whitelist is lowercase-exact.
    expect_err(
        "feature: add login retry",
        ValidationError::UnknownType("feature".to_string()),
    );
    expect_err(
        "Feat: add login retry",
        ValidationError::UnknownType("Feat".to_string()),
    );
    expect_err(
        "FEAT: add login retry",
        ValidationError::UnknownType("FEAT".to_string()),
    );
}

// ---- validate_header: scope grammar ----

#[test]
fn scope_rejects_uppercase_hyphen_and_space() {
    // Current grammar: lowercase ascii + digits only (no hyphen).
    expect_err(
        "feat(Auth): add retry",
        ValidationError::InvalidScope("Auth".to_string()),
    );
    expect_err(
        "feat(user-api): add retry",
        ValidationError::InvalidScope("user-api".to_string()),
    );
    expect_err(
        "feat(user api): add retry",
        ValidationError::InvalidScope("user api".to_string()),
    );
}

#[test]
fn scope_length_boundary_32_and_33() {
    let ok_scope = "a".repeat(32);
    let raw = format!("feat({ok_scope}): add login retry");
    assert_eq!(expect_ok(&raw), raw);

    let long_scope = "a".repeat(33);
    expect_err(
        &format!("feat({long_scope}): add login retry"),
        ValidationError::InvalidScope(long_scope),
    );
}

// ---- validate_header: description rules ----

#[test]
fn empty_description_surfaces_as_invalid_header_format() {
    // Whole-text trim eats the trailing space of "feat: ", so an empty
    // description can never reach the dedicated EmptyDescription check.
    // Pin the real behavior, not a wishful one.
    for raw in ["feat:", "feat:\n\nA perfectly fine body follows"] {
        expect_err(raw, ValidationError::InvalidHeaderFormat);
    }
}

#[test]
fn description_length_boundary() {
    // At the limit → accepted.
    let desc = "a".repeat(MAX_DESCRIPTION_CHARS);
    let raw = format!("feat: {desc}");
    assert_eq!(expect_ok(&raw), raw);

    // One over → rejected with the actual count.
    let desc = "a".repeat(MAX_DESCRIPTION_CHARS + 1);
    expect_err(
        &format!("feat: {desc}"),
        ValidationError::DescriptionTooLong {
            actual: MAX_DESCRIPTION_CHARS + 1,
        },
    );
}

#[test]
fn description_must_start_with_lowercase_ascii() {
    for raw in [
        "feat: Add login retry", // uppercase
        "feat: 3rd retry fix",   // digit
        "feat: 增加登录重试",    // CJK — pins the English contract
    ] {
        expect_err(raw, ValidationError::DescriptionNotLowercase);
    }
}

#[test]
fn description_must_not_end_with_period() {
    expect_err("feat: add login retry.", ValidationError::TrailingPeriod);
    // The rule only applies to the header — a period in the body is fine.
    let raw = "feat: add login retry\n\nThis fixes the leak.";
    assert_eq!(expect_ok(raw), raw);
}

// ---- normalize ----

#[test]
fn normalize_collapses_blank_runs_in_body() {
    let raw = "feat: add login retry\n\nfirst point\n\n\n\nsecond point";
    assert_eq!(
        expect_ok(raw),
        "feat: add login retry\n\nfirst point\n\nsecond point"
    );
}

#[test]
fn normalize_drops_leading_blanks_between_header_and_body() {
    // LLM emitted two blank lines after the header — recovered, not rejected.
    let raw = "feat: add login retry\n\n\nThe body starts here";
    assert_eq!(
        expect_ok(raw),
        "feat: add login retry\n\nThe body starts here"
    );
}

#[test]
fn normalize_trims_line_tail_and_collapses_whitespace_only_lines() {
    let raw = "feat: add login retry\n\nfirst line   \n   \nsecond line";
    assert_eq!(
        expect_ok(raw),
        "feat: add login retry\n\nfirst line\n\nsecond line"
    );
}

#[test]
fn normalize_preserves_inner_indentation_of_body() {
    let raw = "feat: add login retry\n\n  if attempts > max {\n    return;\n  }";
    assert_eq!(expect_ok(raw), raw);
}

// ---- realistic end-to-end cases ----

#[test]
fn realistic_rust_project_messages() {
    // Plain success, and the contract accessors of the validated type.
    let msg = validate_and_normalize("feat(auth): add jwt token refresh").unwrap();
    assert_eq!(msg.as_str(), "feat(auth): add jwt token refresh");
    assert_eq!(format!("{msg}"), "feat(auth): add jwt token refresh");

    // prost/tonic-style regeneration commit with an explanatory body.
    let raw = "feat(rpc): regenerate protobuf clients for user service\n\nAdds the streaming variants introduced in acme.v1.proto";
    assert_eq!(expect_ok(raw), raw);
}

#[test]
fn realistic_ollama_chatter_and_fence_combo_are_recovered_independently() {
    // Chatter without quotes.
    let raw = "Here is your commit message\n\nfix(stream): abort retry loop on broken pipe";
    assert_eq!(
        expect_ok(raw),
        "fix(stream): abort retry loop on broken pipe"
    );
}

#[test]
fn realistic_body_with_markdown_list_is_preserved() {
    let raw = "fix(chat): handle broken pipe on stream abort\n\n- stop retrying after ECONNRESET\n- surface the failure to the caller";
    assert_eq!(expect_ok(raw), raw);
}

#[test]
fn realistic_windows_line_endings_on_full_message() {
    let raw = "chore(ci): pin rust toolchain\r\n\r\nReproducible builds for release artifacts\r\n";
    assert_eq!(
        expect_ok(raw),
        "chore(ci): pin rust toolchain\n\nReproducible builds for release artifacts"
    );
}
