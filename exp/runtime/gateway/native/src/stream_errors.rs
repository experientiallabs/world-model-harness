//! Classification of provider-declared failures by what the provider said.
//!
//! Two failure sources reach the caller with the wrong shape without this:
//!
//! * A provider that opens the stream and then declares its own error inside a
//!   frame (OpenAI `error` / `response.failed`, Anthropic `error`, Gemini's
//!   error envelope, an OpenAI-compatible `error` object) used to become one
//!   `provider_internal` 502 no matter what it said. Half of what providers say
//!   there is the CALLER's fault ("Your input exceeds the context window",
//!   "does not support max tokens > N", content filtering), a quarter is a
//!   throttle, and only the rest is the provider failing. Filing a caller's
//!   over-long prompt as a 502 hides the fix from them, pages the operator, and
//!   burns a failover attempt that fails identically.
//! * A customer-managed (BYOK) rung whose credential the provider rejects, or
//!   whose account is out of quota, is the CUSTOMER's configuration problem.
//!   The house wording ("ask the gateway operator to verify the provider
//!   connection credential") tells them to ask someone else about their own key.
//!
//! Classification reads the RAW provider code and message (never relayed as
//! such); the bounded, sanitized detail is attached separately.

use crate::errors::{Failure, FailureClass};
use crate::upstream::transport_failure;

/// What a provider-declared error means for the caller and the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamErrorKind {
    /// The caller's request is what the provider refused: relay the detail,
    /// never redial, never fail over (the next rung refuses the same input).
    InvalidRequest,
    /// The model's safety layer declined the content.
    Refusal,
    /// Rate limit or overload: fail over, advertise a retry.
    Throttled,
    /// The provider ACCOUNT cannot pay.
    ProviderQuota,
    /// The provider rejected the credential.
    ProviderAuthentication,
    /// The provider does not know the model.
    ProviderNotFound,
    /// The provider itself failed.
    ProviderInternal,
}

const INVALID_REQUEST_CODES: &[&str] = &[
    "invalid_request_error",
    "invalid_request",
    "invalid_prompt",
    "invalid_value",
    "invalid_params",
    "invalid_parameter",
    "invalid_argument",
    "context_length_exceeded",
    "string_above_max_length",
    "unsupported_parameter",
    "unsupported_value",
    "failed_precondition",
    "out_of_range",
];
const REFUSAL_CODES: &[&str] = &[
    "content_filter",
    "content_policy_violation",
    "data_inspection_failed",
    "safety",
    "recitation",
    "prohibited_content",
    "blocklist",
    "spii",
    "moderation_blocked",
    "refusal",
    // OpenAI's cyber-safety policy verdict (gpt-6-astra, 2026-09-06): a model
    // decision on the content, filed as a refusal, never a provider failure.
    "cyber_policy",
];
const THROTTLED_CODES: &[&str] = &[
    "rate_limit_exceeded",
    "rate_limit_error",
    "overloaded_error",
    "resource_exhausted",
    "slow_down",
    "server_overloaded",
];
const QUOTA_CODES: &[&str] = &[
    "insufficient_quota",
    "insufficient_balance",
    "insufficient_credits",
    "billing_hard_limit_reached",
    "billing_not_active",
];
const AUTHENTICATION_CODES: &[&str] = &[
    "authentication_error",
    "invalid_api_key",
    "permission_error",
    "permission_denied",
    "unauthenticated",
    "access_denied",
    "account_deactivated",
    "invalid_authentication",
];
const NOT_FOUND_CODES: &[&str] = &["model_not_found", "not_found", "not_found_error"];

const INVALID_REQUEST_PHRASES: &[&str] = &[
    "exceeds the context window",
    "context length",
    "context window",
    "too long",
    "too many tokens",
    "invalid params",
    "invalid parameter",
    "invalid request",
    "invalid tool schema",
    "invalid schema",
    "does not support",
    "not supported",
    "unsupported",
    "max_tokens",
    "max tokens",
];
const REFUSAL_PHRASES: &[&str] = &[
    "content filter",
    "content filtering",
    "content policy",
    "content management policy",
    "inappropriate content",
    "safety system",
    "flagged as",
    "blocked by",
];
const THROTTLED_PHRASES: &[&str] = &[
    "rate limit",
    "rate-limit",
    "ratelimit",
    "overloaded",
    "too many requests",
    "at capacity",
    "temporarily unavailable due to load",
];
const QUOTA_PHRASES: &[&str] = &[
    "insufficient quota",
    "insufficient credits",
    "insufficient balance",
    "insufficient funds",
    "exceeded your current quota",
    "billing",
];
const AUTHENTICATION_PHRASES: &[&str] = &[
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
    "authentication",
    "unauthorized",
];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

/// Whether one provider code is an AUTHORITATIVE content verdict (a content
/// filter, safety, or data-inspection code). Used where only the code may
/// decide, never a sentence: a pre-stream 4xx body's prose can say "blocked by"
/// about a rate limit or a firewall.
pub fn is_refusal_code(code: Option<&str>) -> bool {
    code.is_some_and(|value| REFUSAL_CODES.contains(&value.trim().to_ascii_lowercase().as_str()))
}

/// Classify one provider-declared error from its raw code and message.
///
/// Content verdicts win first (a content filter may arrive under an
/// `invalid_request_error` type). Then the provider's own code is
/// authoritative (an `authentication_error` whose sentence happens to say
/// "must be provided" is still a credential failure). A numeric code takes the
/// shared HTTP mapping when it names a client-side status; a 5xx does NOT end
/// the search, because an aggregator often re-statuses an upstream 400 as its
/// own 502 and only the sentence says so. Credential, quota, and throttle
/// phrasing are read before caller-input phrasing so the broader input
/// vocabulary never swallows them. Anything left is the provider failing.
pub fn classify_stream_error(code: Option<&str>, message: Option<&str>) -> StreamErrorKind {
    let code_lower = code.map(|value| value.trim().to_ascii_lowercase());
    let message_lower = message.map(|value| value.to_ascii_lowercase());
    let code_ref = code_lower.as_deref().unwrap_or("");
    let message_ref = message_lower.as_deref().unwrap_or("");

    if REFUSAL_CODES.contains(&code_ref) || contains_any(message_ref, REFUSAL_PHRASES) {
        return StreamErrorKind::Refusal;
    }
    if INVALID_REQUEST_CODES.contains(&code_ref) {
        return StreamErrorKind::InvalidRequest;
    }
    if THROTTLED_CODES.contains(&code_ref) {
        return StreamErrorKind::Throttled;
    }
    if QUOTA_CODES.contains(&code_ref) {
        return StreamErrorKind::ProviderQuota;
    }
    if AUTHENTICATION_CODES.contains(&code_ref) {
        return StreamErrorKind::ProviderAuthentication;
    }
    if NOT_FOUND_CODES.contains(&code_ref) {
        return StreamErrorKind::ProviderNotFound;
    }
    if let Ok(status) = code_ref.parse::<u16>() {
        match transport_failure(Some(status)).failure_class {
            FailureClass::InvalidRequest => return StreamErrorKind::InvalidRequest,
            FailureClass::Throttled => return StreamErrorKind::Throttled,
            FailureClass::ProviderQuota => return StreamErrorKind::ProviderQuota,
            FailureClass::ProviderAuthentication => return StreamErrorKind::ProviderAuthentication,
            FailureClass::ProviderNotFound => return StreamErrorKind::ProviderNotFound,
            _ => {}
        }
    }
    if contains_any(message_ref, AUTHENTICATION_PHRASES) {
        return StreamErrorKind::ProviderAuthentication;
    }
    if contains_any(message_ref, QUOTA_PHRASES) {
        return StreamErrorKind::ProviderQuota;
    }
    if contains_any(message_ref, THROTTLED_PHRASES) {
        return StreamErrorKind::Throttled;
    }
    if contains_any(message_ref, INVALID_REQUEST_PHRASES) {
        return StreamErrorKind::InvalidRequest;
    }
    StreamErrorKind::ProviderInternal
}

/// Build the failure for one provider-declared stream error.
///
/// `detail` is the already-sanitized bounded line (or none). Every class keeps
/// it for the ledger; only the invalid-request class relays it to the caller,
/// which is exactly the class where the provider's sentence IS the fix.
pub fn stream_failure(kind: StreamErrorKind, detail: Option<String>) -> Failure {
    let failure = match kind {
        StreamErrorKind::InvalidRequest => Failure::new(
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
        )
        .with_retry(false, false),
        StreamErrorKind::Refusal => {
            Failure::new(FailureClass::Refusal, "provider refused the request")
        }
        StreamErrorKind::Throttled => Failure::new(
            FailureClass::Throttled,
            "provider throttled the request; retry after the delay in the Retry-After header",
        )
        .with_retry(false, true),
        StreamErrorKind::ProviderQuota => transport_failure(Some(402)),
        StreamErrorKind::ProviderAuthentication => transport_failure(Some(401)),
        StreamErrorKind::ProviderNotFound => transport_failure(Some(404)),
        StreamErrorKind::ProviderInternal => {
            Failure::new(FailureClass::ProviderInternal, "provider stream failed")
                .with_retry(true, true)
        }
    };
    failure.with_provider_detail(detail)
}

/// Re-own a credential or account failure on a customer-managed (BYOK) rung.
///
/// On a house rung these are operator deadness. On the customer's own
/// connection they are the customer's configuration: the message names their
/// provider and what to fix, and `customer_owned` makes a terminal answer their
/// 400 (and settlement files it client-side). The class and its failover
/// eligibility are kept: a pool may hold several customer-managed rungs with
/// independent credentials, and a later one may still serve. Anything else
/// passes through unchanged.
pub fn customer_credential_failure(failure: Failure, provider: &str) -> Failure {
    let message = match failure.failure_class {
        FailureClass::ProviderAuthentication => format!(
            "your connected {provider} credential was rejected by the provider; update the \
             key on this organization's {provider} provider connection and resend"
        ),
        FailureClass::ProviderQuota => format!(
            "your connected {provider} account has exhausted its quota or has billing \
             disabled; fund or enable the account at {provider} and resend"
        ),
        _ => return failure,
    };
    Failure {
        safe_message: message,
        customer_owned: true,
        ..failure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_input_errors_classify_as_invalid_request_even_behind_an_aggregator_502() {
        // OpenRouter re-statuses the upstream 400 as its own 502; the sentence wins.
        assert_eq!(
            classify_stream_error(
                Some("502"),
                Some("Your input exceeds the context window of this model. Please adjust your input and try again.")
            ),
            StreamErrorKind::InvalidRequest
        );
        assert_eq!(
            classify_stream_error(
                Some("400"),
                Some("Minimax midstream error: invalid params, model[MiniMax-Text-01] does not support max tokens > 8192")
            ),
            StreamErrorKind::InvalidRequest
        );
        assert_eq!(
            classify_stream_error(Some("invalid_request_error"), Some("Invalid request data")),
            StreamErrorKind::InvalidRequest
        );
        assert_eq!(
            classify_stream_error(Some("context_length_exceeded"), None),
            StreamErrorKind::InvalidRequest
        );
    }

    #[test]
    fn content_verdicts_are_refusals_whatever_type_they_arrive_under() {
        assert_eq!(
            classify_stream_error(
                Some("invalid_request_error"),
                Some("Output blocked by content filtering policy")
            ),
            StreamErrorKind::Refusal
        );
        assert_eq!(
            classify_stream_error(Some("SAFETY"), None),
            StreamErrorKind::Refusal
        );
        assert_eq!(
            classify_stream_error(
                Some("data_inspection_failed"),
                Some("Output data may contain inappropriate content.")
            ),
            StreamErrorKind::Refusal
        );
        assert_eq!(
            classify_stream_error(Some("cyber_policy"), None),
            StreamErrorKind::Refusal
        );
    }

    #[test]
    fn authoritative_codes_outrank_caller_input_phrasing() {
        // "must be provided" / "must be enabled" used to read as caller input;
        // the provider's own code decides first.
        assert_eq!(
            classify_stream_error(
                Some("authentication_error"),
                Some("API key must be provided")
            ),
            StreamErrorKind::ProviderAuthentication
        );
        assert_eq!(
            classify_stream_error(Some("insufficient_quota"), Some("Billing must be enabled")),
            StreamErrorKind::ProviderQuota
        );
        // Code-less sentences: credential and quota phrasing win over input phrasing.
        assert_eq!(
            classify_stream_error(
                None,
                Some("Incorrect API key provided; it is not supported")
            ),
            StreamErrorKind::ProviderAuthentication
        );
        assert_eq!(
            classify_stream_error(
                None,
                Some("You exceeded your current quota, max tokens unsupported")
            ),
            StreamErrorKind::ProviderQuota
        );
        // An unclassified sentence with a numeric 5xx stays provider-internal.
        assert_eq!(
            classify_stream_error(Some("503"), Some("Service Unavailable")),
            StreamErrorKind::ProviderInternal
        );
    }

    #[test]
    fn throttles_quota_auth_and_not_found_classify_by_code_or_status() {
        assert_eq!(
            classify_stream_error(Some("rate_limit_exceeded"), Some("Rate limit reached.")),
            StreamErrorKind::Throttled
        );
        assert_eq!(
            classify_stream_error(Some("overloaded_error"), Some("Overloaded")),
            StreamErrorKind::Throttled
        );
        assert_eq!(
            classify_stream_error(
                None,
                Some("The engine is currently overloaded, please try again later")
            ),
            StreamErrorKind::Throttled
        );
        assert_eq!(
            classify_stream_error(Some("429"), None),
            StreamErrorKind::Throttled
        );
        assert_eq!(
            classify_stream_error(Some("insufficient_quota"), None),
            StreamErrorKind::ProviderQuota
        );
        assert_eq!(
            classify_stream_error(Some("402"), None),
            StreamErrorKind::ProviderQuota
        );
        assert_eq!(
            classify_stream_error(Some("authentication_error"), Some("invalid x-api-key")),
            StreamErrorKind::ProviderAuthentication
        );
        assert_eq!(
            classify_stream_error(Some("model_not_found"), None),
            StreamErrorKind::ProviderNotFound
        );
    }

    #[test]
    fn provider_failures_stay_provider_internal() {
        assert_eq!(
            classify_stream_error(Some("server_error"), Some("The model failed.")),
            StreamErrorKind::ProviderInternal
        );
        assert_eq!(
            classify_stream_error(Some("502"), Some("upstream connect error")),
            StreamErrorKind::ProviderInternal
        );
        assert_eq!(
            classify_stream_error(Some("500"), Some("Internal Server Error")),
            StreamErrorKind::ProviderInternal
        );
        assert_eq!(
            classify_stream_error(None, None),
            StreamErrorKind::ProviderInternal
        );
    }

    #[test]
    fn invalid_request_failures_relay_the_detail_and_never_advance() {
        let failure = stream_failure(
            StreamErrorKind::InvalidRequest,
            Some("Your input exceeds the context window of this model.".to_string()),
        );
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(!failure.retryable_same_deployment);
        assert!(!failure.failover_eligible);
        let public = failure.public_error();
        assert_eq!(public.status_code, 400);
        assert_eq!(
            public.message,
            "provider rejected the request: Your input exceeds the context window of this model."
        );
    }

    #[test]
    fn other_kinds_keep_the_detail_off_the_wire_but_in_the_ledger() {
        let throttled = stream_failure(
            StreamErrorKind::Throttled,
            Some("rate_limit_exceeded".into()),
        );
        assert_eq!(throttled.failure_class, FailureClass::Throttled);
        assert!(throttled.failover_eligible && !throttled.retryable_same_deployment);
        assert_eq!(throttled.public_error().status_code, 429);
        assert!(!throttled
            .public_error()
            .message
            .contains("rate_limit_exceeded"));
        assert_eq!(
            throttled.provider_detail.as_deref(),
            Some("rate_limit_exceeded")
        );

        let internal = stream_failure(
            StreamErrorKind::ProviderInternal,
            Some("server_error".into()),
        );
        assert_eq!(internal.failure_class, FailureClass::ProviderInternal);
        assert!(internal.retryable_same_deployment && internal.failover_eligible);
        assert_eq!(internal.safe_message, "provider stream failed");

        let refusal = stream_failure(StreamErrorKind::Refusal, None);
        assert_eq!(refusal.failure_class, FailureClass::Refusal);
        assert_eq!(refusal.public_error().status_code, 400);
    }

    #[test]
    fn byok_credential_and_quota_failures_become_the_customers_400_but_still_fail_over() {
        let rejected = customer_credential_failure(transport_failure(Some(401)), "openai");
        // Class and ladder semantics are kept: another customer-managed rung
        // with its own credential may still serve.
        assert_eq!(rejected.failure_class, FailureClass::ProviderAuthentication);
        assert!(rejected.failover_eligible && !rejected.retryable_same_deployment);
        assert!(rejected.customer_owned);
        let public = rejected.public_error();
        assert_eq!(public.status_code, 400);
        assert_eq!(public.code, "provider_credential_rejected");
        assert_eq!(public.error_type, "invalid_request_error");
        assert!(public
            .message
            .contains("your connected openai credential was rejected"));

        let quota = customer_credential_failure(transport_failure(Some(402)), "openrouter");
        assert_eq!(quota.failure_class, FailureClass::ProviderQuota);
        assert!(quota.customer_owned && quota.failover_eligible);
        assert_eq!(quota.public_error().code, "provider_account_quota");
        assert!(quota
            .public_error()
            .message
            .contains("openrouter account has exhausted its quota"));

        // Any other class passes through untouched, house 502 shape and all.
        let not_found = customer_credential_failure(transport_failure(Some(404)), "openai");
        assert_eq!(not_found.failure_class, FailureClass::ProviderNotFound);
        assert!(!not_found.customer_owned);
        let internal = customer_credential_failure(transport_failure(Some(500)), "openai");
        assert_eq!(internal.public_error().status_code, 502);
    }
}
