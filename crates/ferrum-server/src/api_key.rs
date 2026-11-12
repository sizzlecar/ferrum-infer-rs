//! Optional bearer-key authentication shared by text and image HTTP serving.
use axum::{
    extract::{Request, State},
    http::{header, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json, Router,
};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::fmt;
use subtle::ConstantTimeEq;

/// Server credential. Debug and configuration exports never contain its value.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(value: impl Into<String>) -> ferrum_types::Result<Self> {
        let key = Self(value.into());
        key.validate()?;
        Ok(key)
    }

    /// Validate separately from deserialization so malformed credentials are not
    /// included in a configuration parser's source-line error diagnostics.
    pub fn validate(&self) -> ferrum_types::Result<()> {
        let value = self.0.as_bytes();
        let unpadded = value.strip_suffix(b"=").unwrap_or(value);
        let valid = !value.is_empty()
            && value
                .iter()
                .all(|b| b.is_ascii_alphanumeric() || b"-._~+/=".contains(b))
            && unpadded
                .iter()
                .position(|b| *b == b'=')
                .is_none_or(|first| unpadded[first..].iter().all(|b| *b == b'='))
            && value[0] != b'=';
        if valid {
            Ok(())
        } else {
            Err(ferrum_types::FerrumError::configuration(
                "API key must be a nonempty bearer token without whitespace or control characters",
            ))
        }
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApiKey([REDACTED])")
    }
}

impl Serialize for ApiKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str("[REDACTED]")
    }
}

pub(crate) fn protect(router: Router, key: Option<&ApiKey>) -> ferrum_types::Result<Router> {
    let Some(key) = key else { return Ok(router) };
    key.validate()?;
    let expected: [u8; 32] = Sha256::digest(key.0.as_bytes()).into();
    Ok(router.layer(middleware::from_fn_with_state(expected, authenticate)))
}

async fn authenticate(State(expected): State<[u8; 32]>, request: Request, next: Next) -> Response {
    // Readiness and static assets remain available. All data, inference and
    // metrics routes require the key. OPTIONS cannot invoke inference handlers.
    if request.method() == Method::OPTIONS
        || matches!(
            request.uri().path(),
            "/health" | "/playground" | "/playground/app.js" | "/playground/style.css"
        )
    {
        return next.run(request).await;
    }
    let mut values = request.headers().get_all(header::AUTHORIZATION).iter();
    let token = values
        .next()
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, token)| scheme.eq_ignore_ascii_case("bearer") && !token.is_empty())
        .map(|(_, token)| token);
    if values.next().is_none()
        && token.is_some_and(|token| {
            let actual: [u8; 32] = Sha256::digest(token.as_bytes()).into();
            bool::from(expected.ct_eq(&actual))
        })
    {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(serde_json::json!({"error": {
                "message": "Invalid or missing API key.",
                "type": "authentication_error", "param": null, "code": "invalid_api_key"
            }})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests;
