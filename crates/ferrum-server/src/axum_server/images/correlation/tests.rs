use super::*;
use axum::http::StatusCode;

#[test]
fn image_correlation_accepts_only_one_bounded_client_attempt() {
    let mut headers = HeaderMap::new();
    assert_eq!(attempt(&headers).unwrap(), None);
    let id = Uuid::new_v4();
    headers.insert(IMAGE_ATTEMPT_ID_HEADER, id.to_string().parse().unwrap());
    assert_eq!(attempt(&headers).unwrap(), Some(id));
    headers.append(IMAGE_ATTEMPT_ID_HEADER, id.to_string().parse().unwrap());
    assert!(attempt(&headers).is_err());
    for invalid in [
        "",
        "prompt-or-image-data",
        "00000000000000000000000000000000",
        "00000000-0000-0000-0000-00000000000g",
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(IMAGE_ATTEMPT_ID_HEADER, invalid.parse().unwrap());
        assert!(attempt(&headers).is_err());
    }
    for name in [IMAGE_REQUEST_ID_HEADER, IMAGE_TRACE_SESSION_HEADER] {
        let mut headers = HeaderMap::new();
        headers.insert(name, id.to_string().parse().unwrap());
        assert!(attempt(&headers).is_err());
    }
}

#[tokio::test]
async fn image_correlation_headers_preserve_original_response_status_and_body() {
    // Header serialization is a protocol test; these UUIDs are not presented
    // as an executed pipeline or a fabricated trace/retirement receipt.
    let correlation = ImageRequestCorrelation::new(Some(Uuid::new_v4()));
    let trace_session = Uuid::new_v4();
    let context = Context {
        correlation: Some(correlation),
        trace_session: Some(trace_session),
    };
    let response = context.response((StatusCode::ACCEPTED, "original body").into_response());
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    for (name, id) in [
        (IMAGE_REQUEST_ID_HEADER, correlation.request_id),
        (
            IMAGE_ATTEMPT_ID_HEADER,
            correlation.client_attempt_id.unwrap(),
        ),
        (IMAGE_TRACE_SESSION_HEADER, trace_session),
    ] {
        assert_eq!(response.headers()[name].to_str().unwrap(), id.to_string());
    }
    let body = axum::body::to_bytes(response.into_body(), 128)
        .await
        .unwrap();
    assert_eq!(body.as_ref(), b"original body");
    let error = context
        .error(ServerError::CapacityExceeded("bounded admission".into()))
        .into_response();
    assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        error.headers()[IMAGE_REQUEST_ID_HEADER].to_str().unwrap(),
        correlation.request_id.to_string()
    );
    let legacy = Context {
        correlation: Some(ImageRequestCorrelation::new(None)),
        trace_session: None,
    }
    .response(StatusCode::OK.into_response());
    assert!(legacy.headers().contains_key(IMAGE_REQUEST_ID_HEADER));
    assert!(!legacy.headers().contains_key(IMAGE_ATTEMPT_ID_HEADER));
    assert!(!legacy.headers().contains_key(IMAGE_TRACE_SESSION_HEADER));
}
