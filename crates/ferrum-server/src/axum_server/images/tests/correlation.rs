use super::*;
use axum::http::{Request, StatusCode};
use ferrum_types::image_trace::{
    IMAGE_ATTEMPT_ID_HEADER, IMAGE_REQUEST_ID_HEADER, IMAGE_TRACE_SESSION_HEADER,
};
use uuid::Uuid;

fn request(attempt: Option<Uuid>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json");
    if let Some(attempt) = attempt {
        builder = builder.header(IMAGE_ATTEMPT_ID_HEADER, attempt.to_string());
    }
    builder
        .body(Body::from(
            r#"{"model":"image","prompt":"protocol fixture","size":"32x32"}"#,
        ))
        .unwrap()
}

#[tokio::test]
async fn image_correlation_routes_reject_forged_or_malformed_ids_before_submission() {
    let (images, calls) = image_service();
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let attempt = Uuid::new_v4();
    for path in ["/v1/images/generations", "/v1/images/edits"] {
        for (name, value) in [
            (IMAGE_ATTEMPT_ID_HEADER, "not-a-uuid".to_owned()),
            (IMAGE_REQUEST_ID_HEADER, attempt.to_string()),
            (IMAGE_TRACE_SESSION_HEADER, attempt.to_string()),
        ] {
            let mut request = request(None);
            *request.uri_mut() = path.parse().unwrap();
            request.headers_mut().insert(name, value.parse().unwrap());
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(!response.headers().contains_key(IMAGE_REQUEST_ID_HEADER));
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    images.shutdown().await.unwrap();
    assert!(!images.take_shutdown().await.unwrap().has_failure());
}

#[tokio::test]
async fn image_correlation_real_reference_success_returns_service_id_and_legacy_omission() {
    // Reuse the original Stored Reference decoder, not a forged successful
    // tensor/receipt. The fixture has no continuous collector, hence no trace ID.
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8 * 1024 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024 * 1024,
                output_bytes: 4 * 1024 * 1024,
                transfer_staging_bytes: 2 * 1024 * 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let service = ImageService::new(
        decoupled_output::Decoder::new(host.clone(), calls.clone()),
        ferrum_engine::image_generation::service::ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Configured,
            maximum_pending: 2,
            maximum_pending_bytes: 4096,
            maximum_prompt_bytes: 64,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let mut config = config();
    config.default_width = 32;
    config.default_height = 32;
    let images = ImageHttpService::new("image", service, config).unwrap();
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let attempt = Uuid::new_v4();
    let mut ids = Vec::new();
    for client_attempt in [Some(attempt), Some(attempt), None] {
        let response = router
            .clone()
            .oneshot(request(client_attempt))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let server_id = Uuid::parse_str(
            response.headers()[IMAGE_REQUEST_ID_HEADER]
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_ne!(server_id, attempt);
        assert!(!ids.contains(&server_id));
        ids.push(server_id);
        assert_eq!(
            response
                .headers()
                .get(IMAGE_ATTEMPT_ID_HEADER)
                .map(|x| x.to_str().unwrap().to_owned()),
            client_attempt.map(|x| x.to_string())
        );
        assert!(!response.headers().contains_key(IMAGE_TRACE_SESSION_HEADER));
        let body = axum::body::to_bytes(response.into_body(), 32 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["data"].as_array().unwrap().len(), 1);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    images.shutdown().await.unwrap();
    assert!(!images.take_shutdown().await.unwrap().has_failure());
    drop((router, images));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn image_correlation_capacity_rejection_retains_actual_service_id() {
    let host = host_account(4 * 1024 * 1024); // Input class has zero capacity.
    let (images, calls) = image_service_with_host(host, None);
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let attempt = Uuid::new_v4();
    let mut request = request(Some(attempt));
    *request.body_mut() = Body::from(r#"{"model":"image","prompt":"x"}"#);
    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let server_id = Uuid::parse_str(
        response.headers()[IMAGE_REQUEST_ID_HEADER]
            .to_str()
            .unwrap(),
    )
    .unwrap();
    assert_ne!(server_id, attempt);
    assert_eq!(
        response.headers()[IMAGE_ATTEMPT_ID_HEADER]
            .to_str()
            .unwrap(),
        attempt.to_string()
    );
    assert!(!response.headers().contains_key(IMAGE_TRACE_SESSION_HEADER));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    images.shutdown().await.unwrap();
    assert!(!images.take_shutdown().await.unwrap().has_failure());
}
