use super::*;

/// Body polling is causally after ingress middleware and before JSON/template
/// processing. A timestamp captured by the engine or handler would be too late.
#[tokio::test]
async fn slo_ingress_precedes_body_read_for_all_generation_routes() {
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
        for streaming in [false, true] {
            let engine = Arc::new(CapturingLlm::new());
            let app = AxumServer::from_llm(engine.clone()).build_router();
            let payload = match path {
                "/v1/chat/completions" => json!({
                    "model": "qwen3", "stream": streaming,
                    "messages": [{"role": "user", "content": "hello"}]
                }),
                "/v1/completions" => json!({
                    "model": "qwen3", "stream": streaming, "prompt": "hello"
                }),
                _ => json!({"model": "qwen3", "stream": streaming, "input": "hello"}),
            };
            let body_polled_at = Arc::new(Mutex::new(None));
            let body_clock = Arc::clone(&body_polled_at);
            let body = Body::from_stream(futures::stream::once(async move {
                *body_clock.lock().unwrap() = Some(Instant::now());
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(payload.to_string()))
            }));
            let sent_at = Instant::now();
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(header::CONTENT_TYPE, "application/json")
                        // Arbitrary wire headers do not grant a service class.
                        .header("x-ferrum-slo-service-class", "unconfigured-admin")
                        .body(body)
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                AxumStatusCode::OK,
                "{path}, stream={streaming}"
            );
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert!(!bytes.is_empty());
            let context = engine
                .last_context
                .lock()
                .unwrap()
                .clone()
                .expect("ingress forwarded");
            assert!(context.ingress() >= sent_at);
            assert!(context.ingress() <= body_polled_at.lock().unwrap().unwrap());
            assert_eq!(context.service_class(), None);
        }
    }
}

#[tokio::test]
async fn slo_unsupported_engine_cannot_silently_ignore_active_context() {
    let mut engine = StubLlm::new("unused");
    engine.config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
    let request = InferenceRequest::new("hello".to_owned(), ModelId::new("stub-model"));
    assert!(engine
        .infer_with_context(request.clone(), InferenceRequestContext::capture())
        .await
        .is_err());
    assert!(engine
        .infer_stream_with_context(request, InferenceRequestContext::capture())
        .await
        .is_err());
}
