use super::*;

#[tokio::test]
async fn retired_modality_routes_are_not_registered() {
    let (loaded, engine) = router_with_capturing_llm();
    for router in [router_without_llm(), loaded] {
        for (path, content_type, body) in [
            (
                "/v1/embeddings",
                "application/json",
                r#"{"model":"qwen3","input":"hello"}"#,
            ),
            (
                "/v1/audio/transcriptions",
                "multipart/form-data; boundary=audio",
                "--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--audio--\r\n",
            ),
            (
                "/v1/audio/speech",
                "application/json",
                r#"{"model":"qwen3","input":"hello","response_format":"wav"}"#,
            ),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(header::CONTENT_TYPE, content_type)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), AxumStatusCode::NOT_FOUND, "{path}");
        }
    }
    assert!(!engine.has_captured_request());
}

#[tokio::test]
async fn retired_modality_routes_do_not_alias_text_generation() {
    let (router, engine) = router_with_capturing_llm();
    let request = json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello"}]
    });
    for path in [
        "/v1/embeddings",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ] {
        let response = post_json(router.clone(), path, request.clone()).await;
        assert_eq!(response.status(), AxumStatusCode::NOT_FOUND, "{path}");
        assert!(!engine.has_captured_request());
    }

    let response = post_json(router, "/v1/chat/completions", request).await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    assert!(engine.has_captured_request());
}
