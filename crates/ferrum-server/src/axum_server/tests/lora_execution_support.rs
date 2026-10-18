use super::*;

fn adapter() -> LoraAdapterModel {
    LoraAdapterModel::new("sql", "public:sql", "/tmp/sql-adapter")
}

fn registry() -> ServedModelRegistry {
    ServedModelRegistry::try_new(
        "qwen3",
        ServedModelKind::Llm,
        vec!["public".to_owned()],
        vec![adapter()],
    )
    .unwrap()
}

#[test]
fn lora_registration_requires_execution_support_not_metrics_or_resource_authority() {
    for resource_authority in [
        ExecutionResourceAuthority::LegacyEngine,
        ExecutionResourceAuthority::PlanRuntime,
    ] {
        let engine = StubLlm {
            resource_authority,
            ..StubLlm::with_lora_metrics("ok", json!({"enabled": true, "adapter_count": 1}))
        };
        let result = AxumServer::from_llm(Arc::new(engine))
            .with_lora_adapters("stub-model", vec![adapter()]);
        assert!(matches!(result, Err(Error::Unsupported { .. })));
    }
    let result =
        AxumServer::from_state(AppState::default()).with_lora_adapters("qwen3", vec![adapter()]);
    assert!(matches!(result, Err(Error::Unsupported { .. })));

    let base = AxumServer::from_llm(Arc::new(StubLlm::new("ok")))
        .with_lora_adapters("stub-model", vec![])
        .unwrap();
    assert_eq!(base.state.served_model_registry.adapter_count(), 0);

    let supported = AxumServer::from_llm(Arc::new(
        StubLlm::new("ok").with_adapter_support(AdapterExecutionSupport::PerSequenceLora),
    ))
    .with_lora_adapters("stub-model", vec![adapter()])
    .unwrap();
    assert_eq!(supported.state.served_model_registry.adapter_count(), 1);
}

#[tokio::test]
async fn injected_lora_registry_is_rejected_before_start_binds_or_sets_running() {
    // Occupying the port distinguishes the capability failure from a later bind
    // failure without allowing an accidentally accepted server to keep running.
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let config = ServerConfig {
        host: "127.0.0.1".to_owned(),
        port: occupied.local_addr().unwrap().port(),
        ..ServerConfig::default()
    };
    for state in [
        AppState::default(),
        AppState::default().with_llm(Arc::new(CapturingLlm::new())),
    ] {
        let server = AxumServer::from_state(state).with_served_model_registry(registry());
        let result = server.start(&config).await;
        assert!(
            matches!(result, Err(Error::Unsupported { .. })),
            "{result:?}"
        );
        assert!(!server.is_running());
    }
}

#[tokio::test]
async fn injected_lora_alias_cannot_fall_through_to_base_inference() {
    let engine = Arc::new(CapturingLlm::new());
    let router = AxumServer::from_llm(engine.clone())
        .with_served_model_registry(registry())
        .build_router();
    for stream in [false, true] {
        for (path, body) in [
            (
                "/v1/chat/completions",
                json!({
                    "model": "public:sql", "stream": stream, "max_tokens": 4,
                    "messages": [{"role": "user", "content": "hello"}]
                }),
            ),
            (
                "/v1/completions",
                json!({"model": "public:sql", "stream": stream, "max_tokens": 4, "prompt": "hello"}),
            ),
            (
                "/v1/responses",
                json!({"model": "public:sql", "stream": stream, "max_output_tokens": 4, "input": "hello"}),
            ),
        ] {
            let response = post_json(router.clone(), path, body).await;
            assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST, "{path}");
            let body = response_json(response).await;
            assert_eq!(body["error"]["param"], "model", "{path}: {body}");
            assert!(body["error"]["message"].as_str().unwrap().contains("LoRA"));
            assert!(!engine.has_captured_request(), "{path}");
        }
    }

    let response = post_json(
        router,
        "/v1/chat/completions",
        json!({"model": "public", "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let request = engine.last_request();
    assert_eq!(request.model_id, ModelId::new("qwen3"));
    assert!(!request.metadata.contains_key("ferrum_lora_adapter"));
}

#[tokio::test]
async fn supported_lora_alias_routes_completions_and_responses() {
    for (path, body) in [
        (
            "/v1/completions",
            json!({"model": "qwen3:sql", "prompt": "hello", "max_tokens": 4}),
        ),
        (
            "/v1/responses",
            json!({"model": "qwen3:sql", "input": "hello", "max_output_tokens": 4}),
        ),
    ] {
        let (router, engine) = router_with_capturing_lora_llm();
        let response = post_json(router, path, body).await;
        assert_eq!(response.status(), AxumStatusCode::OK, "{path}");
        let request = engine.last_request();
        assert_eq!(request.model_id, ModelId::new("qwen3"));
        assert_eq!(request.metadata["ferrum_lora_adapter"], "sql");
        assert_eq!(request.metadata["ferrum_lora_path"], "/tmp/sql-adapter");
    }
}
