use super::*;
use ferrum_output::{OutputBudget, OutputSession};
use ferrum_types::OutputConfig;

fn budgeted_stub(text: &str, config: OutputConfig) -> (Arc<StubLlm>, OutputBudget) {
    let budget = OutputBudget::new(config.clone()).unwrap();
    let mut engine = StubLlm::new(text);
    engine.config.output = config;
    engine.output_budget = budget.clone();
    (Arc::new(engine), budget)
}

fn request_for(path: &str, stream: bool) -> Value {
    match path {
        "/v1/chat/completions" => {
            json!({"model":"stub-model","messages":[{"role":"user","content":"hello"}],"stream":stream})
        }
        "/v1/responses" => {
            json!({"model":"stub-model","input":"hello","store":false,"stream":stream})
        }
        _ => json!({"model":"stub-model","prompt":"hello","stream":stream}),
    }
}

#[tokio::test]
async fn http_semantic_sync_three_protocols_retain_only_the_final_encoded_owner() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/completions"] {
        let (engine, budget) = budgeted_stub("full answer 字节", OutputConfig::default());
        let router = AxumServer::from_llm(engine).build_router();
        let response = post_json(router, path, request_for(path, false)).await;
        assert_eq!(response.status(), AxumStatusCode::OK, "{path}");
        assert_eq!(
            budget.usage().semantic_bytes,
            0,
            "semantic DTO owners finish before returning the wire body"
        );
        let mut body = response.into_body().into_data_stream();
        let bytes = body.next().await.unwrap().unwrap();
        assert_eq!(budget.usage().encoded_bytes, bytes.len());
        assert_eq!(budget.usage().sessions, 1);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        match path {
            "/v1/chat/completions" => {
                assert_eq!(
                    value["choices"][0]["message"]["content"],
                    "full answer 字节"
                );
                assert_eq!(value["usage"]["completion_tokens"], 2);
            }
            "/v1/responses" => {
                assert_eq!(value["output"][0]["content"][0]["text"], "full answer 字节");
                assert_eq!(value["usage"]["output_tokens"], 2);
                assert_eq!(value["status"], "completed");
            }
            _ => {
                assert_eq!(value["choices"][0]["text"], "full answer 字节");
                assert_eq!(value["usage"]["completion_tokens"], 2);
            }
        }
        assert!(body.next().await.is_none());
        drop(body);
        let last = bytes.clone();
        drop(bytes);
        assert_eq!(budget.usage().sessions, 1);
        drop(last);
        assert_eq!(budget.usage().sessions, 0);
        assert_eq!(budget.usage().encoded_bytes, 0);
    }
}

#[tokio::test]
async fn http_semantic_chat_and_responses_report_quota_without_waiting_for_owned_history() {
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let (engine, budget) = budgeted_stub(
            &"x".repeat(5000),
            OutputConfig {
                semantic_bytes_total: 8192,
                semantic_bytes_per_session: 8192,
                ..Default::default()
            },
        );
        let router = AxumServer::from_llm(engine).build_router();
        let response = post_json(router, path, request_for(path, true)).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let wire = tokio::time::timeout(std::time::Duration::from_secs(2), response_text(response)).await.expect("history growth rejects synchronously instead of waiting on its own retained credit");
        assert!(wire.contains("output_limit_exceeded"), "{path}: {wire}");
        assert!(
            !wire.contains("\"finish_reason\":\"stop\""),
            "{path}: {wire}"
        );
        assert!(
            !wire.contains("\"finish_reason\":\"length\""),
            "{path}: {wire}"
        );
        assert!(
            !wire.contains("event: response.completed"),
            "{path}: {wire}"
        );
        assert_eq!(budget.usage().semantic_bytes, 0, "{path}");
        assert_eq!(budget.usage().sessions, 0, "{path}");
    }
}

#[tokio::test]
async fn http_semantic_completions_stream_does_not_retain_prior_text() {
    let chunks = vec!["a".repeat(3000), "字".repeat(1000), "c".repeat(3000)];
    let expected = chunks.concat();
    let (mut engine, budget) = budgeted_stub(
        "",
        OutputConfig {
            semantic_bytes_total: 4096,
            semantic_bytes_per_session: 4096,
            ..Default::default()
        },
    );
    let fixture = Arc::get_mut(&mut engine).expect("unpublished fixture");
    fixture.stream_chunks = Some(chunks);
    fixture.stream_usage = Some(TokenUsage::new(7, 3));
    assert!(expected.len() > budget.config().semantic_bytes_per_session);
    let response = post_json(
        AxumServer::from_llm(engine).build_router(),
        "/v1/completions",
        request_for("/v1/completions", true),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut actual = String::new();
    let mut terminal_reasons = Vec::new();
    let mut usage = None;
    let mut done = 0;
    while let Some(frame) = tokio::time::timeout(std::time::Duration::from_secs(2), body.next())
        .await
        .expect("each current chunk fits independently")
    {
        let frame = frame.unwrap();
        // The HTTP DTO is dropped after encoding; only the wire frame and the
        // current upstream queued chunk remain, not earlier generated text.
        assert_eq!(budget.usage().semantic_bytes, 0);
        let wire = std::str::from_utf8(&frame).unwrap();
        for data in wire.lines().filter_map(|line| line.strip_prefix("data: ")) {
            if data == "[DONE]" {
                done += 1;
                continue;
            }
            let value: Value = serde_json::from_str(data).unwrap();
            assert!(value.get("error").is_none(), "{value}");
            for choice in value["choices"].as_array().unwrap() {
                actual.push_str(choice["text"].as_str().unwrap());
                if let Some(reason) = choice["finish_reason"].as_str() {
                    terminal_reasons.push(reason.to_owned());
                }
            }
            if !value["usage"].is_null() {
                usage = Some(value["usage"].clone());
            }
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(terminal_reasons, ["stop"]);
    assert_eq!(
        usage,
        Some(json!({"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}))
    );
    assert_eq!(done, 1);
    drop(body);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(budget.usage().queued_bytes, 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
}

#[tokio::test]
async fn http_semantic_completions_rejects_a_current_dto_larger_than_its_quota() {
    let (engine, budget) = budgeted_stub(
        &"x".repeat(5000),
        OutputConfig {
            semantic_bytes_total: 4096,
            semantic_bytes_per_session: 4096,
            ..Default::default()
        },
    );
    let response = post_json(
        AxumServer::from_llm(engine).build_router(),
        "/v1/completions",
        request_for("/v1/completions", true),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let wire = tokio::time::timeout(std::time::Duration::from_secs(2), response_text(response))
        .await
        .expect("impossible current DTO rejects without waiting");
    assert!(wire.contains("output_limit_exceeded"), "{wire}");
    assert!(!wire.contains("\"finish_reason\":\"stop\""), "{wire}");
    assert!(
        !wire.contains(&"x".repeat(5000)),
        "oversized payload must not be published"
    );
    assert_eq!(budget.usage().semantic_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

fn cached_chat(session: &OutputSession) -> ChatSyncOutput {
    output::Retained::new(
        session,
        ChatCompletionsResponse {
            id: "cached".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "stub-model".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: Some(ChatMessage {
                    role: MessageRole::Assistant,
                    content: "cached generated answer".into(),
                    reasoning: None,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                }),
                delta: None,
                finish_reason: Some("stop".into()),
            }],
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            }),
        },
    )
    .unwrap()
}

#[test]
fn http_semantic_session_history_shares_original_owner_until_last_reference() {
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let output = cached_chat(&session);
    let charged = session.semantic_usage();
    let cache = CacheRuntimeState::default();
    let policy = CachePolicy {
        prefix_cache_enabled: false,
        session_cache_mode: "memory".into(),
        session_cache_max_entries: 2,
        session_cache_max_tokens: 4096,
    };
    cache.update_session(
        Some(SessionContext {
            id: "conversation".into(),
            prior_messages: vec![],
            incoming_messages: vec![],
            _request_copies: vec![],
        }),
        output.clone(),
        &policy,
    );
    assert_eq!(
        session.semantic_usage(),
        charged,
        "caching shallow clones the charged output owner"
    );
    let retained_history = cache.sessions.lock().unwrap()["conversation"].clone();
    drop(output);
    drop(session);
    assert!(
        budget.try_open_session().is_err(),
        "old generated output still owns the only admission session"
    );
    cache.sessions.lock().unwrap().clear();
    assert_eq!(budget.usage().semantic_bytes, charged);
    assert_eq!(
        retained_history[0].message().content,
        "cached generated answer"
    );
    drop(retained_history);
    assert_eq!(budget.usage().semantic_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
    drop(budget.try_open_session().unwrap());
}
