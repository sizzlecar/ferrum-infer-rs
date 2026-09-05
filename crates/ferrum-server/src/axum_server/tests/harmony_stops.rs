use super::*;

struct TextCase {
    chunks: &'static [&'static str],
    content: &'static str,
    reasoning: Option<&'static str>,
}

fn text_cases() -> [TextCase; 3] {
    // These are the bytes retained by the engine after a matched user stop.
    // Engine tests separately cover finding and removing split/in-token stops.
    [
        TextCase {
            chunks: &["<|channel|>fi", "nal<|message|>答", "案 "],
            content: "答案 ",
            reasoning: None,
        },
        TextCase {
            chunks: &["<|channel|>analysis<|message|>", "Still reasoning "],
            content: "",
            reasoning: Some("Still reasoning "),
        },
        TextCase {
            chunks: &[
                "<|channel|>analysis<|message|>Reason.<|end|>",
                "<|start|>assistant<|channel|>final<|message|>",
                "答案 ",
            ],
            content: "答案 ",
            reasoning: Some("Reason."),
        },
    ]
}

fn router(chunks: &[&str], finish_reason: FinishReason, separate_final: bool) -> Router {
    let mut engine = if separate_final {
        StubLlm::with_separate_final_stream_chunk(chunks)
    } else {
        StubLlm::with_stream_chunks(chunks)
    };
    engine.finish_reason = finish_reason;
    AxumServer::from_llm(Arc::new(engine))
        .with_prompt_template(Some(harmony_json_template()))
        .build_router()
}

fn request(stream: bool, stop: Option<&str>) -> Value {
    let mut wire = json!({
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Continue."}],
        "stream": stream
    });
    if stream {
        wire["stream_options"] = json!({"include_usage": true});
    }
    if let Some(stop) = stop {
        wire["stop"] = json!([stop]);
    }
    wire
}

fn delta_text(events: &[Value], field: &str) -> String {
    events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"][field].as_str())
        .collect()
}

#[tokio::test]
async fn harmony_user_stop_sync_returns_parsed_text_reasoning_and_usage() {
    for case in text_cases() {
        let response = post_json(
            router(case.chunks, FinishReason::Stop, false),
            "/v1/chat/completions",
            request(false, Some("STOP")),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert!(body.get("error").is_none());
        let choice = &body["choices"][0];
        assert_eq!(choice["message"]["content"], case.content);
        assert_eq!(choice["message"]["reasoning"].as_str(), case.reasoning);
        assert!(choice["message"].get("reasoning_content").is_none());
        assert!(choice["message"].get("tool_calls").is_none());
        assert_eq!(choice["finish_reason"], "stop");
        // StubLlm::infer supplies this usage independently of HTTP text/events.
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 9);
    }
}

async fn assert_stream_success(case: &TextCase, separate_final: bool) {
    let response = post_json(
        router(case.chunks, FinishReason::Stop, separate_final),
        "/v1/chat/completions",
        request(true, Some("STOP")),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let body = response_text(response).await;
    let events = responses_sse_json_events(&body);
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "{body}"
    );
    assert_eq!(delta_text(&events, "content"), case.content);
    assert_eq!(
        delta_text(&events, "reasoning"),
        case.reasoning.unwrap_or("")
    );
    assert!(events.iter().all(|event| {
        let delta = &event["choices"][0]["delta"];
        delta.get("tool_calls").is_none() && delta.get("reasoning_content").is_none()
    }));

    let terminals: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| event["choices"][0]["finish_reason"].is_string())
        .collect();
    assert_eq!(terminals.len(), 1, "expected one terminal choice: {body}");
    let (terminal_index, terminal) = terminals[0];
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(terminal["choices"][0]["delta"]["content"], "");
    assert!(terminal["choices"][0]["delta"]["reasoning"].is_null());
    assert!(terminal["usage"].is_null());

    let usage_index = events
        .iter()
        .position(|event| event["usage"].is_object())
        .expect("requested usage event");
    assert!(
        terminal_index < usage_index,
        "usage follows terminal choice: {body}"
    );
    let usage = &events[usage_index];
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 5);
    assert_eq!(usage["usage"]["completion_tokens"], case.chunks.len());
    assert_eq!(usage["usage"]["total_tokens"], 5 + case.chunks.len());
    assert_eq!(
        body.lines().filter(|line| *line == "data: [DONE]").count(),
        1
    );
    assert!(body.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn harmony_user_stop_stream_returns_parsed_text_when_finish_shares_text_chunk() {
    for case in text_cases() {
        assert_stream_success(&case, false).await;
    }
}

#[tokio::test]
async fn harmony_user_stop_stream_flushes_on_separate_empty_terminal_chunk() {
    // This fixture emits the final engine chunk with empty text and token=None.
    // The retained text must still flush before the separate terminal/usage/DONE.
    for case in text_cases() {
        assert_stream_success(&case, true).await;
    }
}

async fn assert_rejected(output: &str, finish_reason: FinishReason, stop: Option<&str>) {
    for stream in [false, true] {
        let response = post_json(
            router(&[output], finish_reason, true),
            "/v1/chat/completions",
            request(stream, stop),
        )
        .await;
        if stream {
            assert_eq!(response.status(), AxumStatusCode::OK);
            let body = response_text(response).await;
            let events = responses_sse_json_events(&body);
            let error = events
                .iter()
                .find(|event| event.get("error").is_some())
                .expect("invalid Harmony must emit an SSE error");
            assert_eq!(error["error"]["type"], "internal_server_error");
            assert_eq!(error["error"]["param"], "model_output");
            assert!(delta_text(&events, "content").is_empty());
            assert!(delta_text(&events, "reasoning").is_empty());
            assert!(
                events.iter().all(|event| {
                    event["choices"][0]["delta"].get("tool_calls").is_none()
                        && event["choices"][0]["finish_reason"].is_null()
                        && event["usage"].is_null()
                }),
                "failed output must not become a successful completion: {body}"
            );
            assert_eq!(
                body.lines().filter(|line| *line == "data: [DONE]").count(),
                1
            );
            assert!(body.trim_end().ends_with("data: [DONE]"));
        } else {
            assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
            let body = response_json(response).await;
            assert_eq!(body["error"]["type"], "internal_server_error");
            assert!(body.get("choices").is_none());
            assert!(body.get("usage").is_none());
        }
    }
}

#[tokio::test]
async fn harmony_natural_eos_requires_terminal_even_with_unmatched_request_stop() {
    for case in text_cases() {
        let output = case.chunks.concat();
        for stop in [None, Some("UNMATCHED_STOP")] {
            assert_rejected(&output, FinishReason::EOS, stop).await;
        }
    }
}

#[tokio::test]
async fn harmony_user_stop_keeps_incomplete_tools_and_unknown_controls_fail_closed() {
    for output in [
        "<|channel|>analysis<|message|>Need weather.<|end|>\
         <|start|>assistant<|channel|>commentary to=functions.weather\
         <|constrain|>json<|message|>{\"city\":",
        "<|channel|>analysis<|message|>Need weather.<|end|>\
         <|start|>assistant<|channel|>commentary to=functions.weather\
         <|constrain|>json<|message|>{\"city\":\"Paris\"}",
        "<|channel|>final<|message|>private payload<|bogus|>",
        "<|channel|>final<|message|>private payload<|",
    ] {
        assert_rejected(output, FinishReason::Stop, Some("STOP")).await;
    }
}
