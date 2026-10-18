//! Consume real HTTP body frames while the fixture engine is unable to finish.
use super::*;
use futures::FutureExt;
use std::time::Duration;

// Pulling the next frame is what advances the upstream engine now. Preserve
// every frame produced before the gate instead of relying on an eager relay.
async fn pull_until_gate(
    body: &mut axum::body::BodyDataStream,
    text: &mut String,
    gate: &StreamGate,
) {
    loop {
        tokio::select! {
            biased;
            _ = gate.entered.notified() => return,
            frame = body.next() => {
                let frame = frame.expect("engine gate precedes HTTP EOF").unwrap();
                text.push_str(std::str::from_utf8(&frame).unwrap());
            }
        }
    }
}

fn template() -> ModelChatTemplate {
    let mut template = ModelChatTemplate::new(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        "native-stream-contract",
    );
    template.tool_call_protocol = ferrum_types::ApiToolCallProtocol::FunctionParameterXml;
    template.reasoning_protocol = ModelReasoningProtocol::ModelGenerated;
    template
}

fn request(responses: bool, streaming: bool) -> (&'static str, Value) {
    let function = json!({"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}
    }});
    if responses {
        (
            "/v1/responses",
            json!({
                "model": "stub-model", "input": "Check the weather.",
                "tools": [{"type": "function", "name": function["name"], "parameters": function["parameters"]}],
                "tool_choice": "auto", "stream": streaming
            }),
        )
    } else {
        let mut body = json!({
            "model": "stub-model", "messages": [{"role": "user", "content": "Check the weather."}],
            "tools": [{"type": "function", "function": function}],
            "tool_choice": "auto", "stream": streaming
        });
        if streaming {
            body["stream_options"] = json!({"include_usage": true});
        }
        ("/v1/chat/completions", body)
    }
}

fn router(stub: StubLlm) -> Router {
    AxumServer::from_llm(Arc::new(stub))
        .with_prompt_template(Some(template()))
        .build_router()
        .unwrap()
}

fn text_deltas(events: &[Value], responses: bool) -> String {
    events
        .iter()
        .filter_map(|event| {
            if responses {
                (event["type"] == "response.output_text.delta")
                    .then(|| event["delta"].as_str())
                    .flatten()
            } else {
                event["choices"][0]["delta"]["content"].as_str()
            }
        })
        .collect()
}

fn arguments(events: &[Value], responses: bool) -> String {
    events
        .iter()
        .filter_map(|event| {
            if responses {
                (event["type"] == "response.function_call_arguments.delta")
                    .then(|| event["delta"].as_str())
                    .flatten()
            } else {
                event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            }
        })
        .collect()
}

#[tokio::test]
async fn native_xml_stream_exposes_body_before_engine_can_finish_without_repeating_it() {
    let payload = "Paris <think>literal</think> </tool_call> literal";
    let first = "<think>Private reasoning.</think>\r\nVisible before <to";
    let rest = format!("ol_call><function=weather><parameter=city>{payload}</parameter></function></tool_call>\nTail.");
    for responses in [false, true] {
        let gate = Arc::new(StreamGate::default());
        let (path, wire) = request(responses, true);
        let response = post_json(
            router(StubLlm {
                stream_after_first_gate: Some(gate.clone()),
                ..StubLlm::with_separate_final_stream_chunk(&[first, &rest])
            }),
            path,
            wire,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        let mut text = String::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let frame = body.next().await.expect("early HTTP frame").unwrap();
                text.push_str(std::str::from_utf8(&frame).unwrap());
                if text_deltas(&responses_sse_json_events(&text), responses) == "Visible before " {
                    break;
                }
            }
            pull_until_gate(&mut body, &mut text, &gate).await;
        })
        .await
        .expect("ordinary body must arrive before the gated engine continuation");
        let events = responses_sse_json_events(&text);
        assert!(arguments(&events, responses).is_empty());
        assert!(!text_deltas(&events, responses).contains("Private reasoning."));
        assert!(!text.contains("[DONE]"));
        gate.resume.notify_one();
        while let Some(frame) = body.next().await {
            text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        let events = responses_sse_json_events(&text);
        assert_eq!(
            text_deltas(&events, responses),
            "Visible before \nTail.",
            "{text}"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&arguments(&events, responses)).unwrap(),
            json!({"city": payload})
        );
        let reasoning = events
            .iter()
            .filter_map(|event| {
                if responses {
                    (event["type"] == "response.reasoning_text.delta")
                        .then(|| event["delta"].as_str())
                        .flatten()
                } else {
                    event["choices"][0]["delta"]["reasoning"].as_str()
                }
            })
            .collect::<String>();
        assert_eq!(reasoning, "Private reasoning.");
        assert!(
            events.iter().all(|event| event["error"].is_null()),
            "{text}"
        );

        // The ordinary non-streaming endpoint has exactly the same projection.
        let (path, wire) = request(responses, false);
        let sync = post_json(router(StubLlm::new(&format!("{first}{rest}"))), path, wire).await;
        let status = sync.status();
        let text = response_text(sync).await;
        assert_eq!(status, AxumStatusCode::OK, "{path}: {text}");
        let sync: Value = serde_json::from_str(&text).unwrap();
        let content = if responses {
            sync["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "message")
                .filter_map(|item| item["content"].as_array())
                .flatten()
                .filter_map(|part| part["text"].as_str())
                .collect::<String>()
        } else {
            sync["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(content, text_deltas(&events, responses));
    }
}

#[tokio::test]
async fn native_xml_stream_body_cannot_be_reclassified_by_late_reasoning_or_bare_json() {
    let first = "Ordinary body. ";
    let rest =
        "</think> <think>literal</think> {\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}";
    for responses in [false, true] {
        let gate = Arc::new(StreamGate::default());
        let (path, wire) = request(responses, true);
        let response = post_json(
            router(StubLlm {
                stream_after_first_gate: Some(gate.clone()),
                ..StubLlm::with_separate_final_stream_chunk(&[first, rest])
            }),
            path,
            wire,
        )
        .await;
        let mut body = response.into_body().into_data_stream();
        let mut text = String::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while text_deltas(&responses_sse_json_events(&text), responses) != first {
                let frame = body.next().await.unwrap().unwrap();
                text.push_str(std::str::from_utf8(&frame).unwrap());
            }
            pull_until_gate(&mut body, &mut text, &gate).await;
        })
        .await
        .expect("body must be readable before the closer is generated");
        gate.resume.notify_one();
        while let Some(frame) = body.next().await {
            text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        let events = responses_sse_json_events(&text);
        assert_eq!(text_deltas(&events, responses), format!("{first}{rest}"));
        assert!(arguments(&events, responses).is_empty());
        assert!(
            events.iter().all(|event| event["error"].is_null()),
            "{text}"
        );
    }
}

#[tokio::test]
async fn native_xml_required_stream_keeps_body_held_and_retains_terminal_error() {
    let gate = Arc::new(StreamGate::default());
    let (path, mut wire) = request(false, true);
    wire["tool_choice"] = json!("required");
    let response = post_json(
        router(StubLlm {
            stream_after_first_gate: Some(gate.clone()),
            ..StubLlm::with_separate_final_stream_chunk(&["Unacceptable prose.", "Still no call."])
        }),
        path,
        wire,
    )
    .await;
    let mut body = response.into_body().into_data_stream();
    let mut before_gate = String::new();
    tokio::time::timeout(
        Duration::from_secs(2),
        pull_until_gate(&mut body, &mut before_gate, &gate),
    )
    .await
    .unwrap();
    assert!(
        before_gate.is_empty(),
        "required output cannot leak before terminal validation"
    );
    assert!(
        body.next().now_or_never().is_none(),
        "required output cannot leak while the engine is gated"
    );
    gate.resume.notify_one();
    let mut text = String::new();
    while let Some(frame) = body.next().await {
        text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
    }
    let events = responses_sse_json_events(&text);
    assert!(text_deltas(&events, false).is_empty());
    assert!(
        events
            .iter()
            .any(|event| event["error"]["param"] == "tool_choice"),
        "{text}"
    );
}

#[tokio::test]
async fn native_xml_stream_length_and_engine_error_never_publish_executable_calls() {
    for fail in [false, true] {
        let gate = Arc::new(StreamGate::default());
        let first = "Safe prose. <tool_";
        let rest = "call><function=weather><parameter=city>Paris";
        let (path, wire) = request(false, true);
        let response = post_json(
            router(StubLlm {
                stream_after_first_gate: Some(gate.clone()),
                stream_terminal_error: fail,
                finish_reason: FinishReason::Length,
                ..StubLlm::with_separate_final_stream_chunk(&[first, rest])
            }),
            path,
            wire,
        )
        .await;
        let mut body = response.into_body().into_data_stream();
        let mut text = String::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while text_deltas(&responses_sse_json_events(&text), false) != "Safe prose. " {
                let frame = body.next().await.unwrap().unwrap();
                text.push_str(std::str::from_utf8(&frame).unwrap());
            }
            pull_until_gate(&mut body, &mut text, &gate).await;
        })
        .await
        .unwrap();
        gate.resume.notify_one();
        while let Some(frame) = body.next().await {
            text.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        let events = responses_sse_json_events(&text);
        assert!(arguments(&events, false).is_empty());
        if fail {
            assert!(
                events.iter().any(|event| !event["error"].is_null()),
                "{text}"
            );
            assert!(events
                .iter()
                .all(|event| event["choices"][0]["finish_reason"].is_null()));
            assert_eq!(text_deltas(&events, false), "Safe prose. ");
        } else {
            assert_eq!(text_deltas(&events, false), format!("{first}{rest}"));
            assert!(events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "length"));
            assert!(events
                .iter()
                .any(|event| event["usage"]["completion_tokens"] == 2));
        }
    }
}

#[tokio::test]
async fn native_xml_stream_terminal_moves_projection_credit_and_preserves_upstream_api() {
    use ferrum_output::{semantic_cost, OutputBudget};

    let local_city = "城市".repeat(1000);
    let raw = format!("<tool_call><function=weather><parameter=city>{local_city}</parameter></function></tool_call>");
    let limit = 15_000;
    for upstream_api in [false, true] {
        let (path, wire) = request(false, true);
        let openai_request: ChatCompletionsRequest = serde_json::from_value(wire.clone()).unwrap();
        let api_request = api_chat_request(
            &openai_request,
            openai_request.tool_choice.as_ref(),
            ferrum_types::ApiToolCallProtocol::FunctionParameterXml,
        );
        let projection = ferrum_types::project_native_chat_output(
            &api_request,
            ModelOutputProtocol::Text,
            false,
            &raw,
            FinishReason::EOS,
        )
        .unwrap();
        let original_projection_bytes =
            semantic_cost(&(&projection.visible.content, &projection.visible.reasoning)).unwrap()
                + semantic_cost(&projection.api_response).unwrap();
        let expected_city = if upstream_api {
            "都会".repeat(1000)
        } else {
            local_city.clone()
        };
        let mut expected_api = projection.api_response.clone().unwrap();
        if upstream_api {
            expected_api.message.tool_calls[0].id = "upstream_call".into();
            expected_api.message.tool_calls[0].function.arguments =
                serde_json::to_string(&json!({"city": expected_city})).unwrap();
        }
        let terminal_bytes =
            semantic_cost(&(&projection.visible.content, &projection.visible.reasoning)).unwrap()
                + semantic_cost(&Some(&expected_api)).unwrap();
        let token_bytes = semantic_cost(&vec![TokenId::new(11)]).unwrap();
        assert!(raw.len() + token_bytes < limit);
        let config = ferrum_types::OutputConfig {
            semantic_bytes_total: limit,
            semantic_bytes_per_session: limit,
            ..Default::default()
        };
        let budget = OutputBudget::new(config.clone()).unwrap();
        let mut stub = StubLlm::with_separate_final_stream_chunk(&[&raw]);
        stub.config.output = config;
        stub.output_budget = budget.clone();
        if upstream_api {
            stub.api_response = Some(ferrum_types::ApiResponse::Chat(expected_api.clone()));
        }
        let response = post_json(router(stub), path, wire).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let text = tokio::time::timeout(Duration::from_secs(2), response_text(response))
            .await
            .expect("moved terminal credit must not wait on its old owner");
        let events = responses_sse_json_events(&text);
        assert!(
            events.iter().all(|event| event["error"].is_null()),
            "{text}"
        );
        assert_eq!(text_deltas(&events, false), "");
        assert_eq!(
            serde_json::from_str::<Value>(&arguments(&events, false)).unwrap(),
            json!({"city": expected_city})
        );
        let tool_events: Vec<_> = events
            .iter()
            .filter(|event| event["choices"][0]["delta"]["tool_calls"].is_array())
            .collect();
        assert_eq!(tool_events.len(), 1);
        assert_eq!(
            tool_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        if upstream_api {
            assert_eq!(
                tool_events[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
                "upstream_call"
            );
        }
        let published: ChatCompletionsResponse =
            serde_json::from_value(tool_events[0].clone()).unwrap();
        let live_bytes = token_bytes + terminal_bytes + semantic_cost(&published).unwrap();
        assert!(
            live_bytes <= limit,
            "the actual terminal and wire DTO both fit"
        );
        assert!(
            live_bytes + original_projection_bytes > limit,
            "an unused second native projection would reject this real event"
        );
        let reasons: Vec<_> = events
            .iter()
            .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(reasons, ["tool_calls"]);
        let usages: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("usage").filter(|usage| !usage.is_null()))
            .collect();
        assert_eq!(
            usages,
            [&json!({"prompt_tokens":5,"completion_tokens":1,"total_tokens":6})]
        );
        assert_eq!(
            text.lines().filter(|line| *line == "data: [DONE]").count(),
            1
        );
        assert_eq!(budget.usage().semantic_bytes, 0);
        assert_eq!(budget.usage().queued_bytes, 0);
        assert_eq!(budget.usage().encoded_bytes, 0);
        assert_eq!(budget.usage().control_bytes, 0);
        assert_eq!(budget.usage().sessions, 0);
    }
}
