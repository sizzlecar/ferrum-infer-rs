use super::*;
use ferrum_types::{GEMMA_THOUGHT_END_TAG as CLOSE, GEMMA_THOUGHT_START_TAG as OPEN};

fn template() -> ModelChatTemplate {
    // The canonical Gemma template has four generation starts depending on
    // thinking and whether the last message is a tool result.
    let source = concat!(
        "{% set enable_thinking = enable_thinking | default(false) %}",
        "{% if enable_thinking %}<|turn>system\n<|think|>\n<turn|>\n{% endif %}",
        "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}",
        "{% if add_generation_prompt %}",
        "{% if messages[-1].role == 'tool' %}",
        "<|turn>model\n<|tool_response>{{ messages[-1].content }}<tool_response|>",
        "{% if enable_thinking %}<|channel>thought\n{% endif %}",
        "{% else %}<|turn>model\n",
        "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}",
        "{% endif %}{% endif %}",
    );
    let mut template = ModelChatTemplate::new(source, "native-thought-template");
    template.set_output_protocol(ModelOutputProtocol::GemmaThought);
    template
}

fn request(tool_result: bool, thinking: bool, stream: bool) -> Value {
    let mut request = json!({
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Add 123 and 456."}],
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
        "max_tokens": 128
    });
    if tool_result {
        request["messages"] = json!([
            {"role": "user", "content": "Add 123 and 456."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "sum", "type": "function", "function": {
                    "name": "calculate", "arguments": "{\"expression\":\"123+456\"}"
                }
            }]},
            {"role": "tool", "tool_call_id": "sum", "content": "579"}
        ]);
        request["tools"] = json!([{"type": "function", "function": {
            "name": "calculate", "parameters": {"type": "object", "properties": {
                "expression": {"type": "string"}
            }, "required": ["expression"]}
        }}]);
        request["tool_choice"] = json!("auto");
    }
    request
}

fn delta_text(events: &[Value], field: &str) -> String {
    events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"][field].as_str())
        .collect()
}

#[test]
fn gemma_structured_requests_follow_all_four_native_generation_starts() {
    for tool_result in [false, true] {
        for thinking in [false, true] {
            let mut wire = request(tool_result, thinking, false);
            // A completed tool round may request a final JSON answer without
            // advertising more tools. This isolates the JSON content contract.
            wire.as_object_mut().unwrap().remove("tools");
            wire.as_object_mut().unwrap().remove("tool_choice");
            wire["response_format"] = json!({"type": "json_object"});
            let wire: ChatCompletionsRequest = serde_json::from_value(wire).unwrap();
            let internal =
                convert_chat_request_with_template_model(&wire, "loaded-model", Some(&template()))
                    .unwrap();
            let expected = match (tool_result, thinking) {
                (false, false) => StructuredOutputStart::Immediate,
                (true, true) => StructuredOutputStart::AfterDelimiter(CLOSE.to_string()),
                _ => StructuredOutputStart::AfterReasoningEnvelope {
                    opening: OPEN.to_string(),
                    closing: CLOSE.to_string(),
                    allow_reasoning: thinking,
                },
            };
            assert_eq!(internal.sampling_params.structured_output_start, expected);
            assert_eq!(
                internal.sampling_params.model_output_protocol,
                ModelOutputProtocol::GemmaThought
            );
            assert_eq!(
                internal.sampling_params.response_completion_boundary,
                if tool_result || thinking {
                    ResponseCompletionBoundary::AfterDelimiterAndPayload {
                        delimiter: CLOSE.to_string(),
                        alternate_envelope: None,
                    }
                } else {
                    ResponseCompletionBoundary::Immediate
                }
            );
            let forbidden = internal
                .metadata
                .get(INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY);
            assert!(
                !forbidden.is_some_and(|value| value
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|token| token == "<|channel>" || token == OPEN)),
                "disabled thinking must still permit the native empty header"
            );
            internal.sampling_params.validate().unwrap();
        }
    }
}

#[tokio::test]
async fn gemma_routes_separate_native_thought_for_plain_and_tool_result_turns() {
    for (tool_result, thinking, output, reasoning) in [
        (false, false, "579", ""),
        (
            false,
            true,
            "<|channel>thought\nCompute.<channel|>579",
            "Compute.",
        ),
        (true, false, "<|channel>thought\n<channel|>579", ""),
        (true, true, "Use result.<channel|>579", "Use result."),
    ] {
        for stream in [false, true] {
            let chunks: Vec<_> = output
                .char_indices()
                .map(|(start, ch)| &output[start..start + ch.len_utf8()])
                .collect();
            let router = AxumServer::from_llm(Arc::new(StubLlm::with_stream_chunks(&chunks)))
                .with_prompt_template(Some(template()))
                .build_router();
            let response = post_json(
                router,
                "/v1/chat/completions",
                request(tool_result, thinking, stream),
            )
            .await;
            assert_eq!(response.status(), AxumStatusCode::OK);
            if stream {
                let body = response_text(response).await;
                let events = responses_sse_json_events(&body);
                assert!(
                    events.iter().all(|event| event.get("error").is_none()),
                    "{body}"
                );
                assert_eq!(delta_text(&events, "content"), "579");
                assert_eq!(delta_text(&events, "reasoning"), reasoning);
                assert!(events
                    .iter()
                    .any(|event| event["choices"][0]["finish_reason"] == "stop"));
                assert!(body.contains("data: [DONE]"));
            } else {
                let body = response_json(response).await;
                assert_eq!(body["choices"][0]["message"]["content"], "579");
                assert_eq!(
                    body["choices"][0]["message"]["reasoning"]
                        .as_str()
                        .unwrap_or(""),
                    reasoning
                );
            }
        }
    }
}

#[tokio::test]
async fn gemma_plain_stream_emits_content_deltas_before_completion() {
    let engine = StubLlm::with_separate_final_stream_chunk(&[
        "<|chan",
        "nel>thought",
        "\nCompute.",
        "<chan",
        "nel|>",
        "5",
        "7",
        "9",
    ]);
    let router = AxumServer::from_llm(Arc::new(engine))
        .with_prompt_template(Some(template()))
        .build_router();
    let response = post_json(router, "/v1/chat/completions", request(false, true, true)).await;
    let body = response_text(response).await;
    let events = responses_sse_json_events(&body);
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .filter(|text| !text.is_empty())
        .collect();
    assert_eq!(deltas, ["5", "7", "9"], "{body}");
    assert_eq!(delta_text(&events, "reasoning"), "Compute.");
    assert!(events
        .iter()
        .any(|event| event["choices"][0]["finish_reason"] == "stop"));
}

#[tokio::test]
async fn gemma_stream_flushes_visible_tail_when_final_chunk_has_partial_framing() {
    for chunks in [
        vec!["579", "<|chan"],
        vec!["579<|chan"],
        vec!["57", "9<|chan"],
    ] {
        let router = AxumServer::from_llm(Arc::new(StubLlm::with_stream_chunks(&chunks)))
            .with_prompt_template(Some(template()))
            .build_router();
        let response = post_json(router, "/v1/chat/completions", request(false, false, true)).await;
        let body = response_text(response).await;
        let events = responses_sse_json_events(&body);
        assert_eq!(delta_text(&events, "content"), "579");
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "stop"),
            "{body}"
        );
        assert!(body.contains("data: [DONE]"));
    }
}

#[tokio::test]
async fn gemma_strict_json_validates_visible_payload_after_native_reasoning() {
    for stream in [false, true] {
        for output in [
            "<|channel>thought\nCompute.<channel|>{\"answer\":42}",
            "<|channel>thought\n<channel|>{\"answer\":42}",
        ] {
            let mut wire = harmony_json_request(stream);
            wire["chat_template_kwargs"] = json!({"enable_thinking": true});
            let response = post_json(
                router_with_stub_and_template(output, template()),
                "/v1/chat/completions",
                wire,
            )
            .await;
            assert_eq!(response.status(), AxumStatusCode::OK);
            let content = if stream {
                let body = response_text(response).await;
                let events = responses_sse_json_events(&body);
                assert!(
                    events.iter().all(|event| event.get("error").is_none()),
                    "{body}"
                );
                assert!(body.contains("data: [DONE]"));
                delta_text(&events, "content")
            } else {
                response_json(response).await["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap()
                    .to_string()
            };
            assert_eq!(
                serde_json::from_str::<Value>(&content).unwrap(),
                json!({"answer":42})
            );
        }
    }
}

#[tokio::test]
async fn gemma_routes_reject_unknown_channel_without_marker_leaks() {
    for stream in [false, true] {
        let response = post_json(
            router_with_stub_and_template("<|channel>final\nsecret<channel|>579", template()),
            "/v1/chat/completions",
            request(false, true, stream),
        )
        .await;
        if stream {
            let body = response_text(response).await;
            let events = responses_sse_json_events(&body);
            assert!(events.iter().any(|event| event.get("error").is_some()));
            assert!(delta_text(&events, "content").is_empty());
            assert!(delta_text(&events, "reasoning").is_empty());
            assert!(!body.contains("secret") && !body.contains("<|channel>"));
            assert!(body.contains("data: [DONE]"));
        } else {
            assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
        }
    }
}
