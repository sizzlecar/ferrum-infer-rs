//! Token-limited automatic tool responses retain their actual terminal evidence.
use super::*;

const REASONING: &str = "I still need to inspect the source.";

fn router(finish_reason: FinishReason, typed_response: bool) -> Router {
    let mut template = ModelChatTemplate::new(
        concat!(
            "{% if tools %}Tools: {{ tools | tojson }}{% endif %}",
            "{% for message in messages %}{{ message.content }}{% endfor %}",
            "{% if add_generation_prompt %}<assistant><think>{% endif %}",
        ),
        "tool-length-contract",
    );
    assert_eq!(
        template.reasoning_protocol,
        ModelReasoningProtocol::PromptOpened
    );
    template.tool_call_protocol = ferrum_types::ApiToolCallProtocol::FunctionParameterXml;
    AxumServer::from_llm(Arc::new(StubLlm {
        finish_reason,
        api_response: typed_response.then(weather_tool_api_response),
        // The terminal chunk has no text; buffered reasoning must still flush.
        ..StubLlm::with_separate_final_stream_chunk(&[REASONING])
    }))
    .with_prompt_template(Some(template))
    .build_router()
}

fn request(responses: bool, stream: bool) -> (&'static str, Value) {
    let function = json!({
        "name": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    });
    let mut body = if responses {
        json!({
            "model": "stub-model", "input": "Use a tool if needed.",
            "tools": [{"type": "function", "name": function["name"], "parameters": function["parameters"]}],
        })
    } else {
        json!({
            "model": "stub-model", "messages": [{"role": "user", "content": "Use a tool if needed."}],
            "tools": [{"type": "function", "function": function}],
        })
    };
    body["stream"] = json!(stream);
    body["tool_choice"] = json!("auto");
    if responses {
        body["reasoning"] = json!({"effort": "medium"});
    } else {
        body["chat_template_kwargs"] = json!({"enable_thinking": true});
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }
    }
    (
        if responses {
            "/v1/responses"
        } else {
            "/v1/chat/completions"
        },
        body,
    )
}

#[tokio::test]
async fn auto_tools_length_preserves_reasoning_and_usage_in_both_endpoints() {
    for responses in [false, true] {
        for stream in [false, true] {
            let (path, body) = request(responses, stream);
            let response = post_json(router(FinishReason::Length, false), path, body).await;
            let status = response.status();
            let text = response_text(response).await;
            eprintln!("checking {path}, stream={stream}, HTTP {status}");
            assert_eq!(
                status,
                AxumStatusCode::OK,
                "{path}, stream={stream}: {text}"
            );
            let events = if stream {
                responses_sse_json_events(&text)
            } else {
                vec![serde_json::from_str::<Value>(&text).unwrap()]
            };
            assert!(
                events.iter().all(|event| event["error"].is_null()),
                "{text}"
            );
            let (input, output) = if stream { (5, 1) } else { (7, 2) };
            if responses {
                let result = if stream {
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| event["type"] == "response.incomplete")
                            .count(),
                        1
                    );
                    assert!(
                        !events.iter().any(|event| event["type"] == "response.failed"
                            || event["type"] == "response.completed")
                    );
                    let deltas: String = events
                        .iter()
                        .filter(|event| event["type"] == "response.reasoning_text.delta")
                        .map(|event| event["delta"].as_str().unwrap())
                        .collect();
                    assert_eq!(deltas, REASONING);
                    &events.last().unwrap()["response"]
                } else {
                    &events[0]
                };
                assert_eq!(result["status"], "incomplete");
                assert_eq!(result["incomplete_details"]["reason"], "max_output_tokens");
                assert_eq!(result["output"].as_array().unwrap().len(), 1);
                assert_eq!(result["output"][0]["type"], "reasoning");
                assert_eq!(result["output"][0]["status"], "incomplete");
                assert_eq!(result["output"][0]["content"][0]["text"], REASONING);
                assert_eq!(result["usage"]["input_tokens"], input);
                assert_eq!(result["usage"]["output_tokens"], output);
                assert_eq!(result["usage"]["total_tokens"], input + output);
            } else {
                let messages = if stream {
                    assert_eq!(text.matches("data: [DONE]").count(), 1);
                    let finishes: Vec<_> = events
                        .iter()
                        .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
                        .collect();
                    assert_eq!(finishes, ["length"]);
                    events
                        .iter()
                        .filter_map(|event| event["choices"][0].get("delta"))
                        .collect::<Vec<_>>()
                } else {
                    assert_eq!(events[0]["choices"][0]["finish_reason"], "length");
                    vec![&events[0]["choices"][0]["message"]]
                };
                let reasoning: String = messages
                    .iter()
                    .filter_map(|message| message["reasoning"].as_str())
                    .collect();
                assert_eq!(reasoning, REASONING);
                assert!(messages.iter().all(|message| message["content"]
                    .as_str()
                    .unwrap_or_default()
                    .is_empty()
                    && message["tool_calls"].is_null()
                    && message["function_call"].is_null()));
                let usages: Vec<_> = events
                    .iter()
                    .filter_map(|event| event.get("usage").filter(|value| !value.is_null()))
                    .collect();
                assert_eq!(usages.len(), 1);
                assert_eq!(usages[0]["prompt_tokens"], input);
                assert_eq!(usages[0]["completion_tokens"], output);
                assert_eq!(usages[0]["total_tokens"], input + output);
            }
        }
    }
}

#[tokio::test]
async fn auto_tools_length_does_not_publish_a_typed_call_from_an_incomplete_turn() {
    let (path, body) = request(false, true);
    let response = post_json(router(FinishReason::Length, true), path, body).await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let text = response_text(response).await;
    let events = responses_sse_json_events(&text);
    assert!(
        events.iter().all(|event| event["error"].is_null()),
        "{text}"
    );
    assert!(events
        .iter()
        .all(|event| event["choices"][0]["delta"]["tool_calls"].is_null()));
    assert_eq!(
        events
            .iter()
            .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
            .collect::<Vec<_>>(),
        ["length"]
    );
    assert_eq!(events.last().unwrap()["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn length_keeps_required_tool_and_strict_content_contracts() {
    for stream in [false, true] {
        for required_tool in [false, true] {
            let (path, mut body) = request(false, stream);
            let expected_param = if required_tool {
                body["tool_choice"] = json!("required");
                "tool_choice"
            } else {
                body["response_format"] = json!({"type": "json_schema", "json_schema": {
                    "name": "answer", "strict": true, "schema": {
                        "type": "object", "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"], "additionalProperties": false,
                    },
                }});
                "response_format.json_schema"
            };
            let response = post_json(router(FinishReason::Length, false), path, body).await;
            let status = response.status();
            let text = response_text(response).await;
            eprintln!(
                "checking {path}, stream={stream}, required_tool={required_tool}, HTTP {status}"
            );
            if stream {
                assert_eq!(
                    status,
                    AxumStatusCode::OK,
                    "required_tool={required_tool}: {text}"
                );
                assert_eq!(text.matches("data: [DONE]").count(), 1);
                let events = responses_sse_json_events(&text);
                assert!(
                    events
                        .iter()
                        .any(|event| event["error"]["param"] == expected_param),
                    "{text}"
                );
                assert!(
                    events.iter().all(|event| event["choices"].is_null()),
                    "{text}"
                );
            } else {
                assert!(
                    !status.is_success(),
                    "required_tool={required_tool}: {text}"
                );
                let body: Value = serde_json::from_str(&text).expect("error response JSON");
                if required_tool {
                    assert_eq!(body["error"]["param"], expected_param, "{body}");
                } else {
                    assert_eq!(body["error"]["type"], "internal_server_error", "{body}");
                    assert!(
                        body["error"]["message"]
                            .as_str()
                            .unwrap()
                            .contains("response_format.json_schema.strict"),
                        "{body}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn empty_auto_tool_completed_turn_keeps_its_existing_error() {
    let (path, body) = request(false, true);
    let response = post_json(router(FinishReason::EOS, false), path, body).await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let text = response_text(response).await;
    assert_openai_stream_error(
        &text,
        "model output did not satisfy tool/function call request",
    );
    assert_eq!(text.matches("data: [DONE]").count(), 1);
}
