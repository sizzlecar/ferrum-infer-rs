//! Supported Harmony analysis-to-function handoff through the production engine.
//! Direct commentary calls without analysis remain a separate compatibility gap.
use super::*;

const REASONING: &str = "Need weather.";
const ARGUMENTS: &str = r#"{"city":"Paris"}"#;
const OUTPUT: &str = concat!(
    "<|channel|>analysis<|message|>Need weather.<|end|>",
    "<|start|>assistant<|channel|>commentary to=functions.weather",
    "<|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>"
);

fn tool(name: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }
        }
    })
}

struct ToolObservation {
    response: Observation,
    generated: Vec<TokenId>,
    tools: Value,
    choice: Value,
}

async fn generate(stream: bool, selected: &str) -> ToolObservation {
    generate_with_choice(
        stream,
        json!({"type": "function", "function": {"name": selected}}),
    )
    .await
}

async fn generate_with_choice(stream: bool, choice: Value) -> ToolObservation {
    let tokenizer = Arc::new(tokenizer(&[]).await);
    // Encode headers canonically: this BPE has no merges for channel names.
    let generated = tokenizer.encode(OUTPUT, false).unwrap();
    assert_eq!(generated.last(), tokenizer.token_id(CALL).as_ref());
    let mut script = generated.clone();
    script.extend(tokenizer.encode("unwanted continuation", false).unwrap());
    script.push(tokenizer.token_id(ORDINARY_EOS).unwrap());
    let max_tokens = script.len() + 4;
    let executor = Arc::new(ScriptedExecutor::new(tokenizer.vocab_size(), script));
    // Both tools are declared, so the negative case isolates the named selector.
    let tools = json!([tool("weather"), tool("calendar")]);
    let mut wire = json!({
        "model": "protocol-contract",
        "messages": [{"role": "user", "content": "Call the selected tool."}],
        "tools": tools,
        "tool_choice": choice,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if stream {
        wire["stream_options"] = json!({"include_usage": true});
    }
    let response = request_with_wire(
        ModelOutputProtocol::HarmonyGptOss,
        tokenizer,
        executor,
        wire,
    )
    .await;
    ToolObservation {
        response,
        generated,
        tools,
        choice,
    }
}

fn assert_engine_call_completed(observation: &ToolObservation) {
    let response = &observation.response;
    response.executor.assert_completed();
    assert_eq!(response.generated_tokens, observation.generated.len());
    assert_eq!(
        response.executor.decoded_inputs(),
        observation.generated[..observation.generated.len() - 1],
        "CALL must finish the engine before it consumes the scripted suffix"
    );
    assert_eq!(
        response.decoded_inputs.concat(),
        OUTPUT.strip_suffix(CALL).unwrap()
    );
}

fn assert_engine_handoff(observation: &ToolObservation) {
    assert_engine_call_completed(observation);
    let response = &observation.response;
    // Inspect actual model input, not an independently rendered expected prompt.
    // This tools-unaware template receives the production fallback system spec.
    let (spec, _) = response
        .prompt
        .strip_prefix("<|start|>system<|message|>")
        .expect("tools must reach the model through its prompt")
        .split_once("<|end|>")
        .unwrap();
    let spec: Value = serde_json::from_str(spec).unwrap();
    let actual_tools = spec["tools"].as_array().unwrap();
    let expected_tools = observation.tools.as_array().unwrap();
    assert_eq!(actual_tools.len(), expected_tools.len());
    for (actual, expected) in actual_tools.iter().zip(expected_tools) {
        assert_eq!(actual["type"], expected["type"]);
        assert_eq!(actual["function"]["name"], expected["function"]["name"]);
        assert_eq!(
            actual["function"]["parameters"],
            expected["function"]["parameters"]
        );
    }
    assert_eq!(spec["tool_choice"], observation.choice);
    assert!(response.prompt.ends_with("<|start|>assistant"));
}

fn assert_empty_text(value: &Value) {
    assert!(value.is_null() || value.as_str() == Some(""), "{value}");
}

fn assert_no_payload(delta: &Value) {
    for field in ["content", "reasoning", "reasoning_content"] {
        assert_empty_text(&delta[field]);
    }
    assert!(
        delta["tool_calls"].is_null() || delta["tool_calls"].as_array().is_some_and(Vec::is_empty)
    );
    assert!(delta["function_call"].is_null());
}

fn assert_usage(usage: &Value, response: &Observation) {
    assert_eq!(usage["prompt_tokens"], response.prompt_tokens);
    assert_eq!(usage["completion_tokens"], response.generated_tokens);
    assert_eq!(
        usage["total_tokens"],
        response.prompt_tokens + response.generated_tokens
    );
}

fn assert_sync_handoff(response: &Observation) {
    let body: Value = serde_json::from_str(&response.body).unwrap();
    assert!(body["error"].is_null(), "{body}");
    assert_eq!(body["choices"].as_array().unwrap().len(), 1);
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let message = &choice["message"];
    assert_eq!(message["role"], "assistant");
    assert_empty_text(&message["content"]);
    assert_eq!(message["reasoning"], REASONING);
    assert_empty_text(&message["reasoning_content"]);
    assert!(message["function_call"].is_null());
    let calls = message["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert!(!call["id"].as_str().unwrap().is_empty());
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "weather");
    assert_eq!(call["function"]["arguments"], ARGUMENTS);
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments, json!({"city": "Paris"}));
    assert_usage(&body["usage"], response);
}

fn assert_sse_handoff(response: &Observation) {
    let events = sse_events(&response.body);
    let mut reasoning = String::new();
    let mut call_id = None;
    let mut function_type = false;
    let mut name = String::new();
    let mut arguments = String::new();
    let mut terminal = None;
    let mut usage = None;
    for (index, event) in events.iter().enumerate() {
        assert!(event["error"].is_null(), "{}", response.body);
        if !event["usage"].is_null() {
            assert!(usage.replace(index).is_none(), "duplicate usage");
            assert!(terminal.is_some_and(|terminal| terminal < index));
            assert_eq!(event["choices"], json!([]));
            assert_usage(&event["usage"], response);
        }
        let choices = event["choices"].as_array().unwrap();
        if event["usage"].is_null() {
            assert_eq!(choices.len(), 1);
        }
        for choice in choices {
            assert_eq!(choice["index"], 0);
            assert!(choice["message"].is_null());
            let delta = &choice["delta"];
            assert!(delta.is_object());
            assert_empty_text(&delta["content"]);
            assert_empty_text(&delta["reasoning_content"]);
            assert!(delta["function_call"].is_null());
            if terminal.is_some() || !choice["finish_reason"].is_null() {
                assert_no_payload(delta);
            }
            if !choice["finish_reason"].is_null() {
                assert_eq!(choice["finish_reason"], "tool_calls");
                assert!(terminal.replace(index).is_none(), "duplicate terminal");
            }
            if let Some(piece) = delta["reasoning"].as_str() {
                reasoning.push_str(piece);
                assert!(REASONING.starts_with(&reasoning));
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    assert_eq!(call["index"], 0);
                    if let Some(id) = call["id"].as_str().filter(|id| !id.is_empty()) {
                        if let Some(previous) = call_id {
                            assert_eq!(id, previous);
                        } else {
                            call_id = Some(id);
                        }
                    }
                    if let Some(kind) = call["type"].as_str().filter(|kind| !kind.is_empty()) {
                        assert_eq!(kind, "function");
                        function_type = true;
                    }
                    if let Some(piece) = call["function"]["name"].as_str() {
                        name.push_str(piece);
                        assert!("weather".starts_with(&name));
                    }
                    if let Some(piece) = call["function"]["arguments"].as_str() {
                        arguments.push_str(piece);
                        assert!(ARGUMENTS.starts_with(&arguments));
                    }
                }
            }
        }
    }
    assert!(terminal.is_some());
    assert_eq!(usage, Some(events.len() - 1));
    assert_eq!(reasoning, REASONING);
    assert!(call_id.is_some());
    assert!(function_type);
    assert_eq!(name, "weather");
    assert_eq!(arguments, ARGUMENTS);
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"city": "Paris"})
    );
}

#[tokio::test]
async fn harmony_function_handoff_reaches_sync_and_sse() {
    for stream in [false, true] {
        let observation = generate(stream, "weather").await;
        let response = &observation.response;
        assert_eq!(response.status, AxumStatusCode::OK, "{}", response.body);
        if stream {
            assert_sse_handoff(response);
        } else {
            assert_sync_handoff(response);
        }
        assert_engine_handoff(&observation);
    }
}

fn assert_handoff_rejected(response: &Observation, stream: bool) {
    if stream {
        assert_eq!(response.status, AxumStatusCode::OK, "{}", response.body);
        let events = sse_events(&response.body);
        let errors: Vec<_> = events
            .iter()
            .filter(|event| !event["error"].is_null())
            .collect();
        assert_eq!(errors.len(), 1, "{}", response.body);
        assert_eq!(errors[0]["error"]["type"], "internal_server_error");
        assert_eq!(errors[0]["error"]["param"], "tool_choice");
        for event in &events {
            assert!(event["usage"].is_null());
            if let Some(choices) = event["choices"].as_array() {
                for choice in choices {
                    assert!(choice["finish_reason"].is_null());
                    assert_no_payload(&choice["delta"]);
                    assert_no_payload(&choice["message"]);
                }
            }
        }
    } else {
        assert_eq!(
            response.status,
            AxumStatusCode::INTERNAL_SERVER_ERROR,
            "{}",
            response.body
        );
        let body: Value = serde_json::from_str(&response.body).unwrap();
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(body["error"]["param"].is_null());
        assert!(body["choices"].is_null());
        assert!(body["usage"].is_null());
    }
}

#[tokio::test]
async fn harmony_function_handoff_rejects_a_different_named_choice() {
    for stream in [false, true] {
        let observation = generate(stream, "calendar").await;
        let response = &observation.response;
        assert_handoff_rejected(response, stream);
        // The engine completes the same valid handoff before HTTP rejects its
        // disagreement with the named selector; this is not a model failure.
        assert_engine_handoff(&observation);
    }
}

#[tokio::test]
async fn harmony_final_json_cannot_impersonate_a_required_tool_call() {
    for stream in [false, true] {
        let router = router_with_stub_and_template(
            "<|channel|>final<|message|>{\"city\":\"Paris\"}<|return|>",
            template(ModelOutputProtocol::HarmonyGptOss),
        );
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Call the selected tool."}],
                "tools": [tool("weather")],
                "tool_choice": {"type": "function", "function": {"name": "weather"}},
                "stream": stream,
            }),
        )
        .await;
        if stream {
            assert_eq!(response.status(), AxumStatusCode::OK);
            let body = response_text(response).await;
            let events = sse_events(&body);
            assert_eq!(events.len(), 1, "{body}");
            assert_eq!(events[0]["error"]["type"], "invalid_request_error");
            assert_eq!(events[0]["error"]["param"], "tool_choice");
            assert!(events[0]["choices"].is_null());
        } else {
            assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
            let body = response_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["param"], "tool_choice");
            assert!(body["choices"].is_null());
        }
    }
}

async fn assert_tool_choice_none_rejects_native_call(stream: bool) {
    let observation = generate_with_choice(stream, json!("none")).await;
    let response = &observation.response;
    assert_handoff_rejected(response, stream);
    assert!(!response.body.contains(REASONING), "{}", response.body);
    assert!(!response.body.contains("<|"), "{}", response.body);
    // Request policy is enforced after this valid native call completes. It
    // must not depend on whether the template still exposes tool definitions.
    assert_engine_call_completed(&observation);
}

#[tokio::test]
async fn harmony_tool_choice_none_rejects_native_call_sync() {
    assert_tool_choice_none_rejects_native_call(false).await;
}

#[tokio::test]
async fn harmony_tool_choice_none_rejects_native_call_sse() {
    assert_tool_choice_none_rejects_native_call(true).await;
}

#[tokio::test]
async fn harmony_tool_choice_none_accepts_native_final_sync_and_sse() {
    const CONTENT: &str = "No tool is needed.";
    for stream in [false, true] {
        let response = post_json(
            router_with_stub_and_template(
                "<|channel|>final<|message|>No tool is needed.<|return|>",
                template(ModelOutputProtocol::HarmonyGptOss),
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Answer without tools."}],
                "tools": [tool("weather")],
                "tool_choice": "none",
                "stream": stream,
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        if stream {
            let body = response_text(response).await;
            let events = sse_events(&body);
            let mut content = String::new();
            let mut terminal = false;
            for event in &events {
                assert!(event["error"].is_null(), "{body}");
                let choices = event["choices"].as_array().unwrap();
                assert_eq!(choices.len(), 1);
                let choice = &choices[0];
                assert_eq!(choice["index"], 0);
                assert!(choice["message"].is_null());
                let delta = &choice["delta"];
                assert!(delta.is_object());
                assert_empty_text(&delta["reasoning"]);
                assert_empty_text(&delta["reasoning_content"]);
                assert!(delta["tool_calls"].is_null());
                assert!(delta["function_call"].is_null());
                if terminal || !choice["finish_reason"].is_null() {
                    assert_no_payload(delta);
                }
                if !choice["finish_reason"].is_null() {
                    assert!(!terminal, "duplicate terminal: {body}");
                    assert_eq!(choice["finish_reason"], "stop");
                    terminal = true;
                }
                if let Some(piece) = delta["content"].as_str() {
                    content.push_str(piece);
                    assert!(CONTENT.starts_with(&content), "{body}");
                }
            }
            assert!(terminal, "{body}");
            assert_eq!(content, CONTENT);
        } else {
            let body = response_json(response).await;
            assert!(body["error"].is_null(), "{body}");
            assert_eq!(body["choices"].as_array().unwrap().len(), 1);
            let choice = &body["choices"][0];
            assert_eq!(choice["finish_reason"], "stop");
            let message = &choice["message"];
            assert_eq!(message["role"], "assistant");
            assert_eq!(message["content"], CONTENT);
            assert_empty_text(&message["reasoning"]);
            assert_empty_text(&message["reasoning_content"]);
            assert!(message["tool_calls"].is_null());
            assert!(message["function_call"].is_null());
        }
    }
}
