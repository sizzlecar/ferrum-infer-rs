//! Function strictness governs automatic tool arguments independently of transport.
//! Model output is scripted; these tests exercise the real HTTP request/response
//! adapters and client tool-result replay, not model coding ability.
use super::*;

const MALFORMED_EDITS: &str = r#"[{"old":"before","new":"after"}]}"#;

fn edit_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["old", "new"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["path", "edits"],
        "additionalProperties": false
    })
}

fn malformed_arguments() -> String {
    json!({"path": "src/lib.rs", "edits": MALFORMED_EDITS}).to_string()
}

fn corrected_arguments() -> String {
    json!({"path": "src/lib.rs", "edits": [{"old": "before", "new": "after"}]}).to_string()
}

fn tool_request(strict: Option<bool>, choice: Option<Value>, stream: bool) -> Value {
    let mut function = json!({"name": "apply_edits", "parameters": edit_schema()});
    if let Some(strict) = strict {
        function["strict"] = json!(strict);
    }
    let mut request = json!({
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Update src/lib.rs with the edit tool."}],
        "tools": [{"type": "function", "function": function}],
        "stream": stream
    });
    if let Some(choice) = choice {
        request["tool_choice"] = choice;
    }
    request
}

fn tool_response(arguments: &str) -> ferrum_types::ApiResponse {
    let mut response = weather_tool_api_response();
    let ferrum_types::ApiResponse::Chat(chat) = &mut response else {
        unreachable!("chat fixture")
    };
    chat.message.tool_calls[0].function.name = "apply_edits".into();
    chat.message.tool_calls[0].function.arguments = arguments.into();
    response
}

async fn accepted_tool_call(response: Response, stream: bool) -> Value {
    let status = response.status();
    let body = response_text(response).await;
    assert_eq!(status, AxumStatusCode::OK, "{body}");
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        assert!(response["error"].is_null(), "{body}");
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        let calls = response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1, "{body}");
        return calls[0].clone();
    }
    assert!(body.contains("data: [DONE]"), "{body}");
    let mut call = json!({
        "id": "", "type": "function", "function": {"name": "", "arguments": ""}
    });
    let mut finish = None;
    for event in responses_sse_json_events(&body) {
        assert!(event["error"].is_null(), "{body}");
        for choice in event["choices"].as_array().into_iter().flatten() {
            if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
                for delta in calls {
                    assert_eq!(delta["index"], 0, "{body}");
                    if let Some(id) = delta["id"].as_str() {
                        call["id"] = json!(format!("{}{id}", call["id"].as_str().unwrap()));
                    }
                    if let Some(kind) = delta["type"].as_str() {
                        assert_eq!(kind, "function");
                    }
                    for field in ["name", "arguments"] {
                        if let Some(piece) = delta["function"][field].as_str() {
                            call["function"][field] = json!(format!(
                                "{}{piece}",
                                call["function"][field].as_str().unwrap()
                            ));
                        }
                    }
                }
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                assert!(finish.replace(reason.to_string()).is_none(), "{body}");
            }
        }
    }
    assert_eq!(finish.as_deref(), Some("tool_calls"), "{body}");
    assert!(!call["id"].as_str().unwrap().is_empty(), "{body}");
    call
}

async fn rejected_tool_call(response: Response, stream: bool, diagnostic: &str) {
    let status = response.status();
    let body = response_text(response).await;
    if !stream {
        assert_eq!(status, AxumStatusCode::INTERNAL_SERVER_ERROR, "{body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "internal_server_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains(diagnostic),
            "{body}"
        );
        assert!(error["choices"].is_null(), "{body}");
        return;
    }
    assert_eq!(status, AxumStatusCode::OK, "{body}");
    assert!(body.contains("data: [DONE]"), "{body}");
    let events = responses_sse_json_events(&body);
    let errors: Vec<_> = events
        .iter()
        .filter(|event| !event["error"].is_null())
        .collect();
    assert_eq!(errors.len(), 1, "{body}");
    assert_eq!(errors[0]["error"]["type"], "internal_server_error");
    assert!(
        errors[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains(diagnostic),
        "{body}"
    );
    for event in &events {
        for choice in event["choices"].as_array().into_iter().flatten() {
            assert!(choice["delta"]["tool_calls"].is_null(), "{body}");
            assert!(choice["finish_reason"].is_null(), "{body}");
        }
    }
}

#[tokio::test]
async fn automatic_tool_schema_validation_obeys_declared_strictness_in_sync_and_sse() {
    let arguments = malformed_arguments();
    for choice in [None, Some(json!("auto"))] {
        for strict in [None, Some(false), Some(true)] {
            for stream in [false, true] {
                let response = post_json(
                    router_with_stub_api_response("", tool_response(&arguments)),
                    "/v1/chat/completions",
                    tool_request(strict, choice.clone(), stream),
                )
                .await;
                if strict == Some(true) {
                    rejected_tool_call(response, stream, "did not satisfy its schema").await;
                } else {
                    let call = accepted_tool_call(response, stream).await;
                    assert_eq!(call["function"]["name"], "apply_edits");
                    assert_eq!(call["function"]["arguments"], arguments);
                    let value: Value =
                        serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                            .unwrap();
                    assert_eq!(value["edits"], MALFORMED_EDITS);
                }
            }
        }
    }
}

#[tokio::test]
async fn strict_automatic_tools_accept_valid_arguments_in_sync_and_sse() {
    let arguments = corrected_arguments();
    for stream in [false, true] {
        let response = post_json(
            router_with_stub_api_response("", tool_response(&arguments)),
            "/v1/chat/completions",
            tool_request(Some(true), Some(json!("auto")), stream),
        )
        .await;
        let call = accepted_tool_call(response, stream).await;
        assert_eq!(call["function"]["arguments"], arguments);
    }
}

#[tokio::test]
async fn native_xml_invalid_array_remains_intact_for_non_strict_tools() {
    let template = ModelChatTemplate::new(
        "{% if tools %}<tool_call><function=name><parameter=key>value</parameter></function></tool_call>{% endif %}{% for message in messages %}{{ message.content }}{% endfor %}",
        "function-parameter-xml-template",
    );
    let output = format!(
        "<tool_call><function=apply_edits><parameter=path>src/lib.rs</parameter><parameter=edits>{MALFORMED_EDITS}</parameter></function></tool_call>"
    );
    for strict in [None, Some(false), Some(true)] {
        for stream in [false, true] {
            let response = post_json(
                router_with_stub_and_template(&output, template.clone()),
                "/v1/chat/completions",
                tool_request(strict, Some(json!("auto")), stream),
            )
            .await;
            if strict == Some(true) {
                rejected_tool_call(response, stream, "did not satisfy its schema").await;
            } else {
                let call = accepted_tool_call(response, stream).await;
                let arguments: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
                assert_eq!(arguments["path"], "src/lib.rs");
                assert_eq!(arguments["edits"], MALFORMED_EDITS);
            }
        }
    }
}

#[tokio::test]
async fn non_strict_tools_preserve_json_envelope_and_selection_constraints() {
    for stream in [false, true] {
        for (arguments, emitted_name, emitted_type, choice, diagnostic) in [
            (
                "{",
                "apply_edits",
                "function",
                "auto",
                "invalid JSON arguments",
            ),
            (
                "[]",
                "apply_edits",
                "function",
                "auto",
                "non-object arguments",
            ),
            (
                "{}",
                "undeclared_edit",
                "function",
                "auto",
                "undeclared tool call",
            ),
            (
                "{}",
                "apply_edits",
                "other",
                "auto",
                "unsupported tool call type",
            ),
            (
                "{}",
                "apply_edits",
                "function",
                "none",
                "tool_choice is 'none'",
            ),
        ] {
            let mut output = tool_response(arguments);
            let ferrum_types::ApiResponse::Chat(chat) = &mut output else {
                unreachable!("chat fixture")
            };
            chat.message.tool_calls[0].function.name = emitted_name.into();
            chat.message.tool_calls[0].tool_type = emitted_type.into();
            let response = post_json(
                router_with_stub_api_response("", output),
                "/v1/chat/completions",
                tool_request(Some(false), Some(json!(choice)), stream),
            )
            .await;
            rejected_tool_call(response, stream, diagnostic).await;
        }
    }
}

#[tokio::test]
async fn required_and_forced_tools_keep_their_existing_argument_contract() {
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "apply_edits"}}),
    ] {
        for stream in [false, true] {
            let response = post_json(
                router_with_stub_api_response("", tool_response(&malformed_arguments())),
                "/v1/chat/completions",
                tool_request(Some(false), Some(choice.clone()), stream),
            )
            .await;
            rejected_tool_call(response, stream, "did not satisfy its schema").await;
        }
    }
}

struct ToolFeedbackLlm {
    base: StubLlm,
    requests: Mutex<Vec<InferenceRequest>>,
}

impl ToolFeedbackLlm {
    fn new() -> Self {
        Self {
            base: StubLlm::new(""),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn reply(&self, request: &InferenceRequest) -> StubLlm {
        self.requests.lock().unwrap().push(request.clone());
        let has_feedback = matches!(
            request.api_request.as_ref(),
            Some(ferrum_types::ApiRequest::Chat(chat))
                if chat.messages.last().is_some_and(|message|
                    message.role == ferrum_types::ApiMessageRole::Tool)
        );
        let arguments = if has_feedback {
            corrected_arguments()
        } else {
            malformed_arguments()
        };
        StubLlm::with_api_response("", tool_response(&arguments))
    }
}

#[async_trait]
impl InferenceEngine for ToolFeedbackLlm {
    async fn status(&self) -> EngineStatus {
        self.base.status().await
    }

    async fn shutdown(&self) -> ferrum_types::Result<()> {
        self.base.shutdown().await
    }

    fn config(&self) -> &EngineConfig {
        self.base.config()
    }

    fn metrics(&self) -> EngineMetrics {
        self.base.metrics()
    }

    async fn health_check(&self) -> EngineHealthStatus {
        self.base.health_check().await
    }
}

#[async_trait]
impl LlmInferenceEngine for ToolFeedbackLlm {
    async fn infer(&self, request: InferenceRequest) -> ferrum_types::Result<InferenceResponse> {
        self.reply(&request).infer(request).await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>>
    {
        self.reply(&request).infer_stream(request).await
    }
}

#[tokio::test]
async fn non_strict_tool_validation_error_reaches_the_next_model_turn() {
    // A scripted engine supplies the bad call and the later corrected call.
    // The client validates the actual HTTP arguments and replays the real
    // resulting error through the server's normal message/template conversion.
    for stream in [false, true] {
        let engine = Arc::new(ToolFeedbackLlm::new());
        let router = AxumServer::from_llm(engine.clone()).build_router();
        let mut request = tool_request(Some(false), Some(json!("auto")), stream);
        let first = post_json(router.clone(), "/v1/chat/completions", request.clone()).await;
        let call = accepted_tool_call(first, stream).await;
        let arguments = call["function"]["arguments"].as_str().unwrap();
        assert_eq!(arguments, malformed_arguments());
        let client_error = validate_json_text_against_schema(&edit_schema(), arguments)
            .expect_err("the client must receive the invalid edits value unchanged");
        let tool_result = json!({"error": client_error}).to_string();
        request["messages"].as_array_mut().unwrap().extend([
            json!({"role": "assistant", "content": null, "tool_calls": [call.clone()]}),
            json!({"role": "tool", "tool_call_id": call["id"], "content": tool_result}),
        ]);
        let second = post_json(router, "/v1/chat/completions", request).await;
        let repaired = accepted_tool_call(second, stream).await;
        validate_json_text_against_schema(
            &edit_schema(),
            repaired["function"]["arguments"].as_str().unwrap(),
        )
        .expect("the scripted correction must satisfy the client schema");

        let requests = engine.requests.lock().unwrap();
        let second_request = requests.last().unwrap();
        let Some(ferrum_types::ApiRequest::Chat(chat)) = second_request.api_request.as_ref() else {
            panic!("tool feedback must reach the structured model request")
        };
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[1].tool_calls[0].function.arguments, arguments);
        assert_eq!(chat.messages[2].role, ferrum_types::ApiMessageRole::Tool);
        assert_eq!(
            chat.messages[2].tool_call_id.as_deref(),
            call["id"].as_str()
        );
        assert_eq!(chat.messages[2].content, tool_result);
        assert_eq!(chat.tools[0].function.strict, Some(false));
        assert!(
            second_request.prompt.contains(&tool_result),
            "{}",
            second_request.prompt
        );
    }
}
