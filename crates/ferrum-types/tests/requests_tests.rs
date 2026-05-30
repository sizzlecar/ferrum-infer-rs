use ferrum_types::*;
use serde_json::json;

#[test]
fn inference_request_builder_defaults() {
    let req = InferenceRequest::new("hello", "llama");
    assert_eq!(req.prompt, "hello");
    assert_eq!(req.model_id.as_str(), "llama");
    assert!(!req.stream);
    assert_eq!(req.priority, Priority::Normal);
}

#[test]
fn inference_request_builder_setters() {
    let params = SamplingParams {
        max_tokens: 16,
        temperature: 0.7,
        ..Default::default()
    };
    let req = InferenceRequest::new("hi", "mistral")
        .with_sampling_params(params.clone())
        .with_stream(true)
        .with_priority(Priority::High)
        .with_client_id("client-1")
        .with_session_id(SessionId::new())
        .with_metadata("k", json!(1));

    assert_eq!(req.sampling_params.max_tokens, 16);
    assert!(req.stream);
    assert_eq!(req.priority, Priority::High);
    assert!(req.client_id.is_some());
    assert!(req.session_id.is_some());
    assert_eq!(req.metadata.get("k").unwrap(), &json!(1));
}

#[test]
fn inference_request_can_carry_structured_chat_api_request() {
    let api_request = ApiRequest::Chat(ApiChatRequest {
        messages: vec![
            ApiChatMessage {
                role: ApiMessageRole::User,
                content: "Use the weather tool".to_string(),
                name: None,
                tool_calls: vec![],
                tool_call_id: None,
                function_call: None,
            },
            ApiChatMessage {
                role: ApiMessageRole::Tool,
                content: "sunny".to_string(),
                name: None,
                tool_calls: vec![],
                tool_call_id: Some("call_1".to_string()),
                function_call: None,
            },
        ],
        tools: vec![ApiTool {
            tool_type: "function".to_string(),
            function: ApiFunction {
                name: "weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
                strict: Some(true),
            },
        }],
        tool_choice: Some(ApiToolChoice::Mode("auto".to_string())),
        legacy_functions: vec![],
        legacy_function_call: None,
        response_format: Some(ApiResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(ApiJsonSchema {
                name: Some("answer".to_string()),
                schema: json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]
                }),
                strict: Some(true),
            }),
        }),
        stream_options: Some(ApiStreamOptions {
            include_usage: Some(true),
        }),
    });

    let req = InferenceRequest::new("rendered prompt", "qwen3").with_api_request(api_request);
    let Some(ApiRequest::Chat(chat)) = req.api_request.as_ref() else {
        panic!("expected structured chat request");
    };
    assert_eq!(chat.messages[1].role, ApiMessageRole::Tool);
    assert_eq!(chat.tools[0].function.name, "weather");
    assert_eq!(
        chat.tool_choice,
        Some(ApiToolChoice::Mode("auto".to_string()))
    );
    assert_eq!(
        chat.response_format
            .as_ref()
            .and_then(|format| format.json_schema.as_ref())
            .and_then(|schema| schema.strict),
        Some(true)
    );
}

#[test]
fn inference_response_can_carry_structured_chat_tool_call() {
    let response = InferenceResponse {
        request_id: RequestId::new(),
        text: String::new(),
        tokens: vec![],
        finish_reason: FinishReason::Stop,
        usage: TokenUsage::new(3, 0),
        latency_ms: 1,
        created_at: chrono::Utc::now(),
        metadata: Default::default(),
        api_response: Some(ApiResponse::Chat(ApiChatResponse {
            message: ApiChatMessage {
                role: ApiMessageRole::Assistant,
                content: String::new(),
                name: None,
                tool_calls: vec![ApiToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: ApiFunctionCall {
                        name: "weather".to_string(),
                        arguments: "{\"city\":\"Paris\"}".to_string(),
                    },
                }],
                tool_call_id: None,
                function_call: None,
            },
            finish_reason: Some("tool_calls".to_string()),
        })),
    };

    let Some(ApiResponse::Chat(chat)) = response.api_response.as_ref() else {
        panic!("expected structured chat response");
    };
    assert_eq!(chat.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(chat.message.tool_calls[0].function.name, "weather");
}

#[test]
fn batch_request_construction() {
    let r1 = InferenceRequest::new("a", "m");
    let r2 = InferenceRequest::new("b", "m").with_sampling_params(SamplingParams {
        max_tokens: 1024,
        ..Default::default()
    });
    let batch = BatchRequest::new(vec![r1, r2]);
    assert_eq!(batch.size(), 2);
    assert!(batch.max_sequence_length >= 512);
    assert!(!batch.is_empty());
}

#[test]
fn scheduled_request_progress_and_state() {
    let req = InferenceRequest::new("a", "m");
    let mut sreq = ScheduledRequest::new(req);
    sreq.update_progress(10);
    sreq.set_state(RequestState::Running);
    assert_eq!(sreq.tokens_processed, 10);
    assert_eq!(sreq.state, RequestState::Running);
}
