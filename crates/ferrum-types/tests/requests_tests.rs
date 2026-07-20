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
        tool_call_protocol: ApiToolCallProtocol::Json,
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
fn stream_chunk_can_carry_structured_chat_tool_call() {
    let chunk = StreamChunk {
        request_id: RequestId::new(),
        text: String::new(),
        token: None,
        finish_reason: Some(FinishReason::Stop),
        usage: Some(TokenUsage::new(3, 0)),
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

    let Some(ApiResponse::Chat(chat)) = chunk.api_response.as_ref() else {
        panic!("expected structured chat response");
    };
    assert_eq!(chat.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(chat.message.tool_calls[0].id, "call_1");
}

fn chat_request_with_tool(tool_choice: Option<ApiToolChoice>) -> InferenceRequest {
    chat_request_with_tool_protocol(tool_choice, ApiToolCallProtocol::Json)
}

fn chat_request_with_tool_protocol(
    tool_choice: Option<ApiToolChoice>,
    tool_call_protocol: ApiToolCallProtocol,
) -> InferenceRequest {
    InferenceRequest::new("rendered prompt", "mock-model").with_api_request(ApiRequest::Chat(
        ApiChatRequest {
            messages: vec![ApiChatMessage {
                role: ApiMessageRole::User,
                content: "Use the weather tool".to_string(),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                function_call: None,
            }],
            tools: vec![ApiTool {
                tool_type: "function".to_string(),
                function: ApiFunction {
                    name: "weather".to_string(),
                    description: None,
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]}
                        },
                        "required": ["city", "unit"],
                        "additionalProperties": false
                    })),
                    strict: None,
                },
            }],
            tool_choice,
            tool_call_protocol,
            legacy_functions: Vec::new(),
            legacy_function_call: None,
            response_format: None,
            stream_options: None,
        },
    ))
}

#[test]
fn function_parameter_xml_becomes_structured_chat_response() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let text = r#"<tool_call>
<function=weather>
<parameter=city>
Paris
</parameter>
<parameter=unit>
c
</parameter>
</function>
</tool_call>"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected XML tool call");
    };
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.tool_calls[0].function.name, "weather");
    assert_eq!(
        response.message.tool_calls[0].function.arguments,
        r#"{"city":"Paris","unit":"c"}"#
    );
}

#[test]
fn function_parameter_xml_rejects_undeclared_tool() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let text =
        r#"<tool_call><function=calendar><parameter=date>today</parameter></function></tool_call>"#;

    assert!(api_response_from_generated_text(&request, text).is_none());
}

#[test]
fn function_parameter_xml_protocol_keeps_forced_json_arguments_fallback() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Function {
            tool_type: "function".to_string(),
            function: ApiToolChoiceFunction {
                name: "weather".to_string(),
            },
        }),
        ApiToolCallProtocol::FunctionParameterXml,
    );

    let Some(ApiResponse::Chat(response)) =
        api_response_from_generated_text(&request, r#"{"city":"Paris","unit":"c"}"#)
    else {
        panic!("expected forced JSON argument fallback");
    };
    assert_eq!(response.message.tool_calls[0].function.name, "weather");
}

#[test]
fn generated_tool_call_json_becomes_structured_chat_response() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected structured chat tool response");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.content, "");
    assert_eq!(response.message.tool_calls.len(), 1);
    let call = &response.message.tool_calls[0];
    assert_eq!(call.id, "call_1");
    assert_eq!(call.function.name, "weather");
    assert_eq!(call.function.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn qwen3_function_parameters_json_becomes_structured_tool_call() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"function":"weather","parameters":{"city":"深圳","unit":"c"}}"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected structured chat tool response");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.content, "");
    assert_eq!(response.message.tool_calls.len(), 1);
    let call = &response.message.tool_calls[0];
    assert_eq!(call.function.name, "weather");
    assert_eq!(call.function.arguments, r#"{"city":"深圳","unit":"c"}"#);
}

#[test]
fn qwen3_function_object_with_top_level_parameters_keeps_arguments() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"function":{"name":"weather"},"parameters":{"city":"北京","unit":"c"}}"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected structured chat tool response");
    };

    let call = &response.message.tool_calls[0];
    assert_eq!(call.function.name, "weather");
    assert_eq!(call.function.arguments, r#"{"city":"北京","unit":"c"}"#);
}

#[test]
fn llama_auto_tool_wrapper_becomes_structured_tool_call() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"auto":{"tool":"weather","parameters":{"city":"beijing","unit":"c"}}}<|reserved_special_token_55|>"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected llama auto/tool wrapper to map to a structured tool call");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.content, "");
    assert_eq!(response.message.tool_calls.len(), 1);
    let call = &response.message.tool_calls[0];
    assert_eq!(call.function.name, "weather");
    assert_eq!(call.function.arguments, r#"{"city":"beijing","unit":"c"}"#);
}

#[test]
fn single_auto_tool_bare_arguments_json_becomes_structured_tool_call() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"city":"深圳","unit":"c"}"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected bare arguments to map to the only available tool");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.content, "");
    assert_eq!(response.message.tool_calls.len(), 1);
    let call = &response.message.tool_calls[0];
    assert_eq!(call.function.name, "weather");
    assert_eq!(call.function.arguments, r#"{"city":"深圳","unit":"c"}"#);
}

#[test]
fn multi_auto_tool_bare_arguments_json_does_not_guess_tool() {
    let mut request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools.push(ApiTool {
        tool_type: "function".to_string(),
        function: ApiFunction {
            name: "calendar".to_string(),
            description: None,
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: None,
        },
    });

    assert!(api_response_from_generated_text(&request, r#"{"city":"深圳","unit":"c"}"#).is_none());
}

#[test]
fn multi_required_tool_bare_arguments_json_does_not_bind_the_first_tool() {
    let mut request = chat_request_with_tool(Some(ApiToolChoice::Mode("required".to_string())));
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools.push(ApiTool {
        tool_type: "function".to_string(),
        function: ApiFunction {
            name: "calendar".to_string(),
            description: None,
            parameters: Some(json!({
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"]
            })),
            strict: None,
        },
    });

    assert!(
        api_response_from_generated_text(&request, r#"{"city":"Paris","unit":"c"}"#)
            .is_none(),
        "tool_choice=required permits any declared tool, so bare arguments must not be assigned to the first tool"
    );
}

#[test]
fn tool_choice_none_keeps_generated_text_unstructured() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("none".to_string())));
    let text = r#"{"name":"weather","arguments":{"city":"Paris"}}"#;

    assert!(api_response_from_generated_text(&request, text).is_none());
}

#[test]
fn unregistered_tool_name_keeps_generated_text_unstructured() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"name":"calendar","arguments":{"city":"Paris"}}"#;

    assert!(api_response_from_generated_text(&request, text).is_none());
}

#[test]
fn forced_tool_choice_accepts_only_selected_tool() {
    let request = InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
        ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: vec![
                ApiTool {
                    tool_type: "function".to_string(),
                    function: ApiFunction {
                        name: "weather".to_string(),
                        description: None,
                        parameters: None,
                        strict: None,
                    },
                },
                ApiTool {
                    tool_type: "function".to_string(),
                    function: ApiFunction {
                        name: "calendar".to_string(),
                        description: None,
                        parameters: None,
                        strict: None,
                    },
                },
            ],
            tool_choice: Some(ApiToolChoice::Function {
                tool_type: "function".to_string(),
                function: ApiToolChoiceFunction {
                    name: "weather".to_string(),
                },
            }),
            tool_call_protocol: ApiToolCallProtocol::Json,
            legacy_functions: Vec::new(),
            legacy_function_call: None,
            response_format: None,
            stream_options: None,
        }),
    );

    assert!(
        api_response_from_generated_text(&request, r#"{"name":"calendar","arguments":{}}"#)
            .is_none()
    );

    let Some(ApiResponse::Chat(response)) =
        api_response_from_generated_text(&request, r#"{"name":"weather","arguments":{}}"#)
    else {
        panic!("expected selected tool call");
    };
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.message.tool_calls[0].function.name, "weather");
}

#[test]
fn generated_legacy_function_call_json_becomes_structured_chat_response() {
    let request = InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
        ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            tool_call_protocol: ApiToolCallProtocol::Json,
            legacy_functions: vec![ApiFunction {
                name: "weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            }],
            legacy_function_call: None,
            response_format: None,
            stream_options: None,
        }),
    );
    let text = r#"```json
{"function_call":{"name":"weather","arguments":{"city":"Paris"}}}
```"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected structured legacy function response");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("function_call"));
    let function_call = response.message.function_call.unwrap();
    assert_eq!(function_call.name, "weather");
    assert_eq!(function_call.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn forced_legacy_function_call_accepts_only_selected_function() {
    let request = InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
        ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            tool_call_protocol: ApiToolCallProtocol::Json,
            legacy_functions: vec![
                ApiFunction {
                    name: "weather".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
                ApiFunction {
                    name: "calendar".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            ],
            legacy_function_call: Some(ApiFunctionCallChoice::Function {
                name: "weather".to_string(),
            }),
            response_format: None,
            stream_options: None,
        }),
    );

    assert!(api_response_from_generated_text(
        &request,
        r#"{"function_call":{"name":"calendar","arguments":{}}}"#
    )
    .is_none());

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(
        &request,
        r#"{"function_call":{"name":"weather","arguments":{}}}"#,
    ) else {
        panic!("expected selected legacy function call");
    };
    let function_call = response.message.function_call.unwrap();
    assert_eq!(response.finish_reason.as_deref(), Some("function_call"));
    assert_eq!(function_call.name, "weather");
}

#[test]
fn legacy_function_call_still_parses_when_tools_are_present() {
    let request = InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
        ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: vec![ApiTool {
                tool_type: "function".to_string(),
                function: ApiFunction {
                    name: "weather".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }],
            tool_choice: Some(ApiToolChoice::Mode("auto".to_string())),
            tool_call_protocol: ApiToolCallProtocol::Json,
            legacy_functions: vec![ApiFunction {
                name: "legacy_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            }],
            legacy_function_call: Some(ApiFunctionCallChoice::Mode("auto".to_string())),
            response_format: None,
            stream_options: None,
        }),
    );
    let text = r#"{"function_call":{"name":"legacy_weather","arguments":{"city":"Paris"}}}"#;

    let Some(ApiResponse::Chat(response)) = api_response_from_generated_text(&request, text) else {
        panic!("expected structured legacy function response");
    };

    assert_eq!(response.finish_reason.as_deref(), Some("function_call"));
    let function_call = response.message.function_call.unwrap();
    assert_eq!(function_call.name, "legacy_weather");
    assert_eq!(function_call.arguments, r#"{"city":"Paris"}"#);
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
