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
        execution_evidence: None,
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
        execution_evidence: None,
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

fn api_response_after_stop(request: &InferenceRequest, text: &str) -> Option<ApiResponse> {
    api_response_from_generated_text(request, text, FinishReason::Stop)
}

fn parse_single_xml_parameter(
    parameter_schema: serde_json::Value,
    value: &str,
) -> serde_json::Value {
    let mut request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools[0].function.name = "types".to_string();
    chat.tools[0].function.parameters = Some(parameter_schema);
    let text = format!(
        "<tool_call><function=types><parameter=value>{value}</parameter></function></tool_call>"
    );
    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, &text) else {
        panic!("expected XML tool call");
    };
    let arguments: serde_json::Value =
        serde_json::from_str(&response.message.tool_calls[0].function.arguments)
            .expect("tool arguments must be JSON");
    arguments["value"].clone()
}

#[test]
fn typed_tool_envelope_is_exposed_only_for_an_enabled_xml_protocol() {
    let xml_request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let envelope = xml_request
        .api_request
        .as_ref()
        .and_then(ApiRequest::generated_response_envelope)
        .expect("enabled XML tool protocol must expose its response envelope");
    assert_eq!(envelope.open_token_text, "<tool_call>");
    assert_eq!(envelope.close_token_text, "</tool_call>");
    assert_eq!(envelope.max_envelopes, 32);

    let json_request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::Json,
    );
    assert!(json_request
        .api_request
        .as_ref()
        .and_then(ApiRequest::generated_response_envelope)
        .is_none());

    let disabled_request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("none".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    assert!(disabled_request
        .api_request
        .as_ref()
        .and_then(ApiRequest::generated_response_envelope)
        .is_none());
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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
fn function_parameter_xml_preserves_opencode_edit_whitespace() {
    let mut request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools[0].function.name = "edit".to_string();
    chat.tools[0].function.parameters = Some(json!({
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"}
        },
        "required": ["filePath", "oldString", "newString"]
    }));

    let text = concat!(
        "<tool_call>\n",
        "<function=edit>\n",
        "<parameter=filePath>\n",
        "/workspace/src/main.rs\n",
        "</parameter>\n",
        "<parameter=oldString>\n",
        "    if x:\n",
        "        return 1\n",
        "\n",
        "</parameter>\n",
        "<parameter=newString>\n",
        "    if x:\n",
        "        return 2\n",
        "\n",
        "</parameter>\n",
        "<parameter=replaceAll>\n",
        "true\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    );

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
        panic!("expected XML tool call");
    };
    let arguments: serde_json::Value =
        serde_json::from_str(&response.message.tool_calls[0].function.arguments)
            .expect("tool arguments must be JSON");
    assert_eq!(arguments["filePath"], json!("/workspace/src/main.rs"));
    assert_eq!(
        arguments["oldString"],
        json!("    if x:\n        return 1\n")
    );
    assert_eq!(
        arguments["newString"],
        json!("    if x:\n        return 2\n")
    );
    assert_eq!(arguments["replaceAll"], json!(true));
}

#[test]
fn function_parameter_xml_strips_only_matching_wrapper_newlines() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );

    for (case, value, expected) in [
        ("lf", "\nParis\n", "Paris"),
        ("crlf", "\r\nParis\r\n", "Paris"),
        ("lf payload newline", "\nParis\n\n", "Paris\n"),
        ("crlf payload newline", "\r\nParis\r\n\r\n", "Paris\r\n"),
        ("unwrapped", "  Paris \n", "  Paris \n"),
        ("lf opening with crlf ending", "\nParis\r\n", "Paris\r\n"),
        ("crlf opening with lf ending", "\r\nParis\n", "Paris\n"),
    ] {
        let text = format!(
            "<tool_call><function=weather><parameter=city>{value}</parameter></function></tool_call>"
        );
        let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, &text) else {
            panic!("expected XML tool call for {case}");
        };
        let arguments: serde_json::Value =
            serde_json::from_str(&response.message.tool_calls[0].function.arguments)
                .expect("tool arguments must be JSON");
        assert_eq!(arguments["city"], json!(expected), "case: {case}");
    }
}

#[test]
fn function_parameter_xml_decodes_values_using_the_declared_schema() {
    let mut request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools[0].function.name = "types".to_string();
    chat.tools[0].function.parameters = Some(json!({
        "type": "object",
        "$defs": {
            "Cfg": {
                "type": "object",
                "properties": {"flag": {"type": "boolean"}},
                "required": ["flag"]
            },
            "a/b~c": {"type": "object"}
        },
        "definitions": {
            "Legacy": {"type": "array", "items": {"type": "integer"}}
        },
        "properties": {
            "obj": {"type": "object"},
            "arr": {"type": "array", "items": {"type": "object"}},
            "enabled": {"type": "boolean"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "nothing": {"type": "null"},
            "cfg": {"$ref": "#/$defs/Cfg", "description": "ref siblings remain untouched"},
            "escaped": {"$ref": "#/$defs/a~1b~0c"},
            "legacy": {"$ref": "#/definitions/Legacy"},
            "nullable_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "string_or_null": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "text": {"type": "string"}
        }
    }));
    let schema_before = chat.tools[0].function.parameters.clone();
    let text = concat!(
        "<tool_call>\n",
        "<function=types>\n",
        "<parameter=obj>\n{\"k\":\"v\"}\n</parameter>\n",
        "<parameter=arr>\n[{\"type\":\"web\"}]\n</parameter>\n",
        "<parameter=enabled>\ntrue\n</parameter>\n",
        "<parameter=count>\n42\n</parameter>\n",
        "<parameter=ratio>\n2.5\n</parameter>\n",
        "<parameter=nothing>\nnull\n</parameter>\n",
        "<parameter=cfg>\n{\"flag\":false}\n</parameter>\n",
        "<parameter=escaped>\n{\"ok\":true}\n</parameter>\n",
        "<parameter=legacy>\n[1,2]\n</parameter>\n",
        "<parameter=nullable_bool>\nnull\n</parameter>\n",
        "<parameter=string_or_null>\nnull\n</parameter>\n",
        "<parameter=text>\n42\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    );

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
        panic!("expected typed XML tool call");
    };
    let arguments: serde_json::Value =
        serde_json::from_str(&response.message.tool_calls[0].function.arguments)
            .expect("tool arguments must be JSON");
    assert_eq!(arguments["obj"], json!({"k": "v"}));
    assert_eq!(arguments["arr"], json!([{"type": "web"}]));
    assert_eq!(arguments["enabled"], json!(true));
    assert_eq!(arguments["count"], json!(42));
    assert_eq!(arguments["ratio"], json!(2.5));
    assert_eq!(arguments["nothing"], serde_json::Value::Null);
    assert_eq!(arguments["cfg"], json!({"flag": false}));
    assert_eq!(arguments["escaped"], json!({"ok": true}));
    assert_eq!(arguments["legacy"], json!([1, 2]));
    assert_eq!(arguments["nullable_bool"], serde_json::Value::Null);
    assert_eq!(arguments["string_or_null"], json!("null"));
    assert_eq!(arguments["text"], json!("42"));
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_ref() else {
        panic!("expected chat request");
    };
    assert_eq!(chat.tools[0].function.parameters, schema_before);
}

#[test]
fn function_parameter_xml_keeps_text_when_schema_cannot_prove_a_json_type() {
    let mut request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        panic!("expected chat request");
    };
    chat.tools[0].function.name = "types".to_string();
    chat.tools[0].function.parameters = Some(json!({
        "$ref": "#/$defs/Arguments",
        "$defs": {
            "Arguments": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "invalid_bool": {"type": "boolean"},
                    "unknown": {},
                    "external": {"$ref": "https://example.com/schema.json"},
                    "cycle": {"$ref": "#/$defs/Cycle"}
                }
            },
            "Cycle": {"$ref": "#/$defs/Cycle"}
        }
    }));
    let text = concat!(
        "<tool_call><function=types>",
        "<parameter=enabled>true</parameter>",
        "<parameter=invalid_bool>yes</parameter>",
        "<parameter=unknown>true</parameter>",
        "<parameter=external>{\"k\":1}</parameter>",
        "<parameter=cycle>[1]</parameter>",
        "<parameter=undeclared>false</parameter>",
        "</function></tool_call>",
    );

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
        panic!("expected XML tool call");
    };
    let arguments: serde_json::Value =
        serde_json::from_str(&response.message.tool_calls[0].function.arguments)
            .expect("tool arguments must be JSON");
    assert_eq!(arguments["enabled"], json!(true));
    assert_eq!(arguments["invalid_bool"], json!("yes"));
    assert_eq!(arguments["unknown"], json!("true"));
    assert_eq!(arguments["external"], json!(r#"{"k":1}"#));
    assert_eq!(arguments["cycle"], json!("[1]"));
    assert_eq!(arguments["undeclared"], json!("false"));
}

#[test]
fn function_parameter_xml_combines_root_schema_branches_without_order_dependence() {
    for (case, schema, value, expected) in [
        (
            "allOf unconstrained first",
            json!({
                "type": "object",
                "allOf": [
                    {"properties": {"value": {}}},
                    {"properties": {"value": {"type": "boolean"}}}
                ]
            }),
            "true",
            json!(true),
        ),
        (
            "allOf unconstrained last",
            json!({
                "type": "object",
                "allOf": [
                    {"properties": {"value": {"type": "boolean"}}},
                    {"properties": {"value": {}}}
                ]
            }),
            "true",
            json!(true),
        ),
        (
            "anyOf native types",
            json!({
                "type": "object",
                "anyOf": [
                    {"properties": {"value": {"type": "boolean"}}},
                    {"properties": {"value": {"type": "null"}}}
                ]
            }),
            "null",
            serde_json::Value::Null,
        ),
        (
            "anyOf includes string",
            json!({
                "type": "object",
                "anyOf": [
                    {"properties": {"value": {"type": "string"}}},
                    {"properties": {"value": {"type": "null"}}}
                ]
            }),
            "null",
            json!("null"),
        ),
        (
            "oneOf native types",
            json!({
                "type": "object",
                "oneOf": [
                    {"properties": {"value": {"type": "array"}}},
                    {"properties": {"value": {"type": "object"}}}
                ]
            }),
            "[]",
            json!([]),
        ),
        (
            "oneOf includes string",
            json!({
                "type": "object",
                "oneOf": [
                    {"properties": {"value": {"type": "string"}}},
                    {"properties": {"value": {"type": "object"}}}
                ]
            }),
            "{}",
            json!("{}"),
        ),
    ] {
        assert_eq!(
            parse_single_xml_parameter(schema, value),
            expected,
            "case: {case}"
        );
    }
}

#[test]
fn function_parameter_xml_honors_ref_dialects_and_typed_additional_properties() {
    for (case, schema, expected) in [
        (
            "draft 7 ignores ref sibling",
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$defs": {"Any": {}},
                "type": "object",
                "properties": {
                    "value": {"$ref": "#/$defs/Any", "type": "boolean"}
                }
            }),
            json!("true"),
        ),
        (
            "default 2020-12 applies ref sibling",
            json!({
                "$defs": {"Any": {}},
                "type": "object",
                "properties": {
                    "value": {"$ref": "#/$defs/Any", "type": "boolean"}
                }
            }),
            json!(true),
        ),
        (
            "typed additionalProperties",
            json!({
                "type": "object",
                "additionalProperties": {"type": "boolean"}
            }),
            json!(true),
        ),
        (
            "patternProperties remains conservative",
            json!({
                "type": "object",
                "patternProperties": {"^value$": {}},
                "additionalProperties": {"type": "boolean"}
            }),
            json!("true"),
        ),
        (
            "unknown type remains conservative",
            json!({
                "type": "object",
                "properties": {"value": {"type": "booolean"}}
            }),
            json!("true"),
        ),
        (
            "unknown schema dialect remains conservative",
            json!({
                "$schema": "https://example.com/custom-schema",
                "type": "object",
                "properties": {"value": {"type": "boolean"}}
            }),
            json!("true"),
        ),
        (
            "root id keeps the root dialect",
            json!({
                "$id": "https://example.com/tool-schema",
                "type": "object",
                "properties": {"value": {"type": "boolean"}}
            }),
            json!(true),
        ),
        (
            "nested schema declaration is a dialect boundary",
            json!({
                "type": "object",
                "properties": {
                    "value": {
                        "$schema": "http://json-schema.org/draft-04/schema#",
                        "const": true
                    }
                }
            }),
            json!("true"),
        ),
        (
            "legacy nested id is a resource boundary",
            json!({
                "type": "object",
                "properties": {
                    "value": {"id": "nested-schema", "type": "boolean"}
                }
            }),
            json!("true"),
        ),
        (
            "nested dialect does not inherit root const semantics",
            json!({
                "type": "object",
                "properties": {
                    "value": {
                        "$id": "nested-schema",
                        "$schema": "http://json-schema.org/draft-04/schema#",
                        "const": true
                    }
                }
            }),
            json!("true"),
        ),
        (
            "nested schema resource keeps refs local",
            json!({
                "type": "object",
                "$defs": {
                    "Scoped": {
                        "$id": "nested-schema",
                        "$defs": {"Flag": {"type": "boolean"}}
                    }
                },
                "properties": {
                    "value": {"$ref": "#/$defs/Scoped/$defs/Flag"}
                }
            }),
            json!("true"),
        ),
        (
            "percent encoded ref is not treated as a literal key",
            json!({
                "type": "object",
                "$defs": {
                    "Flag value": {"type": "string"},
                    "Flag%20value": {"type": "boolean"}
                },
                "properties": {
                    "value": {"$ref": "#/$defs/Flag%20value"}
                }
            }),
            json!("true"),
        ),
    ] {
        assert_eq!(
            parse_single_xml_parameter(schema, "true"),
            expected,
            "case: {case}"
        );
    }
}

#[test]
fn function_parameter_xml_bounds_repeated_ref_graph_traversal() {
    let mut definitions = serde_json::Map::new();
    definitions.insert("Level0".to_string(), json!({"type": "boolean"}));
    for level in 1..=24 {
        let previous = format!("#/$defs/Level{}", level - 1);
        definitions.insert(
            format!("Level{level}"),
            json!({"anyOf": [{"$ref": previous}, {"$ref": previous}]}),
        );
    }
    let schema = json!({
        "type": "object",
        "$defs": definitions,
        "properties": {"value": {"$ref": "#/$defs/Level24"}}
    });

    assert_eq!(parse_single_xml_parameter(schema, "true"), json!(true));
}

#[test]
fn function_parameter_xml_rejects_undeclared_tool() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let text =
        r#"<tool_call><function=calendar><parameter=date>today</parameter></function></tool_call>"#;

    assert!(api_response_after_stop(&request, text).is_none());
}

#[test]
fn structured_tool_response_requires_a_successful_terminal_reason() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let text =
        r#"<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"#;

    for finish_reason in [
        FinishReason::Length,
        FinishReason::Cancelled,
        FinishReason::Error,
        FinishReason::ContentFilter,
    ] {
        assert!(
            api_response_from_generated_text(&request, text, finish_reason).is_none(),
            "{finish_reason:?} must remain authoritative"
        );
    }
}

#[test]
fn function_parameter_xml_rejects_incomplete_blocks() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    for text in [
        "<tool_call><function=weather><parameter=city>Paris</parameter></function>",
        "<tool_call><function=weather><parameter=city>Paris</parameter></tool_call>",
        "<tool_call><function=weather><parameter=city>Paris</function></tool_call>",
    ] {
        assert!(
            api_response_after_stop(&request, text).is_none(),
            "incomplete XML must fail closed: {text}"
        );
    }
}

#[test]
fn function_parameter_xml_rejects_duplicate_or_unbounded_calls() {
    let request = chat_request_with_tool_protocol(
        Some(ApiToolChoice::Mode("auto".to_string())),
        ApiToolCallProtocol::FunctionParameterXml,
    );
    let call = |city: &str| {
        format!(
            "<tool_call><function=weather><parameter=city>{city}</parameter></function></tool_call>"
        )
    };
    let duplicate = format!("{}{}", call("Paris"), call("Paris"));
    assert!(api_response_after_stop(&request, &duplicate).is_none());

    let too_many = (0..33)
        .map(|index| call(&format!("city-{index}")))
        .collect::<String>();
    assert!(api_response_after_stop(&request, &too_many).is_none());
}

#[test]
fn generated_tool_call_json_rejects_partial_duplicate_or_unbounded_calls() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let call = |city: &str| {
        json!({
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": {"city": city, "unit": "c"}
            }
        })
    };

    let duplicate = json!({"tool_calls": [call("Paris"), call("Paris")]}).to_string();
    assert!(api_response_after_stop(&request, &duplicate).is_none());

    let partial = json!({
        "tool_calls": [
            call("Paris"),
            {"type": "function", "function": {"name": "undeclared", "arguments": {}}}
        ]
    })
    .to_string();
    assert!(api_response_after_stop(&request, &partial).is_none());

    let too_many = json!({
        "tool_calls": (0..33)
            .map(|index| call(&format!("city-{index}")))
            .collect::<Vec<_>>()
    })
    .to_string();
    assert!(api_response_after_stop(&request, &too_many).is_none());
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
        api_response_after_stop(&request, r#"{"city":"Paris","unit":"c"}"#)
    else {
        panic!("expected forced JSON argument fallback");
    };
    assert_eq!(response.message.tool_calls[0].function.name, "weather");
}

#[test]
fn generated_tool_call_json_becomes_structured_chat_response() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}"#;

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    assert!(api_response_after_stop(&request, r#"{"city":"深圳","unit":"c"}"#).is_none());
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
        api_response_after_stop(&request, r#"{"city":"Paris","unit":"c"}"#)
            .is_none(),
        "tool_choice=required permits any declared tool, so bare arguments must not be assigned to the first tool"
    );
}

#[test]
fn tool_choice_none_keeps_generated_text_unstructured() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("none".to_string())));
    let text = r#"{"name":"weather","arguments":{"city":"Paris"}}"#;

    assert!(api_response_after_stop(&request, text).is_none());
}

#[test]
fn unregistered_tool_name_keeps_generated_text_unstructured() {
    let request = chat_request_with_tool(Some(ApiToolChoice::Mode("auto".to_string())));
    let text = r#"{"name":"calendar","arguments":{"city":"Paris"}}"#;

    assert!(api_response_after_stop(&request, text).is_none());
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

    assert!(api_response_after_stop(&request, r#"{"name":"calendar","arguments":{}}"#).is_none());

    let Some(ApiResponse::Chat(response)) =
        api_response_after_stop(&request, r#"{"name":"weather","arguments":{}}"#)
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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

    assert!(api_response_after_stop(
        &request,
        r#"{"function_call":{"name":"calendar","arguments":{}}}"#
    )
    .is_none());

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(
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

    let Some(ApiResponse::Chat(response)) = api_response_after_stop(&request, text) else {
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
