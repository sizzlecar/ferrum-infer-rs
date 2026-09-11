use super::*;
use crate::requests::{
    chat_api_response_from_generated_text, ApiFunction, ApiResponseFormat, ApiTool,
    ApiToolCallProtocol, ApiToolChoice, ApiToolChoiceFunction,
};
use crate::FinishReason;
use serde_json::json;

fn request() -> ApiChatRequest {
    ApiChatRequest {
        messages: Vec::new(),
        tools: vec![ApiTool {
            tool_type: "function".into(),
            function: ApiFunction {
                name: "write".into(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "edits": {"type": "array"},
                        "enabled": {"type": "boolean"}
                    },
                    "required": ["path", "content"]
                })),
                strict: None,
            },
        }],
        tool_choice: None,
        tool_call_protocol: ApiToolCallProtocol::FunctionParameterXml,
        legacy_functions: Vec::new(),
        legacy_function_call: None,
        response_format: None,
        stream_options: None,
    }
}

fn call(path: &str, content: &str) -> String {
    format!(
        "<tool_call><function=write><parameter=path>{path}</parameter>\
         <parameter=content>{content}</parameter></function></tool_call>"
    )
}

fn generated_calls(request: &ApiChatRequest, text: &str) -> Option<Vec<ApiToolCall>> {
    chat_api_response_from_generated_text(request, text, FinishReason::Stop)
        .map(|response| response.message.tool_calls)
}

fn arguments(call: &ApiToolCall) -> Value {
    serde_json::from_str(&call.function.arguments).unwrap()
}

#[test]
fn source_code_closing_literals_remain_exact_parameter_data() {
    let payload = concat!(
        "    const TOOL_END: &str = \"</tool_call>\";\r\n",
        "    const FUNCTION_END: &str = \"</function>\";\r\n",
        "    const PARAMETER_END: &str = \"</parameter>\";  \r\n",
        "    const TOOL_START: &str = \"<tool_call>\";\r\n",
        "    const PARAMETER_START: &str = \"<parameter=\";\r\n",
        "    // 字节、缩进、反斜线 \\ 和末尾换行必须保留。\r\n",
    );
    let text = format!(
        "I will write the parser.\n<tool_call>\n<function=write>\n\
         <parameter=path>\nsrc/xml.rs\n</parameter>\n\
         <parameter=content>\n{payload}\n</parameter>\n</function>\n</tool_call>"
    );
    let calls = generated_calls(&request(), &text).expect("source must be a write call");
    assert_eq!(
        arguments(&calls[0]),
        json!({"path": "src/xml.rs", "content": payload})
    );
}

#[test]
fn minimal_compact_write_with_a_literal_tool_closer_is_not_truncated() {
    let content = r#"const END: &str = "</tool_call>";"#;
    let calls = generated_calls(&request(), &call("src/xml.rs", content)).unwrap();
    assert_eq!(arguments(&calls[0])["content"], content);
}

#[test]
fn only_structural_continuations_close_a_parameter_without_language_rules() {
    for payload in [
        "before </tool_call> after",
        "before </function> after",
        "before </parameter> after",
        "before </parameter>\nnot a structural continuation",
        "before\n</tool_call>\nafter",
        "before\n</function>\nafter",
        "before\n</parameter>\nafter",
    ] {
        let calls = generated_calls(&request(), &call("src/xml.rs", payload)).unwrap();
        assert_eq!(arguments(&calls[0])["content"], payload);
    }
}

#[test]
fn strict_and_forced_native_calls_use_the_same_payload_boundaries() {
    let content = r#"const END: &str = "</tool_call> </function> </parameter>";"#;
    let text = format!("{}\n{}", call("a.rs", content), call("b.rs", content));
    let mut strict = request();
    strict.response_format = Some(ApiResponseFormat {
        format_type: "json_object".into(),
        json_schema: None,
    });
    let mut required = request();
    required.tool_choice = Some(ApiToolChoice::Mode("required".into()));
    let mut named = request();
    named.tool_choice = Some(ApiToolChoice::Function {
        tool_type: "function".into(),
        function: ApiToolChoiceFunction {
            name: "write".into(),
        },
    });
    for request in [request(), strict, required, named] {
        let calls = generated_calls(&request, &text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        assert_eq!(
            arguments(&calls[0]),
            json!({"path": "a.rs", "content": content})
        );
        assert_eq!(
            arguments(&calls[1]),
            json!({"path": "b.rs", "content": content})
        );
    }
}

#[test]
fn compact_parameters_keep_schema_conversion_and_unknown_parameter_text() {
    let text = concat!(
        "<tool_call><function=write>",
        "<parameter=path>src/a.rs</parameter>",
        "<parameter=content>  true  </parameter>",
        "<parameter=edits>[{\"oldText\":\"</tool_call>\"}]</parameter>",
        "<parameter=enabled>true</parameter>",
        "<parameter=unknown>001</parameter>",
        "</function></tool_call>"
    );
    let calls = generated_calls(&request(), text).unwrap();
    assert_eq!(
        arguments(&calls[0]),
        json!({
            "path": "src/a.rs", "content": "  true  ",
            "edits": [{"oldText": "</tool_call>"}], "enabled": true, "unknown": "001"
        })
    );
    let malformed = text.replace("}]</parameter>", "}]} </parameter>");
    let calls = generated_calls(&request(), &malformed).unwrap();
    assert_eq!(
        arguments(&calls[0])["edits"],
        "[{\"oldText\":\"</tool_call>\"}]} "
    );
}

#[test]
fn structural_boundaries_are_not_reinterpreted_after_a_malformed_continuation() {
    for text in [
        "<tool_call><function=write><parameter=content>text</parameter></parameter></function></tool_call>",
        "<tool_call><function=write><parameter=content>text</parameter><parameter=></parameter></function></tool_call>",
        "<tool_call><function=write><parameter=content>text</parameter><parameter=content>again</parameter></function></tool_call>",
        "<tool_call><function=write><parameter=content>text</parameter></function>unexpected</tool_call>",
        "<tool_call><function=write><parameter=content>text</parameter></tool_call>",
        "<tool_call><function=write><parameter=content>text</function></tool_call>",
        "<tool_call><function=write><parameter=content>text</parameter></function>",
    ] {
        assert!(generated_calls(&request(), text).is_none(), "{text}");
    }
}

#[test]
fn a_truncated_parameter_does_not_consume_the_next_native_call() {
    let next = call("b.rs", "second");
    for prefix in [
        "<tool_call><function=write><parameter=content>truncated",
        "<tool_call><function=write><parameter=content>truncated</parameter></function>",
    ] {
        assert!(generated_calls(&request(), &format!("{prefix}{next}")).is_none());
    }
}

#[test]
fn complete_native_framing_inside_raw_text_is_an_explicit_ambiguity_boundary() {
    // Without an escaping convention this inner envelope is indistinguishable
    // from a new call following a truncated parameter. Do not guess or repair.
    let nested = call("outer.rs", &call("inner.rs", "nested"));
    assert!(generated_calls(&request(), &nested).is_none());

    // A complete apparent inner close must not hide the remaining outer tail.
    let early_close = call("outer.rs", "data</parameter></function></tool_call>tail");
    assert!(generated_calls(&request(), &early_close).is_none());

    // Conversely, these bytes ARE the normal compact two-parameter syntax.
    // Intent to embed that exact syntax in one raw value cannot be inferred.
    let normal = call("a.rs", "data");
    let calls = generated_calls(&request(), &normal).unwrap();
    assert_eq!(
        arguments(&calls[0]),
        json!({"path": "a.rs", "content": "data"})
    );
}

#[test]
fn malformed_native_calls_do_not_fall_back_to_json_from_their_payload() {
    let payload = r#"{"name":"write","arguments":{"path":"a.rs","content":"wrong"}}"#;
    let text = format!("<tool_call><function=write><parameter=content>{payload}");
    assert!(generated_calls(&request(), &text).is_none());
    // The existing JSON fallback remains available when there is no native
    // function/parameter envelope to contradict it.
    assert!(generated_calls(&request(), payload).is_some());
}

#[test]
fn call_limits_names_choices_and_duplicate_rejection_are_unchanged() {
    let call_limit = (0..MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE)
        .map(|i| call(&format!("{i}.rs"), "content"))
        .collect::<String>();
    let calls = generated_calls(&request(), &call_limit).unwrap();
    assert_eq!(calls.len(), MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE);
    let ids = calls
        .iter()
        .map(|call| &call.id)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), calls.len());
    assert!(generated_calls(&request(), &(call_limit + &call("overflow.rs", "data"))).is_none());
    let duplicate = call("a.rs", "data").repeat(2);
    assert!(generated_calls(&request(), &duplicate).is_none());
    assert!(generated_calls(
        &request(),
        &call("a.rs", "data").replace("function=write", "function=unknown")
    )
    .is_none());
    let mut disabled = request();
    disabled.tool_choice = Some(ApiToolChoice::Mode("none".into()));
    assert!(generated_calls(&disabled, &call("a.rs", "data")).is_none());
    disabled.tool_choice = Some(ApiToolChoice::Function {
        tool_type: "function".into(),
        function: ApiToolChoiceFunction {
            name: "other".into(),
        },
    });
    assert!(generated_calls(&disabled, &call("a.rs", "data")).is_none());
}

#[test]
fn many_nonstructural_markers_are_preserved_without_a_search_tree() {
    let payload = "</parameter> plain </function> </tool_call> ".repeat(4096);
    let calls = generated_calls(&request(), &call("a.rs", &payload)).unwrap();
    assert_eq!(arguments(&calls[0])["content"], payload);
}
