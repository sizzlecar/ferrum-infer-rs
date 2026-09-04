use super::{chat_completions_handler, json_rejection_detail, AppState, ServerError};
use crate::openai::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, ChatToolCall, Usage,
};
use axum::{
    body::to_bytes,
    extract::{rejection::JsonRejection, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(test)]
use crate::openai::{ChatFunctionCall, MessageRole};
#[cfg(test)]
use std::collections::HashSet;

mod request;
mod stream;
mod tools;

use request::ResponsesRequest;
use stream::adapt_chat_stream;

#[cfg(test)]
use request::parse_input;
#[cfg(test)]
use stream::SseDecoder;
#[cfg(test)]
use tools::{allocate_namespace_alias, stable_namespace_alias_hash, MAX_CHAT_TOOL_NAME_BYTES};

const MAX_SYNC_ADAPTER_BODY_BYTES: usize = 16 * 1024 * 1024;

struct ConvertedRequest {
    chat: ChatCompletionsRequest,
    response: ResponseContext,
    stream: bool,
}

#[derive(Clone)]
struct ResponseContext {
    id: String,
    created_at: u64,
    model: String,
    instructions: Option<String>,
    max_output_tokens: u32,
    temperature: f32,
    top_p: f32,
    parallel_tool_calls: bool,
    tools: Vec<Value>,
    tool_choice: Value,
    metadata: Map<String, Value>,
    text: Value,
    reasoning: Value,
    prompt_cache_key: Option<String>,
    presence_penalty: f32,
    frequency_penalty: f32,
    user: Option<String>,
    include_encrypted_reasoning: bool,
    tool_names: ToolNameMap,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResponseToolName {
    namespace: Option<String>,
    name: String,
}

#[derive(Debug, Clone, Default)]
struct ToolNameMap {
    response_to_chat: HashMap<ResponseToolName, String>,
    chat_to_response: HashMap<String, ResponseToolName>,
}

impl ToolNameMap {
    fn insert(&mut self, response_name: ResponseToolName, chat_name: String) {
        self.response_to_chat
            .insert(response_name.clone(), chat_name.clone());
        self.chat_to_response.insert(chat_name, response_name);
    }

    fn chat_name(&self, namespace: Option<&str>, name: &str) -> Option<&str> {
        self.response_to_chat
            .get(&ResponseToolName {
                namespace: namespace.map(str::to_string),
                name: name.to_string(),
            })
            .map(String::as_str)
    }

    fn response_name(&self, chat_name: &str) -> ResponseToolName {
        self.chat_to_response
            .get(chat_name)
            .cloned()
            .unwrap_or_else(|| ResponseToolName {
                namespace: None,
                name: chat_name.to_string(),
            })
    }
}

pub(super) async fn responses_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: std::result::Result<Json<ResponsesRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(request) = request.map_err(|error| {
        ServerError::invalid_request(
            format!(
                "invalid responses request: {}",
                json_rejection_detail(&error)
            ),
            None,
        )
    })?;
    let converted = request.convert()?;
    let chat_response =
        chat_completions_handler(State(state), headers, Ok(Json(converted.chat))).await?;
    if converted.stream {
        Ok(adapt_chat_stream(chat_response, converted.response))
    } else {
        adapt_chat_sync(chat_response, converted.response).await
    }
}

impl ResponseContext {
    fn response(
        &self,
        status: &str,
        output: Vec<Value>,
        usage: Option<&Usage>,
        error: Option<Value>,
    ) -> Value {
        let incomplete_details = (status == "incomplete")
            .then(|| json!({"reason": "max_output_tokens"}))
            .unwrap_or(Value::Null);
        let completed_at = matches!(status, "completed" | "incomplete" | "failed")
            .then(|| chrono::Utc::now().timestamp() as u64)
            .map(Value::from)
            .unwrap_or(Value::Null);
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": completed_at,
            "status": status,
            "background": false,
            "error": error,
            "incomplete_details": incomplete_details,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": null,
            "metadata": self.metadata,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "previous_response_id": null,
            "reasoning": self.reasoning,
            "service_tier": "default",
            "store": false,
            "temperature": self.temperature,
            "text": self.text,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_logprobs": 0,
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": usage.map(response_usage),
            "user": self.user,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "prompt_cache_key": self.prompt_cache_key,
            "safety_identifier": null
        })
    }
}

fn response_usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens
    })
}

async fn adapt_chat_sync(
    response: Response,
    context: ResponseContext,
) -> std::result::Result<Response, ServerError> {
    let bytes = to_bytes(response.into_body(), MAX_SYNC_ADAPTER_BODY_BYTES)
        .await
        .map_err(|error| {
            ServerError::InternalError(format!("failed to read chat response: {error}"))
        })?;
    let chat: ChatCompletionsResponse = serde_json::from_slice(&bytes).map_err(|error| {
        ServerError::InternalError(format!("failed to decode chat response: {error}"))
    })?;
    let choice = chat.choices.into_iter().next().ok_or_else(|| {
        ServerError::InternalError("chat response did not contain a choice".to_string())
    })?;
    let message = choice.message.ok_or_else(|| {
        ServerError::InternalError("chat response did not contain a message".to_string())
    })?;
    let status = if choice.finish_reason.as_deref() == Some("length") {
        "incomplete"
    } else {
        "completed"
    };
    let mut output = response_output_from_message(
        message,
        context.parallel_tool_calls,
        context.include_encrypted_reasoning,
        &context.tool_names,
    )?;
    if status == "incomplete" {
        mark_last_output_incomplete(&mut output);
    }
    Ok(Json(context.response(status, output, chat.usage.as_ref(), None)).into_response())
}

fn mark_last_output_incomplete(output: &mut [Value]) {
    if let Some(item) = output.last_mut().and_then(Value::as_object_mut) {
        item.insert("status".to_string(), json!("incomplete"));
    }
}

fn response_output_from_message(
    message: ChatMessage,
    parallel_tool_calls: bool,
    include_encrypted_reasoning: bool,
    tool_names: &ToolNameMap,
) -> std::result::Result<Vec<Value>, ServerError> {
    let mut output = Vec::new();
    let has_reasoning = message
        .reasoning
        .as_ref()
        .is_some_and(|value| !value.is_empty());
    let call_count = message
        .tool_calls
        .as_ref()
        .map(Vec::len)
        .unwrap_or_default()
        + usize::from(message.function_call.is_some());
    if !parallel_tool_calls && call_count > 1 {
        return Err(ServerError::InternalError(
            "model emitted multiple function calls while parallel_tool_calls=false".to_string(),
        ));
    }
    if let Some(reasoning) = message.reasoning.filter(|value| !value.is_empty()) {
        output.push(reasoning_output_item(
            format!("rs_{}", Uuid::new_v4().simple()),
            reasoning,
            "completed",
            include_encrypted_reasoning,
        ));
    }
    let has_calls = message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
        || message.function_call.is_some();
    if !message.content.is_empty() || (!has_calls && !has_reasoning) {
        output.push(message_output_item(
            format!("msg_{}", Uuid::new_v4().simple()),
            message.content,
            "completed",
        ));
    }
    for call in message.tool_calls.unwrap_or_default() {
        output.push(function_output_item(&call, "completed", tool_names));
    }
    if let Some(function) = message.function_call {
        let call = ChatToolCall {
            index: None,
            id: format!("call_{}", Uuid::new_v4().simple()),
            tool_type: "function".to_string(),
            function,
        };
        output.push(function_output_item(&call, "completed", tool_names));
    }
    Ok(output)
}

fn reasoning_output_item(
    id: String,
    reasoning: String,
    status: &str,
    include_encrypted: bool,
) -> Value {
    let mut item = Map::new();
    item.insert("id".to_string(), json!(id));
    item.insert("type".to_string(), json!("reasoning"));
    item.insert("status".to_string(), json!(status));
    item.insert("summary".to_string(), json!([]));
    item.insert(
        "content".to_string(),
        json!([{"type": "reasoning_text", "text": reasoning}]),
    );
    if include_encrypted {
        // Local model reasoning has no provider-owned encrypted state. A null
        // expansion tells clients to replay the readable content instead.
        item.insert("encrypted_content".to_string(), Value::Null);
    }
    Value::Object(item)
}

fn message_output_item(id: String, text: String, status: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": []
        }]
    })
}

fn function_output_item(call: &ChatToolCall, status: &str, tool_names: &ToolNameMap) -> Value {
    let response_name = tool_names.response_name(&call.function.name);
    function_call_item(
        format!("fc_{}", Uuid::new_v4().simple()),
        status,
        &call.id,
        &response_name,
        &call.function.arguments,
    )
}

fn function_call_item(
    id: String,
    status: &str,
    call_id: &str,
    response_name: &ResponseToolName,
    arguments: &str,
) -> Value {
    let mut item = json!({
        "id": id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": response_name.name,
        "arguments": arguments
    });
    if let Some(namespace) = response_name.namespace.as_deref() {
        item.as_object_mut()
            .expect("function call item is an object")
            .insert("namespace".to_string(), json!(namespace));
    }
    item
}

fn function_call_arguments_done(
    item_id: &str,
    output_index: usize,
    call_id: &str,
    response_name: &ResponseToolName,
    arguments: &str,
) -> Value {
    let mut event = json!({
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "name": response_name.name,
        "arguments": arguments
    });
    if let Some(namespace) = response_name.namespace.as_deref() {
        event
            .as_object_mut()
            .expect("function arguments event is an object")
            .insert("namespace".to_string(), json!(namespace));
    }
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_request_maps_instructions_and_sampling_to_chat() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "instructions": "Be concise",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 17,
            "temperature": 0.2,
            "top_p": 0.8
        }))
        .expect("request");
        let converted = request.convert().expect("converted request");
        assert_eq!(converted.chat.messages.len(), 2);
        assert_eq!(converted.chat.messages[0].role, MessageRole::System);
        assert_eq!(converted.chat.messages[0].content, "Be concise");
        assert_eq!(converted.chat.messages[1].role, MessageRole::User);
        assert_eq!(converted.chat.max_completion_tokens, Some(17));
        assert_eq!(converted.chat.temperature, Some(0.2));
        assert_eq!(converted.chat.top_p, Some(0.8));
    }

    #[test]
    fn responses_input_maps_messages_and_function_history() {
        let messages = parse_input(&json!([
            {"role": "user", "content": [{"type": "input_text", "text": "weather"}]},
            {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}
        ]))
        .expect("responses input");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "weather");
        assert_eq!(messages[1].tool_calls.as_ref().unwrap()[0].id, "call_1");
        assert_eq!(messages[2].role, MessageRole::Tool);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[2].content, "sunny");
    }

    #[test]
    fn responses_input_accepts_real_caller_owned_second_turn_history() {
        let messages = parse_input(&json!([
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}]
            },
            {
                "type": "reasoning",
                "encrypted_content": null,
                "summary": []
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What was my first message?"}]
            }
        ]))
        .expect("caller-owned second turn");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].content, "Hi there!");
        assert_eq!(messages[1].reasoning, None);
        assert_eq!(messages[2].role, MessageRole::User);
    }

    #[test]
    fn responses_input_bundles_readable_reasoning_calls_and_typed_tool_output() {
        let messages = parse_input(&json!([
            {"role": "user", "content": "Check Paris"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "I should check the weather."}],
                "content": [{"type": "reasoning_text", "text": "Use the weather tool."}],
                "encrypted_content": null
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{\"city\":\"Paris\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "input_text", "text": "sunny"}]
            }
        ]))
        .expect("reasoning tool history");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(
            messages[1].reasoning.as_deref(),
            Some("Use the weather tool.")
        );
        assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(messages[2].role, MessageRole::Tool);
        assert_eq!(messages[2].content, "sunny");
    }

    #[test]
    fn responses_input_accepts_empty_function_arguments_and_output() {
        let messages = parse_input(&json!([
            {"role": "user", "content": "Check"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "check",
                "arguments": ""
            },
            {"type": "function_call_output", "call_id": "call_1", "output": ""}
        ]))
        .expect("zero-argument function history");

        assert_eq!(
            messages[1].tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            ""
        );
        assert_eq!(messages[2].content, "");
    }

    #[test]
    fn responses_input_rejects_orphan_and_duplicate_function_history() {
        for (input, expected_param) in [
            (
                json!([
                    {"role": "user", "content": "Check"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "done"}
                ]),
                "input[1].call_id",
            ),
            (
                json!([
                    {"type": "function_call", "call_id": "call_1", "name": "check", "arguments": "{}"},
                    {"type": "function_call", "call_id": "call_1", "name": "check", "arguments": "{}"}
                ]),
                "input[1].call_id",
            ),
            (
                json!([
                    {"type": "function_call", "call_id": "call_1", "name": "check", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "first"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "second"}
                ]),
                "input[2].call_id",
            ),
        ] {
            let error = parse_input(&input).expect_err("invalid function history");
            match error {
                ServerError::InvalidRequest { param, .. } => {
                    assert_eq!(param.as_deref(), Some(expected_param));
                }
                other => panic!("expected invalid function history, got {other:?}"),
            }
        }
    }

    #[test]
    fn responses_request_accepts_captured_codex_http_tool_round_trip() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "instructions": "You are a coding agent.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Use repository tools carefully."}]
                },
                {
                    "type": "message",
                    "id": "msg_u1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run pwd"}]
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Inspect with a command."}],
                    "content": [{"type": "reasoning_text", "text": "raw reasoning"}],
                    "encrypted_content": null
                },
                {
                    "type": "message",
                    "id": "msg_a1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will inspect."}],
                    "phase": "commentary"
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}",
                    "call_id": "call_1"
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": "/tmp"
                }
            ],
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": false
                },
                "strict": false
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "thread-1",
            "client_metadata": {"thread_id": "thread-1", "turn_id": "turn-2"}
        }))
        .expect("captured Codex request");

        let converted = request.convert().expect("Codex HTTP request conversion");
        assert!(converted.stream);
        assert_eq!(converted.chat.messages.len(), 4);
        assert_eq!(converted.chat.messages[0].role, MessageRole::System);
        assert_eq!(
            converted.chat.messages[0].content,
            "You are a coding agent.\n\nUse repository tools carefully."
        );
        let assistant = &converted.chat.messages[2];
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(assistant.content, "I will inspect.");
        assert_eq!(assistant.reasoning.as_deref(), Some("raw reasoning"));
        assert_eq!(assistant.tool_calls.as_ref().unwrap()[0].id, "call_1");
        assert_eq!(converted.chat.messages[3].role, MessageRole::Tool);
        assert_eq!(
            converted.chat.messages[3].tool_call_id.as_deref(),
            Some("call_1")
        );
    }

    #[test]
    fn responses_namespace_tools_use_collision_free_aliases_and_round_trip() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "input": [
                {"role": "user", "content": "Inspect agents"},
                {
                    "type": "function_call",
                    "namespace": "collaboration",
                    "name": "wait_agent",
                    "arguments": "{\"targets\":[]}",
                    "call_id": "call_1"
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "done"}
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "description": "Inspect agent state.",
                    "tools": [{
                        "type": "function",
                        "name": "wait_agent",
                        "description": "Wait for an agent.",
                        "parameters": {"type": "object"},
                        "strict": false
                    }]
                },
                {
                    "type": "function",
                    "name": "collaboration__wait_agent",
                    "parameters": {"type": "object"}
                },
                {
                    "type": "namespace",
                    "name": "other",
                    "tools": [{
                        "type": "function",
                        "name": "wait_agent",
                        "parameters": {"type": "object"}
                    }]
                }
            ],
            "tool_choice": {
                "type": "function",
                "namespace": "collaboration",
                "name": "wait_agent"
            }
        }))
        .expect("namespace request");

        let converted = request.convert().expect("namespace conversion");
        let chat_tools = converted.chat.tools.as_ref().expect("chat tools");
        assert_eq!(chat_tools.len(), 3);
        let collaboration_alias = chat_tools[0].function.name.clone();
        assert_ne!(collaboration_alias, "collaboration__wait_agent");
        assert!(collaboration_alias.len() <= MAX_CHAT_TOOL_NAME_BYTES);
        assert_eq!(chat_tools[1].function.name, "collaboration__wait_agent");
        assert_eq!(chat_tools[2].function.name, "other__wait_agent");
        assert_eq!(
            chat_tools[0].function.description.as_deref(),
            Some("Inspect agent state.\n\nWait for an agent.")
        );
        assert_eq!(
            converted.chat.messages[1].tool_calls.as_ref().unwrap()[0]
                .function
                .name,
            collaboration_alias
        );
        assert_eq!(
            serde_json::to_value(converted.chat.tool_choice.as_ref().unwrap()).unwrap(),
            json!({
                "type": "function",
                "function": {"name": collaboration_alias}
            })
        );

        assert_eq!(converted.response.tools[0]["type"], "namespace");
        assert_eq!(
            converted.response.tools[0]["tools"][0]["name"],
            "wait_agent"
        );
        assert_eq!(
            converted.response.tool_choice,
            json!({
                "type": "function",
                "namespace": "collaboration",
                "name": "wait_agent"
            })
        );

        let backend_call = ChatToolCall {
            index: None,
            id: "call_2".to_string(),
            tool_type: "function".to_string(),
            function: ChatFunctionCall {
                name: collaboration_alias.clone(),
                arguments: "{}".to_string(),
            },
        };
        let item = function_output_item(&backend_call, "completed", &converted.response.tool_names);
        assert_eq!(item["namespace"], "collaboration");
        assert_eq!(item["name"], "wait_agent");
        assert_ne!(item["name"], "collaboration__wait_agent__2");

        let arguments_done = function_call_arguments_done(
            "fc_1",
            0,
            "call_2",
            &converted
                .response
                .tool_names
                .response_name(&collaboration_alias),
            "{}",
        );
        assert_eq!(arguments_done["namespace"], "collaboration");
        assert_eq!(arguments_done["name"], "wait_agent");

        let unknown = converted
            .response
            .tool_names
            .response_name("collaboration__made_up");
        assert_eq!(unknown.namespace, None);
        assert_eq!(unknown.name, "collaboration__made_up");
    }

    #[test]
    fn responses_namespace_alias_is_stable_and_bounded() {
        let namespace = "namespace".repeat(12);
        let name = "function".repeat(12);
        let first = allocate_namespace_alias(&namespace, &name, &mut HashSet::new());
        let second = allocate_namespace_alias(&namespace, &name, &mut HashSet::new());
        assert_eq!(first, second);
        assert!(first.len() <= MAX_CHAT_TOOL_NAME_BYTES);
        assert!(first.ends_with(&format!(
            "__{:016x}",
            stable_namespace_alias_hash(&namespace, &name, 0)
        )));
    }

    #[test]
    fn responses_namespace_tools_reject_unimplemented_semantic_fields() {
        for (tool, expected_param) in [
            (
                json!({
                    "type": "namespace",
                    "name": "files",
                    "allowed_callers": ["code_interpreter"],
                    "tools": []
                }),
                "tools[0].allowed_callers",
            ),
            (
                json!({
                    "type": "namespace",
                    "name": "files",
                    "tools": [{
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"},
                        "defer_loading": true
                    }]
                }),
                "tools[0].tools[0].defer_loading",
            ),
        ] {
            let request: ResponsesRequest = serde_json::from_value(json!({
                "model": "local-model",
                "input": "hello",
                "tools": [tool]
            }))
            .expect("namespace request shape");
            let error = match request.convert() {
                Ok(_) => panic!("semantic namespace field must not be discarded"),
                Err(error) => error,
            };
            match error {
                ServerError::UnsupportedFeature { param, .. } => {
                    assert_eq!(param.as_deref(), Some(expected_param));
                }
                other => panic!("expected unsupported namespace field, got {other:?}"),
            }
        }
    }

    #[test]
    fn responses_namespace_tools_reject_duplicate_identity() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "input": "hello",
            "tools": [
                {
                    "type": "namespace",
                    "name": "files",
                    "tools": [{
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"}
                    }]
                },
                {
                    "type": "namespace",
                    "name": "files",
                    "tools": [{
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"}
                    }]
                }
            ]
        }))
        .expect("duplicate namespace request");

        let error = match request.convert() {
            Ok(_) => panic!("duplicate namespace identity must be rejected"),
            Err(error) => error,
        };
        match error {
            ServerError::InvalidRequest { param, .. } => {
                assert_eq!(param.as_deref(), Some("tools[1].tools[0].name"));
            }
            other => panic!("expected invalid duplicate namespace tool, got {other:?}"),
        }
    }

    #[test]
    fn responses_namespace_history_requires_an_advertised_identity() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "input": [
                {"role": "user", "content": "read"},
                {
                    "type": "function_call",
                    "namespace": "files",
                    "name": "missing",
                    "arguments": "{}",
                    "call_id": "call_1"
                }
            ],
            "tools": [{
                "type": "namespace",
                "name": "files",
                "tools": [{
                    "type": "function",
                    "name": "read",
                    "parameters": {"type": "object"}
                }]
            }]
        }))
        .expect("unknown namespace history request");

        let error = match request.convert() {
            Ok(_) => panic!("unadvertised namespace identity must be rejected"),
            Err(error) => error,
        };
        match error {
            ServerError::InvalidRequest { param, .. } => {
                assert_eq!(param.as_deref(), Some("input[1].namespace"));
            }
            other => panic!("expected invalid namespace history, got {other:?}"),
        }
    }

    #[test]
    fn responses_request_accepts_stateless_agent_controls_and_structured_text() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "instructions": "Be concise",
            "input": [{"role": "user", "content": "answer"}],
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": false,
            "prompt_cache_key": "thread-1",
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
        }))
        .expect("responses request");

        let converted = request.convert().expect("agent controls");
        assert_eq!(
            converted.response.prompt_cache_key.as_deref(),
            Some("thread-1")
        );
        assert!(!converted.response.parallel_tool_calls);
        assert_eq!(
            converted
                .chat
                .chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("reasoning_effort")),
            Some(&json!("high"))
        );
        let format = converted.chat.response_format.expect("response format");
        assert_eq!(format.format_type, "json_schema");
        let schema = format.json_schema.expect("json schema");
        assert_eq!(schema.name.as_deref(), Some("answer"));
        assert_eq!(schema.strict, Some(true));
    }

    #[test]
    fn responses_input_rejects_opaque_reasoning_that_cannot_be_replayed() {
        let error = parse_input(&json!([
            {"role": "user", "content": "hello"},
            {
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "opaque-provider-state"
            },
            {"role": "user", "content": "continue"}
        ]))
        .expect_err("opaque reasoning must not be discarded");

        match error {
            ServerError::UnsupportedFeature { param, .. } => {
                assert_eq!(param.as_deref(), Some("input[1].encrypted_content"));
            }
            other => panic!("expected unsupported encrypted reasoning, got {other:?}"),
        }
    }

    #[test]
    fn responses_input_rejects_encrypted_reasoning_even_when_summary_exists() {
        let error = parse_input(&json!([
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "A lossy summary"}],
                "encrypted_content": "opaque-provider-state"
            }
        ]))
        .expect_err("summary must not replace opaque reasoning state");

        match error {
            ServerError::UnsupportedFeature { param, .. } => {
                assert_eq!(param.as_deref(), Some("input[0].encrypted_content"));
            }
            other => panic!("expected unsupported encrypted reasoning, got {other:?}"),
        }
    }

    #[test]
    fn responses_request_rejects_unimplemented_json_schema_description() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local-model",
            "input": "answer",
            "text": {"format": {
                "type": "json_schema",
                "name": "answer",
                "description": "Use this schema to select an answer",
                "schema": {"type": "object"}
            }}
        }))
        .expect("request shape");
        let error = match request.convert() {
            Ok(_) => panic!("description must not be discarded"),
            Err(error) => error,
        };
        match error {
            ServerError::UnsupportedFeature { param, .. } => {
                assert_eq!(param.as_deref(), Some("text.format.description"));
            }
            other => panic!("expected unsupported description, got {other:?}"),
        }
    }

    #[test]
    fn responses_sse_decoder_handles_split_frames() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: {\"a\":").is_empty());
        assert_eq!(
            decoder.push(b"1}\n\ndata: [DONE]\n\n"),
            vec!["{\"a\":1}", "[DONE]"]
        );
    }
}
