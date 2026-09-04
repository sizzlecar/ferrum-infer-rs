use super::super::{
    ServerError, DEFAULT_COMPLETION_MAX_TOKENS, DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
};
use super::tools::{parse_tool_choice, parse_tools};
use super::{ConvertedRequest, ResponseContext, ToolNameMap};
use crate::openai::{
    AssistantMessagePhase, ChatCompletionsRequest, ChatFunctionCall, ChatMessage, ChatToolCall,
    MessageRole, OpenAiJsonSchema, OpenAiResponseFormat, StreamOptions,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize)]
pub(in crate::axum_server) struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    metadata: Option<Map<String, Value>>,
    /// OpenResponses text controls use a different JSON shape from Chat
    /// Completions' `response_format`, so they are validated during conversion.
    #[serde(default)]
    text: Option<Value>,
    /// Model reasoning controls. Ferrum maps supported effort values to its
    /// model-owned chat-template options.
    #[serde(default)]
    reasoning: Option<Value>,
    /// Best-effort cache affinity hint. Prefix caching remains automatic and
    /// correctness never depends on this value.
    #[serde(default)]
    prompt_cache_key: Option<String>,
    /// Codex transport telemetry. It is intentionally opaque and has no
    /// inference semantics.
    #[serde(default)]
    client_metadata: Option<Map<String, Value>>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    user: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl ResponsesRequest {
    pub(super) fn convert(self) -> std::result::Result<ConvertedRequest, ServerError> {
        self.validate_stateless_scope()?;

        let model = self
            .model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| ServerError::invalid_request("model is required", Some("model")))?
            .to_string();
        let (chat_tools, response_tools, tool_names) = parse_tools(self.tools.as_deref())?;
        let ParsedInput {
            mut messages,
            mut phases,
        } = parse_input_with_tool_names(&self.input, &tool_names)?;
        merge_leading_system_messages(&mut messages, &mut phases, self.instructions.as_deref());
        if messages.is_empty() {
            return Err(ServerError::invalid_request(
                "input must contain at least one text message or function item",
                Some("input"),
            ));
        }

        let (chat_tool_choice, response_tool_choice) =
            parse_tool_choice(self.tool_choice.as_ref(), &tool_names)?;
        let (response_format, response_text) = parse_text_controls(self.text.as_ref())?;
        let (chat_template_kwargs, response_reasoning) =
            parse_reasoning_controls(self.reasoning.as_ref())?;
        let include_encrypted_reasoning = parse_include(self.include.as_deref())?;
        validate_prompt_cache_key(self.prompt_cache_key.as_deref())?;
        // `client_metadata` is transport telemetry used by some coding agents.
        // Accepting an object is safe because it never changes model input or
        // scheduling; serde has already rejected non-object values.
        let _client_metadata = self.client_metadata.as_ref();
        let stream = self.stream.unwrap_or(false);
        let max_output_tokens = self
            .max_output_tokens
            .unwrap_or(DEFAULT_COMPLETION_MAX_TOKENS);
        let temperature = self.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE);
        let top_p = self.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P);

        let chat = ChatCompletionsRequest {
            model: model.clone(),
            messages,
            max_tokens: None,
            max_completion_tokens: Some(max_output_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            n: Some(1),
            stream: Some(stream),
            ignore_eos: None,
            stop: None,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            user: self.user.clone(),
            seed: None,
            response_format,
            tools: (!chat_tools.is_empty()).then_some(chat_tools),
            tool_choice: chat_tool_choice,
            stream_options: stream.then_some(StreamOptions {
                include_usage: Some(true),
            }),
            functions: None,
            function_call: None,
            metadata: None,
            chat_template_kwargs,
        };
        let response = ResponseContext {
            id: format!("resp_{}", Uuid::new_v4().simple()),
            created_at: chrono::Utc::now().timestamp() as u64,
            model,
            instructions: self.instructions,
            max_output_tokens,
            temperature,
            top_p,
            parallel_tool_calls: self.parallel_tool_calls.unwrap_or(true),
            tools: response_tools,
            tool_choice: response_tool_choice,
            metadata: self.metadata.unwrap_or_default(),
            text: response_text,
            reasoning: response_reasoning,
            prompt_cache_key: self.prompt_cache_key,
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            user: self.user,
            include_encrypted_reasoning,
            tool_names,
        };

        Ok(ConvertedRequest {
            chat,
            message_phases: phases,
            response,
            stream,
        })
    }

    fn validate_stateless_scope(&self) -> std::result::Result<(), ServerError> {
        if self.store == Some(true) {
            return Err(unsupported(
                "Ferrum's stateless Responses API requires store=false",
                "store",
            ));
        }
        if self.previous_response_id.is_some() {
            return Err(unsupported(
                "previous_response_id requires response state storage",
                "previous_response_id",
            ));
        }
        if self.conversation.is_some() {
            return Err(unsupported(
                "conversation requires response state storage",
                "conversation",
            ));
        }
        if self.background == Some(true) {
            return Err(unsupported(
                "background responses are not supported by the stateless endpoint",
                "background",
            ));
        }
        if self
            .truncation
            .as_deref()
            .is_some_and(|truncation| truncation != "disabled")
        {
            return Err(unsupported(
                "only truncation=disabled is supported",
                "truncation",
            ));
        }
        if let Some((name, _)) = self.extra.iter().find(|(_, value)| !value.is_null()) {
            return Err(ServerError::unsupported_feature(
                format!("Responses API field `{name}` is not supported yet"),
                Some(name),
            ));
        }
        Ok(())
    }
}

pub(super) fn unsupported(message: impl Into<String>, param: &str) -> ServerError {
    ServerError::unsupported_feature(message, Some(param))
}

fn parse_include(include: Option<&[String]>) -> std::result::Result<bool, ServerError> {
    let mut encrypted_reasoning = false;
    for (index, value) in include.unwrap_or_default().iter().enumerate() {
        match value.as_str() {
            "reasoning.encrypted_content" => encrypted_reasoning = true,
            unsupported_value => {
                return Err(unsupported(
                    format!("Responses API include value `{unsupported_value}` is not supported"),
                    &format!("include[{index}]"),
                ))
            }
        }
    }
    Ok(encrypted_reasoning)
}

fn validate_prompt_cache_key(key: Option<&str>) -> std::result::Result<(), ServerError> {
    if key.is_some_and(|key| key.chars().count() > 64) {
        return Err(ServerError::invalid_request(
            "prompt_cache_key must contain at most 64 characters",
            Some("prompt_cache_key"),
        ));
    }
    Ok(())
}

fn parse_text_controls(
    text: Option<&Value>,
) -> std::result::Result<(Option<OpenAiResponseFormat>, Value), ServerError> {
    let Some(text) = text.filter(|value| !value.is_null()) else {
        return Ok((None, json!({"format": {"type": "text"}})));
    };
    let object = text
        .as_object()
        .ok_or_else(|| ServerError::invalid_request("text must be an object", Some("text")))?;
    reject_unknown_object_fields(object, &["format", "verbosity"], "text")?;
    if let Some(verbosity) = object.get("verbosity").filter(|value| !value.is_null()) {
        if !matches!(verbosity.as_str(), Some("low" | "medium" | "high")) {
            return Err(ServerError::invalid_request(
                "text.verbosity must be low, medium, or high",
                Some("text.verbosity"),
            ));
        }
        return Err(unsupported(
            "text.verbosity is not supported by local model templates",
            "text.verbosity",
        ));
    }

    let Some(format) = object.get("format").filter(|value| !value.is_null()) else {
        return Ok((None, json!({"format": {"type": "text"}})));
    };
    let format = format.as_object().ok_or_else(|| {
        ServerError::invalid_request("text.format must be an object", Some("text.format"))
    })?;
    let format_type = required_string(format, "type", "text.format.type")?;
    match format_type.as_str() {
        "text" => {
            reject_unknown_object_fields(format, &["type"], "text.format")?;
            Ok((None, json!({"format": {"type": "text"}})))
        }
        "json_object" => {
            reject_unknown_object_fields(format, &["type"], "text.format")?;
            Ok((
                Some(OpenAiResponseFormat {
                    format_type: "json_object".to_string(),
                    json_schema: None,
                }),
                json!({"format": {"type": "json_object"}}),
            ))
        }
        "json_schema" => {
            reject_unknown_object_fields(
                format,
                &["type", "name", "description", "schema", "strict"],
                "text.format",
            )?;
            let name = required_string(format, "name", "text.format.name")?;
            let schema = format
                .get("schema")
                .filter(|value| !value.is_null())
                .cloned()
                .ok_or_else(|| {
                    ServerError::invalid_request(
                        "text.format.schema is required",
                        Some("text.format.schema"),
                    )
                })?;
            let strict = optional_bool(format, "strict", "text.format.strict")?.unwrap_or(false);
            let description = optional_string(format, "description", "text.format.description")?;
            if description
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            {
                return Err(unsupported(
                    "text.format.description is not supported by local model templates",
                    "text.format.description",
                ));
            }
            let mut normalized = Map::new();
            normalized.insert("type".to_string(), json!("json_schema"));
            normalized.insert("name".to_string(), json!(name));
            normalized.insert("schema".to_string(), schema.clone());
            normalized.insert("strict".to_string(), json!(strict));
            if let Some(description) = description {
                normalized.insert("description".to_string(), json!(description));
            }
            Ok((
                Some(OpenAiResponseFormat {
                    format_type: "json_schema".to_string(),
                    json_schema: Some(OpenAiJsonSchema {
                        name: Some(name),
                        schema: Some(schema),
                        strict: Some(strict),
                    }),
                }),
                json!({"format": Value::Object(normalized)}),
            ))
        }
        unsupported_type => Err(unsupported(
            format!("text format `{unsupported_type}` is not supported"),
            "text.format.type",
        )),
    }
}

fn parse_reasoning_controls(
    reasoning: Option<&Value>,
) -> std::result::Result<(Option<HashMap<String, Value>>, Value), ServerError> {
    let Some(reasoning) = reasoning.filter(|value| !value.is_null()) else {
        return Ok((None, Value::Null));
    };
    let object = reasoning.as_object().ok_or_else(|| {
        ServerError::invalid_request("reasoning must be an object", Some("reasoning"))
    })?;
    reject_unknown_object_fields(object, &["effort", "summary"], "reasoning")?;

    let mut kwargs = HashMap::new();
    if let Some(effort) = object.get("effort").filter(|value| !value.is_null()) {
        let effort = effort.as_str().ok_or_else(|| {
            ServerError::invalid_request(
                "reasoning.effort must be a string",
                Some("reasoning.effort"),
            )
        })?;
        match effort {
            "none" => {
                kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
            }
            "minimal" | "low" | "medium" | "high" | "xhigh" => {
                kwargs.insert("reasoning_effort".to_string(), json!(effort));
            }
            unsupported_effort => {
                return Err(unsupported(
                    format!("reasoning effort `{unsupported_effort}` is not supported"),
                    "reasoning.effort",
                ))
            }
        }
    }
    if let Some(summary) = object.get("summary").filter(|value| !value.is_null()) {
        if !matches!(
            summary.as_str(),
            Some("auto" | "concise" | "detailed" | "none")
        ) {
            return Err(ServerError::invalid_request(
                "reasoning.summary must be auto, concise, detailed, or none",
                Some("reasoning.summary"),
            ));
        }
    }

    Ok(((!kwargs.is_empty()).then_some(kwargs), reasoning.clone()))
}

pub(super) fn reject_unknown_object_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    parent: &str,
) -> std::result::Result<(), ServerError> {
    if let Some((name, _)) = object
        .iter()
        .find(|(name, value)| !allowed.contains(&name.as_str()) && !value.is_null())
    {
        let param = format!("{parent}.{name}");
        return Err(unsupported(
            format!("Responses API field `{param}` is not supported yet"),
            &param,
        ));
    }
    Ok(())
}

fn chat_message(role: MessageRole, content: String) -> ChatMessage {
    ChatMessage {
        role,
        content,
        reasoning: None,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        function_call: None,
    }
}

fn merge_leading_system_messages(
    messages: &mut Vec<ChatMessage>,
    phases: &mut Vec<Option<AssistantMessagePhase>>,
    instructions: Option<&str>,
) {
    debug_assert_eq!(messages.len(), phases.len());
    let leading_systems = messages
        .iter()
        .take_while(|message| message.role == MessageRole::System)
        .count();
    if leading_systems == 0 && instructions.is_none() {
        return;
    }

    let mut parts = Vec::with_capacity(leading_systems + usize::from(instructions.is_some()));
    if let Some(instructions) = instructions.filter(|value| !value.is_empty()) {
        parts.push(instructions.to_string());
    }
    parts.extend(
        messages
            .drain(..leading_systems)
            .map(|message| message.content)
            .filter(|content| !content.is_empty()),
    );
    phases.drain(..leading_systems);
    if !parts.is_empty() {
        messages.insert(0, chat_message(MessageRole::System, parts.join("\n\n")));
        phases.insert(0, None);
    }
}

struct ParsedInput {
    messages: Vec<ChatMessage>,
    phases: Vec<Option<AssistantMessagePhase>>,
}

#[cfg(test)]
pub(super) fn parse_input(input: &Value) -> std::result::Result<Vec<ChatMessage>, ServerError> {
    parse_input_with_tool_names(input, &ToolNameMap::default()).map(|parsed| parsed.messages)
}

#[cfg(test)]
pub(super) fn parse_input_phases(
    input: &Value,
) -> std::result::Result<Vec<Option<AssistantMessagePhase>>, ServerError> {
    parse_input_with_tool_names(input, &ToolNameMap::default()).map(|parsed| parsed.phases)
}

fn parse_input_with_tool_names(
    input: &Value,
    tool_names: &ToolNameMap,
) -> std::result::Result<ParsedInput, ServerError> {
    match input {
        Value::String(text) => Ok(ParsedInput {
            messages: vec![chat_message(MessageRole::User, text.clone())],
            phases: vec![None],
        }),
        Value::Array(items) => {
            let mut messages = Vec::new();
            let mut phases = Vec::new();
            let mut function_calls = HashSet::new();
            let mut function_outputs = HashSet::new();
            for (index, item) in items.iter().enumerate() {
                parse_input_item(
                    item,
                    index,
                    &mut messages,
                    &mut phases,
                    &mut function_calls,
                    &mut function_outputs,
                    tool_names,
                )?;
            }
            Ok(ParsedInput { messages, phases })
        }
        _ => Err(ServerError::invalid_request(
            "input must be a string or an array of input items",
            Some("input"),
        )),
    }
}

fn parse_input_item(
    item: &Value,
    index: usize,
    messages: &mut Vec<ChatMessage>,
    phases: &mut Vec<Option<AssistantMessagePhase>>,
    function_calls: &mut HashSet<String>,
    function_outputs: &mut HashSet<String>,
    tool_names: &ToolNameMap,
) -> std::result::Result<(), ServerError> {
    let object = item.as_object().ok_or_else(|| {
        ServerError::invalid_request(
            "each input item must be an object",
            Some(&format!("input[{index}]")),
        )
    })?;
    let item_type = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    match item_type {
        "message" => {
            let role_param = format!("input[{index}].role");
            let role = match required_string(object, "role", &role_param)?.as_str() {
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "system" | "developer" => MessageRole::System,
                _ => {
                    return Err(ServerError::invalid_request(
                        "message role must be user, assistant, system, or developer",
                        Some(&role_param),
                    ))
                }
            };
            let content_param = format!("input[{index}].content");
            let content = parse_message_content(object.get("content"), &content_param)?;
            let phase_param = format!("input[{index}].phase");
            let phase = match optional_string(object, "phase", &phase_param)?.as_deref() {
                None => None,
                Some(_) if role != MessageRole::Assistant => {
                    return Err(ServerError::invalid_request(
                        "phase is only valid for assistant messages",
                        Some(&phase_param),
                    ))
                }
                Some("commentary") => Some(AssistantMessagePhase::Commentary),
                Some("final_answer") => Some(AssistantMessagePhase::FinalAnswer),
                Some(_) => {
                    return Err(ServerError::invalid_request(
                        "assistant message phase must be commentary or final_answer",
                        Some(&phase_param),
                    ))
                }
            };
            let pending_reasoning = if role == MessageRole::Assistant {
                messages.last().is_some_and(|message| {
                    message.role == MessageRole::Assistant
                        && message.content.is_empty()
                        && message.reasoning.is_some()
                        && phases.last().is_some_and(Option::is_none)
                        && message.tool_calls.is_none()
                        && message.function_call.is_none()
                })
            } else {
                false
            };
            if pending_reasoning {
                let previous = messages.last_mut().expect("pending reasoning message");
                previous.content = content;
                *phases.last_mut().expect("pending reasoning phase") = phase;
            } else {
                messages.push(chat_message(role, content));
                phases.push(phase);
            }
        }
        "reasoning" => {
            if let Some(reasoning) = parse_reasoning_input_item(object, index)? {
                let append_to_previous = messages.last().is_some_and(|message| {
                    message.role == MessageRole::Assistant
                        && message.content.is_empty()
                        && message.tool_calls.is_none()
                        && message.function_call.is_none()
                        && phases.last().copied().flatten()
                            != Some(AssistantMessagePhase::FinalAnswer)
                });
                if append_to_previous {
                    let previous = messages.last_mut().expect("previous assistant message");
                    append_reasoning_text(&mut previous.reasoning, &reasoning);
                } else {
                    let mut message = chat_message(MessageRole::Assistant, String::new());
                    message.reasoning = Some(reasoning);
                    messages.push(message);
                    phases.push(None);
                }
            }
        }
        "function_call" => {
            let call_id_param = format!("input[{index}].call_id");
            let name_param = format!("input[{index}].name");
            let namespace_param = format!("input[{index}].namespace");
            let arguments_param = format!("input[{index}].arguments");
            let call_id = required_string(object, "call_id", &call_id_param)?;
            if !function_calls.insert(call_id.clone()) {
                return Err(ServerError::invalid_request(
                    "function call_id must be unique within input history",
                    Some(&call_id_param),
                ));
            }
            let name = required_string(object, "name", &name_param)?;
            let namespace = optional_string(object, "namespace", &namespace_param)?;
            if namespace.as_deref() == Some("") {
                return Err(ServerError::invalid_request(
                    "function namespace must be a non-empty string",
                    Some(&namespace_param),
                ));
            }
            let chat_name = match namespace.as_deref() {
                Some(namespace) => tool_names
                    .chat_name(Some(namespace), &name)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        ServerError::invalid_request(
                            format!(
                                "input function `{namespace}.{name}` is not present in request tools"
                            ),
                            Some(&namespace_param),
                        )
                    })?,
                None => tool_names
                    .chat_name(None, &name)
                    .unwrap_or(&name)
                    .to_string(),
            };
            let call = ChatToolCall {
                index: None,
                id: call_id,
                tool_type: "function".to_string(),
                function: ChatFunctionCall {
                    name: chat_name,
                    arguments: required_text(object, "arguments", &arguments_param)?,
                },
            };
            let append_to_previous = messages.last().is_some_and(|message| {
                message.role == MessageRole::Assistant
                    && phases.last().copied().flatten() != Some(AssistantMessagePhase::FinalAnswer)
            });
            if append_to_previous {
                let last = messages.last_mut().expect("previous assistant message");
                last.tool_calls.get_or_insert_with(Vec::new).push(call);
            } else {
                let mut message = chat_message(MessageRole::Assistant, String::new());
                message.tool_calls = Some(vec![call]);
                messages.push(message);
                phases.push(None);
            }
        }
        "function_call_output" => {
            let call_id_param = format!("input[{index}].call_id");
            let output_param = format!("input[{index}].output");
            let call_id = required_string(object, "call_id", &call_id_param)?;
            if !function_calls.contains(&call_id) {
                return Err(ServerError::invalid_request(
                    "function_call_output must reference an earlier function_call",
                    Some(&call_id_param),
                ));
            }
            if !function_outputs.insert(call_id.clone()) {
                return Err(ServerError::invalid_request(
                    "function_call_output must be unique for each call_id",
                    Some(&call_id_param),
                ));
            }
            let mut message = chat_message(
                MessageRole::Tool,
                parse_function_call_output(object.get("output"), &output_param)?,
            );
            message.tool_call_id = Some(call_id);
            messages.push(message);
            phases.push(None);
        }
        unsupported_type => {
            return Err(ServerError::unsupported_feature(
                format!("input item type `{unsupported_type}` is not supported yet"),
                Some(&format!("input[{index}].type")),
            ))
        }
    }
    Ok(())
}

fn parse_function_call_output(
    output: Option<&Value>,
    param: &str,
) -> std::result::Result<String, ServerError> {
    match output {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(_)) => parse_message_content(output, param),
        Some(_) => Err(ServerError::invalid_request(
            "function call output must be a string or an array of text parts",
            Some(param),
        )),
        None => Err(ServerError::invalid_request(
            "function call output is required",
            Some(param),
        )),
    }
}

fn parse_reasoning_input_item(
    object: &Map<String, Value>,
    index: usize,
) -> std::result::Result<Option<String>, ServerError> {
    reject_unknown_object_fields(
        object,
        &[
            "id",
            "type",
            "summary",
            "content",
            "encrypted_content",
            "status",
        ],
        &format!("input[{index}]"),
    )?;
    let raw = parse_reasoning_parts(
        object.get("content"),
        "reasoning_text",
        &format!("input[{index}].content"),
    )?;
    let summary = parse_reasoning_parts(
        object.get("summary"),
        "summary_text",
        &format!("input[{index}].summary"),
    )?;
    let encrypted_param = format!("input[{index}].encrypted_content");
    let encrypted = optional_string(object, "encrypted_content", &encrypted_param)?;
    if raw.is_empty() && encrypted.as_deref().is_some_and(|value| !value.is_empty()) {
        return Err(unsupported(
            "encrypted reasoning cannot be replayed without readable reasoning content",
            &encrypted_param,
        ));
    }
    let readable = (!raw.is_empty())
        .then_some(raw)
        .or_else(|| (!summary.is_empty()).then_some(summary));
    Ok(readable)
}

fn parse_reasoning_parts(
    value: Option<&Value>,
    expected_type: &str,
    param: &str,
) -> std::result::Result<String, ServerError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(String::new());
    };
    let parts = value.as_array().ok_or_else(|| {
        ServerError::invalid_request(format!("{param} must be an array"), Some(param))
    })?;
    let mut text = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let part_param = format!("{param}[{index}]");
        let object = part.as_object().ok_or_else(|| {
            ServerError::invalid_request(
                format!("{part_param} must be an object"),
                Some(&part_param),
            )
        })?;
        let type_param = format!("{part_param}.type");
        let part_type = required_string(object, "type", &type_param)?;
        if part_type != expected_type {
            return Err(unsupported(
                format!("reasoning content type `{part_type}` is not supported"),
                &type_param,
            ));
        }
        text.push(required_text(
            object,
            "text",
            &format!("{part_param}.text"),
        )?);
    }
    Ok(text.join("\n"))
}

fn append_reasoning_text(target: &mut Option<String>, reasoning: &str) {
    match target {
        Some(existing) if !existing.is_empty() && !reasoning.is_empty() => {
            existing.push('\n');
            existing.push_str(reasoning);
        }
        Some(existing) => existing.push_str(reasoning),
        None => *target = Some(reasoning.to_string()),
    }
}

fn parse_message_content(
    content: Option<&Value>,
    param: &str,
) -> std::result::Result<String, ServerError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                let object = part.as_object().ok_or_else(|| {
                    ServerError::invalid_request(
                        "message content parts must be objects",
                        Some(&format!("{param}[{index}]")),
                    )
                })?;
                let part_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    ServerError::invalid_request(
                        "message content part is missing type",
                        Some(&format!("{param}[{index}].type")),
                    )
                })?;
                if !matches!(part_type, "input_text" | "output_text" | "text") {
                    return Err(ServerError::unsupported_feature(
                        format!("message content type `{part_type}` is not supported yet"),
                        Some(&format!("{param}[{index}].type")),
                    ));
                }
                texts.push(required_text(
                    object,
                    "text",
                    &format!("{param}[{index}].text"),
                )?);
            }
            Ok(texts.join("\n"))
        }
        Some(Value::Null) | None => Ok(String::new()),
        _ => Err(ServerError::invalid_request(
            "message content must be a string or an array of text parts",
            Some(param),
        )),
    }
}

pub(super) fn required_string(
    object: &Map<String, Value>,
    key: &str,
    param: &str,
) -> std::result::Result<String, ServerError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            ServerError::invalid_request(format!("{param} must be a non-empty string"), Some(param))
        })
}

fn required_text(
    object: &Map<String, Value>,
    key: &str,
    param: &str,
) -> std::result::Result<String, ServerError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            ServerError::invalid_request(format!("{param} must be a string"), Some(param))
        })
}

pub(super) fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    param: &str,
) -> std::result::Result<Option<String>, ServerError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        _ => Err(ServerError::invalid_request(
            format!("{param} must be a string"),
            Some(param),
        )),
    }
}

pub(super) fn optional_bool(
    object: &Map<String, Value>,
    key: &str,
    param: &str,
) -> std::result::Result<Option<bool>, ServerError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        _ => Err(ServerError::invalid_request(
            format!("{param} must be a boolean"),
            Some(param),
        )),
    }
}
