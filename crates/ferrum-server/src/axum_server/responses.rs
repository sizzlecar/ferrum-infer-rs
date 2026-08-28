use super::{
    chat_completions_handler, json_rejection_detail, AppState, ServerError,
    DEFAULT_COMPLETION_MAX_TOKENS, DEFAULT_SAMPLING_TEMPERATURE, DEFAULT_SAMPLING_TOP_P,
};
use crate::openai::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatFunction, ChatFunctionCall, ChatMessage,
    ChatTool, ChatToolCall, MessageRole, StreamOptions, ToolChoice, ToolChoiceFunction, Usage,
};
use axum::{
    body::to_bytes,
    extract::{rejection::JsonRejection, State},
    http::HeaderMap,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{collections::BTreeMap, convert::Infallible};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

const MAX_SYNC_ADAPTER_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesRequest {
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
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

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

impl ResponsesRequest {
    fn convert(self) -> std::result::Result<ConvertedRequest, ServerError> {
        self.validate_stateless_scope()?;

        let model = self
            .model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| ServerError::invalid_request("model is required", Some("model")))?
            .to_string();
        let mut messages = parse_input(&self.input)?;
        if let Some(instructions) = self.instructions.as_ref() {
            messages.insert(0, chat_message(MessageRole::System, instructions.clone()));
        }
        if messages.is_empty() {
            return Err(ServerError::invalid_request(
                "input must contain at least one text message or function item",
                Some("input"),
            ));
        }

        let (chat_tools, response_tools) = parse_tools(self.tools.as_deref())?;
        let (chat_tool_choice, response_tool_choice) =
            parse_tool_choice(self.tool_choice.as_ref())?;
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
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            user: None,
            seed: None,
            response_format: None,
            tools: (!chat_tools.is_empty()).then_some(chat_tools),
            tool_choice: chat_tool_choice,
            stream_options: stream.then_some(StreamOptions {
                include_usage: Some(true),
            }),
            functions: None,
            function_call: None,
            metadata: None,
            chat_template_kwargs: None,
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
        };

        Ok(ConvertedRequest {
            chat,
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
            .include
            .as_ref()
            .is_some_and(|include| !include.is_empty())
        {
            return Err(unsupported(
                "include expansions are not supported by the stateless endpoint",
                "include",
            ));
        }
        if self.parallel_tool_calls == Some(false) {
            return Err(unsupported(
                "parallel_tool_calls=false is not supported yet",
                "parallel_tool_calls",
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

fn unsupported(message: impl Into<String>, param: &str) -> ServerError {
    ServerError::unsupported_feature(message, Some(param))
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

fn parse_input(input: &Value) -> std::result::Result<Vec<ChatMessage>, ServerError> {
    match input {
        Value::String(text) => Ok(vec![chat_message(MessageRole::User, text.clone())]),
        Value::Array(items) => {
            let mut messages = Vec::new();
            for (index, item) in items.iter().enumerate() {
                parse_input_item(item, index, &mut messages)?;
            }
            Ok(messages)
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
            messages.push(chat_message(role, content));
        }
        "function_call" => {
            let call_id_param = format!("input[{index}].call_id");
            let name_param = format!("input[{index}].name");
            let arguments_param = format!("input[{index}].arguments");
            let call = ChatToolCall {
                index: None,
                id: required_string(object, "call_id", &call_id_param)?,
                tool_type: "function".to_string(),
                function: ChatFunctionCall {
                    name: required_string(object, "name", &name_param)?,
                    arguments: required_string(object, "arguments", &arguments_param)?,
                },
            };
            if let Some(last) = messages.last_mut().filter(|message| {
                message.role == MessageRole::Assistant
                    && message.content.is_empty()
                    && message.tool_calls.is_some()
            }) {
                last.tool_calls.get_or_insert_with(Vec::new).push(call);
            } else {
                let mut message = chat_message(MessageRole::Assistant, String::new());
                message.tool_calls = Some(vec![call]);
                messages.push(message);
            }
        }
        "function_call_output" => {
            let call_id_param = format!("input[{index}].call_id");
            let output_param = format!("input[{index}].output");
            let mut message = chat_message(
                MessageRole::Tool,
                required_string(object, "output", &output_param)?,
            );
            message.tool_call_id = Some(required_string(object, "call_id", &call_id_param)?);
            messages.push(message);
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
                texts.push(required_string(
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

fn required_string(
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

fn parse_tools(
    tools: Option<&[Value]>,
) -> std::result::Result<(Vec<ChatTool>, Vec<Value>), ServerError> {
    let mut chat_tools = Vec::new();
    let mut response_tools = Vec::new();
    for (index, tool) in tools.unwrap_or_default().iter().enumerate() {
        let object = tool.as_object().ok_or_else(|| {
            ServerError::invalid_request(
                "tools entries must be objects",
                Some(&format!("tools[{index}]")),
            )
        })?;
        let type_param = format!("tools[{index}].type");
        let tool_type = required_string(object, "type", &type_param)?;
        if tool_type != "function" {
            return Err(ServerError::unsupported_feature(
                format!(
                    "tool type `{tool_type}` is not supported; only function tools are supported"
                ),
                Some(&type_param),
            ));
        }
        let name_param = format!("tools[{index}].name");
        let name = required_string(object, "name", &name_param)?;
        let description = optional_string(
            object,
            "description",
            &format!("tools[{index}].description"),
        )?;
        let parameters = object.get("parameters").cloned();
        if parameters.as_ref().is_some_and(|value| !value.is_object()) {
            return Err(ServerError::invalid_request(
                "function parameters must be a JSON object",
                Some(&format!("tools[{index}].parameters")),
            ));
        }
        let strict = optional_bool(object, "strict", &format!("tools[{index}].strict"))?;
        chat_tools.push(ChatTool {
            tool_type: "function".to_string(),
            function: ChatFunction {
                name: name.clone(),
                description: description.clone(),
                parameters: parameters.clone(),
                strict,
            },
        });
        let mut normalized = Map::new();
        normalized.insert("type".to_string(), json!("function"));
        normalized.insert("name".to_string(), json!(name));
        if let Some(description) = description {
            normalized.insert("description".to_string(), json!(description));
        }
        if let Some(parameters) = parameters {
            normalized.insert("parameters".to_string(), parameters);
        }
        if let Some(strict) = strict {
            normalized.insert("strict".to_string(), json!(strict));
        }
        response_tools.push(Value::Object(normalized));
    }
    Ok((chat_tools, response_tools))
}

fn optional_string(
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

fn optional_bool(
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

fn parse_tool_choice(
    choice: Option<&Value>,
) -> std::result::Result<(Option<ToolChoice>, Value), ServerError> {
    let Some(choice) = choice else {
        return Ok((None, json!("auto")));
    };
    match choice {
        Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
            Ok((Some(ToolChoice::Mode(mode.clone())), json!(mode)))
        }
        Value::Object(object) => {
            let tool_type = required_string(object, "type", "tool_choice.type")?;
            if tool_type != "function" {
                return Err(unsupported(
                    "only function tool_choice objects are supported",
                    "tool_choice.type",
                ));
            }
            let name = required_string(object, "name", "tool_choice.name")?;
            Ok((
                Some(ToolChoice::Function {
                    tool_type: "function".to_string(),
                    function: ToolChoiceFunction { name: name.clone() },
                }),
                json!({"type": "function", "name": name}),
            ))
        }
        _ => Err(ServerError::invalid_request(
            "tool_choice must be auto, none, required, or a function selector",
            Some("tool_choice"),
        )),
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
            "reasoning": null,
            "service_tier": "default",
            "store": false,
            "temperature": self.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_logprobs": 0,
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": usage.map(response_usage),
            "user": null
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
    let output = response_output_from_message(message);
    let status = if choice.finish_reason.as_deref() == Some("length") {
        "incomplete"
    } else {
        "completed"
    };
    Ok(Json(context.response(status, output, chat.usage.as_ref(), None)).into_response())
}

fn response_output_from_message(message: ChatMessage) -> Vec<Value> {
    let mut output = Vec::new();
    let has_calls = message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
        || message.function_call.is_some();
    if !message.content.is_empty() || !has_calls {
        output.push(message_output_item(
            format!("msg_{}", Uuid::new_v4().simple()),
            message.content,
            "completed",
        ));
    }
    for call in message.tool_calls.unwrap_or_default() {
        output.push(function_output_item(&call, "completed"));
    }
    if let Some(function) = message.function_call {
        let call = ChatToolCall {
            index: None,
            id: format!("call_{}", Uuid::new_v4().simple()),
            tool_type: "function".to_string(),
            function,
        };
        output.push(function_output_item(&call, "completed"));
    }
    output
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

fn function_output_item(call: &ChatToolCall, status: &str) -> Value {
    json!({
        "id": format!("fc_{}", Uuid::new_v4().simple()),
        "type": "function_call",
        "status": status,
        "call_id": call.id,
        "name": call.function.name,
        "arguments": call.function.arguments
    })
}

type EventSender = mpsc::UnboundedSender<std::result::Result<Event, Infallible>>;

fn adapt_chat_stream(response: Response, context: ResponseContext) -> Response {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut state = ResponsesStreamState::new(context);
        state.send_initial_events(&tx);
        let mut decoder = SseDecoder::default();
        let mut body = response.into_body().into_data_stream();
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => {
                    for data in decoder.push(&bytes) {
                        state.consume_chat_event(&data, &tx);
                    }
                }
                Err(error) => {
                    state.fail(format!("failed to read chat stream: {error}"), &tx);
                    return;
                }
            }
            if state.terminal {
                return;
            }
        }
        for data in decoder.finish() {
            state.consume_chat_event(&data, &tx);
        }
        if !state.terminal {
            state.fail("chat stream ended before [DONE]".to_string(), &tx);
        }
    });
    Sse::new(UnboundedReceiverStream::new(rx)).into_response()
}

struct ResponsesStreamState {
    context: ResponseContext,
    sequence: u64,
    text_item_id: String,
    text: String,
    text_started: bool,
    tool_calls: BTreeMap<u32, ToolCallAccumulator>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    terminal: bool,
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

impl ResponsesStreamState {
    fn new(context: ResponseContext) -> Self {
        Self {
            context,
            sequence: 0,
            text_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            text: String::new(),
            text_started: false,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            usage: None,
            terminal: false,
        }
    }

    fn send_initial_events(&mut self, tx: &EventSender) {
        let response = self.context.response("in_progress", vec![], None, None);
        self.send(tx, "response.created", json!({"response": response}));
        let response = self.context.response("in_progress", vec![], None, None);
        self.send(tx, "response.in_progress", json!({"response": response}));
    }

    fn consume_chat_event(&mut self, data: &str, tx: &EventSender) {
        if self.terminal {
            return;
        }
        if data == "[DONE]" {
            self.complete(tx);
            return;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => {
                self.fail(format!("invalid chat stream event: {error}"), tx);
                return;
            }
        };
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("chat stream failed")
                .to_string();
            self.fail(message, tx);
            return;
        }
        let chunk: ChatCompletionsResponse = match serde_json::from_value(value) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.fail(format!("invalid chat completion chunk: {error}"), tx);
                return;
            }
        };
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                if !delta.content.is_empty() {
                    self.push_text(&delta.content, tx);
                }
                for (position, call) in delta.tool_calls.unwrap_or_default().into_iter().enumerate()
                {
                    let index = call.index.unwrap_or(position as u32);
                    let accumulated = self.tool_calls.entry(index).or_default();
                    if !call.id.is_empty() {
                        accumulated.id = call.id;
                    }
                    if !call.function.name.is_empty() {
                        accumulated.name = call.function.name;
                    }
                    accumulated.arguments.push_str(&call.function.arguments);
                }
                if let Some(function) = delta.function_call {
                    let accumulated = self.tool_calls.entry(0).or_default();
                    accumulated.id = accumulated
                        .id
                        .is_empty()
                        .then(|| format!("call_{}", Uuid::new_v4().simple()))
                        .unwrap_or_else(|| accumulated.id.clone());
                    accumulated.name = function.name;
                    accumulated.arguments.push_str(&function.arguments);
                }
            }
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
        }
    }

    fn push_text(&mut self, delta: &str, tx: &EventSender) {
        if !self.text_started {
            self.text_started = true;
            self.send(
                tx,
                "response.output_item.added",
                json!({
                    "output_index": 0,
                    "item": {
                        "id": self.text_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress"
                    }
                }),
            );
            self.send(
                tx,
                "response.content_part.added",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                }),
            );
        }
        self.text.push_str(delta);
        self.send(
            tx,
            "response.output_text.delta",
            json!({
                "item_id": self.text_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "logprobs": []
            }),
        );
    }

    fn complete(&mut self, tx: &EventSender) {
        if self.terminal {
            return;
        }
        let mut output = Vec::new();
        if self.text_started || self.tool_calls.is_empty() {
            if !self.text_started {
                self.push_text("", tx);
            }
            self.send(
                tx,
                "response.output_text.done",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": self.text,
                    "logprobs": []
                }),
            );
            let item =
                message_output_item(self.text_item_id.clone(), self.text.clone(), "completed");
            self.send(
                tx,
                "response.content_part.done",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": item["content"][0].clone()
                }),
            );
            self.send(
                tx,
                "response.output_item.done",
                json!({"output_index": 0, "item": item.clone()}),
            );
            output.push(item);
        }

        let calls = std::mem::take(&mut self.tool_calls);
        for (_, call) in calls {
            let output_index = output.len();
            let call_id = if call.id.is_empty() {
                format!("call_{}", Uuid::new_v4().simple())
            } else {
                call.id
            };
            let item_id = format!("fc_{}", Uuid::new_v4().simple());
            let in_progress = json!({
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": call.name,
                "arguments": ""
            });
            self.send(
                tx,
                "response.output_item.added",
                json!({"output_index": output_index, "item": in_progress}),
            );
            if !call.arguments.is_empty() {
                self.send(
                    tx,
                    "response.function_call_arguments.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": call.arguments
                    }),
                );
            }
            self.send(
                tx,
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "call_id": call_id,
                    "name": call.name,
                    "arguments": call.arguments
                }),
            );
            let item = json!({
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": call.name,
                "arguments": call.arguments
            });
            self.send(
                tx,
                "response.output_item.done",
                json!({"output_index": output_index, "item": item.clone()}),
            );
            output.push(item);
        }

        let status = if self.finish_reason.as_deref() == Some("length") {
            "incomplete"
        } else {
            "completed"
        };
        let response = self
            .context
            .response(status, output, self.usage.as_ref(), None);
        self.send(tx, "response.completed", json!({"response": response}));
        self.terminal = true;
    }

    fn fail(&mut self, message: String, tx: &EventSender) {
        if self.terminal {
            return;
        }
        let response = self.context.response(
            "failed",
            vec![],
            self.usage.as_ref(),
            Some(json!({"code": "internal_server_error", "message": message})),
        );
        self.send(tx, "response.failed", json!({"response": response}));
        self.terminal = true;
    }

    fn send(&mut self, tx: &EventSender, event_type: &str, mut event: Value) {
        if let Some(object) = event.as_object_mut() {
            object.insert("type".to_string(), json!(event_type));
            object.insert("sequence_number".to_string(), json!(self.sequence));
        }
        self.sequence += 1;
        let _ = tx.send(Ok(Event::default()
            .event(event_type)
            .data(event.to_string())));
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut data = Vec::new();
        while let Some((start, separator_len)) = find_sse_separator(&self.buffer) {
            let frame = self.buffer.drain(..start).collect::<Vec<_>>();
            self.buffer.drain(..separator_len);
            if let Some(value) = sse_data(&frame) {
                data.push(value);
            }
        }
        data
    }

    fn finish(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let frame = std::mem::take(&mut self.buffer);
        sse_data(&frame).into_iter().collect()
    }
}

fn find_sse_separator(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes.windows(2).position(|window| window == b"\n\n");
    let crlf = bytes.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left <= right => Some((left, 2)),
        (Some(_), Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, Some(right)) => Some((right, 4)),
        (None, None) => None,
    }
}

fn sse_data(frame: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(frame);
    let lines = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
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
    fn responses_sse_decoder_handles_split_frames() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: {\"a\":").is_empty());
        assert_eq!(
            decoder.push(b"1}\n\ndata: [DONE]\n\n"),
            vec!["{\"a\":1}", "[DONE]"]
        );
    }
}
