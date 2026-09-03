//! Request and response types for inference

mod xml_parameter_schema;

use crate::{
    ids::*, models::TokenUsage, FinishReason, Priority, ResponseCompletionEnvelope, SamplingParams,
    TokenId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xml_parameter_schema::XmlParameterSchemaProbe;

pub const PROMPT_TOKENS_METADATA_KEY: &str = "ferrum_prompt_tokens";
pub const DEFAULT_MAX_TOKENS_METADATA_KEY: &str = "ferrum_default_max_tokens";

/// Explicit request for execution evidence that is expensive or sensitive to retain.
///
/// The default keeps the inference hot path unchanged. Product entrypoints opt in
/// only when the user enables a diagnostic artifact such as a request replay dump.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceEvidenceRequest {
    #[serde(default)]
    pub capture_prompt_token_ids: bool,
    /// Capture engine token-commit timestamps from the request's monotonic
    /// clock. Disabled by default so ordinary inference does not allocate or
    /// read an additional clock in the token hot path.
    #[serde(default)]
    pub capture_engine_token_timing: bool,
}

/// Engine-boundary timing for one inference request.
///
/// Every token timestamp is an offset from the same `std::time::Instant`
/// captured at request admission. The wall anchor exists only to correlate
/// this monotonic domain with profile events; durations and ITL must be
/// computed from `token_commit_nanos_since_request_start`, never by
/// subtracting wall-clock timestamps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineDecodeStage {
    /// Host scheduling work that selected this request for a decode wave.
    DecodeScheduling,
    /// The model-executor call for one decode wave, including preparation,
    /// device submission/completion, readback, and executor-side retirement.
    DecodeExecution,
    /// Host-side validation, sampling, and state commit after device execution.
    DecodePostprocess,
}

/// A measured request-local engine decode stage in the request's monotonic
/// clock domain. These are explicit producer boundaries, not intervals inferred
/// by filling gaps between other profile events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineDecodeStageInterval {
    pub stage: EngineDecodeStage,
    pub start_nanos_since_request_start: u64,
    pub end_nanos_since_request_start: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineTokenTimingEvidence {
    pub clock_source: String,
    pub wall_anchor_unix_nanos: i64,
    pub wall_anchor_max_error_nanos: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ready_nanos_since_request_start: Option<u64>,
    pub token_commit_nanos_since_request_start: Vec<u64>,
    /// Opt-in decode-stage evidence captured only with engine token timing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decode_stage_intervals: Vec<EngineDecodeStageInterval>,
}

impl EngineTokenTimingEvidence {
    pub fn validate(&self, output_tokens: usize) -> Result<(), String> {
        if self.clock_source != "rust_std_instant" {
            return Err("engine token timing clock_source must be rust_std_instant".to_string());
        }
        if self.wall_anchor_unix_nanos <= 0 {
            return Err("engine token timing wall anchor must be positive".to_string());
        }
        if self.token_commit_nanos_since_request_start.len() != output_tokens {
            return Err(format!(
                "engine token timing has {} commits for {output_tokens} output tokens",
                self.token_commit_nanos_since_request_start.len()
            ));
        }
        if self
            .token_commit_nanos_since_request_start
            .windows(2)
            .any(|window| window[1] < window[0])
        {
            return Err("engine token commit timestamps must be monotonic".to_string());
        }
        if self.decode_stage_intervals.iter().any(|interval| {
            interval.end_nanos_since_request_start < interval.start_nanos_since_request_start
        }) {
            return Err("engine decode stage interval end precedes start".to_string());
        }
        if self.decode_stage_intervals.windows(2).any(|window| {
            window[1].start_nanos_since_request_start < window[0].start_nanos_since_request_start
        }) {
            return Err("engine decode stage intervals must be ordered by start".to_string());
        }
        Ok(())
    }

    pub fn ttft_nanos(&self) -> Option<u64> {
        self.token_commit_nanos_since_request_start.first().copied()
    }

    pub fn inter_token_nanos(&self) -> Vec<u64> {
        self.token_commit_nanos_since_request_start
            .windows(2)
            .map(|window| window[1].saturating_sub(window[0]))
            .collect()
    }

    pub fn decode_wall_nanos(&self) -> Option<u64> {
        let start = self.decode_ready_nanos_since_request_start?;
        let end = self
            .token_commit_nanos_since_request_start
            .last()
            .copied()?;
        (end >= start).then_some(end - start)
    }
}

/// Evidence captured at the engine execution boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceExecutionEvidence {
    #[serde(default)]
    pub prompt_token_ids: Vec<TokenId>,
    /// Complete generated-token history at the engine boundary. Streaming
    /// chunks cannot reconstruct this reliably because special tokens and
    /// deferred multi-byte pieces may not produce a visible text delta.
    #[serde(default)]
    pub output_token_ids: Vec<TokenId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_token_timing: Option<EngineTokenTimingEvidence>,
}

/// Inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier
    pub id: RequestId,
    /// Input prompt text
    pub prompt: String,
    /// Model to use for inference
    pub model_id: ModelId,
    /// Sampling parameters
    pub sampling_params: SamplingParams,
    /// Whether to stream response
    pub stream: bool,
    /// Request priority
    pub priority: Priority,
    /// Client identifier
    pub client_id: Option<ClientId>,
    /// Session identifier for stateful interactions
    pub session_id: Option<SessionId>,
    /// Request creation timestamp
    pub created_at: DateTime<Utc>,
    /// Structured product/API request context. `prompt` remains the rendered
    /// model input for current engines; this carries the original semantic
    /// request boundary for API features such as tools and response formats.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_request: Option<ApiRequest>,
    /// Explicitly requested execution evidence. Disabled by default so normal
    /// inference does not retain or copy prompt token IDs after completion.
    #[serde(default)]
    pub evidence_request: InferenceEvidenceRequest,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApiRequest {
    Chat(ApiChatRequest),
    Completion(ApiCompletionRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApiResponse {
    Chat(ApiChatResponse),
    Completion(ApiCompletionResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiChatRequest {
    pub messages: Vec<ApiChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ApiTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ApiToolChoice>,
    /// Wire protocol emitted by the model's chat template for automatic tool calls.
    /// Forced/named calls may still use a hard argument grammar and the JSON fallback.
    #[serde(default)]
    pub tool_call_protocol: ApiToolCallProtocol,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub legacy_functions: Vec<ApiFunction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_function_call: Option<ApiFunctionCallChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ApiResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ApiStreamOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiCompletionRequest {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ApiResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiChatResponse {
    pub message: ApiChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiCompletionResponse {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiChatMessage {
    pub role: ApiMessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ApiToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ApiFunctionCall>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApiMessageRole {
    System,
    User,
    Assistant,
    Function,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ApiFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ApiToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: ApiToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiToolCallProtocol {
    #[default]
    Json,
    FunctionParameterXml,
}

impl ApiToolCallProtocol {
    /// Atomic tokenizer entries that this wire protocol may need to emit.
    ///
    /// The sampling layer resolves these texts against the loaded tokenizer
    /// and allows only exact matches. Other added-vocabulary tokens remain
    /// masked.
    pub const fn generated_control_token_texts(self) -> &'static [&'static str] {
        match self {
            Self::Json => &[],
            Self::FunctionParameterXml => &["<tool_call>", "</tool_call>"],
        }
    }

    /// Lexical envelope that may satisfy a pending response boundary.
    pub fn generated_response_envelope(self) -> Option<ResponseCompletionEnvelope> {
        match self {
            Self::Json => None,
            Self::FunctionParameterXml => Some(ResponseCompletionEnvelope {
                open_token_text: "<tool_call>".to_string(),
                close_token_text: "</tool_call>".to_string(),
                max_envelopes: MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE,
            }),
        }
    }
}

impl ApiChatRequest {
    /// Protocol controls are generatable only when this request can emit a
    /// modern tool call. `tool_choice: none` must not widen token sampling.
    pub fn generated_control_token_texts(&self) -> &'static [&'static str] {
        if self.tools.is_empty() || api_tool_choice_is_none(self) {
            return &[];
        }
        self.tool_call_protocol.generated_control_token_texts()
    }

    /// Alternate complete response envelope available to this request.
    pub fn generated_response_envelope(&self) -> Option<ResponseCompletionEnvelope> {
        if self.tools.is_empty() || api_tool_choice_is_none(self) {
            return None;
        }
        self.tool_call_protocol.generated_response_envelope()
    }
}

impl ApiRequest {
    pub fn generated_control_token_texts(&self) -> &'static [&'static str] {
        match self {
            Self::Chat(request) => request.generated_control_token_texts(),
            Self::Completion(_) => &[],
        }
    }

    pub fn generated_response_envelope(&self) -> Option<ResponseCompletionEnvelope> {
        match self {
            Self::Chat(request) => request.generated_response_envelope(),
            Self::Completion(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ApiFunctionCallChoice {
    Mode(String),
    Function { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ApiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<ApiJsonSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiJsonSchema {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiStreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

const MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE: usize = 32;

pub fn api_response_from_generated_text(
    request: &InferenceRequest,
    text: &str,
    finish_reason: FinishReason,
) -> Option<ApiResponse> {
    let ApiRequest::Chat(chat_request) = request.api_request.as_ref()? else {
        return None;
    };
    chat_api_response_from_generated_text(chat_request, text, finish_reason).map(ApiResponse::Chat)
}

pub fn chat_api_may_emit_tool_or_function_call(chat_request: &ApiChatRequest) -> bool {
    (!chat_request.tools.is_empty() && !api_tool_choice_is_none(chat_request))
        || (!chat_request.legacy_functions.is_empty()
            && !api_function_call_choice_is_none(chat_request))
}

pub fn chat_api_response_from_generated_text(
    chat_request: &ApiChatRequest,
    text: &str,
    finish_reason: FinishReason,
) -> Option<ApiChatResponse> {
    if !matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) {
        return None;
    }

    if !chat_request.tools.is_empty() && !api_tool_choice_is_none(chat_request) {
        if let Some(tool_calls) = parse_tool_calls_from_generated_text(text, chat_request) {
            return Some(ApiChatResponse {
                message: ApiChatMessage {
                    role: ApiMessageRole::Assistant,
                    content: String::new(),
                    name: None,
                    tool_calls,
                    tool_call_id: None,
                    function_call: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            });
        }
    }

    if !chat_request.legacy_functions.is_empty() && !api_function_call_choice_is_none(chat_request)
    {
        if let Some(function_call) =
            parse_legacy_function_call_from_generated_text(text, chat_request)
        {
            return Some(ApiChatResponse {
                message: ApiChatMessage {
                    role: ApiMessageRole::Assistant,
                    content: String::new(),
                    name: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    function_call: Some(function_call),
                },
                finish_reason: Some("function_call".to_string()),
            });
        }
    }

    None
}

fn api_tool_choice_is_none(chat_request: &ApiChatRequest) -> bool {
    matches!(
        chat_request.tool_choice.as_ref(),
        Some(ApiToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none")
    )
}

fn api_function_call_choice_is_none(chat_request: &ApiChatRequest) -> bool {
    matches!(
        chat_request.legacy_function_call.as_ref(),
        Some(ApiFunctionCallChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none")
    )
}

fn parse_tool_calls_from_generated_text(
    text: &str,
    chat_request: &ApiChatRequest,
) -> Option<Vec<ApiToolCall>> {
    if chat_request.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml {
        if let Some(calls) = parse_function_parameter_xml_tool_calls(text, chat_request) {
            return validate_parsed_tool_calls(calls);
        }
    }

    let value = parse_json_value_from_generated_text(text)?;
    if let Some(calls) = value.get("tool_calls").and_then(|value| value.as_array()) {
        if calls.len() > MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE {
            return None;
        }
        let parsed = calls
            .iter()
            .enumerate()
            .map(|(index, value)| parse_tool_call_value(value, index, chat_request))
            .collect::<Option<Vec<_>>>()?;
        return validate_parsed_tool_calls(parsed);
    }
    if let Some(tool_call) = value.get("tool_call") {
        return parse_tool_call_value(tool_call, 0, chat_request)
            .and_then(|call| validate_parsed_tool_calls(vec![call]));
    }
    if let Some(tool_call) = parse_wrapped_tool_call_value(&value, 0, chat_request) {
        return validate_parsed_tool_calls(vec![tool_call]);
    }
    parse_tool_call_value(&value, 0, chat_request)
        .or_else(|| parse_forced_tool_arguments_value(&value, 0, chat_request))
        .and_then(|call| validate_parsed_tool_calls(vec![call]))
}

fn validate_parsed_tool_calls(calls: Vec<ApiToolCall>) -> Option<Vec<ApiToolCall>> {
    if calls.is_empty() || calls.len() > MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE {
        return None;
    }
    for (index, call) in calls.iter().enumerate() {
        if calls[..index].iter().any(|previous| {
            previous.function.name == call.function.name
                && previous.function.arguments == call.function.arguments
        }) {
            return None;
        }
    }
    Some(calls)
}

fn parse_function_parameter_xml_tool_calls(
    text: &str,
    chat_request: &ApiChatRequest,
) -> Option<Vec<ApiToolCall>> {
    const TOOL_START: &str = "<tool_call>";
    const TOOL_END: &str = "</tool_call>";
    const FUNCTION_START: &str = "<function=";
    const FUNCTION_END: &str = "</function>";

    let mut remaining = text;
    let mut calls = Vec::new();
    while let Some(tool_start) = remaining.find(TOOL_START) {
        if calls.len() == MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE {
            return None;
        }
        remaining = &remaining[tool_start + TOOL_START.len()..];
        let tool_end = remaining.find(TOOL_END)?;
        let block = &remaining[..tool_end];
        remaining = &remaining[tool_end + TOOL_END.len()..];

        let Some(function_start) = block.find(FUNCTION_START) else {
            return None;
        };
        if !block[..function_start].trim().is_empty() {
            return None;
        }
        let function = &block[function_start + FUNCTION_START.len()..];
        let Some(name_end) = function.find('>') else {
            return None;
        };
        let name = function[..name_end].trim();
        if !api_tool_name_allowed(chat_request, name) {
            return None;
        }
        let arguments_end = function[name_end + 1..]
            .find(FUNCTION_END)
            .map(|offset| name_end + 1 + offset)?;
        if !function[arguments_end + FUNCTION_END.len()..]
            .trim()
            .is_empty()
        {
            return None;
        }
        let parameter_schema = chat_request
            .tools
            .iter()
            .find(|tool| tool.tool_type == "function" && tool.function.name == name)
            .and_then(|tool| tool.function.parameters.as_ref());
        let arguments = parse_function_parameter_xml_arguments(
            &function[name_end + 1..arguments_end],
            parameter_schema,
        )?;
        let arguments = serde_json::to_string(&arguments).ok()?;
        calls.push(ApiToolCall {
            id: format!("call_{}", calls.len()),
            tool_type: "function".to_string(),
            function: ApiFunctionCall {
                name: name.to_string(),
                arguments,
            },
        });
    }

    (!calls.is_empty()).then_some(calls)
}

fn parse_function_parameter_xml_arguments(
    text: &str,
    parameter_schema: Option<&serde_json::Value>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    const PARAMETER_START: &str = "<parameter=";
    const PARAMETER_END: &str = "</parameter>";

    let mut arguments = serde_json::Map::new();
    let mut schema_probe = parameter_schema.map(XmlParameterSchemaProbe::new);
    let mut remaining = text;
    while let Some(parameter_start) = remaining.find(PARAMETER_START) {
        remaining = &remaining[parameter_start + PARAMETER_START.len()..];
        let Some(name_end) = remaining.find('>') else {
            return None;
        };
        let name = remaining[..name_end].trim();
        remaining = &remaining[name_end + 1..];
        if name.is_empty() {
            return None;
        }
        let value_end = remaining.find(PARAMETER_END)?;
        if arguments.contains_key(name) {
            return None;
        }
        let value = strip_xml_parameter_wrapper_newlines(&remaining[..value_end]);
        let value = schema_probe.as_mut().map_or_else(
            || serde_json::Value::String(value.to_string()),
            |probe| probe.decode(name, value),
        );
        arguments.insert(name.to_string(), value);
        remaining = &remaining[value_end + PARAMETER_END.len()..];
    }
    Some(arguments)
}

/// Qwen-style function XML renders one structural newline immediately inside
/// each parameter tag. Remove only that framing while preserving whitespace
/// that belongs to the argument itself, such as code indentation or a final
/// newline used by exact-match edit tools.
fn strip_xml_parameter_wrapper_newlines(value: &str) -> &str {
    if let Some(value) = value.strip_prefix("\r\n") {
        return value.strip_suffix("\r\n").unwrap_or(value);
    }
    if let Some(value) = value.strip_prefix('\n') {
        if value.ends_with("\r\n") {
            return value;
        }
        return value.strip_suffix('\n').unwrap_or(value);
    }
    value
}

fn parse_wrapped_tool_call_value(
    value: &serde_json::Value,
    index: usize,
    chat_request: &ApiChatRequest,
) -> Option<ApiToolCall> {
    for key in ["auto", "tool", "tool_call", "auto_tool_response"] {
        if let Some(wrapped) = value.get(key) {
            if let Some(call) = parse_tool_call_value(wrapped, index, chat_request) {
                return Some(call);
            }
        }
    }
    None
}

fn parse_tool_call_value(
    value: &serde_json::Value,
    index: usize,
    chat_request: &ApiChatRequest,
) -> Option<ApiToolCall> {
    let tool_type = value
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("function");
    if tool_type != "function" {
        return None;
    }
    let function = value.get("function").unwrap_or(value);
    let name = function
        .as_str()
        .or_else(|| function.get("name").and_then(|value| value.as_str()))
        .or_else(|| function.get("tool").and_then(|value| value.as_str()))
        .or_else(|| value.get("name").and_then(|value| value.as_str()))?;
    if !api_tool_name_allowed(chat_request, name) {
        return None;
    }
    let arguments = api_arguments_to_string(
        function
            .get("arguments")
            .or_else(|| function.get("parameters"))
            .or_else(|| value.get("arguments"))
            .or_else(|| value.get("parameters")),
    );
    let id = value
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{index}"));

    Some(ApiToolCall {
        id,
        tool_type: "function".to_string(),
        function: ApiFunctionCall {
            name: name.to_string(),
            arguments,
        },
    })
}

fn parse_forced_tool_arguments_value(
    value: &serde_json::Value,
    index: usize,
    chat_request: &ApiChatRequest,
) -> Option<ApiToolCall> {
    let tool = unwrapped_tool_arguments_target(chat_request, value)?;
    if value.get("tool_calls").is_some()
        || value.get("tool_call").is_some()
        || value.get("function").is_some()
        || value.get("name").is_some()
    {
        return None;
    }

    Some(ApiToolCall {
        id: format!("call_{index}"),
        tool_type: "function".to_string(),
        function: ApiFunctionCall {
            name: tool.function.name.clone(),
            arguments: serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
        },
    })
}

fn unwrapped_tool_arguments_target<'a>(
    chat_request: &'a ApiChatRequest,
    value: &serde_json::Value,
) -> Option<&'a ApiTool> {
    if let Some(name) = forced_tool_choice_name(chat_request) {
        return chat_request
            .tools
            .iter()
            .find(|tool| tool.tool_type == "function" && tool.function.name == name);
    }

    if matches!(
        chat_request.tool_choice.as_ref(),
        Some(ApiToolChoice::Mode(mode)) if !mode.eq_ignore_ascii_case("auto")
    ) {
        return None;
    }

    let tool = single_function_tool(chat_request)?;
    if !value_looks_like_tool_arguments(value, tool) {
        return None;
    }
    Some(tool)
}

fn value_looks_like_tool_arguments(value: &serde_json::Value, tool: &ApiTool) -> bool {
    let Some(arguments) = value.as_object() else {
        return false;
    };
    if arguments.is_empty() {
        return false;
    }
    let Some(properties) = tool
        .function
        .parameters
        .as_ref()
        .and_then(|parameters| parameters.get("properties"))
        .and_then(|properties| properties.as_object())
    else {
        return false;
    };
    arguments.keys().all(|key| properties.contains_key(key))
}

fn forced_tool_choice_name(chat_request: &ApiChatRequest) -> Option<&str> {
    match chat_request.tool_choice.as_ref() {
        Some(ApiToolChoice::Function {
            tool_type,
            function,
        }) if tool_type == "function" && api_tool_name_allowed(chat_request, &function.name) => {
            Some(function.name.as_str())
        }
        Some(ApiToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("required") => {
            single_function_tool(chat_request).map(|tool| tool.function.name.as_str())
        }
        _ => None,
    }
}

fn single_function_tool(chat_request: &ApiChatRequest) -> Option<&ApiTool> {
    let mut tools = chat_request
        .tools
        .iter()
        .filter(|tool| tool.tool_type == "function");
    let tool = tools.next()?;
    tools.next().is_none().then_some(tool)
}

fn parse_legacy_function_call_from_generated_text(
    text: &str,
    chat_request: &ApiChatRequest,
) -> Option<ApiFunctionCall> {
    let value = parse_json_value_from_generated_text(text)?;
    let function = value.get("function_call").unwrap_or(&value);
    let name = function.get("name").and_then(|value| value.as_str())?;
    if !api_function_name_allowed(chat_request, name) {
        return None;
    }
    Some(ApiFunctionCall {
        name: name.to_string(),
        arguments: api_arguments_to_string(function.get("arguments")),
    })
}

fn api_tool_name_allowed(chat_request: &ApiChatRequest, name: &str) -> bool {
    match chat_request.tool_choice.as_ref() {
        Some(ApiToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none") => false,
        Some(ApiToolChoice::Function {
            tool_type,
            function,
        }) => {
            tool_type == "function"
                && function.name == name
                && chat_request
                    .tools
                    .iter()
                    .any(|tool| tool.function.name == name)
        }
        _ => chat_request
            .tools
            .iter()
            .any(|tool| tool.function.name == name),
    }
}

fn api_function_name_allowed(chat_request: &ApiChatRequest, name: &str) -> bool {
    match chat_request.legacy_function_call.as_ref() {
        Some(ApiFunctionCallChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none") => false,
        Some(ApiFunctionCallChoice::Function { name: selected }) => {
            selected == name
                && chat_request
                    .legacy_functions
                    .iter()
                    .any(|function| function.name == name)
        }
        _ => chat_request
            .legacy_functions
            .iter()
            .any(|function| function.name == name),
    }
}

fn parse_json_value_from_generated_text(text: &str) -> Option<serde_json::Value> {
    let trimmed = strip_single_json_fence(text.trim());
    serde_json::from_str(trimmed).ok().or_else(|| {
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        (start <= end)
            .then(|| serde_json::from_str(&trimmed[start..=end]).ok())
            .flatten()
    })
}

fn strip_single_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest).trim_start();
    rest.strip_suffix("```").map(str::trim).unwrap_or(text)
}

fn api_arguments_to_string(arguments: Option<&serde_json::Value>) -> String {
    match arguments {
        Some(serde_json::Value::String(raw)) => raw.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    }
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(prompt: impl Into<String>, model_id: impl Into<ModelId>) -> Self {
        Self {
            id: RequestId::new(),
            prompt: prompt.into(),
            model_id: model_id.into(),
            sampling_params: SamplingParams::default(),
            stream: false,
            priority: Priority::default(),
            client_id: None,
            session_id: None,
            created_at: Utc::now(),
            api_request: None,
            evidence_request: InferenceEvidenceRequest::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set sampling parameters
    pub fn with_sampling_params(mut self, params: SamplingParams) -> Self {
        self.sampling_params = params;
        self
    }

    /// Enable streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set client ID
    pub fn with_client_id(mut self, client_id: impl Into<ClientId>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// Set session ID
    pub fn with_session_id(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set structured product/API request context.
    pub fn with_api_request(mut self, api_request: ApiRequest) -> Self {
        self.api_request = Some(api_request);
        self
    }

    /// Request prompt-token evidence from the execution boundary.
    pub fn with_prompt_token_evidence(mut self) -> Self {
        self.evidence_request.capture_prompt_token_ids = true;
        self
    }

    /// Request engine token-commit timing evidence.
    pub fn with_engine_token_timing_evidence(mut self) -> Self {
        self.evidence_request.capture_engine_token_timing = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID this response corresponds to
    pub request_id: RequestId,
    /// Generated text
    pub text: String,
    /// Generated token IDs
    pub tokens: Vec<TokenId>,
    /// Reason for completion
    pub finish_reason: FinishReason,
    /// Token usage statistics
    pub usage: TokenUsage,
    /// Total latency in milliseconds
    pub latency_ms: u64,
    /// Response creation timestamp
    pub created_at: DateTime<Utc>,
    /// Additional response metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Structured product/API response context. Engines that can produce
    /// product-native outputs, such as assistant tool calls, can populate
    /// this without overloading plain text or ad hoc metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_response: Option<ApiResponse>,
    /// Optional engine-boundary evidence requested by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_evidence: Option<InferenceExecutionEvidence>,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Request ID this chunk corresponds to
    pub request_id: RequestId,
    /// Text delta for this chunk
    pub text: String,
    /// Token ID for this chunk (if available)
    pub token: Option<TokenId>,
    /// Finish reason if this is the final chunk
    pub finish_reason: Option<FinishReason>,
    /// Token usage (typically only in final chunk)
    pub usage: Option<TokenUsage>,
    /// Chunk creation timestamp
    pub created_at: DateTime<Utc>,
    /// Chunk metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Structured product/API response context for final streaming chunks.
    /// This mirrors `InferenceResponse::api_response` so streaming endpoints
    /// can return native tool/function-call payloads without reparsing text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_response: Option<ApiResponse>,
    /// Optional engine-boundary evidence, emitted on the final chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_evidence: Option<InferenceExecutionEvidence>,
}

/// Batch request for processing multiple requests together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Batch identifier
    pub batch_id: BatchId,
    /// Requests in this batch
    pub requests: Vec<InferenceRequest>,
    /// Maximum sequence length for this batch
    pub max_sequence_length: usize,
    /// Batch creation timestamp
    pub created_at: DateTime<Utc>,
}

impl BatchRequest {
    /// Create a new batch request
    pub fn new(requests: Vec<InferenceRequest>) -> Self {
        let max_sequence_length = requests
            .iter()
            .map(|r| r.sampling_params.max_tokens)
            .max()
            .unwrap_or(512);

        Self {
            batch_id: BatchId::new(),
            requests,
            max_sequence_length,
            created_at: Utc::now(),
        }
    }

    /// Get the number of requests in this batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Request state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    /// Request is waiting in queue
    Waiting,
    /// Request is being processed
    Running,
    /// Request was preempted and is waiting to resume
    Preempted,
    /// Request completed successfully
    Completed,
    /// Request failed with error
    Failed,
    /// Request was cancelled
    Cancelled,
}

/// Scheduled request with additional state information
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// The original request
    pub request: InferenceRequest,
    /// Current state in scheduler
    pub state: RequestState,
    /// Allocated cache blocks
    pub allocated_blocks: Vec<crate::BlockId>,
    /// Number of tokens processed so far
    pub tokens_processed: usize,
    /// Estimated completion time
    pub estimated_completion: Option<DateTime<Utc>>,
}

impl ScheduledRequest {
    /// Create a new scheduled request
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            state: RequestState::Waiting,
            allocated_blocks: Vec::new(),
            tokens_processed: 0,
            estimated_completion: None,
        }
    }

    /// Update request state
    pub fn set_state(&mut self, state: RequestState) {
        self.state = state;
    }

    /// Add allocated cache blocks
    pub fn add_blocks(&mut self, blocks: Vec<crate::BlockId>) {
        self.allocated_blocks.extend(blocks);
    }

    /// Update tokens processed
    pub fn update_progress(&mut self, tokens_processed: usize) {
        self.tokens_processed = tokens_processed;
    }
}
