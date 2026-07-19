//! OpenAI API compatibility types
//!
//! This module defines types that match the OpenAI API specification
//! for chat completions, completions, and model management.

use serde::{de, Deserialize, Serialize};
use std::collections::HashMap;

/// Chat completions request (OpenAI compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionsRequest {
    /// Model to use for completion
    pub model: String,

    /// List of messages
    pub messages: Vec<ChatMessage>,

    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Newer OpenAI chat field replacing `max_tokens` for completion budget.
    /// When both are supplied, Ferrum uses this value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// vLLM-compatible top-k sampling extension. Values `-1` and `0`
    /// disable top-k filtering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,

    /// vLLM-compatible minimum probability sampling extension. A value of
    /// `0` disables minimum-probability filtering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    /// vLLM-compatible repetition penalty extension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// vLLM-compatible extension for benchmark/throughput workloads.
    /// When true, Ferrum ignores model EOS tokens and stops only on the
    /// requested token budget or explicit user stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,

    /// Stop sequences
    #[serde(default, deserialize_with = "deserialize_stop_sequences")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Logit bias
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Return log probabilities. Ferrum rejects this until implemented so
    /// clients get an explicit OpenAI-style error instead of silent ignore.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Response format constraint (e.g., `{"type": "json_object"}`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiResponseFormat>,

    /// OpenAI tool definitions. Function tools are parsed, carried through
    /// structured request data, and can shape model-emitted tool-call JSON.
    /// Tool execution itself stays caller-owned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,

    /// OpenAI tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Streaming response options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Legacy OpenAI functions compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatFunction>>,

    /// Legacy OpenAI function-call selector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallChoice>,

    /// Ferrum extension metadata. Used for opt-in product features such as
    /// `metadata.ferrum_session_id` when callers prefer body metadata over
    /// the `X-Ferrum-Session` header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// vLLM-compatible chat-template variables. Ferrum forwards supported
    /// values to the model-provided chat template; templates that do not read
    /// a variable are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
}

/// OpenAI streaming options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Tool definition in OpenAI chat-completion requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatFunction,
}

/// Function schema for `tools[].function` and legacy `functions[]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// OpenAI `tool_choice` accepts either a simple mode string or a specific
/// function-tool selector object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Legacy `function_call` accepts a simple mode string or a named function.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallChoice {
    Mode(String),
    Function { name: String },
}

/// OpenAI-compatible response format specifier.
///
/// Mirrors OpenAI's `response_format` field on `/v1/chat/completions`:
///   - `{"type": "text"}`         — default, no constraint
///   - `{"type": "json_object"}`  — output must be valid JSON
///   - `{"type": "json_schema", "json_schema": {"name":..., "strict":true,
///      "schema": {...}}}` — output must conform to the inline JSON Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    /// Present only when `format_type == "json_schema"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<OpenAiJsonSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiJsonSchema {
    /// Optional name for the schema (ignored internally, kept for round-trip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The actual JSON Schema. Stored as raw JSON value so callers can pass
    /// any valid schema object; we re-serialise when forwarding to the
    /// guided-decoding pipeline. Optional at deserialization time so the
    /// HTTP layer can return an OpenAI-shaped `param` error for missing
    /// schemas instead of Axum's generic JSON rejection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// OpenAI's `strict` flag. When true, Ferrum rejects schemas outside the
    /// currently supported guided-decoding subset instead of silently falling
    /// back to best-effort JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role
    pub role: MessageRole,

    /// Message content. Accepts either a plain string or the OpenAI
    /// "typed parts" array form (`[{"type":"text","text":"..."}]`)
    /// — both shapes deserialize into a single String. Non-text parts
    /// fail deserialization so multimodal input is rejected instead of
    /// silently dropped.
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_message_content")]
    pub content: String,

    /// vLLM-compatible parsed reasoning text. When Ferrum parses
    /// `<think>...</think>`, `content` contains only the final visible
    /// answer and this field contains the reasoning block text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Message name (for function calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Assistant tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,

    /// Tool response correlation id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Legacy assistant function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatFunctionCall>,
}

/// Assistant tool call in OpenAI responses and historical conversation input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatFunctionCall,
}

/// Function call payload. OpenAI serializes arguments as a JSON string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Deserialize chat message content from either a plain string or the
/// OpenAI typed-parts array form. Real OpenAI clients (and `vllm bench
/// serve`'s openai-chat backend) send `content` as
/// `[{"type":"text","text":"..."}]` even for plain text; refusing that
/// breaks every standard client.
fn deserialize_message_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Array(parts) => {
            let mut text_parts = Vec::with_capacity(parts.len());
            for part in parts {
                let ty = part
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| de::Error::custom("message content part missing type"))?;
                if ty != "text" {
                    return Err(de::Error::custom(format!(
                        "unsupported message content part type `{ty}`"
                    )));
                }
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            Ok(text_parts.join("\n"))
        }
        _ => Err(de::Error::custom(
            "message content must be a string, null, or an array of text parts",
        )),
    }
}

fn deserialize_stop_sequences<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(stop)) => Ok(Some(vec![stop])),
        Some(serde_json::Value::Array(values)) => {
            let mut stops = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    serde_json::Value::String(stop) => stops.push(stop),
                    _ => {
                        return Err(de::Error::custom(
                            "stop must be a string or an array of strings",
                        ))
                    }
                }
            }
            Ok(Some(stops))
        }
        _ => Err(de::Error::custom(
            "stop must be a string or an array of strings",
        )),
    }
}

/// Message roles
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Function,
    Tool,
}

/// Chat completions response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionsResponse {
    /// Response ID
    pub id: String,

    /// Object type
    pub object: String,

    /// Creation timestamp
    pub created: u64,

    /// Model used
    pub model: String,

    /// Choices array
    pub choices: Vec<ChatChoice>,

    /// Token usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Chat choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: u32,

    /// Message content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessage>,

    /// Delta for streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChatMessage>,

    /// Finish reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Legacy completions request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsRequest {
    /// Model to use
    pub model: String,

    /// Prompt text. OpenAI's legacy completions endpoint also accepts prompt
    /// arrays, but Ferrum currently supports only a single string and rejects
    /// other shapes with `param=prompt`.
    #[serde(default)]
    pub prompt: CompletionPrompt,

    /// Maximum tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate. Ferrum currently supports only
    /// `n=1` and rejects larger values explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Stream responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences
    #[serde(default, deserialize_with = "deserialize_stop_sequences")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Legacy completions log probabilities. Explicitly rejected until
    /// implemented so clients don't mistake a silent ignore for support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Logit bias.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
}

/// Legacy completions prompt. Kept as a parsed enum so the HTTP layer can
/// return an OpenAI-shaped field error instead of a generic JSON rejection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Unsupported(serde_json::Value),
}

impl Default for CompletionPrompt {
    fn default() -> Self {
        Self::Unsupported(serde_json::Value::Null)
    }
}

impl CompletionPrompt {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Unsupported(_) => None,
        }
    }
}

/// Completions response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
}

/// Completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Model list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<ModelPermission>,
    pub root: Option<String>,
    pub parent: Option<String>,
}

/// Model permission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

// ======================== Embeddings API ========================

/// Embeddings request (OpenAI-compatible, extended for images)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model identifier
    pub model: String,

    /// Input to embed — text string, array of strings, or objects with text/image fields
    pub input: EmbeddingInput,

    /// Encoding format: "float" (default) or "base64"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

/// Polymorphic embedding input.
/// Supports: single string, array of strings, single object, array of objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text string (OpenAI standard)
    Single(String),
    /// Batch of text strings (OpenAI standard)
    Batch(Vec<String>),
    /// Single multimodal item (Jina-style extension)
    SingleObject(EmbeddingItem),
    /// Batch of multimodal items
    BatchObjects(Vec<EmbeddingItem>),
}

/// A single embedding input item — text or image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingItem {
    /// Text to embed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Image: file path or base64 data URI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// Embeddings response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Single embedding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Token usage for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ======================== Audio Transcription API ========================

/// Transcription response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

// ======================== Error types ========================

/// OpenAI API error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiError {
    pub error: OpenAiErrorDetail,
}

/// OpenAI error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// OpenAI error types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiErrorType {
    InvalidRequestError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RateLimitError,
    InternalServerError,
    ServiceUnavailableError,
}

/// Server-sent event for streaming
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u32>,
}

impl SseEvent {
    pub fn data(data: String) -> Self {
        Self {
            event: None,
            data,
            id: None,
            retry: None,
        }
    }

    pub fn json(value: &serde_json::Value) -> Result<Self, serde_json::Error> {
        Ok(Self::data(serde_json::to_string(value)?))
    }

    pub fn to_string(&self) -> String {
        let mut result = String::new();

        if let Some(event) = &self.event {
            result.push_str(&format!("event: {}\n", event));
        }

        if let Some(id) = &self.id {
            result.push_str(&format!("id: {}\n", id));
        }

        if let Some(retry) = self.retry {
            result.push_str(&format!("retry: {}\n", retry));
        }

        result.push_str(&format!("data: {}\n\n", self.data));
        result
    }
}

/// TTS speech request (OpenAI compatible /v1/audio/speech)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// Model name (e.g., "qwen3-tts", "tts-1")
    #[serde(default = "default_tts_model")]
    pub model: String,

    /// Text to synthesize
    pub input: String,

    /// Voice preset (ignored for now — uses default speaker)
    #[serde(default = "default_voice")]
    pub voice: String,

    /// Response format: "wav", "pcm" (default: "wav")
    #[serde(default = "default_audio_format")]
    pub response_format: String,

    /// Language hint: "auto", "chinese", "english"
    #[serde(default = "default_language")]
    pub language: String,

    /// Enable streaming (chunked transfer)
    #[serde(default)]
    pub stream: bool,
}

fn default_tts_model() -> String {
    "qwen3-tts".to_string()
}
fn default_voice() -> String {
    "default".to_string()
}
fn default_audio_format() -> String {
    "wav".to_string()
}
fn default_language() -> String {
    "auto".to_string()
}
