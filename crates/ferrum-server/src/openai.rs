//! OpenAI API compatibility types
//!
//! This module defines types that match the OpenAI API specification
//! for chat completions, completions, and model management.

use serde::{Deserialize, Serialize};
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

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences
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

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Response format constraint (e.g., `{"type": "json_object"}`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiResponseFormat>,
}

/// OpenAI-compatible response format specifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponseFormat {
    /// Format type: "text" or "json_object"
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role
    pub role: MessageRole,

    /// Message content
    pub content: String,

    /// Message name (for function calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message roles
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Function,
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

    /// Prompt text
    pub prompt: String,

    /// Maximum tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Stream responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
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
