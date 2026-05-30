//! Request and response types for inference

use crate::{ids::*, models::TokenUsage, FinishReason, Priority, SamplingParams, TokenId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PROMPT_TOKENS_METADATA_KEY: &str = "ferrum_prompt_tokens";

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
