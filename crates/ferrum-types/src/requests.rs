//! Request and response types for inference

use crate::{ids::*, models::TokenUsage, FinishReason, Priority, SamplingParams, TokenId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
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
