//! OpenAI-compatible API request and response types
//!
//! This module defines the JSON structures that match OpenAI's API specification
//! for seamless compatibility with existing OpenAI clients and tools.

use crate::inference::{InferenceRequest, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI-compatible chat completions request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier
    pub model: String,
    /// List of messages for the conversation
    pub messages: Vec<ChatMessage>,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Nucleus sampling (0.0 to 1.0)
    pub top_p: Option<f32>,
    /// Number of completions to generate
    pub n: Option<usize>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,
    /// Logit bias
    pub logit_bias: Option<HashMap<String, f32>>,
    /// User identifier
    pub user: Option<String>,
}

/// Chat message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: String,
    /// Content of the message
    pub content: String,
    /// Optional name of the user
    pub name: Option<String>,
}

/// OpenAI-compatible completions request (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Nucleus sampling
    pub top_p: Option<f32>,
    /// Number of completions to generate
    pub n: Option<usize>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// User identifier
    pub user: Option<String>,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (always "chat.completion")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of completion choices
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics
    pub usage: Usage,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Index of the choice
    pub index: usize,
    /// Generated message
    pub message: ChatMessage,
    /// Reason for finishing
    pub finish_reason: String,
}

/// OpenAI-compatible completion response (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (always "text_completion")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of completion choices
    pub choices: Vec<CompletionChoice>,
    /// Token usage statistics
    pub usage: Usage,
}

/// Completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Index of the choice
    pub index: usize,
    /// Generated text
    pub text: String,
    /// Reason for finishing
    pub finish_reason: String,
}

/// Streaming chat completion chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of delta choices
    pub choices: Vec<ChatDelta>,
}

/// Chat completion delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Index of the choice
    pub index: usize,
    /// Message delta
    pub delta: ChatMessage,
    /// Reason for finishing (if applicable)
    pub finish_reason: Option<String>,
}

/// Model information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Object type (always "model")
    pub object: String,
    /// Unix timestamp when the model was created
    pub created: u64,
    /// Organization that owns the model
    pub owned_by: String,
}

/// List of models response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Object type (always "list")
    pub object: String,
    /// List of available models
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Status of the service
    pub status: String,
    /// Additional status information
    pub details: HashMap<String, serde_json::Value>,
}

impl ChatCompletionRequest {
    /// Convert to internal inference request
    pub fn to_inference_request(&self) -> InferenceRequest {
        // Convert messages to a single prompt
        let prompt = self.messages_to_prompt();

        InferenceRequest {
            id: None,
            prompt,
            model: Some(self.model.clone()),
            generation_config: None,
            stream: self.stream,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            repetition_penalty: self.presence_penalty,
            stop: self.stop.clone(),
            user: self.user.clone(),
        }
    }

    /// Convert messages to a formatted prompt
    fn messages_to_prompt(&self) -> String {
        self.messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl CompletionRequest {
    /// Convert to internal inference request
    pub fn to_inference_request(&self) -> InferenceRequest {
        InferenceRequest {
            id: None,
            prompt: self.prompt.clone(),
            model: Some(self.model.clone()),
            generation_config: None,
            stream: self.stream,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            repetition_penalty: self.presence_penalty,
            stop: self.stop.clone(),
            user: self.user.clone(),
        }
    }
}

impl From<crate::inference::InferenceResponse> for ChatCompletionResponse {
    fn from(response: crate::inference::InferenceResponse) -> Self {
        Self {
            id: response.id,
            object: "chat.completion".to_string(),
            created: response.created,
            model: response.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: response.text,
                    name: None,
                },
                finish_reason: response.finish_reason,
            }],
            usage: response.usage,
        }
    }
}

impl From<crate::inference::InferenceResponse> for CompletionResponse {
    fn from(response: crate::inference::InferenceResponse) -> Self {
        Self {
            id: response.id,
            object: "text_completion".to_string(),
            created: response.created,
            model: response.model,
            choices: vec![CompletionChoice {
                index: 0,
                text: response.text,
                finish_reason: response.finish_reason,
            }],
            usage: response.usage,
        }
    }
}

impl From<crate::inference::StreamChunk> for ChatCompletionChunk {
    fn from(chunk: crate::inference::StreamChunk) -> Self {
        Self {
            id: chunk.id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: "ferrum-infer".to_string(), // This should come from the actual model
            choices: vec![ChatDelta {
                index: 0,
                delta: ChatMessage {
                    role: "assistant".to_string(),
                    content: chunk.delta,
                    name: None,
                },
                finish_reason: chunk.finish_reason,
            }],
        }
    }
}

impl From<crate::models::ModelInfo> for ModelInfo {
    fn from(info: crate::models::ModelInfo) -> Self {
        Self {
            id: info.name,
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            owned_by: "ferrum-infer".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_to_inference() {
        let request = ChatCompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                name: None,
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            n: None,
            stream: Some(false),
            stop: None,
            presence_penalty: Some(0.1),
            frequency_penalty: None,
            logit_bias: None,
            user: Some("test_user".to_string()),
        };

        let inference_req = request.to_inference_request();
        assert_eq!(inference_req.prompt, "user: Hello!");
        assert_eq!(inference_req.model, Some("gpt-3.5-turbo".to_string()));
        assert_eq!(inference_req.max_tokens, Some(100));
        assert_eq!(inference_req.temperature, Some(0.7));
        assert_eq!(inference_req.user, Some("test_user".to_string()));
    }

    #[test]
    fn test_completion_request_to_inference() {
        let request = CompletionRequest {
            model: "text-davinci-003".to_string(),
            prompt: "Once upon a time".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.8),
            top_p: Some(0.95),
            n: None,
            stream: Some(true),
            stop: Some(vec!["END".to_string()]),
            presence_penalty: None,
            frequency_penalty: Some(0.2),
            user: None,
        };

        let inference_req = request.to_inference_request();
        assert_eq!(inference_req.prompt, "Once upon a time");
        assert_eq!(inference_req.model, Some("text-davinci-003".to_string()));
        assert_eq!(inference_req.max_tokens, Some(50));
        assert_eq!(inference_req.stream, Some(true));
        assert_eq!(inference_req.stop, Some(vec!["END".to_string()]));
    }
}
