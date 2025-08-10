//! Core inference engine for LLM processing
//!
//! This module contains the main inference engine that orchestrates model loading,
//! request processing, and response generation with caching support.

use crate::config::Config;
use crate::error::{EngineError, Result};
#[cfg(feature = "ml")]
use crate::models::{
    GenerationConfig, GenerationResult, InferenceCache, Model, ModelInfo, ModelManager,
};
#[cfg(not(feature = "ml"))]
use crate::models::{GenerationConfig, InferenceCache, MockModel, Model, ModelInfo, ModelManager};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{error, info};
use uuid::Uuid;

/// Main inference engine
pub struct InferenceEngine {
    config: Config,
    #[cfg(feature = "ml")]
    model_manager: Arc<ModelManager>,
    #[cfg(feature = "ml")]
    cache: Arc<RwLock<Box<dyn InferenceCache>>>,
    #[cfg(not(feature = "ml"))]
    mock_model: Arc<MockModel>,
    stats: Arc<RwLock<EngineStats>>,
}

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: Option<String>,
    pub prompt: String,
    pub model: Option<String>,
    pub generation_config: Option<GenerationConfig>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub user: Option<String>,
}

/// Response from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub text: String,
    pub created: i64,
    pub model: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub metadata: ResponseMetadata,
}

/// Metadata for inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub model: String,
    pub inference_time_ms: u64,
    pub tokens_per_second: f32,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk for real-time responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub text: String,
    pub finish_reason: Option<String>,
    pub created: i64,
    pub model: String,
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_inference_time_ms: f64,
    pub total_tokens_generated: u64,
    pub uptime_seconds: u64,
}

/// Health status of the engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub model_status: String,
    pub cache_status: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub uptime_seconds: u64,
    pub cache_hit_rate: f32,
    pub memory_usage_mb: usize,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing inference engine");

        #[cfg(feature = "ml")]
        let model_manager = Arc::new(ModelManager::new(config.clone()).await?);

        #[cfg(not(feature = "ml"))]
        let mock_model = Arc::new(MockModel::new());

        let stats = Arc::new(RwLock::new(EngineStats::default()));

        Ok(Self {
            config,
            #[cfg(feature = "ml")]
            model_manager,
            #[cfg(not(feature = "ml"))]
            mock_model,
            stats,
        })
    }

    /// Process an inference request
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        let request_id = request
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        info!("Processing inference request {}", request_id);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
        }

        let result = self.process_request_internal(request, &request_id).await;

        // Update stats based on result
        let inference_time = start_time.elapsed();
        {
            let mut stats = self.stats.write();
            match &result {
                Ok(response) => {
                    stats.successful_requests += 1;
                    stats.total_tokens_generated += response
                        .usage
                        .as_ref()
                        .map(|u| u.completion_tokens)
                        .unwrap_or(0) as u64;
                }
                Err(_) => {
                    stats.failed_requests += 1;
                }
            }

            // Update average inference time
            let total_completed = stats.successful_requests + stats.failed_requests;
            if total_completed > 0 {
                stats.avg_inference_time_ms = (stats.avg_inference_time_ms
                    * (total_completed - 1) as f64
                    + inference_time.as_millis() as f64)
                    / total_completed as f64;
            }
        }

        result
    }

    /// Process inference request with streaming support
    pub async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<mpsc::Receiver<Result<StreamChunk>>> {
        let (tx, rx) = mpsc::channel(100);
        let request_id = request
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // For now, just send a single chunk as a mock implementation
        let chunk = StreamChunk {
            id: request_id.clone(),
            text: "Hello, this is a streaming response!".to_string(),
            finish_reason: Some("stop".to_string()),
            created: chrono::Utc::now().timestamp(),
            model: request
                .model
                .unwrap_or_else(|| self.config.model.name.clone()),
        };

        tokio::spawn(async move {
            if let Err(_) = tx.send(Ok(chunk)).await {
                error!("Failed to send stream chunk");
            }
        });

        Ok(rx)
    }

    async fn process_request_internal(
        &self,
        request: InferenceRequest,
        request_id: &str,
    ) -> Result<InferenceResponse> {
        // Validate request
        self.validate_request(&request)?;

        let model_name = request.model.as_ref().unwrap_or(&self.config.model.name);

        #[cfg(feature = "ml")]
        {
            // Get model
            let model = self.model_manager.load_model(model_name).await?;

            // Process with real model
            let prompt_tokens = model.tokenize(&request.prompt)?;
            let generation_config = request.generation_config.unwrap_or_default();

            let generation_result = model
                .generate(&prompt_tokens, &generation_config, None)
                .await?;

            Ok(InferenceResponse {
                id: request_id.to_string(),
                text: generation_result.text,
                created: chrono::Utc::now().timestamp(),
                model: model_name.clone(),
                finish_reason: Some(format!("{:?}", generation_result.finish_reason)),
                usage: Some(Usage {
                    prompt_tokens: generation_result.generation_stats.prompt_tokens,
                    completion_tokens: generation_result.generation_stats.completion_tokens,
                    total_tokens: generation_result.generation_stats.total_tokens,
                }),
                metadata: ResponseMetadata {
                    model: model_name.clone(),
                    inference_time_ms: generation_result.generation_stats.inference_time_ms,
                    tokens_per_second: generation_result.generation_stats.tokens_per_second,
                },
            })
        }

        #[cfg(not(feature = "ml"))]
        {
            // Mock implementation for CI
            let response_text = format!("Mock response for: {}", request.prompt);
            let prompt_tokens = request.prompt.split_whitespace().count();
            let completion_tokens = response_text.split_whitespace().count();

            Ok(InferenceResponse {
                id: request_id.to_string(),
                text: response_text,
                created: chrono::Utc::now().timestamp(),
                model: model_name.clone(),
                finish_reason: Some("stop".to_string()),
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                }),
                metadata: ResponseMetadata {
                    model: model_name.clone(),
                    inference_time_ms: 100,
                    tokens_per_second: 50.0,
                },
            })
        }
    }

    fn validate_request(&self, request: &InferenceRequest) -> Result<()> {
        if request.prompt.trim().is_empty() {
            return Err(EngineError::invalid_request("Prompt cannot be empty"));
        }

        if let Some(max_tokens) = request.max_tokens {
            if max_tokens == 0 {
                return Err(EngineError::invalid_request(
                    "max_tokens must be greater than 0",
                ));
            }
            if max_tokens > self.config.model.max_sequence_length {
                return Err(EngineError::invalid_request(format!(
                    "max_tokens ({}) exceeds model limit ({})",
                    max_tokens, self.config.model.max_sequence_length
                )));
            }
        }

        if let Some(temperature) = request.temperature {
            if temperature < 0.0 || temperature > 2.0 {
                return Err(EngineError::invalid_request(
                    "temperature must be between 0.0 and 2.0",
                ));
            }
        }

        if let Some(top_p) = request.top_p {
            if top_p < 0.0 || top_p > 1.0 {
                return Err(EngineError::invalid_request(
                    "top_p must be between 0.0 and 1.0",
                ));
            }
        }

        Ok(())
    }

    /// Get model information
    #[cfg(feature = "ml")]
    pub fn get_models_info(&self) -> Vec<ModelInfo> {
        self.model_manager.get_all_model_info()
    }

    #[cfg(not(feature = "ml"))]
    pub fn get_models_info(&self) -> Vec<ModelInfo> {
        vec![self.mock_model.model_info()]
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> EngineStats {
        self.stats.read().clone()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus> {
        #[cfg(feature = "ml")]
        let model_status = if self.model_manager.is_healthy().await {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        };

        #[cfg(not(feature = "ml"))]
        let model_status = "healthy".to_string();

        let stats = self.stats.read();
        let cache_hit_rate = if stats.total_requests > 0 {
            stats.cache_hits as f32 / stats.total_requests as f32
        } else {
            0.0
        };

        Ok(HealthStatus {
            status: "healthy".to_string(),
            model_status,
            cache_status: "healthy".to_string(),
            total_requests: stats.total_requests,
            successful_requests: stats.successful_requests,
            uptime_seconds: 0, // TODO: Track actual uptime
            cache_hit_rate,
            memory_usage_mb: 0, // TODO: Implement actual memory usage
        })
    }

    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine");
        info!("Inference engine shutdown complete");
        Ok(())
    }
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_inference_time_ms: 0.0,
            total_tokens_generated: 0,
            uptime_seconds: 0,
        }
    }
}

impl InferenceRequest {
    /// Validate the inference request
    pub fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() {
            return Err(EngineError::invalid_request("Prompt cannot be empty"));
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err(EngineError::invalid_request(
                    "max_tokens must be greater than 0",
                ));
            }
        }

        if let Some(temperature) = self.temperature {
            if temperature < 0.0 || temperature > 2.0 {
                return Err(EngineError::invalid_request(
                    "temperature must be between 0.0 and 2.0",
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if top_p < 0.0 || top_p > 1.0 {
                return Err(EngineError::invalid_request(
                    "top_p must be between 0.0 and 1.0",
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_validation() {
        let mut request = InferenceRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            model: None,
            generation_config: None,
            stream: None,
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            repetition_penalty: None,
            stop_sequences: None,
            user: None,
        };

        assert!(request.validate().is_ok());

        request.prompt = "".to_string();
        assert!(request.validate().is_err());

        request.prompt = "Hello".to_string();
        request.temperature = Some(3.0);
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_usage_calculation() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };

        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }
}
