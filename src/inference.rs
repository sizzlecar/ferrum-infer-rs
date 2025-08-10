//! Core inference engine for the LLM inference system
//!
//! This module contains the main InferenceEngine that orchestrates model loading,
//! caching, and request processing with optimizations for performance and scalability.

use crate::cache::{CacheFactory, LRUCache};
use crate::config::Config;
use crate::error::{EngineError, Result};
#[cfg(feature = "ml")]
use crate::models::{
    GenerationConfig, GenerationResult, InferenceCache, Model, ModelManager, ModelInfo
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Main inference engine that coordinates all operations
pub struct InferenceEngine {
    config: Config,
    #[cfg(feature = "ml")]
    model_manager: Arc<ModelManager>,
    #[cfg(feature = "ml")]
    cache: Arc<RwLock<Box<dyn InferenceCache>>>,
    stats: Arc<RwLock<EngineStats>>,
}

/// Request structure for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: Option<String>,
    /// Input prompt text
    pub prompt: String,
    /// Model name to use for inference
    pub model: Option<String>,
    /// Generation configuration
    #[cfg(feature = "ml")]
    pub generation_config: Option<GenerationConfig>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-p nucleus sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Repetition penalty
    pub repetition_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
}

/// Response structure for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID
    pub id: String,
    /// Generated text
    pub text: String,
    /// Generation metadata
    pub metadata: ResponseMetadata,
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Model used
    pub model: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Cache hit flag
    pub cache_hit: bool,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_generation_time_ms: u64,
    pub total_tokens_generated: u64,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing inference engine with config: {:?}", config);

        #[cfg(feature = "ml")]
        let model_manager = Arc::new(ModelManager::new(config.clone()).await?);
        
        #[cfg(feature = "ml")]
        let cache = Arc::new(RwLock::new(
            CacheFactory::create_cache(&config.cache_type, config.cache_size)
                .map_err(|e| EngineError::CacheError(e.to_string()))?,
        ));

        let stats = Arc::new(RwLock::new(EngineStats::default()));

        Ok(InferenceEngine {
            config,
            #[cfg(feature = "ml")]
            model_manager,
            #[cfg(feature = "ml")]
            cache,
            stats,
        })
    }

    /// Process an inference request
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        let request_id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        
        info!("Processing inference request: {}", request_id);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
        }

        #[cfg(feature = "ml")]
        {
            // Real ML inference would happen here
            self.process_with_ml(request, request_id, start_time).await
        }

        #[cfg(not(feature = "ml"))]
        {
            // Mock response for CI/testing
            let response = InferenceResponse {
                id: request_id.clone(),
                text: format!("Mock response for: {}", request.prompt),
                metadata: ResponseMetadata {
                    model: request.model.unwrap_or_else(|| "mock-model".to_string()),
                    tokens_generated: 50,
                    generation_time_ms: start_time.elapsed().as_millis() as u64,
                    cache_hit: false,
                },
            };

            {
                let mut stats = self.stats.write();
                stats.successful_requests += 1;
                stats.total_tokens_generated += 50;
                stats.total_generation_time_ms += start_time.elapsed().as_millis() as u64;
            }

            Ok(response)
        }
    }

    #[cfg(feature = "ml")]
    async fn process_with_ml(
        &self,
        request: InferenceRequest,
        request_id: String,
        start_time: Instant,
    ) -> Result<InferenceResponse> {
        // This would contain the real ML processing logic
        let model_name = request.model.unwrap_or_else(|| self.config.default_model.clone());
        
        // Check cache first
        let cache_key = format!("{}:{}", model_name, request.prompt);
        if let Some(cached_response) = self.check_cache(&cache_key) {
            info!("Cache hit for request: {}", request_id);
            
            let mut stats = self.stats.write();
            stats.successful_requests += 1;
            stats.cache_hits += 1;
            
            return Ok(cached_response);
        }

        // Load model and generate response
        let model = self.model_manager.get_model(&model_name).await?;
        let generation_config = request.generation_config.unwrap_or_default();
        
        let result = model.generate(&request.prompt, &generation_config).await?;
        
        let response = InferenceResponse {
            id: request_id.clone(),
            text: result.text.clone(),
            metadata: ResponseMetadata {
                model: model_name.clone(),
                tokens_generated: result.tokens_generated,
                generation_time_ms: start_time.elapsed().as_millis() as u64,
                cache_hit: false,
            },
        };

        // Cache the response
        self.store_in_cache(cache_key, response.clone());

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.successful_requests += 1;
            stats.cache_misses += 1;
            stats.total_tokens_generated += result.tokens_generated as u64;
            stats.total_generation_time_ms += start_time.elapsed().as_millis() as u64;
        }

        info!("Successfully generated response for request: {}", request_id);
        Ok(response)
    }

    #[cfg(feature = "ml")]
    fn check_cache(&self, key: &str) -> Option<InferenceResponse> {
        self.cache.read().get(key)
    }

    #[cfg(feature = "ml")]
    fn store_in_cache(&self, key: String, response: InferenceResponse) {
        self.cache.write().put(key, response);
    }

    /// Get inference engine statistics
    pub fn get_stats(&self) -> EngineStats {
        self.stats.read().clone()
    }

    /// Get available models
    #[cfg(feature = "ml")]
    pub fn get_models_info(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models()
    }

    #[cfg(not(feature = "ml"))]
    pub fn get_models_info(&self) -> Vec<crate::models::ModelInfo> {
        vec![crate::models::ModelInfo {
            id: "mock-model".to_string(),
            name: "Mock Model".to_string(),
            description: Some("Mock model for testing".to_string()),
            context_length: 2048,
            created: chrono::Utc::now().timestamp(),
        }]
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        
        #[cfg(feature = "ml")]
        let model_status = if self.model_manager.is_healthy().await {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        };

        #[cfg(not(feature = "ml"))]
        let model_status = "healthy".to_string();

        Ok(HealthStatus {
            status: "healthy".to_string(),
            model_status,
            cache_status: "healthy".to_string(),
            total_requests: stats.total_requests,
            successful_requests: stats.successful_requests,
            uptime_seconds: 0, // Would track actual uptime
        })
    }
}

/// Health status response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub model_status: String,
    pub cache_status: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub uptime_seconds: u64,
}

impl InferenceRequest {
    /// Validate the inference request
    pub fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() {
            return Err(EngineError::ValidationError(
                "Prompt cannot be empty".to_string(),
            ));
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > 4096 {
                return Err(EngineError::ValidationError(
                    "max_tokens must be between 1 and 4096".to_string(),
                ));
            }
        }

        if let Some(temperature) = self.temperature {
            if temperature < 0.0 || temperature > 2.0 {
                return Err(EngineError::ValidationError(
                    "temperature must be between 0.0 and 2.0".to_string(),
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if top_p < 0.0 || top_p > 1.0 {
                return Err(EngineError::ValidationError(
                    "top_p must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_inference_request_validation() {
        let valid_request = InferenceRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            model: Some("test-model".to_string()),
            #[cfg(feature = "ml")]
            generation_config: None,
            stream: Some(false),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
        };

        assert!(valid_request.validate().is_ok());

        let invalid_request = InferenceRequest {
            id: None,
            prompt: "".to_string(), // Empty prompt
            model: Some("test-model".to_string()),
            #[cfg(feature = "ml")]
            generation_config: None,
            stream: Some(false),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
        };

        assert!(invalid_request.validate().is_err());
    }

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let config = Config::default();
        let engine = InferenceEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_mock_inference() {
        let config = Config::default();
        let engine = InferenceEngine::new(config).await.unwrap();

        let request = InferenceRequest {
            id: Some("test-123".to_string()),
            prompt: "Test prompt".to_string(),
            model: Some("test-model".to_string()),
            #[cfg(feature = "ml")]
            generation_config: None,
            stream: Some(false),
            max_tokens: Some(50),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
        };

        let response = engine.infer(request).await;
        assert!(response.is_ok());
        
        let response = response.unwrap();
        assert_eq!(response.id, "test-123");
        assert!(!response.text.is_empty());
    }
}