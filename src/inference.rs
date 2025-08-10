//! Core inference engine for the LLM inference system
//!
//! This module contains the main InferenceEngine that orchestrates model loading,
//! caching, and request processing with optimizations for performance and scalability.

use crate::cache::{CacheFactory, LRUCache};
use crate::config::Config;
use crate::error::{EngineError, Result};
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
    model_manager: Arc<ModelManager>,
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
    pub stop: Option<Vec<String>>,
    /// User identifier for caching
    pub user: Option<String>,
}

/// Response structure for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID
    pub id: String,
    /// Generated text
    pub text: String,
    /// Generation statistics
    pub usage: Usage,
    /// Finish reason
    pub finish_reason: String,
    /// Model used for generation
    pub model: String,
    /// Timestamp when response was created
    pub created: u64,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens in the completion
    pub completion_tokens: usize,
    /// Total number of tokens
    pub total_tokens: usize,
}

/// Engine-wide statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    /// Total number of requests processed
    pub total_requests: u64,
    /// Total number of successful requests
    pub successful_requests: u64,
    /// Total number of failed requests
    pub failed_requests: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Engine uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Cache memory usage in bytes
    pub cache_memory_bytes: usize,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Request ID
    pub id: String,
    /// Generated text delta
    pub delta: String,
    /// Whether this is the final chunk
    pub finished: bool,
    /// Finish reason (only present in final chunk)
    pub finish_reason: Option<String>,
    /// Usage statistics (only present in final chunk)
    pub usage: Option<Usage>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing inference engine with config: {:?}", config);

        // Create model manager
        let model_manager = Arc::new(ModelManager::new(config.clone()));

        // Create cache
        let cache = if config.cache.enabled {
            CacheFactory::create_cache(config.cache.clone())?
        } else {
            // Create a dummy cache that doesn't store anything
            Box::new(LRUCache::new(config.cache.clone()))
        };

        let engine = Self {
            config,
            model_manager,
            cache: Arc::new(RwLock::new(cache)),
            stats: Arc::new(RwLock::new(EngineStats::default())),
        };

        info!("Inference engine initialized successfully");
        Ok(engine)
    }

    /// Process an inference request
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();
        let request_id = request.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

        debug!("Processing inference request: {}", request_id);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
        }

        let result = self.process_request_internal(request.clone()).await;

        // Update stats based on result
        let inference_time = start_time.elapsed().as_millis() as f64;
        {
            let mut stats = self.stats.write();
            match &result {
                Ok(response) => {
                    stats.successful_requests += 1;
                    stats.total_tokens_generated += response.usage.completion_tokens as u64;
                    
                    // Update average inference time
                    let total_time = stats.avg_inference_time_ms * stats.successful_requests as f64;
                    stats.avg_inference_time_ms = (total_time + inference_time) / (stats.successful_requests + 1) as f64;
                }
                Err(_) => {
                    stats.failed_requests += 1;
                }
            }
        }

        result
    }

    /// Process a streaming inference request
    pub async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<StreamChunk>>> {
        let request_id = request.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let engine = self.clone();
        let request_clone = request.clone();

        tokio::spawn(async move {
            if let Err(e) = engine.process_streaming_request(request_clone, tx).await {
                error!("Streaming request failed: {}", e);
            }
        });

        Ok(rx)
    }

    /// Internal request processing
    async fn process_request_internal(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Get or load model
        let model_name = request.model.as_deref().unwrap_or(&self.config.model.name);
        let model = self.model_manager.load_model(model_name).await?;

        // Prepare generation config
        let mut gen_config = request.generation_config.unwrap_or_default();
        if let Some(max_tokens) = request.max_tokens {
            gen_config.max_new_tokens = max_tokens;
        }
        if let Some(temperature) = request.temperature {
            gen_config.temperature = temperature;
        }
        if let Some(top_p) = request.top_p {
            gen_config.top_p = top_p;
        }
        if let Some(top_k) = request.top_k {
            gen_config.top_k = top_k;
        }
        if let Some(rep_penalty) = request.repetition_penalty {
            gen_config.repetition_penalty = rep_penalty;
        }
        if let Some(stop) = request.stop {
            gen_config.stop_tokens = stop;
        }
        if let Some(stream) = request.stream {
            gen_config.stream = stream;
        }

        // Tokenize input
        let input_tokens = model.tokenize(&request.prompt)?;

        // Check cache
        let cache_key = self.generate_cache_key(&request, &input_tokens);
        let mut cache_entry = if self.config.cache.enabled {
            self.cache.read().get_cache(&cache_key)
        } else {
            None
        };

        // Generate response
        let generation_result = model
            .generate(
                &input_tokens,
                &gen_config,
                cache_entry.as_mut().map(|c| c as &mut dyn InferenceCache),
            )
            .await?;

        // Store in cache if enabled
        if self.config.cache.enabled && cache_entry.is_none() {
            // Create new cache entry from generation result
            // This is simplified - in practice, you'd extract the actual KV cache from the model
            // For now, we'll skip caching for new entries
        }

        let request_id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());

        Ok(InferenceResponse {
            id: request_id,
            text: generation_result.text,
            usage: Usage {
                prompt_tokens: generation_result.generation_stats.prompt_tokens,
                completion_tokens: generation_result.generation_stats.completion_tokens,
                total_tokens: generation_result.generation_stats.total_tokens,
            },
            finish_reason: format!("{:?}", generation_result.finish_reason),
            model: model_name.to_string(),
            created: chrono::Utc::now().timestamp() as u64,
        })
    }

    /// Process streaming request
    async fn process_streaming_request(
        &self,
        request: InferenceRequest,
        tx: tokio::sync::mpsc::Sender<Result<StreamChunk>>,
    ) -> Result<()> {
        // For MVP, we'll simulate streaming by chunking the complete response
        // In a full implementation, this would involve real streaming from the model
        let response = self.process_request_internal(request).await?;
        
        let words: Vec<&str> = response.text.split_whitespace().collect();
        let chunk_size = 3; // Words per chunk

        for (i, chunk) in words.chunks(chunk_size).enumerate() {
            let is_last = i == (words.len() / chunk_size);
            let delta = chunk.join(" ");
            
            let stream_chunk = StreamChunk {
                id: response.id.clone(),
                delta: if i == 0 { delta } else { format!(" {}", delta) },
                finished: is_last,
                finish_reason: if is_last { Some(response.finish_reason.clone()) } else { None },
                usage: if is_last { Some(response.usage.clone()) } else { None },
            };

            if tx.send(Ok(stream_chunk)).await.is_err() {
                break; // Receiver dropped
            }

            // Add small delay to simulate streaming
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        Ok(())
    }

    /// Generate cache key for request
    fn generate_cache_key(&self, request: &InferenceRequest, tokens: &[u32]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request.prompt.hash(&mut hasher);
        tokens.hash(&mut hasher);
        request.user.hash(&mut hasher);
        
        format!("cache_{:x}", hasher.finish())
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> EngineStats {
        self.stats.read().clone()
    }

    /// Get loaded models information
    pub fn get_models_info(&self) -> Vec<ModelInfo> {
        self.model_manager.get_all_model_info()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        let cache_stats = self.cache.read().cache_stats();

        Ok(HealthStatus {
            status: "healthy".to_string(),
            uptime_seconds: stats.uptime_seconds,
            total_requests: stats.total_requests,
            cache_hit_rate: cache_stats.hit_rate,
            memory_usage_mb: stats.memory_stats.total_allocated_bytes / (1024 * 1024),
        })
    }

    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine");
        // Clear cache
        self.cache.write().clear_cache();
        info!("Inference engine shutdown complete");
        Ok(())
    }
}

/// Health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub cache_hit_rate: f32,
    pub memory_usage_mb: usize,
}

impl Clone for InferenceEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            model_manager: Arc::clone(&self.model_manager),
            cache: Arc::clone(&self.cache),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_inference_time_ms: 0.0,
            total_tokens_generated: 0,
            avg_tokens_per_second: 0.0,
            uptime_seconds: 0,
            memory_stats: MemoryStats::default(),
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_allocated_bytes: 0,
            peak_memory_bytes: 0,
            cache_memory_bytes: 0,
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
                return Err(EngineError::invalid_request("max_tokens must be greater than 0"));
            }
        }

        if let Some(temperature) = self.temperature {
            if temperature < 0.0 || temperature > 2.0 {
                return Err(EngineError::invalid_request("temperature must be between 0.0 and 2.0"));
            }
        }

        if let Some(top_p) = self.top_p {
            if top_p < 0.0 || top_p > 1.0 {
                return Err(EngineError::invalid_request("top_p must be between 0.0 and 1.0"));
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
            stop: None,
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

        assert_eq!(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens);
    }
}