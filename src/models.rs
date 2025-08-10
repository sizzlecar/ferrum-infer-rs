//! Model management and loading for the LLM inference engine
//!
//! This module provides abstractions for loading and managing LLM models,
//! supporting different model types and backends through a trait-based design.

use crate::config::{Config, ModelConfig};
use crate::error::{EngineError, Result};
#[cfg(feature = "ml")]
use candle_core::{Device, Tensor};
#[cfg(feature = "ml")]
use candle_transformers::models::llama::{Cache, LlamaConfig};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;

/// Trait for model loading and management
#[async_trait::async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load a model from the specified configuration
    async fn load_model(&self, config: &ModelConfig) -> Result<Box<dyn Model>>;

    /// Validate if the model can be loaded
    async fn validate_model(&self, config: &ModelConfig) -> Result<()>;

    /// Get supported model types
    fn supported_model_types(&self) -> Vec<String>;
}

/// Trait for LLM model inference
#[async_trait::async_trait]
pub trait Model: Send + Sync {
    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Tokenize input text
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;

    /// Detokenize token IDs to text
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;

    /// Generate tokens from input
    async fn generate(
        &self,
        input_tokens: &[u32],
        generation_config: &GenerationConfig,
        cache: Option<&mut dyn InferenceCache>,
    ) -> Result<GenerationResult>;

    /// Get the model's vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the model's context length
    fn context_length(&self) -> usize;

    /// Check if the model supports a specific feature
    fn supports_feature(&self, feature: ModelFeature) -> bool;
}

/// Trait for inference caching
pub trait InferenceCache: Send + Sync {
    /// Get cached response for a key
    fn get(&self, key: &str) -> Option<crate::inference::InferenceResponse>;

    /// Store response for a key
    fn put(&mut self, key: String, value: crate::inference::InferenceResponse);

    /// Clear all cache entries
    fn clear(&mut self);

    /// Get cached KV pairs for a sequence
    fn get_cache(&self, sequence_id: &str) -> Option<CacheEntry>;

    /// Store KV pairs for a sequence
    fn store_cache(&mut self, sequence_id: &str, cache: CacheEntry);

    /// Remove cache entry
    fn remove_cache(&mut self, sequence_id: &str);

    /// Clear all cache entries
    fn clear_cache(&mut self);

    /// Get cache statistics
    fn cache_stats(&self) -> CacheStats;
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub parameter_count: Option<u64>,
    pub context_length: usize,
    pub vocab_size: usize,
    pub device: String,
    pub dtype: String,
    pub supports_streaming: bool,
    pub supports_batching: bool,
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub stop_tokens: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub stream: bool,
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub generation_stats: GenerationStats,
}

/// Cache entry for KV caching
#[derive(Debug, Clone)]
pub struct CacheEntry {
    #[cfg(feature = "ml")]
    pub key_cache: Vec<Tensor>,
    #[cfg(feature = "ml")]
    pub value_cache: Vec<Tensor>,
    #[cfg(not(feature = "ml"))]
    pub key_cache: Vec<String>, // Mock data for CI
    #[cfg(not(feature = "ml"))]
    pub value_cache: Vec<String>, // Mock data for CI
    pub sequence_length: usize,
    pub created_at: std::time::Instant,
    pub last_accessed: std::time::Instant,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f32,
    pub miss_rate: f32,
    pub eviction_count: u64,
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub inference_time_ms: u64,
    pub tokens_per_second: f32,
    pub time_to_first_token_ms: Option<u64>,
}

/// Finish reason for generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    EOS,
    Error(String),
}

/// Model features enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFeature {
    Streaming,
    Batching,
    FlashAttention,
    KVCaching,
    TensorParallelism,
}

/// Model manager for handling multiple models
pub struct ModelManager {
    config: Config,
    #[cfg(feature = "ml")]
    models: RwLock<HashMap<String, Arc<dyn Model>>>,
    #[cfg(feature = "ml")]
    loaders: Vec<Box<dyn ModelLoader>>,
}

impl ModelManager {
    /// Create a new model manager
    #[cfg(feature = "ml")]
    pub async fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            models: RwLock::new(HashMap::new()),
            loaders: Vec::new(),
        })
    }

    #[cfg(not(feature = "ml"))]
    pub async fn new(config: Config) -> Result<Self> {
        Ok(Self { config })
    }

    /// Add a model loader
    #[cfg(feature = "ml")]
    pub fn add_loader(&mut self, loader: Box<dyn ModelLoader>) {
        self.loaders.push(loader);
    }

    /// Load a model with the given name
    #[cfg(feature = "ml")]
    pub async fn load_model(&self, name: &str) -> Result<Arc<dyn Model>> {
        // Check if model is already loaded
        {
            let models = self.models.read();
            if let Some(model) = models.get(name) {
                return Ok(Arc::clone(model));
            }
        }

        // Find appropriate loader and load model
        for loader in &self.loaders {
            if loader
                .supported_model_types()
                .contains(&self.config.model.name)
            {
                let model = loader.load_model(&self.config.model).await?;
                let model_arc = Arc::from(model);

                // Store in cache
                {
                    let mut models = self.models.write();
                    models.insert(name.to_string(), Arc::clone(&model_arc));
                }

                return Ok(model_arc);
            }
        }

        Err(EngineError::model(format!(
            "No loader found for model: {}",
            name
        )))
    }

    /// Get a loaded model
    #[cfg(feature = "ml")]
    pub fn get_model(&self, name: &str) -> Option<Arc<dyn Model>> {
        let models = self.models.read();
        models.get(name).map(Arc::clone)
    }

    /// List all loaded models
    #[cfg(feature = "ml")]
    pub fn list_models(&self) -> Vec<String> {
        let models = self.models.read();
        models.keys().cloned().collect()
    }

    #[cfg(not(feature = "ml"))]
    pub fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            name: "mock-model".to_string(),
            model_type: "mock".to_string(),
            parameter_count: Some(1000000),
            context_length: 2048,
            vocab_size: 50000,
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            supports_streaming: true,
            supports_batching: false,
        }]
    }

    /// Unload a model
    #[cfg(feature = "ml")]
    pub fn unload_model(&self, name: &str) -> bool {
        let mut models = self.models.write();
        models.remove(name).is_some()
    }

    /// Get model information for all loaded models
    #[cfg(feature = "ml")]
    pub fn get_all_model_info(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().map(|model| model.model_info()).collect()
    }

    #[cfg(not(feature = "ml"))]
    pub fn get_all_model_info(&self) -> Vec<ModelInfo> {
        self.list_models()
    }

    #[cfg(feature = "ml")]
    pub async fn is_healthy(&self) -> bool {
        // Check if any models are loaded and responsive
        let models = self.models.read();
        !models.is_empty()
    }

    #[cfg(not(feature = "ml"))]
    pub async fn is_healthy(&self) -> bool {
        true // Mock is always healthy
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.1,
            do_sample: true,
            stop_tokens: vec!["</s>".to_string()],
            stop_token_ids: vec![],
            stream: false,
        }
    }
}

impl CacheEntry {
    #[cfg(feature = "ml")]
    pub fn new(key_cache: Vec<Tensor>, value_cache: Vec<Tensor>, sequence_length: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            key_cache,
            value_cache,
            sequence_length,
            created_at: now,
            last_accessed: now,
        }
    }

    #[cfg(not(feature = "ml"))]
    pub fn new(key_cache: Vec<String>, value_cache: Vec<String>, sequence_length: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            key_cache,
            value_cache,
            sequence_length,
            created_at: now,
            last_accessed: now,
        }
    }

    pub fn touch(&mut self) {
        self.last_accessed = std::time::Instant::now();
    }

    pub fn age(&self) -> std::time::Duration {
        std::time::Instant::now().duration_since(self.created_at)
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            total_size_bytes: 0,
            hit_rate: 0.0,
            miss_rate: 0.0,
            eviction_count: 0,
        }
    }
}

/// Mock model implementation for non-ML builds
#[cfg(not(feature = "ml"))]
pub struct MockModel {
    pub info: ModelInfo,
}

#[cfg(not(feature = "ml"))]
impl MockModel {
    pub fn new() -> Self {
        Self {
            info: ModelInfo {
                name: "mock-model".to_string(),
                model_type: "mock".to_string(),
                parameter_count: Some(1000000),
                context_length: 2048,
                vocab_size: 50000,
                device: "cpu".to_string(),
                dtype: "f32".to_string(),
                supports_streaming: true,
                supports_batching: false,
            },
        }
    }
}

#[cfg(not(feature = "ml"))]
#[async_trait::async_trait]
impl Model for MockModel {
    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Simple mock tokenization
        Ok(text.chars().map(|c| c as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Simple mock detokenization
        Ok(tokens
            .iter()
            .map(|&t| char::from_u32(t).unwrap_or('?'))
            .collect())
    }

    async fn generate(
        &self,
        _input_tokens: &[u32],
        _config: &GenerationConfig,
        _cache: Option<&mut dyn InferenceCache>,
    ) -> Result<GenerationResult> {
        Ok(GenerationResult {
            tokens: vec![72, 101, 108, 108, 111], // "Hello" in ASCII
            text: "Hello".to_string(),
            finish_reason: FinishReason::Length,
            generation_stats: GenerationStats {
                prompt_tokens: 5,
                completion_tokens: 5,
                total_tokens: 10,
                inference_time_ms: 100,
                tokens_per_second: 50.0,
                time_to_first_token_ms: Some(20),
            },
        })
    }

    fn vocab_size(&self) -> usize {
        self.info.vocab_size
    }

    fn context_length(&self) -> usize {
        self.info.context_length
    }

    fn supports_feature(&self, _feature: ModelFeature) -> bool {
        true // Mock supports everything
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert_eq!(config.temperature, 0.7);
        assert!(!config.stream);
    }

    #[test]
    fn test_cache_entry_creation() {
        #[cfg(feature = "ml")]
        let cache_entry = CacheEntry::new(vec![], vec![], 10);
        #[cfg(not(feature = "ml"))]
        let cache_entry = CacheEntry::new(vec![], vec![], 10);

        assert_eq!(cache_entry.sequence_length, 10);
        assert!(cache_entry.age().as_millis() < 100);
    }

    #[test]
    fn test_finish_reason() {
        assert_eq!(FinishReason::Length, FinishReason::Length);
        assert_ne!(FinishReason::Length, FinishReason::Stop);
    }
}
