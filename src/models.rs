//! Model management and loading for the LLM inference engine
//!
//! This module provides abstractions for loading and managing LLM models,
//! supporting different model types and backends through a trait-based design.

use crate::config::{Config, ModelConfig};
use crate::error::{EngineError, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::llama::{Cache, LlamaConfig};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
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
    pub key_cache: Vec<Tensor>,
    pub value_cache: Vec<Tensor>,
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
    models: RwLock<HashMap<String, Arc<dyn Model>>>,
    loaders: Vec<Box<dyn ModelLoader>>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(config: Config) -> Self {
        Self {
            config,
            models: RwLock::new(HashMap::new()),
            loaders: Vec::new(),
        }
    }

    /// Add a model loader
    pub fn add_loader(&mut self, loader: Box<dyn ModelLoader>) {
        self.loaders.push(loader);
    }

    /// Load a model with the given name
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
            if loader.supported_model_types().contains(&self.config.model.name) {
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

        Err(EngineError::model(format!("No loader found for model: {}", name)))
    }

    /// Get a loaded model
    pub fn get_model(&self, name: &str) -> Option<Arc<dyn Model>> {
        let models = self.models.read();
        models.get(name).map(Arc::clone)
    }

    /// List all loaded models
    pub fn list_models(&self) -> Vec<String> {
        let models = self.models.read();
        models.keys().cloned().collect()
    }

    /// Unload a model
    pub fn unload_model(&self, name: &str) -> bool {
        let mut models = self.models.write();
        models.remove(name).is_some()
    }

    /// Get model information for all loaded models
    pub fn get_all_model_info(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().map(|model| model.model_info()).collect()
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