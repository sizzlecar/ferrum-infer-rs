//! Model registry for managing available models

use ferrum_core::{ModelType, Result, Error};
use std::collections::HashMap;
use parking_lot::RwLock;
use tracing::info;

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_type: ModelType,
    pub description: String,
    pub size_gb: f32,
    pub requires_gpu: bool,
    pub max_sequence_length: usize,
    pub default_config: HashMap<String, serde_json::Value>,
}

/// Model registry
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelMetadata>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            models: RwLock::new(HashMap::new()),
        };
        
        // Register default models
        registry.register_default_models();
        registry
    }
    
    /// Register default models
    fn register_default_models(&mut self) {
        // Llama models
        self.register(ModelMetadata {
            model_id: "meta-llama/Llama-2-7b-hf".to_string(),
            model_type: ModelType::Llama,
            description: "Llama 2 7B base model".to_string(),
            size_gb: 13.5,
            requires_gpu: true,
            max_sequence_length: 4096,
            default_config: HashMap::new(),
        });
        
        self.register(ModelMetadata {
            model_id: "meta-llama/Llama-2-13b-hf".to_string(),
            model_type: ModelType::Llama,
            description: "Llama 2 13B base model".to_string(),
            size_gb: 26.0,
            requires_gpu: true,
            max_sequence_length: 4096,
            default_config: HashMap::new(),
        });
        
        // Mistral models
        self.register(ModelMetadata {
            model_id: "mistralai/Mistral-7B-v0.1".to_string(),
            model_type: ModelType::Mistral,
            description: "Mistral 7B v0.1 base model".to_string(),
            size_gb: 14.5,
            requires_gpu: true,
            max_sequence_length: 8192,
            default_config: HashMap::new(),
        });
        
        self.register(ModelMetadata {
            model_id: "mistralai/Mixtral-8x7B-v0.1".to_string(),
            model_type: ModelType::Mistral,
            description: "Mixtral 8x7B MoE model".to_string(),
            size_gb: 87.0,
            requires_gpu: true,
            max_sequence_length: 32768,
            default_config: HashMap::new(),
        });
        
        // Qwen models
        self.register(ModelMetadata {
            model_id: "Qwen/Qwen-7B".to_string(),
            model_type: ModelType::Qwen,
            description: "Qwen 7B base model".to_string(),
            size_gb: 14.0,
            requires_gpu: true,
            max_sequence_length: 8192,
            default_config: HashMap::new(),
        });
    }
    
    /// Register a model
    pub fn register(&self, metadata: ModelMetadata) {
        info!("Registering model: {}", metadata.model_id);
        self.models.write().insert(metadata.model_id.clone(), metadata);
    }
    
    /// Get model metadata
    pub fn get(&self, model_id: &str) -> Option<ModelMetadata> {
        self.models.read().get(model_id).cloned()
    }
    
    /// List all registered models
    pub fn list(&self) -> Vec<ModelMetadata> {
        self.models.read().values().cloned().collect()
    }
    
    /// Check if model is registered
    pub fn contains(&self, model_id: &str) -> bool {
        self.models.read().contains_key(model_id)
    }
    
    /// Get model type
    pub fn get_model_type(&self, model_id: &str) -> Result<ModelType> {
        self.get(model_id)
            .map(|m| m.model_type)
            .ok_or_else(|| Error::not_found(format!("Model {} not registered", model_id)))
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
