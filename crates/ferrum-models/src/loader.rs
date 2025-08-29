//! Model loader implementation using Candle

use async_trait::async_trait;
use ferrum_core::{
    ModelLoader, Model, ModelConfig, ModelInfo, ModelId, Result, Error,
};
use candle_core::{Device, DType};
use std::sync::Arc;
use std::path::Path;
use tracing::{info, warn, debug};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Loader configuration
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Model cache directory
    pub cache_dir: String,
    
    /// Enable model caching
    pub enable_cache: bool,
    
    /// Download models from HuggingFace if not found locally
    pub enable_download: bool,
    
    /// Maximum number of models to keep in memory
    pub max_loaded_models: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: ".cache/models".to_string(),
            enable_cache: true,
            enable_download: true,
            max_loaded_models: 2,
        }
    }
}

/// Candle-based model loader
pub struct CandleModelLoader {
    config: LoaderConfig,
    loaded_models: Arc<RwLock<HashMap<ModelId, Arc<dyn Model>>>>,
    device_cache: Arc<RwLock<Device>>,
}

impl CandleModelLoader {
    /// Create a new model loader
    pub fn new(config: LoaderConfig) -> Self {
        info!("Initializing CandleModelLoader with cache dir: {}", config.cache_dir);
        
        // Initialize device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        Self {
            config,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            device_cache: Arc::new(RwLock::new(device)),
        }
    }
    
    /// Get or create device from config
    fn get_device(&self, config: &ModelConfig) -> Result<Device> {
        match &config.device {
            ferrum_core::Device::CPU => Ok(Device::Cpu),
            ferrum_core::Device::CUDA(id) => {
                Device::cuda(*id)
                    .map_err(|e| Error::model_loading(format!("Failed to get CUDA device: {}", e)))
            }
            ferrum_core::Device::ROCm(_) => {
                Err(Error::unsupported("ROCm is not yet supported"))
            }
        }
    }
    
    /// Convert data type
    fn convert_dtype(&self, dtype: ferrum_core::DataType) -> DType {
        match dtype {
            ferrum_core::DataType::FP32 => DType::F32,
            ferrum_core::DataType::FP16 => DType::F16,
            ferrum_core::DataType::BF16 => DType::BF16,
            ferrum_core::DataType::INT8 => DType::U8,
            ferrum_core::DataType::FP8 => DType::U8, // FP8 not directly supported
        }
    }
    
    /// Download model from HuggingFace if needed
    async fn download_model_if_needed(&self, model_id: &str, model_path: &str) -> Result<String> {
        let path = Path::new(model_path);
        
        if path.exists() {
            debug!("Model found at: {}", model_path);
            return Ok(model_path.to_string());
        }
        
        if !self.config.enable_download {
            return Err(Error::not_found(format!("Model not found at {} and download is disabled", model_path)));
        }
        
        info!("Downloading model {} from HuggingFace...", model_id);
        
        // Use hf-hub to download
        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());
        
        // Download model files
        let model_file = repo.get("pytorch_model.bin").await
            .or_else(|_| async { repo.get("model.safetensors").await }.await)
            .map_err(|e| Error::model_loading(format!("Failed to download model: {}", e)))?;
        
        Ok(model_file.to_string_lossy().to_string())
    }
    
    /// Load model based on type
    async fn load_model_impl(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let device = self.get_device(config)?;
        let dtype = self.convert_dtype(config.dtype);
        
        // Download model if needed
        let model_path = self.download_model_if_needed(
            &config.model_id.0,
            &config.model_path
        ).await?;
        
        // Load based on model type
        let model: Arc<dyn Model> = match &config.model_type {
            ferrum_core::ModelType::Llama => {
                info!("Loading Llama model from {}", model_path);
                let llama_config = crate::llama::LlamaConfig::from_model_config(config)?;
                let model = crate::llama::LlamaModel::load(
                    &model_path,
                    llama_config,
                    device,
                    dtype,
                ).await?;
                Arc::new(model)
            }
            ferrum_core::ModelType::Mistral => {
                info!("Loading Mistral model from {}", model_path);
                let mistral_config = crate::mistral::MistralConfig::from_model_config(config)?;
                let model = crate::mistral::MistralModel::load(
                    &model_path,
                    mistral_config,
                    device,
                    dtype,
                ).await?;
                Arc::new(model)
            }
            ferrum_core::ModelType::Qwen => {
                // Qwen can use similar architecture to Llama
                info!("Loading Qwen model from {}", model_path);
                let llama_config = crate::llama::LlamaConfig::from_model_config(config)?;
                let model = crate::llama::LlamaModel::load(
                    &model_path,
                    llama_config,
                    device,
                    dtype,
                ).await?;
                Arc::new(model)
            }
            ferrum_core::ModelType::Custom(name) => {
                return Err(Error::unsupported(format!("Custom model type {} not supported", name)));
            }
        };
        
        Ok(model)
    }
    
    /// Evict least recently used model if needed
    fn evict_if_needed(&self) {
        let models = self.loaded_models.read();
        if models.len() >= self.config.max_loaded_models {
            drop(models);
            
            // Simple eviction: remove first model (could be improved with LRU)
            let mut models = self.loaded_models.write();
            if let Some(key) = models.keys().next().cloned() {
                info!("Evicting model {:?} to make room", key);
                models.remove(&key);
            }
        }
    }
}

#[async_trait]
impl ModelLoader for CandleModelLoader {
    async fn load_model(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        // Check if already loaded
        {
            let models = self.loaded_models.read();
            if let Some(model) = models.get(&config.model_id) {
                debug!("Model {:?} already loaded, returning cached version", config.model_id);
                return Ok(Arc::clone(model));
            }
        }
        
        // Evict old models if needed
        self.evict_if_needed();
        
        // Load the model
        info!("Loading model {:?}...", config.model_id);
        let model = self.load_model_impl(config).await?;
        
        // Cache the model
        if self.config.enable_cache {
            let mut models = self.loaded_models.write();
            models.insert(config.model_id.clone(), Arc::clone(&model));
            info!("Model {:?} loaded and cached", config.model_id);
        }
        
        Ok(model)
    }
    
    async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.loaded_models.write();
        
        if models.remove(&ModelId(model_id.to_string())).is_some() {
            info!("Model {} unloaded", model_id);
            Ok(())
        } else {
            Err(Error::not_found(format!("Model {} not found", model_id)))
        }
    }
    
    async fn get_model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        let models = self.loaded_models.read();
        models.get(&ModelId(model_id.to_string())).cloned()
    }
    
    async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.loaded_models.read();
        models.values().map(|m| m.info().clone()).collect()
    }
}
