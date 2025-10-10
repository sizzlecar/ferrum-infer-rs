//! Model source resolution and downloading
//!
//! Supports multiple model sources:
//! - HuggingFace Hub (with authentication)
//! - Local filesystem paths
//! - HTTP/HTTPS URLs

use ferrum_types::{FerrumError, ModelSource, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Configuration for model source resolution
#[derive(Debug, Clone)]
pub struct ModelSourceConfig {
    /// Cache directory for downloaded models
    pub cache_dir: Option<PathBuf>,
    /// HuggingFace API token
    pub hf_token: Option<String>,
    /// Offline mode - only use cached models
    pub offline_mode: bool,
    /// Maximum retries for downloads
    pub max_retries: usize,
    /// Download timeout in seconds
    pub download_timeout: u64,
    /// Use file locks during downloads
    pub use_file_lock: bool,
}

impl Default for ModelSourceConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            hf_token: Self::get_hf_token(),
            offline_mode: false,
            max_retries: 3,
            download_timeout: 300,
            use_file_lock: true,
        }
    }
}

impl ModelSourceConfig {
    /// Get HuggingFace token from environment
    pub fn get_hf_token() -> Option<String> {
        std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok()
    }
}

/// Model file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format (preferred)
    SafeTensors,
    /// PyTorch .bin files
    PyTorchBin,
    /// GGUF format (quantized)
    GGUF,
    /// Unknown/unsupported format
    Unknown,
}

/// Resolved model source with local path
#[derive(Debug, Clone)]
pub struct ResolvedModelSource {
    /// Original model identifier
    pub original: String,
    /// Local path to model files
    pub local_path: PathBuf,
    /// Detected model format
    pub format: ModelFormat,
    /// Whether loaded from cache
    pub from_cache: bool,
}

impl From<ResolvedModelSource> for ModelSource {
    fn from(value: ResolvedModelSource) -> Self {
        ModelSource::Local(value.local_path.display().to_string())
    }
}

/// Trait for model source resolution
#[async_trait::async_trait]
pub trait ModelSourceResolver: Send + Sync {
    /// Resolve a model identifier to a local path
    ///
    /// # Arguments
    /// * `id` - Model identifier (HF repo, local path, etc.)
    /// * `revision` - Optional git revision/branch/tag
    async fn resolve(&self, id: &str, revision: Option<&str>) -> Result<ResolvedModelSource>;
}

/// Default implementation using HuggingFace Hub
pub struct DefaultModelSourceResolver {
    config: ModelSourceConfig,
    api: Api,
}

impl DefaultModelSourceResolver {
    pub fn new(config: ModelSourceConfig) -> Self {
        let mut builder = ApiBuilder::new();
        
        // Set cache directory
        if let Some(cache_dir) = &config.cache_dir {
            builder = builder.with_cache_dir(cache_dir.clone());
        }
        
        // Set API token
        if let Some(token) = &config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        
        let api = builder.build().unwrap_or_else(|e| {
            warn!("Failed to build HF API with custom config: {}, using default", e);
            Api::new().expect("Failed to create default HF API")
        });
        
        Self { config, api }
    }
    
    /// Check if path is a local directory
    fn is_local_path(id: &str) -> bool {
        Path::new(id).exists()
    }
    
    /// Detect model format from directory
    fn detect_format(path: &Path) -> ModelFormat {
        if path.join("model.safetensors").exists() 
            || path.join("model-00001-of-00001.safetensors").exists()
            || std::fs::read_dir(path)
                .ok()
                .and_then(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
                })
                .is_some()
        {
            ModelFormat::SafeTensors
        } else if path.join("pytorch_model.bin").exists() {
            ModelFormat::PyTorchBin
        } else if std::fs::read_dir(path)
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"))
            })
            .is_some()
        {
            ModelFormat::GGUF
        } else {
            ModelFormat::Unknown
        }
    }
    
    /// Resolve from local filesystem
    async fn resolve_local(&self, path: &str) -> Result<ResolvedModelSource> {
        let path_buf = PathBuf::from(path);
        
        if !path_buf.exists() {
            return Err(FerrumError::model(format!("Model path does not exist: {}", path)));
        }
        
        let format = Self::detect_format(&path_buf);
        
        Ok(ResolvedModelSource {
            original: path.to_string(),
            local_path: path_buf,
            format,
            from_cache: true,
        })
    }
    
    /// Resolve from HuggingFace Hub
    async fn resolve_huggingface(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<ResolvedModelSource> {
        info!("Resolving HuggingFace model: {}", repo_id);
        
        let repo = if let Some(rev) = revision {
            self.api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            ))
        } else {
            self.api.repo(hf_hub::Repo::new(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
            ))
        };
        
        // Try to download essential files
        debug!("Downloading config.json...");
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| FerrumError::model(format!("Failed to download config.json: {}", e)))?;
        
        let model_dir = config_path
            .parent()
            .ok_or_else(|| FerrumError::model("Invalid cache path"))?
            .to_path_buf();
        
        debug!("Model cached at: {:?}", model_dir);
        
        // Try to detect and download model weights
        let format = self.download_weights(&repo, &model_dir).await?;
        
        Ok(ResolvedModelSource {
            original: repo_id.to_string(),
            local_path: model_dir,
            format,
            from_cache: false,
        })
    }
    
    /// Download model weights
    async fn download_weights(&self, repo: &ApiRepo, model_dir: &Path) -> Result<ModelFormat> {
        // Try SafeTensors first (preferred)
        if let Ok(_) = repo.get("model.safetensors").await {
            info!("Downloaded model.safetensors");
            return Ok(ModelFormat::SafeTensors);
        }
        
        // Try sharded SafeTensors
        if let Ok(_) = repo.get("model.safetensors.index.json").await {
            info!("Found sharded SafeTensors model");
            // Download all shards
            if let Ok(index_path) = repo.get("model.safetensors.index.json").await {
                if let Ok(index_content) = std::fs::read_to_string(&index_path) {
                    if let Ok(index) = serde_json::from_str::<serde_json::Value>(&index_content) {
                        if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                            let shard_files: std::collections::HashSet<_> = weight_map
                                .values()
                                .filter_map(|v| v.as_str())
                                .collect();
                            
                            for shard in shard_files {
                                debug!("Downloading shard: {}", shard);
                                repo.get(shard).await.map_err(|e| {
                                    FerrumError::model(format!("Failed to download shard {}: {}", shard, e))
                                })?;
                            }
                        }
                    }
                }
            }
            return Ok(ModelFormat::SafeTensors);
        }
        
        // Try PyTorch .bin format
        if let Ok(_) = repo.get("pytorch_model.bin").await {
            warn!("Using PyTorch .bin format (SafeTensors preferred)");
            return Ok(ModelFormat::PyTorchBin);
        }
        
        // Check for GGUF
        if Self::detect_format(model_dir) == ModelFormat::GGUF {
            return Ok(ModelFormat::GGUF);
        }
        
        Err(FerrumError::model(
            "No supported model format found (expected SafeTensors or PyTorch .bin)",
        ))
    }
}

#[async_trait::async_trait]
impl ModelSourceResolver for DefaultModelSourceResolver {
    async fn resolve(&self, id: &str, revision: Option<&str>) -> Result<ResolvedModelSource> {
        // Check if it's a local path
        if Self::is_local_path(id) {
            return self.resolve_local(id).await;
        }
        
        // Otherwise treat as HuggingFace repo
        self.resolve_huggingface(id, revision).await
    }
}
