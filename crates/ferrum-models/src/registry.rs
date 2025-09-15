//! Model registry implementation
//!
//! This module provides a registry for model builders,
//! allowing different backends to register their implementations.
//! Enhanced with dynamic discovery, alias management, and validation.

use crate::traits::{Architecture, ModelBuilder, ModelRegistry};
use crate::source::{ModelFormat, ModelSourceConfig};
use ferrum_core::{Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Model alias configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAlias {
    /// Alias name
    pub name: String,
    /// Target model ID
    pub target: String,
    /// Description
    pub description: Option<String>,
}

/// Model discovery result
#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    /// Model ID
    pub id: String,
    /// Local path
    pub path: PathBuf,
    /// Detected format
    pub format: ModelFormat,
    /// Architecture (if detected)
    pub architecture: Option<Architecture>,
    /// Whether validation passed
    pub is_valid: bool,
}

/// Enhanced model registry with dynamic discovery and alias management
pub struct DefaultModelRegistry {
    /// Registered model builders
    builders: Arc<RwLock<HashMap<Architecture, Arc<dyn ModelBuilder>>>>,
    /// Model aliases (alias_name -> target_model_id)
    aliases: Arc<RwLock<HashMap<String, ModelAlias>>>,
    /// Discovered models cache
    discovered_models: Arc<RwLock<HashMap<String, DiscoveredModel>>>,
    /// Source configuration
    #[allow(dead_code)]
    source_config: ModelSourceConfig,
}

impl DefaultModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            builders: Arc::new(RwLock::new(HashMap::new())),
            aliases: Arc::new(RwLock::new(HashMap::new())),
            discovered_models: Arc::new(RwLock::new(HashMap::new())),
            source_config: ModelSourceConfig::default(),
        }
    }

    /// Create with custom source configuration
    pub fn with_config(config: ModelSourceConfig) -> Self {
        Self {
            builders: Arc::new(RwLock::new(HashMap::new())),
            aliases: Arc::new(RwLock::new(HashMap::new())),
            discovered_models: Arc::new(RwLock::new(HashMap::new())),
            source_config: config,
        }
    }

    /// Create with default builders (none in this abstract crate)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.load_default_aliases();
        registry
    }
    
    /// Load default model aliases
    fn load_default_aliases(&mut self) {
        let default_aliases = vec![
            ModelAlias {
                name: "llama2-7b".to_string(),
                target: "meta-llama/Llama-2-7b-hf".to_string(),
                description: Some("Llama 2 7B model".to_string()),
            },
            ModelAlias {
                name: "llama2-13b".to_string(),
                target: "meta-llama/Llama-2-13b-hf".to_string(),
                description: Some("Llama 2 13B model".to_string()),
            },
            ModelAlias {
                name: "mistral-7b".to_string(),
                target: "mistralai/Mistral-7B-v0.1".to_string(),
                description: Some("Mistral 7B model".to_string()),
            },
        ];
        
        for alias in default_aliases {
            self.aliases.write().insert(alias.name.clone(), alias);
        }
    }
}

impl ModelRegistry for DefaultModelRegistry {
    fn register_builder(&mut self, builder: Box<dyn ModelBuilder>) {
        let arc_builder: Arc<dyn ModelBuilder> = Arc::from(builder);
        let mut builders = self.builders.write();
        for arch in arc_builder.supported_architectures() {
            debug!("Registering builder for architecture: {:?}", arch);
            builders.insert(arch, arc_builder.clone());
        }
    }

    fn get_builder_arc(&self, architecture: &Architecture) -> Option<Arc<dyn ModelBuilder>> {
        self.builders.read().get(architecture).cloned()
    }

    fn supported_architectures(&self) -> Vec<Architecture> {
        let builders = self.builders.read();
        builders.keys().cloned().collect()
    }
}

impl DefaultModelRegistry {
    /// Resolve model ID (handle aliases)
    pub fn resolve_model_id(&self, id_or_alias: &str) -> String {
        if let Some(alias) = self.aliases.read().get(id_or_alias) {
            debug!("Resolved alias '{}' to '{}'", id_or_alias, alias.target);
            alias.target.clone()
        } else {
            id_or_alias.to_string()
        }
    }
    
    /// Add model alias
    pub fn add_alias(&mut self, alias: ModelAlias) -> Result<()> {
        info!("Adding model alias: {} -> {}", alias.name, alias.target);
        self.aliases.write().insert(alias.name.clone(), alias);
        Ok(())
    }
    
    /// Remove model alias
    pub fn remove_alias(&mut self, alias_name: &str) -> Result<()> {
        if self.aliases.write().remove(alias_name).is_some() {
            info!("Removed model alias: {}", alias_name);
            Ok(())
        } else {
            Err(Error::invalid_request(format!(
                "Alias '{}' not found",
                alias_name
            )))
        }
    }
    
    /// List all aliases
    pub fn list_aliases(&self) -> Vec<ModelAlias> {
        self.aliases.read().values().cloned().collect()
    }
    
    /// Discover models in a directory
    pub async fn discover_models(&mut self, directory: &PathBuf) -> Result<Vec<DiscoveredModel>> {
        info!("Discovering models in directory: {:?}", directory);
        let mut discovered = Vec::new();
        
        if !directory.exists() || !directory.is_dir() {
            return Err(Error::invalid_request(format!(
                "Directory does not exist or is not a directory: {:?}",
                directory
            )));
        }
        
        let entries = std::fs::read_dir(directory)
            .map_err(|e| Error::internal(format!("Failed to read directory: {}", e)))?;
            
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(model) = self.discover_single_model(&path).await {
                    discovered.push(model.clone());
                    self.discovered_models.write().insert(model.id.clone(), model);
                }
            }
        }
        
        info!("Discovered {} models", discovered.len());
        Ok(discovered)
    }
    
    /// Discover single model
    async fn discover_single_model(&self, path: &PathBuf) -> Result<DiscoveredModel> {
        let model_id = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
            
        let format = self.detect_model_format(path);
        let architecture = self.detect_architecture_from_path(path).await;
        let is_valid = self.validate_model_directory(path);
        
        Ok(DiscoveredModel {
            id: model_id,
            path: path.clone(),
            format,
            architecture,
            is_valid,
        })
    }
    
    /// Detect model format from directory (使用统一的工具函数)
    fn detect_model_format(&self, path: &PathBuf) -> ModelFormat {
        crate::source::detect_format(path)
    }
    
    /// Detect architecture from path
    async fn detect_architecture_from_path(&self, path: &PathBuf) -> Option<Architecture> {
        // Try to detect from config file
        let config_path = path.join("config.json");
        if config_path.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&config_path).await {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                        return self.parse_architecture_string(model_type);
                    }
                    if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
                        if let Some(arch) = architectures.first().and_then(|v| v.as_str()) {
                            return self.parse_architecture_string(arch);
                        }
                    }
                }
            }
        }
        
        // Fall back to directory name heuristics
        let dir_name = path.file_name()?.to_str()?.to_lowercase();
        if dir_name.contains("llama") {
            Some(Architecture::Llama)
        } else if dir_name.contains("mistral") {
            Some(Architecture::Mistral)
        } else if dir_name.contains("qwen") {
            Some(Architecture::Qwen)
        } else {
            None
        }
    }
    
    /// Parse architecture string
    fn parse_architecture_string(&self, arch_str: &str) -> Option<Architecture> {
        let arch_lower = arch_str.to_lowercase();
        if arch_lower.contains("llama") {
            Some(Architecture::Llama)
        } else if arch_lower.contains("mistral") {
            Some(Architecture::Mistral)
        } else if arch_lower.contains("qwen2") {
            Some(Architecture::Qwen2)
        } else if arch_lower.contains("qwen") {
            Some(Architecture::Qwen)
        } else if arch_lower.contains("phi") {
            Some(Architecture::Phi)
        } else if arch_lower.contains("gemma") {
            Some(Architecture::Gemma)
        } else {
            None
        }
    }
    
    /// Validate model directory
    fn validate_model_directory(&self, path: &PathBuf) -> bool {
        // Basic validation: check for essential files
        let has_config = path.join("config.json").exists() || 
                         path.join("params.json").exists();
        let has_tokenizer = path.join("tokenizer.json").exists();
        let has_weights = path.join("model.safetensors").exists() ||
                          std::fs::read_dir(path)
                              .map(|entries| {
                                  entries.filter_map(|e| e.ok())
                                      .any(|e| {
                                          let file_name = e.file_name();
                                          let name = file_name.to_string_lossy();
                                          name.ends_with(".safetensors") || name.ends_with(".bin")
                                      })
                              })
                              .unwrap_or(false);
                              
        has_config && has_tokenizer && has_weights
    }
    
    /// Get discovered models
    pub fn get_discovered_models(&self) -> Vec<DiscoveredModel> {
        self.discovered_models.read().values().cloned().collect()
    }
    
    /// Clear discovery cache
    pub fn clear_discovery_cache(&mut self) {
        self.discovered_models.write().clear();
    }
    
    // Note: get_builder_arc is now implemented via the ModelRegistry trait
}

impl Default for DefaultModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global model registry instance
static GLOBAL_REGISTRY: once_cell::sync::Lazy<Arc<RwLock<DefaultModelRegistry>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(DefaultModelRegistry::new())));

/// Get the global model registry
pub fn global_registry() -> Arc<RwLock<DefaultModelRegistry>> {
    GLOBAL_REGISTRY.clone()
}

/// Register a model builder globally
pub fn register_global_builder(builder: Box<dyn ModelBuilder>) {
    let mut registry = GLOBAL_REGISTRY.write();
    registry.register_builder(builder);
}

/// Model registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Whether to use global registry
    pub use_global: bool,

    /// Custom builders to register
    pub custom_builders: Vec<String>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            use_global: true,
            custom_builders: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_registry_creation() {
        let registry = DefaultModelRegistry::new();
        assert_eq!(registry.supported_architectures().len(), 0);
    }

    #[test]
    fn test_alias_resolution() {
        let registry = DefaultModelRegistry::with_defaults();
        assert_eq!(
            registry.resolve_model_id("llama2-13b"),
            "meta-llama/Llama-2-13b-hf"
        );
        assert_eq!(registry.resolve_model_id("unknown"), "unknown");
    }

    #[tokio::test]
    async fn test_discover_models_minimal_valid() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().to_path_buf();

        let mdir = root.join("model_a");
        fs::create_dir_all(&mdir).unwrap();
        fs::write(mdir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(mdir.join("tokenizer.json"), "{}\n").unwrap();
        fs::write(mdir.join("model.safetensors"), "stub").unwrap();

        let mut registry = DefaultModelRegistry::new();
        let found = registry.discover_models(&root).await.unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].is_valid);
        assert!(matches!(found[0].architecture, Some(Architecture::Llama)));
        assert_eq!(found[0].format, ModelFormat::HuggingFace);
    }
}
