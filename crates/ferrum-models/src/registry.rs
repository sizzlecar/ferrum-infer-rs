//! Model registry and alias management

use ferrum_types::{ModelId, Result, FerrumError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Model alias entry
#[derive(Debug, Clone)]
pub struct ModelAlias {
    /// Alias name (short name)
    pub name: String,
    /// Target model identifier
    pub target: String,
    /// Optional description
    pub description: Option<String>,
}

/// Architecture types for models
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Architecture {
    Llama,
    Qwen2,
    Mistral,
    Phi,
    GPT2,
    Unknown,
}

impl Architecture {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llamaforcausallm" => Architecture::Llama,
            "qwen2" | "qwen2forcausallm" => Architecture::Qwen2,
            "mistral" | "mistralforcausallm" => Architecture::Mistral,
            "phi" | "phiforcausallm" => Architecture::Phi,
            "gpt2" | "gpt2lmheadmodel" => Architecture::GPT2,
            _ => Architecture::Unknown,
        }
    }
}

/// Model format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormatType {
    SafeTensors,
    PyTorch,
    GGUF,
    Unknown,
}

/// Discovered model entry
#[derive(Debug, Clone)]
pub struct ModelDiscoveryEntry {
    /// Model identifier
    pub id: String,
    /// Local path to model
    pub path: PathBuf,
    /// Model format
    pub format: ModelFormatType,
    /// Architecture type (if detected)
    pub architecture: Option<Architecture>,
    /// Whether model passes validation
    pub is_valid: bool,
}

/// Model registry for managing models and aliases
#[derive(Debug)]
pub struct DefaultModelRegistry {
    /// Model aliases
    aliases: HashMap<String, String>,
    /// Discovered models cache
    discovered_models: Vec<ModelDiscoveryEntry>,
}

impl DefaultModelRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            discovered_models: Vec::new(),
        }
    }
    
    /// Create registry with common aliases
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        
        // Common model aliases
        registry.register_alias("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        registry.register_alias("llama2-7b", "meta-llama/Llama-2-7b-hf");
        registry.register_alias("llama2-7b-chat", "meta-llama/Llama-2-7b-chat-hf");
        registry.register_alias("llama3-8b", "meta-llama/Meta-Llama-3-8B");
        registry.register_alias("llama3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct");
        registry.register_alias("qwen2-7b", "Qwen/Qwen2-7B");
        registry.register_alias("qwen2-7b-instruct", "Qwen/Qwen2-7B-Instruct");
        registry.register_alias("mistral-7b", "mistralai/Mistral-7B-v0.1");
        registry.register_alias("mistral-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2");
        registry.register_alias("phi3-mini", "microsoft/Phi-3-mini-4k-instruct");
        
        registry
    }
    
    /// Register a model alias
    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        let alias_str = alias.into();
        let target_str = target.into();
        debug!("Registering alias: {} -> {}", alias_str, target_str);
        self.aliases.insert(alias_str, target_str);
    }
    
    /// Add alias from struct
    pub fn add_alias(&mut self, alias: ModelAlias) -> Result<()> {
        self.register_alias(alias.name, alias.target);
        Ok(())
    }
    
    /// Resolve model ID through aliases
    pub fn resolve_model_id(&self, name: &str) -> String {
        self.aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
    
    /// List all registered aliases
    pub fn list_aliases(&self) -> Vec<ModelAlias> {
        self.aliases
            .iter()
            .map(|(name, target)| ModelAlias {
                name: name.clone(),
                target: target.clone(),
                description: None,
            })
            .collect()
    }
    
    /// Discover models in a directory
    pub async fn discover_models(&mut self, root: &Path) -> Result<Vec<ModelDiscoveryEntry>> {
        if !root.exists() || !root.is_dir() {
            return Ok(Vec::new());
        }
        
        info!("Discovering models in: {:?}", root);
        
        let mut discovered = Vec::new();
        
        // First check if root itself is a model directory
        if let Some(model_entry) = self.inspect_model_dir(root).await {
            discovered.push(model_entry);
        } else {
            // Otherwise scan subdirectories
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(model_entry) = self.inspect_model_dir(&path).await {
                            discovered.push(model_entry);
                        }
                    }
                }
            }
        }
        
        self.discovered_models = discovered.clone();
        Ok(discovered)
    }
    
    /// Inspect a directory to see if it contains a model
    async fn inspect_model_dir(&self, path: &Path) -> Option<ModelDiscoveryEntry> {
        // Check for config.json
        let config_path = path.join("config.json");
        if !config_path.exists() {
            debug!("No config.json in: {:?}", path);
            return None;
        }
        
        // Detect format
        let format = self.detect_model_format(path);
        if format == ModelFormatType::Unknown {
            debug!("Unknown format in: {:?}", path);
            return None;
        }
        
        debug!("Found valid model at: {:?}, format: {:?}", path, format);
        
        // Try to read architecture from config
        let architecture = self.read_architecture(&config_path);
        
        // Extract model ID from path - try to get friendly name from parent directory
        let id = if let Some(parent) = path.parent() {
            if let Some(grandparent) = parent.parent() {
                // Extract from models--org--name format
                if let Some(name) = grandparent.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("models--") {
                        name[8..].replace("--", "/")
                    } else {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    }
                } else {
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string()
                }
            } else {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            }
        } else {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        };
        
        Some(ModelDiscoveryEntry {
            id,
            path: path.to_path_buf(),
            format,
            architecture,
            is_valid: true,
        })
    }
    
    /// Detect model format in directory
    fn detect_model_format(&self, path: &Path) -> ModelFormatType {
        if path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists()
        {
            ModelFormatType::SafeTensors
        } else if path.join("pytorch_model.bin").exists()
            || path.join("pytorch_model.bin.index.json").exists()
        {
            ModelFormatType::PyTorch
        } else if std::fs::read_dir(path).ok().and_then(|entries| {
            entries
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        == Some("gguf")
                })
        }).is_some() {
            ModelFormatType::GGUF
        } else {
            ModelFormatType::Unknown
        }
    }
    
    /// Read architecture type from config.json
    fn read_architecture(&self, config_path: &Path) -> Option<Architecture> {
        let content = std::fs::read_to_string(config_path).ok()?;
        let config: serde_json::Value = serde_json::from_str(&content).ok()?;
        
        // Try "model_type" field
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            return Some(Architecture::from_str(model_type));
        }
        
        // Try "architectures" array
        if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
            if let Some(arch) = architectures.first().and_then(|v| v.as_str()) {
                return Some(Architecture::from_str(arch));
            }
        }
        
        None
    }
}

impl Default for DefaultModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
