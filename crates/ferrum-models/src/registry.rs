//! Model registry and alias management

use ferrum_types::{FerrumError, ModelId, Result};
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
        } else if std::fs::read_dir(path)
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"))
            })
            .is_some()
        {
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

// ============================================================================
// 内联单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_str() {
        assert_eq!(Architecture::from_str("llama"), Architecture::Llama);
        assert_eq!(
            Architecture::from_str("LlamaForCausalLM"),
            Architecture::Llama
        );
        assert_eq!(Architecture::from_str("qwen2"), Architecture::Qwen2);
        assert_eq!(Architecture::from_str("mistral"), Architecture::Mistral);
        assert_eq!(Architecture::from_str("phi"), Architecture::Phi);
        assert_eq!(Architecture::from_str("gpt2"), Architecture::GPT2);
        assert_eq!(
            Architecture::from_str("unknown_arch"),
            Architecture::Unknown
        );
    }

    #[test]
    fn test_architecture_copy() {
        let arch = Architecture::Llama;
        let arch2 = arch;
        assert_eq!(arch, arch2);
    }

    #[test]
    fn test_model_format_type_eq() {
        assert_eq!(ModelFormatType::SafeTensors, ModelFormatType::SafeTensors);
        assert_ne!(ModelFormatType::SafeTensors, ModelFormatType::PyTorch);
    }

    #[test]
    fn test_model_alias_creation() {
        let alias = ModelAlias {
            name: "test".to_string(),
            target: "test/model".to_string(),
            description: Some("Test model".to_string()),
        };

        assert_eq!(alias.name, "test");
        assert_eq!(alias.target, "test/model");
        assert!(alias.description.is_some());
    }

    #[test]
    fn test_model_alias_clone() {
        let alias = ModelAlias {
            name: "test".to_string(),
            target: "test/model".to_string(),
            description: None,
        };

        let cloned = alias.clone();
        assert_eq!(alias.name, cloned.name);
        assert_eq!(alias.target, cloned.target);
    }

    #[test]
    fn test_model_discovery_entry() {
        let entry = ModelDiscoveryEntry {
            id: "test-model".to_string(),
            path: PathBuf::from("/path/to/model"),
            format: ModelFormatType::SafeTensors,
            architecture: Some(Architecture::Llama),
            is_valid: true,
        };

        assert_eq!(entry.id, "test-model");
        assert_eq!(entry.format, ModelFormatType::SafeTensors);
        assert!(entry.is_valid);
    }

    #[test]
    fn test_registry_creation() {
        let registry = DefaultModelRegistry::new();
        assert_eq!(registry.aliases.len(), 0);
        assert_eq!(registry.discovered_models.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let registry = DefaultModelRegistry::default();
        assert_eq!(registry.aliases.len(), 0);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = DefaultModelRegistry::with_defaults();

        // 应该有一些默认别名
        assert!(registry.aliases.len() > 0);

        // 测试一些常见别名
        assert!(registry.aliases.contains_key("tinyllama"));
        assert!(registry.aliases.contains_key("llama2-7b"));
    }

    #[test]
    fn test_registry_register_alias() {
        let mut registry = DefaultModelRegistry::new();

        registry.register_alias("test", "test/model");

        assert_eq!(
            registry.aliases.get("test"),
            Some(&"test/model".to_string())
        );
    }

    #[test]
    fn test_registry_resolve_model_id() {
        let mut registry = DefaultModelRegistry::new();

        registry.register_alias("mymodel", "org/actual-model");

        let resolved = registry.resolve_model_id("mymodel");
        assert_eq!(resolved, "org/actual-model");

        // 未注册的别名应该返回原始值
        let unresolved = registry.resolve_model_id("unknown");
        assert_eq!(unresolved, "unknown");
    }

    #[test]
    fn test_registry_list_aliases() {
        let mut registry = DefaultModelRegistry::new();

        registry.register_alias("model1", "org/model1");
        registry.register_alias("model2", "org/model2");

        let aliases = registry.list_aliases();
        assert_eq!(aliases.len(), 2);
    }

    #[test]
    fn test_architecture_debug() {
        let arch = Architecture::Llama;
        let debug_str = format!("{:?}", arch);
        assert!(debug_str.contains("Llama"));
    }

    #[test]
    fn test_model_format_debug() {
        let format = ModelFormatType::SafeTensors;
        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("SafeTensors"));
    }

    #[test]
    fn test_model_discovery_entry_clone() {
        let entry = ModelDiscoveryEntry {
            id: "test".to_string(),
            path: PathBuf::from("/path"),
            format: ModelFormatType::GGUF,
            architecture: Some(Architecture::Mistral),
            is_valid: false,
        };

        let cloned = entry.clone();
        assert_eq!(entry.id, cloned.id);
        assert_eq!(entry.format, cloned.format);
        assert_eq!(entry.is_valid, cloned.is_valid);
    }

    #[test]
    fn test_registry_multiple_aliases_same_target() {
        let mut registry = DefaultModelRegistry::new();

        registry.register_alias("alias1", "org/model");
        registry.register_alias("alias2", "org/model");

        assert_eq!(registry.resolve_model_id("alias1"), "org/model");
        assert_eq!(registry.resolve_model_id("alias2"), "org/model");
    }

    #[test]
    fn test_architecture_serialization() {
        let arch = Architecture::Qwen2;
        let json = serde_json::to_string(&arch).unwrap();
        assert!(json.contains("Qwen2"));

        let deserialized: Architecture = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, arch);
    }
}
