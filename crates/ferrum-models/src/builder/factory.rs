//! Model builder factory implementation

use crate::builder::{LlamaModelBuilder, MistralModelBuilder};
use ferrum_interfaces::{ModelBuilder, ModelCapabilities, ModelConfig, ModelInfo};
use ferrum_types::{Architecture, FerrumError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Default model builder factory
#[derive(Debug)]
pub struct DefaultModelBuilderFactory {
    /// Registered builders by architecture
    builders: HashMap<Architecture, Arc<dyn ModelBuilder + Send + Sync>>,
}

impl DefaultModelBuilderFactory {
    /// Create new model builder factory
    pub fn new() -> Self {
        let mut builders: HashMap<Architecture, Arc<dyn ModelBuilder + Send + Sync>> =
            HashMap::new();

        // Register built-in builders
        builders.insert(Architecture::Llama, Arc::new(LlamaModelBuilder::new()));
        builders.insert(Architecture::Llama2, Arc::new(LlamaModelBuilder::new()));
        builders.insert(Architecture::CodeLlama, Arc::new(LlamaModelBuilder::new()));
        builders.insert(Architecture::Mistral, Arc::new(MistralModelBuilder::new()));
        builders.insert(Architecture::Mixtral, Arc::new(MistralModelBuilder::new()));

        debug!(
            "Created model builder factory with {} builders",
            builders.len()
        );

        Self { builders }
    }

    /// Register a custom builder for an architecture
    pub fn register_builder<B>(&mut self, architecture: Architecture, builder: B)
    where
        B: ModelBuilder + Send + Sync + 'static,
    {
        self.builders.insert(architecture, Arc::new(builder));
        debug!("Registered custom builder for {:?}", architecture);
    }

    /// Get builder for architecture
    pub fn get_builder(
        &self,
        architecture: Architecture,
    ) -> Result<Arc<dyn ModelBuilder + Send + Sync>> {
        self.builders.get(&architecture).cloned().ok_or_else(|| {
            FerrumError::not_found(format!("No builder for architecture: {:?}", architecture))
        })
    }

    /// List supported architectures
    pub fn supported_architectures(&self) -> Vec<Architecture> {
        self.builders.keys().cloned().collect()
    }

    /// Check if architecture is supported
    pub fn supports_architecture(&self, architecture: Architecture) -> bool {
        self.builders.contains_key(&architecture)
    }

    /// Create model executor from config
    pub async fn build_model(
        &self,
        config: &ModelConfig,
        backend: Arc<dyn ferrum_interfaces::ComputeBackend + Send + Sync>,
        weight_loader: Arc<dyn ferrum_interfaces::WeightLoader + Send + Sync>,
    ) -> Result<Box<dyn ferrum_interfaces::ModelExecutor + Send + Sync>> {
        let builder = self.get_builder(config.architecture)?;
        builder.build(config, backend, weight_loader).await
    }

    /// Get model info from config (without building)
    pub fn get_model_info(&self, config: &ModelConfig) -> Result<ModelInfo> {
        let builder = self.get_builder(config.architecture)?;
        builder.model_info(config)
    }

    /// Get model capabilities
    pub fn get_capabilities(&self, architecture: Architecture) -> Result<ModelCapabilities> {
        let builder = self.get_builder(architecture)?;
        Ok(builder.capabilities())
    }
}

impl Default for DefaultModelBuilderFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Weight format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    GGUF,
    Pickle,
}

impl WeightFormat {
    /// Get file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            WeightFormat::SafeTensors => &["safetensors"],
            WeightFormat::GGUF => &["gguf"],
            WeightFormat::Pickle => &["pt", "pth", "pkl"],
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "safetensors" => Some(WeightFormat::SafeTensors),
            "gguf" => Some(WeightFormat::GGUF),
            "pt" | "pth" | "pkl" => Some(WeightFormat::Pickle),
            _ => None,
        }
    }

    /// Detect format from filename
    pub fn from_filename(filename: &str) -> Option<Self> {
        filename.rsplit('.').next().and_then(Self::from_extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let factory = DefaultModelBuilderFactory::new();
        let supported = factory.supported_architectures();

        assert!(supported.contains(&Architecture::Llama));
        assert!(supported.contains(&Architecture::Mistral));
        assert!(!supported.is_empty());
    }

    #[test]
    fn test_architecture_support() {
        let factory = DefaultModelBuilderFactory::new();

        assert!(factory.supports_architecture(Architecture::Llama));
        assert!(factory.supports_architecture(Architecture::Mistral));
        assert!(!factory.supports_architecture(Architecture::Mamba)); // Not registered by default
    }

    #[test]
    fn test_get_builder() {
        let factory = DefaultModelBuilderFactory::new();

        let builder = factory.get_builder(Architecture::Llama);
        assert!(builder.is_ok());

        let builder = factory.get_builder(Architecture::Mamba);
        assert!(builder.is_err());
    }

    #[test]
    fn test_weight_format_detection() {
        assert_eq!(
            WeightFormat::from_filename("model.safetensors"),
            Some(WeightFormat::SafeTensors)
        );
        assert_eq!(
            WeightFormat::from_filename("model.gguf"),
            Some(WeightFormat::GGUF)
        );
        assert_eq!(
            WeightFormat::from_filename("model.pt"),
            Some(WeightFormat::Pickle)
        );
        assert_eq!(WeightFormat::from_filename("model.unknown"), None);
    }

    #[test]
    fn test_weight_format_extensions() {
        assert_eq!(WeightFormat::SafeTensors.extensions(), &["safetensors"]);
        assert_eq!(WeightFormat::GGUF.extensions(), &["gguf"]);
        assert_eq!(WeightFormat::Pickle.extensions(), &["pt", "pth", "pkl"]);
    }
}
