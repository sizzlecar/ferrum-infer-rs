//! Model registry implementation
//!
//! This module provides a registry for model builders,
//! allowing different backends to register their implementations.

use crate::traits::{Architecture, ModelBuilder, ModelRegistry};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Default implementation of model registry
pub struct DefaultModelRegistry {
    /// Registered model builders
    builders: Arc<RwLock<HashMap<Architecture, Arc<dyn ModelBuilder>>>>,
}

impl DefaultModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            builders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default builders (none in this abstract crate)
    pub fn with_defaults() -> Self {
        Self::new()
    }
}

impl ModelRegistry for DefaultModelRegistry {
    fn register_builder(&mut self, builder: Box<dyn ModelBuilder>) {
        let arc_builder: Arc<dyn ModelBuilder> = Arc::from(builder);
        let mut builders = self.builders.write();
        for arch in arc_builder.supported_architectures() {
            builders.insert(arch, arc_builder.clone());
        }
    }

    fn get_builder(&self, _architecture: &Architecture) -> Option<&dyn ModelBuilder> {
        let _builders = self.builders.read();
        // This is a bit tricky due to lifetime issues
        // In a real implementation, we might need to return Arc<dyn ModelBuilder>
        // For now, we return None as this is just the interface
        None
    }

    fn supported_architectures(&self) -> Vec<Architecture> {
        let builders = self.builders.read();
        builders.keys().cloned().collect()
    }
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

    #[test]
    fn test_registry_creation() {
        let registry = DefaultModelRegistry::new();
        assert_eq!(registry.supported_architectures().len(), 0);
    }
}
