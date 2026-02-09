//! # Ferrum Runtime
//!
//! Device runtime and compute backend implementations for LLM inference.

// Re-export traits from ferrum-interfaces
pub use ferrum_interfaces::{
    backend::KernelExecutor, ComputeBackend, DeviceMemoryManager, TensorFactory, TensorLike,
    TensorOps, TensorRef,
};

// Re-export types from ferrum-types
pub use ferrum_types::{DataType, Device, Result};

// Backend implementations
pub mod backends;
pub mod memory;

// Re-exports
pub use backends::*;
pub use memory::*;

use once_cell::sync::Lazy;
use std::sync::Arc;

/// Global backend registry
static BACKEND_REGISTRY: Lazy<Arc<DefaultBackendRegistry>> =
    Lazy::new(|| Arc::new(DefaultBackendRegistry::new()));

/// Get the global backend registry
pub fn global_backend_registry() -> Arc<DefaultBackendRegistry> {
    BACKEND_REGISTRY.clone()
}

/// Default backend registry
pub struct DefaultBackendRegistry {
    compute_backends:
        parking_lot::RwLock<std::collections::HashMap<String, Arc<dyn ComputeBackend>>>,
}

impl DefaultBackendRegistry {
    pub fn new() -> Self {
        Self {
            compute_backends: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn register_compute_backend(
        &self,
        name: &str,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<()> {
        self.compute_backends
            .write()
            .insert(name.to_string(), backend);
        Ok(())
    }

    pub fn get_compute_backend(&self, name: &str) -> Option<Arc<dyn ComputeBackend>> {
        self.compute_backends.read().get(name).cloned()
    }

    pub fn list_backends(&self) -> Vec<String> {
        self.compute_backends.read().keys().cloned().collect()
    }
}

impl std::fmt::Debug for DefaultBackendRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultBackendRegistry")
            .field("backend_count", &self.compute_backends.read().len())
            .finish()
    }
}

impl Default for DefaultBackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
