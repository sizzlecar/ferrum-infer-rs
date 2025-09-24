//! # Ferrum Runtime
//!
//! Device runtime and compute backend implementations for LLM inference.
//!
//! ## Overview
//!
//! This crate provides concrete implementations of:
//! - `TensorFactory` and `TensorOps` for different backends (Candle, CPU, Metal)
//! - `ComputeBackend` for hardware abstraction
//! - `DeviceMemoryManager` for efficient GPU/CPU memory management
//! - `KernelExecutor` for custom GPU kernels
//!
//! ## Supported Backends
//!
//! - **Candle**: Default ML framework backend with CUDA/Metal support
//! - **CPU**: Optimized CPU operations using ndarray
//! - **Metal**: Custom Metal Performance Shaders implementation (coming soon)
//! - **CUDA**: Direct CUDA operations (coming soon)

// Core traits are re-exported from ferrum-interfaces
pub use ferrum_interfaces::{
    ComputeBackend, DeviceMemoryManager, TensorFactory, TensorLike, TensorOps,
    TensorRef,
};
pub use ferrum_interfaces::backend::KernelExecutor;

// Core types from ferrum-types
pub use ferrum_types::{DataType, Device, Result};

// Backend implementations
pub mod backends;
pub mod memory;

// Re-exports of concrete implementations
pub use backends::*;
pub use memory::*;

/// Lightweight handle around a tensor factory implementation.
#[derive(Clone)]
pub struct TensorFactoryHandle(pub Arc<dyn TensorFactory + Send + Sync>);

impl TensorFactoryHandle {
    /// Create a new handle from a concrete factory.
    pub fn new(factory: Arc<dyn TensorFactory + Send + Sync>) -> Self {
        Self(factory)
    }

    /// Convenience: create a handle for the default factory bound to CPU.
    pub fn default_cpu() -> Self {
        Self(default_tensor_factory())
    }

    /// Clone the underlying factory handle.
    pub fn clone_handle(&self) -> Self {
        Self(self.0.clone())
    }

    /// Borrow the inner factory reference.
    pub fn as_ref(&self) -> &(dyn TensorFactory + Send + Sync) {
        self.0.as_ref()
    }

    /// Merge additional factories into the global registry and return the base handle.
    pub fn merge_with(self, factories: Vec<(Device, Arc<dyn TensorFactory + Send + Sync>)>) -> Self {
        for (device, factory) in factories {
            backends::candle::register_tensor_factory(device, factory);
        }
        self
    }
}

impl Default for TensorFactoryHandle {
    fn default() -> Self {
        Self::default_cpu()
    }
}

impl std::ops::Deref for TensorFactoryHandle {
    type Target = dyn TensorFactory + Send + Sync;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

// Default backend factory
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Global backend registry
static BACKEND_REGISTRY: Lazy<Arc<DefaultBackendRegistry>> =
    Lazy::new(|| Arc::new(DefaultBackendRegistry::new()));

/// Get the global backend registry
pub fn global_backend_registry() -> Arc<DefaultBackendRegistry> {
    BACKEND_REGISTRY.clone()
}

/// Default backend registry implementation
#[derive(Debug)]
pub struct DefaultBackendRegistry {
    compute_backends: parking_lot::RwLock<std::collections::HashMap<String, Box<dyn ComputeBackend>>>,
}

impl DefaultBackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            compute_backends: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Register a compute backend
    pub fn register_compute_backend(
        &self,
        name: &str,
        backend: Box<dyn ComputeBackend>,
    ) -> Result<()> {
        let mut backends = self.compute_backends.write();
        backends.insert(name.to_string(), backend);
        Ok(())
    }

    /// Get compute backend by name
    pub fn get_compute_backend(&self, name: &str) -> Option<&dyn ComputeBackend> {
        // Note: This is a simplified implementation
        // In practice, we need to handle the lifetime correctly
        let backends = self.compute_backends.read();
        backends.get(name).map(|b| b.as_ref())
    }

    /// Create default backend for device
    pub async fn create_default_backend(&self, device: &Device) -> Result<Box<dyn ComputeBackend>> {
        match device {
            #[cfg(feature = "candle")]
            Device::CUDA(_) | Device::Metal => {
                Ok(Box::new(backends::candle::CandleBackend::new(*device).await?))
            }
            Device::CPU => {
                #[cfg(feature = "cpu")]
                {
                    Ok(Box::new(backends::cpu::CpuBackend::new()))
                }
                #[cfg(not(feature = "cpu"))]
                {
                    #[cfg(feature = "candle")]
                    {
                        Ok(Box::new(backends::candle::CandleBackend::new(*device).await?))
                    }
                    #[cfg(not(feature = "candle"))]
                    {
                        Err(ferrum_types::FerrumError::backend(
                            "No backend available for CPU device",
                        ))
                    }
                }
            }
        }
    }
}

impl Default for DefaultBackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

static DEFAULT_TENSOR_FACTORY: Lazy<Arc<dyn TensorFactory + Send + Sync>> = Lazy::new(|| {
    backends::candle::get_tensor_factory(&Device::CPU)
});

pub fn default_tensor_factory() -> Arc<dyn TensorFactory + Send + Sync> {
    DEFAULT_TENSOR_FACTORY.clone()
}
