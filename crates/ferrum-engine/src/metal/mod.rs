//! Metal backend for Apple GPU acceleration
//!
//! This module provides a hybrid approach:
//! - Uses Metal shaders for performance-critical operations (attention, RoPE, sampling)
//! - Uses MPSGraph for linear operations (matmul, embedding)
//! - Integrates with existing Candle infrastructure for model loading and tokenization

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod backend;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod benchmark;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod compute_pipeline;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod context;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod error;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_executor;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_llama;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_model;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_qwen2;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod quantization;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod sampling_ops;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use backend::MetalBackend;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use benchmark::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use compute_pipeline::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use context::MetalContext;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use error::MetalError;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use metal_executor::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use metal_llama::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use metal_model::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use metal_qwen2::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use quantization::*;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use sampling_ops::*;

// Stub implementations for non-Apple platforms
#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
pub struct MetalBackend;

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
impl MetalBackend {
    pub fn new(_device: ferrum_types::Device) -> ferrum_types::Result<Self> {
        Err(ferrum_types::FerrumError::internal(
            "Metal backend not available on this platform",
        ))
    }
}
