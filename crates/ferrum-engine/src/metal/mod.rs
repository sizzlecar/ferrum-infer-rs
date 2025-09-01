//! Metal backend for Apple GPU acceleration
//!
//! This module provides a hybrid approach:
//! - Uses Metal shaders for performance-critical operations (attention, RoPE, sampling)
//! - Uses MPSGraph for linear operations (matmul, embedding)
//! - Integrates with existing Candle infrastructure for model loading and tokenization

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod backend;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod context;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod error;
// mod kernels; // TODO: Re-enable when Metal kernels are ready

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use backend::MetalBackend;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use context::MetalContext;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use error::MetalError;

// Stub implementations for non-Apple platforms
#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
pub struct MetalBackend;

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
impl MetalBackend {
    pub fn new(_device: ferrum_core::Device) -> ferrum_core::Result<Self> {
        Err(ferrum_core::Error::internal("Metal backend not available on this platform"))
    }
}