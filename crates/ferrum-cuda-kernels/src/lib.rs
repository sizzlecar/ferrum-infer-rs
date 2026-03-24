//! Ferrum custom CUDA kernels for high-performance inference.
//!
//! This crate provides fused CUDA kernels that bypass candle's per-op dispatch,
//! reducing kernel launch overhead and memory bandwidth usage.
//!
//! Kernels are compiled from CUDA source via nvrtc at first use, then cached.

#[cfg(feature = "cuda")]
mod fused_add_rms_norm;
#[cfg(feature = "cuda")]
pub use fused_add_rms_norm::fused_add_rms_norm;

#[cfg(feature = "cuda")]
mod fused_silu_mul;
#[cfg(feature = "cuda")]
pub use fused_silu_mul::fused_silu_mul;

#[cfg(feature = "cuda")]
mod kernel_loader;
#[cfg(feature = "cuda")]
pub use kernel_loader::KernelStore;
