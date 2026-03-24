//! Ferrum custom CUDA kernels for high-performance inference.
//!
//! This crate provides fused CUDA kernels that bypass candle's per-op dispatch,
//! reducing kernel launch overhead and memory bandwidth usage.
//!
//! On CUDA builds, kernels are compiled to PTX during `cargo build` and loaded
//! on demand at runtime.

#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[cfg(feature = "cuda")]
mod fused_add_rms_norm;
#[cfg(feature = "cuda")]
pub use fused_add_rms_norm::fused_add_rms_norm;

#[cfg(feature = "cuda")]
mod fused_silu_mul;
#[cfg(feature = "cuda")]
pub use fused_silu_mul::fused_silu_mul;
