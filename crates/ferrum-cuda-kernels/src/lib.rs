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

#[cfg(feature = "cuda")]
mod rms_norm;
#[cfg(feature = "cuda")]
pub use rms_norm::rms_norm;

#[cfg(feature = "cuda")]
mod rope;
#[cfg(feature = "cuda")]
pub use rope::rope;

#[cfg(feature = "cuda")]
mod decode_attention;
#[cfg(feature = "cuda")]
pub use decode_attention::decode_attention;

#[cfg(feature = "cuda")]
mod residual_add;
#[cfg(feature = "cuda")]
pub use residual_add::residual_add;

#[cfg(feature = "cuda")]
pub mod cublas;

#[cfg(feature = "cuda")]
pub mod decode_buffers;

#[cfg(feature = "cuda")]
pub mod weight_store;

#[cfg(feature = "cuda")]
pub mod cuda_graph;

#[cfg(feature = "cuda")]
pub mod quant;

#[cfg(feature = "cuda")]
pub mod marlin;

#[cfg(feature = "cuda")]
pub mod gpu_paged_kv;

#[cfg(feature = "cuda")]
pub mod cuda_decode;
