//! Unified Backend trait for CUDA, Metal, and CPU compute.
//!
//! Each backend implements the same set of transformer-layer primitives
//! (GEMM, norms, RoPE, attention, activations). `layer_forward()` and
//! `ModelRunner` are generic over `Backend`, so one forward path serves
//! all hardware targets.

mod traits;
pub use traits::*;

mod kv_layer;
pub use kv_layer::*;

pub mod dtype;
pub use dtype::Dtype;

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
