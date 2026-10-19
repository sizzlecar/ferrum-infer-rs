//! Unified Backend trait for CUDA, Metal, and CPU compute.
//!
//! Each backend implements the same set of transformer-layer primitives
//! (GEMM, norms, RoPE, attention, activations). `layer_forward()` and
//! `ModelRunner` are generic over `Backend`, so one forward path serves
//! all hardware targets.

mod paged_attention;
pub use paged_attention::PagedAttentionLaunchGeometry;

mod traits;
pub use traits::*;

mod completion;
pub use completion::BackendCompletion;

mod types;
pub use types::*;

mod capabilities;
mod device_attributes;
#[cfg(feature = "cuda")]
pub use device_attributes::probe_cuda_device_attributes;
pub use device_attributes::CudaDeviceAttributes;
mod memory;
pub use memory::probe_device_memory;
#[cfg(any(feature = "cuda", test))]
mod native_status;

#[cfg(any(feature = "cuda", test))]
pub(crate) mod enqueue_timing;

#[cfg(any(feature = "cuda", test))]
mod reusable_execution;

#[cfg(any(feature = "cuda", test))]
mod program_binding_rows;

#[cfg(any(feature = "cuda", test))]
mod program_binding_upload;

mod kv_layer;
pub use kv_layer::*;

pub mod dtype;
pub use dtype::{Dtype, HostDtype};

pub mod buffer;
pub use buffer::CpuBuf;
#[cfg(feature = "cuda")]
pub use buffer::CudaBuf;

pub mod cpu;

pub mod reference;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod timer;
