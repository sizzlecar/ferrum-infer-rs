//! Optimized Kernel Implementations
//!
//! This module provides high-performance kernel definitions and registry for
//! critical inference operations. Features include:
//!
//! - Flash Attention: Memory-efficient attention computation
//! - Fused Operations: RoPE + Attention, QKV projection + split
//! - Backend Abstraction: CPU, Metal, CUDA implementations
//! - Kernel Registry: Runtime discovery and selection of optimal kernels

pub mod attention;
pub mod cuda_kernel_ops;
pub mod fused;
pub mod ops;
pub mod registry;

pub use attention::{
    create_attention_info, AttentionConfig, AttentionKernel, AttentionType, FlashAttentionInfo,
    PagedAttentionInfo, StandardAttentionInfo,
};
#[cfg(feature = "cuda")]
pub use cuda_kernel_ops::flash_attn_varlen;
pub use cuda_kernel_ops::CudaKernelOps;
pub use fused::{
    CpuFusedOpsInfo, FusedOpType, FusedOps, FusedOpsConfig, FusedRopeAttention,
    FusedRopeAttentionConfig, RopeCache,
};
pub use ops::{
    create_attention_op, AttentionOp, AttentionOutput, CpuAttentionOp, DecodeAttentionInput,
    PrefillAttentionInput,
};
pub use registry::{global_kernel_registry, KernelInfo, KernelRegistry, PerformanceHint};
