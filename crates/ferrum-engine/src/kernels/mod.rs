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
pub mod fused;
pub mod registry;

pub use attention::{
    create_attention_info, AttentionConfig, AttentionKernel, AttentionType, FlashAttentionInfo,
    PagedAttentionInfo, StandardAttentionInfo,
};
pub use fused::{
    CpuFusedOpsInfo, FusedOpType, FusedOps, FusedOpsConfig, FusedRopeAttention,
    FusedRopeAttentionConfig, RopeCache,
};
pub use registry::{global_kernel_registry, KernelInfo, KernelRegistry, PerformanceHint};
