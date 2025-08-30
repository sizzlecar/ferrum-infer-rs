//! # Ferrum Runtime
//! 
//! Execution runtime and GPU operations abstractions for LLM inference.
//! 
//! ## Overview
//! 
//! This module defines the core traits for runtime execution, GPU operations,
//! and hardware abstraction in the Ferrum inference framework.
//! 
//! ## Design Principles
//! 
//! - **Hardware Agnostic**: Support for CUDA, ROCm, Metal, and CPU backends
//! - **Async Operations**: Non-blocking GPU operations with proper synchronization
//! - **Memory Management**: Efficient GPU memory allocation and transfer
//! - **Stream Management**: Support for multiple GPU streams and queues
//! - **Kernel Execution**: Abstract interface for custom GPU kernels

pub mod traits;
pub mod types;

// Re-exports
pub use traits::{
    Runtime, DeviceManager, MemoryManager, StreamManager,
    KernelExecutor, TensorOps, ComputeBackend
};

pub use types::{
    DeviceInfo, MemoryInfo, StreamHandle, KernelHandle,
    ComputeCapability, RuntimeConfig, ExecutionContext,
    MemoryTransfer, SynchronizationPoint
};
