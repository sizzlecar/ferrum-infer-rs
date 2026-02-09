//! Multi-GPU Parallelism Module
//!
//! This module provides support for distributing model inference across
//! multiple GPUs using various parallelism strategies:
//!
//! - Tensor Parallelism: Split tensor operations across GPUs
//! - Pipeline Parallelism: Split model layers across GPUs
//! - Data Parallelism: Process different batches on different GPUs
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    DeviceManager                         │
//! │  - Device discovery and capability detection             │
//! │  - Resource allocation and monitoring                    │
//! └─────────────────────────────────────────────────────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          ▼                   ▼                   ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │ TensorParallel  │ │ PipelineParallel│ │  DataParallel   │
//! │   Executor      │ │   Executor      │ │   Executor      │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```

pub mod config;
pub mod device;
pub mod executor;
pub mod tensor_parallel;

pub use config::{LayerDistribution, LayerRange, ParallelConfig, ParallelismType};
pub use device::{global_device_manager, DeviceCapability, DeviceInfo, DeviceManager};
pub use executor::{ParallelExecutor, ParallelExecutorFactory, ParallelStrategySelector};
pub use tensor_parallel::{
    LayerParallelType, TensorParallelConfig, TensorParallelGroup, TransformerParallelMapping,
    WeightShard,
};
