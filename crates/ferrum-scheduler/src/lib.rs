//! # Ferrum Scheduler
//! 
//! Request scheduling and batching abstractions for LLM inference.
//! 
//! ## Overview
//! 
//! This module defines the core traits for implementing various scheduling strategies
//! and batching algorithms to optimize throughput and latency in LLM inference.
//! 
//! ## Design Principles
//! 
//! - **Strategy Pattern**: Multiple scheduling algorithms (FIFO, Priority, Fair, etc.)
//! - **Continuous Batching**: Dynamic request batching for optimal GPU utilization
//! - **Preemption Support**: Ability to preempt long-running requests
//! - **SLA Awareness**: Consider latency SLAs and priorities
//! - **Resource Awareness**: Schedule based on available resources

pub mod traits;
pub mod types;

// Re-exports
pub use traits::{
    Scheduler, BatchScheduler, RequestQueue, LoadBalancer,
    PreemptionManager, SlaManager, ResourceAwareScheduler
};

pub use types::{
    SchedulingPolicy, BatchingStrategy, PreemptionPolicy,
    SlaConfig, QueueConfig, SchedulerMetrics, BatchRequest
};
