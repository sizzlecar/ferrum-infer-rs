//! # Ferrum Scheduler
//!
//! Request scheduling and batching implementations for LLM inference.
//!
//! ## Overview
//!
//! This crate provides concrete implementations of the scheduler interfaces defined
//! in ferrum-interfaces, including:
//!
//! - FIFO scheduler for simple first-come-first-served scheduling
//! - Priority scheduler for priority-based request handling
//! - Resource-aware scheduler for optimal resource utilization
//! - SLA-aware scheduler for meeting service level agreements
//!
//! ## Design Principles
//!
//! - **Strategy Pattern**: Multiple scheduling algorithms (FIFO, Priority, Fair, etc.)
//! - **Continuous Batching**: Dynamic request batching for optimal GPU utilization
//! - **Preemption Support**: Ability to preempt long-running requests
//! - **SLA Awareness**: Consider latency SLAs and priorities
//! - **Resource Awareness**: Schedule based on available resources

pub mod implementations;
pub mod metrics;
pub mod queue;

// Re-exports of interfaces from ferrum-interfaces
pub use ferrum_interfaces::{
    scheduler::{
        BatchResourceRequirements, PreemptionResult, PreemptionState, ResourceConstraints,
        ScheduledRequest,
    },
    BatchHint, BatchPlan, SchedulerInterface as Scheduler,
};

// Re-export types from ferrum-types
pub use ferrum_types::{SchedulerConfig, SchedulerStats};

pub use ferrum_types::{BatchId, InferenceRequest, InferenceResponse, Priority, RequestId, Result};

// Re-exports of implementations
pub use implementations::*;
