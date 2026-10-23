//! # Ferrum Scheduler
//!
//! Request scheduling and batching implementations for LLM inference.
//!
//! ## Overview
//!
//! This crate implements continuous batching using the scheduler contracts defined
//! in ferrum-interfaces:
//!
//! - Iteration-level prefill and decode scheduling
//! - Capacity-aware admission, preemption, and recovery
//! - Request priorities within the pressure-management policy
//! - Optional feedback-driven latency targets
//!
//! ## Design Principles
//!
//! - **Continuous Batching**: Dynamic request batching for optimal GPU utilization
//! - **Preemption Support**: Ability to preempt long-running requests
//! - **SLA Awareness**: Consider latency SLAs and priorities
//! - **Resource Awareness**: Schedule based on available resources

pub mod cost_prediction;
pub mod implementations;
pub mod joint_policy;
pub mod metrics;
pub mod vnext;

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
pub use cost_prediction::{BatchSelection, TimePlanningRound};
pub use implementations::*;
