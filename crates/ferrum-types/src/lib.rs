//! Core type definitions for the Ferrum inference framework
//!
//! This crate contains all the fundamental types, IDs, configurations, and error definitions
//! that are shared across the entire Ferrum ecosystem. It is designed to be lightweight
//! and dependency-free to avoid circular dependencies.

pub mod auto_config;
pub mod backend_context;
pub mod benchmark_correlation;
pub mod capacity;
pub mod config;
pub mod devices;
pub mod errors;
pub mod execution_policy;
pub mod harmony;
pub mod ids;
pub mod image_batching;
pub mod llama_execution;
pub mod metrics;
pub mod models;
pub mod native_operator;
pub mod numerical;
pub mod observability_profile;
pub mod output;
pub mod process_memory;
pub mod reasoning;
pub mod reasoning_controls;
pub mod request_clock;
pub mod requests;
pub mod resource_change;
pub mod resource_trace;
pub mod runtime_config;
pub mod sampling;
pub mod slo_config;
pub mod startup_memory;

// Re-export commonly used types
pub use auto_config::*;
pub use backend_context::*;
pub use benchmark_correlation::*;
pub use capacity::AllocatorCapacityDomain;
pub use config::*;
pub use devices::*;
pub use errors::*;
pub use execution_policy::*;
pub use harmony::*;
pub use ids::*;
pub use image_batching::*;
pub use llama_execution::*;
pub use metrics::*;
pub use models::*;
pub use native_operator::*;
pub use numerical::*;
pub use observability_profile::*;
pub use output::*;
pub use process_memory::*;
pub use reasoning::*;
pub use reasoning_controls::*;
pub use request_clock::*;
pub use requests::*;
pub use resource_change::{ResourceChangeNotifier, ResourceChangeSubscription};
pub use resource_trace::*;
pub use runtime_config::*;
pub use sampling::*;
pub use slo_config::*;
pub use startup_memory::*;

/// Result type used throughout Ferrum
pub type Result<T> = std::result::Result<T, FerrumError>;

/// Block identifier type  
pub type BlockId = u32;
