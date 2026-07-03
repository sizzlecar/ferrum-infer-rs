//! Core type definitions for the Ferrum inference framework
//!
//! This crate contains all the fundamental types, IDs, configurations, and error definitions
//! that are shared across the entire Ferrum ecosystem. It is designed to be lightweight
//! and dependency-free to avoid circular dependencies.

pub mod auto_config;
pub mod config;
pub mod devices;
pub mod errors;
pub mod ids;
pub mod metrics;
pub mod models;
pub mod native_operator;
pub mod observability_profile;
pub mod process_memory;
pub mod requests;
pub mod resource_trace;
pub mod runtime_config;
pub mod sampling;

// Re-export commonly used types
pub use auto_config::*;
pub use config::*;
pub use devices::*;
pub use errors::*;
pub use ids::*;
pub use metrics::*;
pub use models::*;
pub use native_operator::*;
pub use observability_profile::*;
pub use process_memory::*;
pub use requests::*;
pub use resource_trace::*;
pub use runtime_config::*;
pub use sampling::*;

/// Result type used throughout Ferrum
pub type Result<T> = std::result::Result<T, FerrumError>;

/// Block identifier type  
pub type BlockId = u32;
