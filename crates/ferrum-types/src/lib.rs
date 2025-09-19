//! Core type definitions for the Ferrum inference framework
//!
//! This crate contains all the fundamental types, IDs, configurations, and error definitions
//! that are shared across the entire Ferrum ecosystem. It is designed to be lightweight
//! and dependency-free to avoid circular dependencies.

pub mod ids;
pub mod config;
pub mod requests;
pub mod devices;
pub mod errors;
pub mod sampling;
pub mod models;
pub mod metrics;

// Re-export commonly used types
pub use ids::*;
pub use config::*;
pub use requests::*;
pub use devices::*;
pub use errors::*;
pub use sampling::*;
pub use models::*;
pub use metrics::*;

/// Result type used throughout Ferrum
pub type Result<T> = std::result::Result<T, FerrumError>;

/// Token identifier type
pub type TokenId = u32;

/// Block identifier type  
pub type BlockId = u32;
