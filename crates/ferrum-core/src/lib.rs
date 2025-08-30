//! # Ferrum Core
//!
//! Core trait definitions and fundamental types for the Ferrum inference framework.
//! This crate provides the foundation for all other modules.

pub mod error;
pub mod traits;
pub mod types;

// Re-exports
pub use error::{Error, Result};
pub use traits::*;
pub use types::*;
