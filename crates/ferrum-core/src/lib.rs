//! # Ferrum Core
//! 
//! Core trait definitions and fundamental types for the Ferrum inference framework.
//! This crate provides the foundation for all other modules.

pub mod error;
pub mod types;
pub mod traits;

// Re-exports
pub use error::{Error, Result};
pub use types::*;
pub use traits::*;
