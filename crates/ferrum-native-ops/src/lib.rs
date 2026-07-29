//! Native operator ABI and manifest resolver.
//!
//! This crate intentionally does not contain third-party CUDA/C++ operator
//! sources. It only defines the ABI surface and fail-closed artifact resolver
//! used by later release gates and runtime integration work.

pub mod abi;
pub mod artifact_set;
pub mod build_cache;
pub mod build_unit;
pub mod manifest;
pub mod registry;
pub mod resolver;

pub use abi::*;
pub use artifact_set::*;
pub use build_cache::*;
pub use build_unit::*;
pub use manifest::*;
pub use registry::*;
pub use resolver::*;
