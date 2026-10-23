//! Backend-neutral execution, resource, operation and numerical contracts.
//!
//! Runtime authority and evidence are validated independently of product
//! request orchestration and concrete device backends.

#[cfg(any(test, feature = "test-support"))]
extern crate self as ferrum_runtime;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

mod admission;
mod completion;
mod device;
mod error;
mod event;
mod execution;
mod execution_facts;
mod host_memory;
mod identity;
mod model;
mod numerical;
mod numerical_resolution;
mod operation;
mod oracle;
mod resolved;
mod resource;
mod standard_operations;
mod static_initialization;
mod weight_source;

pub use admission::*;
pub use completion::*;
pub use device::*;
pub use error::*;
pub use event::*;
pub use execution::*;
pub use execution_facts::*;
pub use host_memory::*;
pub use identity::*;
pub use model::*;
pub use numerical::*;
pub use numerical_resolution::*;
pub use operation::*;
pub use oracle::*;
pub use resolved::*;
pub use resource::*;
pub use standard_operations::*;
pub use static_initialization::*;
pub use weight_source::*;
