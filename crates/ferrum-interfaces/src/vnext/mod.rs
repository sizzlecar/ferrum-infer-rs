//! Isolated vNext contracts for the runtime redesign.
//!
//! These types are intentionally not wired into legacy product or execution
//! paths. They define GPU-free, fail-closed boundaries for later migration.

mod admission;
mod completion;
mod device;
mod error;
mod event;
mod execution;
mod identity;
mod model;
mod operation;
mod oracle;
mod resolved;
mod resource;

pub use admission::*;
pub use completion::*;
pub use device::*;
pub use error::*;
pub use event::*;
pub use execution::*;
pub use identity::*;
pub use model::*;
pub use operation::*;
pub use oracle::*;
pub use resolved::*;
pub use resource::*;
