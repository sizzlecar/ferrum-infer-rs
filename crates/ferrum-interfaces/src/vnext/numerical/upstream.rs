//! Declarations for an explicit upstream projection experiment.
//!
//! No provider, family profile or Auto candidate is registered here. A static
//! plan is not evidence of execution or numerical-domain qualification. The
//! provider must rebuild native geometry, retain live resource authorization,
//! and implement the declared V1 domain check or V2 device marker protocol.

mod arithmetic;
mod plan;
mod scratch;
pub use arithmetic::*;
pub use plan::*;
pub use scratch::*;

#[cfg(test)]
mod tests;
