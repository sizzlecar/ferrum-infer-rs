//! Explicit, finite open-arrival capacity experiments. This module plans work
//! and evaluates raw observations; it neither runs a server nor certifies an
//! unmeasured rate or an asymptotic steady-state capacity.

mod acquisition;
mod evaluate;
mod plan;
mod server_queue;
mod session;
mod types;

pub use evaluate::evaluate_run;
pub use plan::{CapacitySearch, SearchProgress};
pub use session::{
    capacity_evidence_sha256, AuthorizedCapacitySessionBlock, CapacityProcessBirth,
    CapacityServerSession, CapacitySessionBlock, CapacityWarmupAcquisition,
};
pub use types::*;

#[cfg(test)]
mod tests;
