//! Pure, bounded rolling SLO planning. Nothing here allocates KV, reserves
//! output, changes a queue, or grants execution authority.
//!
//! A witness covers a declared finite horizon under an empirical cost model.
//! It is not an admission promise for an entire request or a hard-time bound.

mod candidates;
mod clock;
mod cost_shape;
mod obligations;
mod output;
mod resources;
mod search;
mod shape;
mod simulation;
mod types;
mod validation;

pub use clock::{
    AnchoredPlanningCostModel, PlanningCostClockAnchor, PlanningTimeError, PlanningTimeOrigin,
};
pub use cost_shape::{actual_cost_shape, canonical_cost_shape};
pub use obligations::{
    historical_violation, ForwardObligation, PlanningObligationSet, RecoveryServiceDebt,
    RequestObligation,
};
pub use search::BoundedSloPlanner;
pub use types::*;

#[cfg(test)]
mod tests;
