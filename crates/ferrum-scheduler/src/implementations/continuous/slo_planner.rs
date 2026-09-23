//! Pure SLO planning contracts, recovery obligations and checked time mappings.
//! These read-only values do not reserve resources or grant execution authority.

mod clock;
mod cost_shape;
mod obligations;
mod types;

pub use clock::{
    AnchoredPlanningCostModel, PlanningCostClockAnchor, PlanningTimeError, PlanningTimeOrigin,
};
pub use cost_shape::{actual_cost_shape, canonical_cost_shape};
pub use obligations::{
    historical_violation, ForwardObligation, PlanningObligationSet, RecoveryServiceDebt,
    RequestObligation,
};
pub use types::*;
