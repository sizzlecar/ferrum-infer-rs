//! Planning requests for content-dependent host work. Only an executor that
//! proves every represented subset has the same device route can answer Known.
use super::ExecutionCostRouteProjection;
use crate::execution_cost::{HostContentForecastV2, HostPendingConstraintV2};
use crate::model_executor::LogitsReturnPolicy;

#[derive(Debug, Clone, Copy)]
pub struct FutureHostPendingRowV2<'a> {
    /// Position in this wave's physical order, not the registry participant index.
    pub physical_position: u32,
    /// The installed normal policy after generated history exists. The provider
    /// compares this policy with FullLogits through its real product selectors.
    pub clean_policy: &'a LogitsReturnPolicy,
}

#[derive(Debug, Clone, Copy)]
pub struct FutureHostPendingQueryV2<'a> {
    /// Sorted unique decode positions whose future text is unresolved.
    pub eligible_rows: &'a [FutureHostPendingRowV2<'a>],
    pub constraint: HostPendingConstraintV2,
}

#[derive(Debug, Clone)]
pub struct ExecutionCostRouteForecastV2 {
    /// The original once-advanced numeric state and its selected recipe.
    pub projection: ExecutionCostRouteProjection,
    /// Bound to the original attached recipe; never an actual host settlement.
    pub host_content: HostContentForecastV2,
}
