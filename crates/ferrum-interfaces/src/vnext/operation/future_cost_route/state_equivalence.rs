use super::*;
use crate::vnext::ResourcePlanningBudget;

impl ExecutionCostRouteState {
    /// Whether two modeled successful waves have identical future behavior
    /// under the same captured route. This compares all state consumed by a
    /// subsequent wave, including allocator layout and weak mask-source
    /// identity. It grants no authority, keeps no source alive, and cannot join
    /// different captures.
    /// The current waves' costs and host forecasts still require separate
    /// evaluation even when their successors compare equal.
    pub fn same_future_state(
        &self,
        other: &Self,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<bool, ExecutionCostRouteUnknown> {
        core::poll(budget)?;
        if !Arc::ptr_eq(&self.fence, &other.fence)
            || self.projected_graph_state != other.projected_graph_state
            || !same_values(&self.frontiers, &other.frontiers, budget)?
            || !same_values(&self.initialized, &other.initialized, budget)?
        {
            return Ok(false);
        }
        // last_token_mask_uploads describes the completed wave's transfers.
        // append_complete_eager_cost_route resets it before evaluating the next
        // wave, whose uploads depend on token_masks instead. The caller keeps
        // both current costs even when their persistent successors coincide.
        match (&self.token_masks, &other.token_masks) {
            (Some(a), Some(b)) if a.same_future_state(b, budget)? => {}
            (None, None) => {}
            _ => return Ok(false),
        }
        let equal = self
            .resources
            .same_future_state(&other.resources, budget)
            .map_err(ExecutionCostRouteUnknown::Resource)?;
        core::poll(budget)?;
        Ok(equal)
    }
}

pub(super) fn same_values<T: PartialEq>(
    left: &[T],
    right: &[T],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<bool, ExecutionCostRouteUnknown> {
    core::poll(budget)?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (a, b) in left.iter().zip(right) {
        core::poll(budget)?;
        if a != b {
            return Ok(false);
        }
    }
    core::poll(budget)?;
    Ok(true)
}
