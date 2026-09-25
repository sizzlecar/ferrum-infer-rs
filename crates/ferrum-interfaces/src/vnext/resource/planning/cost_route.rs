use super::*;
use crate::vnext::{
    ExecutionCostRouteAvailability, ExecutionCostRouteUnknown, ExecutionCostRouteView,
};

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Immutable layout only; borrowing it grants no slot, backing or submit
    /// authority and cannot pin an invocation's retained physical storage.
    pub(crate) fn planning_program_binding_layout(
        &self,
        bucket: &ReusableExecutionBucketId,
    ) -> Option<&ProgramBindingLayout> {
        self.dynamic_pools
            .program_binding_layout(bucket)
            .map(Arc::as_ref)
    }

    /// Captures core numerical evidence while the caller holds its model
    /// registry/operation read guards. No view retains the supplied sessions.
    pub fn execution_cost_route_view(
        self: &Arc<Self>,
        sessions: &[&SequenceSession<R>],
        frontiers: &[u64],
        lane: &ExecutionLane<R>,
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteView> {
        use ExecutionCostRouteAvailability as A;
        use ExecutionCostRouteUnknown as U;
        if sessions.len() != frontiers.len() {
            return A::Unknown(U::InvalidInput);
        }
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc()) {
            return A::Unknown(U::StaleView);
        }
        let (resources, graph_stream_state) =
            match self.resource_planning_view_with_graph_on_lane(sessions, lane, limits, budget) {
                Ok(value) => value,
                Err(reason) => return A::Unknown(U::Resource(reason)),
            };
        if resources
            .participants()
            .iter()
            .zip(frontiers)
            .any(|(participant, frontier)| {
                *frontier > participant.maximum_tokens()
                    || participant.pending_zero_commands().is_none()
            })
        {
            return A::Unknown(U::InitializationState);
        }
        let Some(readback_available_bytes) = lane.cost_readback_available_bytes() else {
            return A::Unknown(U::ReadbackState);
        };
        if !budget.has_budget() {
            return A::Unknown(U::BudgetExhausted);
        }
        A::Known(ExecutionCostRouteView {
            fence: Arc::new(()),
            resources,
            structured_capture: false,
            initial_frontiers: frontiers.to_vec(),
            readback_available_bytes,
            lane_id: lane.id(),
            token_masks: None,
            graph_stream_state,
        })
    }
}
