use super::*;
use crate::vnext::{
    ExecutionCostRouteAvailability, ExecutionCostRouteUnknown, ExecutionCostRouteView,
};

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Completion evidence is captured first under this fresh lane bracket.
    /// Optional work may fail without erasing that successful resource view.
    pub fn completion_planning_capture<T>(
        self: &Arc<Self>,
        sessions: &[&SequenceSession<R>],
        lane: &ExecutionLane<R>,
        limits: ResourcePlanningLimits,
        forecast_limits: ResourcePlanningLimits,
        observer: &mut dyn crate::model_executor::ExecutorPlanningCapture,
        prepare_forecast: &mut dyn FnMut(
            &mut dyn ResourcePlanningBudget,
        )
            -> Result<(Vec<u64>, T), ExecutionCostRouteUnknown>,
        finalize_forecast: &mut dyn FnMut(
            ExecutionCostRouteView,
            T,
            &mut dyn ResourcePlanningBudget,
        ) -> Result<
            ExecutionCostRouteView,
            ExecutionCostRouteUnknown,
        >,
    ) -> ResourcePlanningAvailability<crate::model_executor::ExecutorCompletionPlanningCapture>
    {
        use crate::model_executor::ExecutorCompletionPlanningCapture;
        use ExecutionCostRouteAvailability as A;
        use ExecutionCostRouteUnknown as U;
        use ResourcePlanningAvailability as R;
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc()) {
            return R::Unknown(ResourcePlanningUnknown::StaleIdentity);
        }
        let result = lane.try_with_completion_planning_lane(
            |epoch, graph_stream_state, readback, catalog| {
                let resources = self.capture_resource_planning_view(
                    sessions,
                    Some((lane.id(), epoch)),
                    limits,
                    observer.resource_budget(),
                )?;
                // Once only. The callback finalizes numeric completion work and
                // fixes its reserve before any optional graph/mask work begins.
                let forecast = if observer.completion_ready(&resources) {
                    let projected = (|| {
                        let budget = observer.forecast_budget();
                        if !budget.has_budget() {
                            return Err(U::BudgetExhausted);
                        }
                        let resources = resources
                            .with_projection_limits(forecast_limits)
                            .map_err(U::Resource)?;
                        let (frontiers, metadata) = prepare_forecast(budget)?;
                        if frontiers.len() != resources.participants().len() {
                            return Err(U::InvalidInput);
                        }
                        let graph_catalog =
                            catalog(forecast_limits, budget).map_err(U::Resource)?;
                        if resources.participants().iter().zip(&frontiers).any(
                            |(participant, frontier)| {
                                *frontier > participant.maximum_tokens()
                                    || participant.pending_zero_commands().is_none()
                            },
                        ) {
                            return Err(U::InitializationState);
                        }
                        let readback_available_bytes = readback.ok_or(U::ReadbackState)?;
                        if !budget.has_budget() {
                            return Err(U::BudgetExhausted);
                        }
                        let view = ExecutionCostRouteView {
                            fence: Arc::new(()),
                            resources,
                            structured_capture: false,
                            initial_frontiers: frontiers,
                            readback_available_bytes,
                            lane_id: lane.id(),
                            token_masks: None,
                            graph_stream_state,
                            graph_catalog,
                        };
                        let view = finalize_forecast(view, metadata, budget)?;
                        if !budget.has_budget() {
                            return Err(U::BudgetExhausted);
                        }
                        Ok(view)
                    })();
                    // A single result exit pairs every accepted optional phase,
                    // including metadata, graph, budget and policy failures.
                    observer.forecast_finished();
                    Some(match projected {
                        Ok(view) => A::Known(view),
                        Err(reason) => A::Unknown(reason),
                    })
                } else {
                    None
                };
                Ok(ExecutorCompletionPlanningCapture {
                    resources,
                    forecast,
                })
            },
        );
        match result {
            Ok(capture) => R::Known(capture),
            Err(reason) => R::Unknown(reason),
        }
    }

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
        let (resources, graph_stream_state, graph_catalog) =
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
            graph_catalog,
        })
    }
}
