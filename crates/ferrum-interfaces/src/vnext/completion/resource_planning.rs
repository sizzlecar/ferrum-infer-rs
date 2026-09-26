//! A nonblocking quiescent-lane bracket for numerical resource capture.
use super::*;
use crate::vnext::{
    DeviceCostGraphStreamState, ResourcePlanningReadStage, ResourcePlanningUnknown,
};
impl<R: DeviceRuntime> ExecutionLane<R> {
    /// Inspect this lane's actual graph configuration without configuring it.
    /// Unknown backend evidence stays `None`; inspection still requires the
    /// same stable, quiescent lane as other reusable-execution queries.
    pub fn cost_graph_stream_state(
        &self,
    ) -> Result<Option<DeviceCostGraphStreamState>, VNextError> {
        self.with_quiescent_stream("inspect graph stream state", |runtime, stream| {
            Ok(runtime.cost_graph_stream_state(stream))
        })
    }

    pub(crate) fn try_with_resource_planning_lane<T>(
        &self,
        capture: impl FnOnce(
            u64,
            Option<DeviceCostGraphStreamState>,
        ) -> Result<T, ResourcePlanningUnknown>,
    ) -> Result<T, ResourcePlanningUnknown> {
        let state = self.state.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => {
                ResourcePlanningUnknown::ReadUnavailable(ResourcePlanningReadStage::ExecutionLane)
            }
            std::sync::TryLockError::Poisoned(_) => ResourcePlanningUnknown::BusyOrUnavailable,
        })?;
        if state.in_flight != 0
            || state.fail_closed
            || self.fail_closed.load(Ordering::Acquire)
            || !self.current_descriptor_matches_snapshot()
            || self.runtime.stream_state(&state.stream) != StreamState::Ready
        {
            return Err(ResourcePlanningUnknown::BusyOrUnavailable);
        }
        capture(
            self.reusable_execution_epoch(),
            self.runtime.cost_graph_stream_state(&state.stream),
        )
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    /// Extends the same quiescent bracket to the actual stream catalog. Its
    /// epoch is supplied by core, never manufactured by a backend producer.
    pub(crate) fn try_with_cost_planning_lane<T>(
        &self,
        limits: crate::vnext::ResourcePlanningLimits,
        budget: &mut dyn crate::vnext::ResourcePlanningBudget,
        capture: impl FnOnce(
            u64,
            Option<DeviceCostGraphStreamState>,
            Option<crate::vnext::DeviceCostGraphCatalog>,
            &mut dyn crate::vnext::ResourcePlanningBudget,
        ) -> Result<T, ResourcePlanningUnknown>,
    ) -> Result<T, ResourcePlanningUnknown> {
        use crate::vnext::DeviceCostGraphCatalogLimits;
        use ResourcePlanningUnknown as U;
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        if !limits.is_valid() {
            return Err(U::InvalidInput);
        }
        // Reuse existing capture bounds: descriptors cap program rows; extents
        // cap total topology nodes and total logical command rows separately.
        let graph_limits = DeviceCostGraphCatalogLimits::new(
            limits.maximum_descriptors,
            limits.maximum_free_extents,
            limits.maximum_free_extents,
        )
        .map_err(|_| U::InvalidInput)?;
        let state = self.state.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => {
                U::ReadUnavailable(ResourcePlanningReadStage::ExecutionLane)
            }
            std::sync::TryLockError::Poisoned(_) => U::BusyOrUnavailable,
        })?;
        if state.in_flight != 0
            || state.fail_closed
            || self.fail_closed.load(Ordering::Acquire)
            || !self.current_descriptor_matches_snapshot()
            || self.runtime.stream_state(&state.stream) != StreamState::Ready
        {
            return Err(U::BusyOrUnavailable);
        }
        let graph = self.runtime.cost_graph_stream_state(&state.stream);
        let mut exhausted = false;
        let inventory =
            self.runtime
                .cost_reusable_graph_catalog(&state.stream, graph_limits, &mut || {
                    if !budget.has_budget() {
                        exhausted = true;
                        Err(VNextError::InvalidExecutionPlan {
                            reason: "graph catalog capture budget exhausted".to_owned(),
                        })
                    } else {
                        Ok(())
                    }
                });
        if exhausted || !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        let catalog = inventory.transpose().map_err(|_| U::ReusableExecution)?;
        if let Some(catalog) = &catalog {
            if !catalog.fits(graph_limits) {
                return Err(U::LimitExceeded);
            }
            if graph != Some(catalog.stream_state()) {
                return Err(U::StaleIdentity);
            }
            for program in catalog.programs() {
                if !budget.has_budget() {
                    return Err(U::BudgetExhausted);
                }
                if program.program().program_id().lane_id() != self.id() {
                    return Err(U::StaleIdentity);
                }
            }
        }
        capture(self.reusable_execution_epoch(), graph, catalog, budget)
    }
}
