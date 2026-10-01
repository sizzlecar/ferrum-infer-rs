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
    /// One nonblocking lane bracket. The caller captures completion resources
    /// before requesting optional catalog work through the bounded callback.
    pub(crate) fn try_with_completion_planning_lane<T>(
        &self,
        capture: impl FnOnce(
            u64,
            Option<DeviceCostGraphStreamState>,
            Option<u64>,
            &mut dyn FnMut(
                crate::vnext::ResourcePlanningLimits,
                &mut dyn crate::vnext::ResourcePlanningBudget,
            ) -> Result<
                Option<crate::vnext::DeviceCostGraphCatalogSnapshot>,
                ResourcePlanningUnknown,
            >,
        ) -> Result<T, ResourcePlanningUnknown>,
    ) -> Result<T, ResourcePlanningUnknown> {
        use ResourcePlanningUnknown as U;
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
        capture(
            self.reusable_execution_epoch(),
            graph,
            self.readback_staging.available_for_cost_planning(),
            &mut |limits, budget| self.capture_planning_catalog(&state, graph, limits, budget),
        )
    }

    pub(crate) fn try_with_cost_planning_lane<T>(
        &self,
        limits: crate::vnext::ResourcePlanningLimits,
        budget: &mut dyn crate::vnext::ResourcePlanningBudget,
        capture: impl FnOnce(
            u64,
            Option<DeviceCostGraphStreamState>,
            Option<crate::vnext::DeviceCostGraphCatalogSnapshot>,
            &mut dyn crate::vnext::ResourcePlanningBudget,
        ) -> Result<T, ResourcePlanningUnknown>,
    ) -> Result<T, ResourcePlanningUnknown> {
        if !budget.has_budget() {
            return Err(ResourcePlanningUnknown::BudgetExhausted);
        }
        if !limits.is_valid() {
            return Err(ResourcePlanningUnknown::InvalidInput);
        }
        self.try_with_completion_planning_lane(|epoch, graph, _, catalog| {
            let catalog = catalog(limits, budget)?;
            capture(epoch, graph, catalog, budget)
        })
    }

    fn capture_planning_catalog(
        &self,
        state: &ExecutionLaneState<R::Stream>,
        graph: Option<DeviceCostGraphStreamState>,
        limits: crate::vnext::ResourcePlanningLimits,
        budget: &mut dyn crate::vnext::ResourcePlanningBudget,
    ) -> Result<Option<crate::vnext::DeviceCostGraphCatalogSnapshot>, ResourcePlanningUnknown> {
        use ResourcePlanningUnknown as U;
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        if !limits.is_valid() {
            return Err(U::InvalidInput);
        }
        let mut exhausted = false;
        let prepared =
            self.runtime
                .cost_prepared_reusable_graph_catalog(&state.stream, &mut || {
                    if budget.has_budget() {
                        Ok(())
                    } else {
                        exhausted = true;
                        Err(VNextError::InvalidExecutionPlan {
                            reason: "prepared graph catalog capture budget exhausted".to_owned(),
                        })
                    }
                });
        if exhausted || !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        match prepared.map_err(|_| U::ReusableExecution)? {
            crate::vnext::DevicePreparedCostGraphCatalogAvailability::Ready(root) => {
                if graph != Some(root.catalog().stream_state())
                    || !root.matches_runtime_lane(
                        &self.descriptor.runtime_implementation_fingerprint,
                        self.id(),
                    )
                {
                    return Err(U::StaleIdentity);
                }
                if !budget.has_budget() {
                    return Err(U::BudgetExhausted);
                }
                return Ok(Some(
                    crate::vnext::DeviceCostGraphCatalogSnapshot::Prepared(root),
                ));
            }
            crate::vnext::DevicePreparedCostGraphCatalogAvailability::Unprepared => {
                return Ok(None)
            }
            crate::vnext::DevicePreparedCostGraphCatalogAvailability::Unsupported => {}
        }
        let graph_limits = crate::vnext::DeviceCostGraphCatalogLimits::new(
            limits.maximum_descriptors,
            limits.maximum_free_extents,
            limits.maximum_free_extents,
        )
        .map_err(|_| U::InvalidInput)?;
        let mut exhausted = false;
        let inventory = self.runtime.cost_reusable_graph_catalog_shared(
            &state.stream,
            graph_limits,
            &mut || {
                if !budget.has_budget() {
                    exhausted = true;
                    Err(VNextError::InvalidExecutionPlan {
                        reason: "graph catalog capture budget exhausted".to_owned(),
                    })
                } else {
                    Ok(())
                }
            },
        );
        if exhausted || !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        let catalog = inventory.transpose().map_err(|error| {
            // Keep the backend's failure reason and inventory measurements.
            // Budget cancellation is classified above and is not logged here.
            tracing::debug!(
                target: "ferrum::cost_catalog_diagnostics",
                error = %error,
                lane_id = ?self.id(),
                graph = ?graph,
                maximum_programs = graph_limits.maximum_programs(),
                maximum_nodes = graph_limits.maximum_nodes(),
                maximum_logical_commands = graph_limits.maximum_logical_commands(),
                "reusable graph catalog capture failed"
            );
            U::ReusableExecution
        })?;
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
        Ok(catalog.map(crate::vnext::DeviceCostGraphCatalogSnapshot::Legacy))
    }
}
