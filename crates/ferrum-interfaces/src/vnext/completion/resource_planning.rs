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
