//! One fresh completion capture, with an independently optional forecast view.
use crate::vnext::{
    ExecutionCostRouteAvailability, ExecutionCostRouteView, ResourcePlanningBudget,
    ResourcePlanningView,
};

/// Numeric evidence only. Failure to obtain optional forecast evidence cannot
/// erase an already captured completion view.
pub struct ExecutorCompletionPlanningCapture {
    pub resources: ResourcePlanningView,
    pub forecast: Option<ExecutionCostRouteAvailability<ExecutionCostRouteView>>,
}

/// Synchronous, bounded callbacks. A shared-capture provider retains its read
/// bracket across completion and optional work. The default resource-only
/// provider reports completion after its original capture and returns no forecast.
/// `completion_ready` is called exactly once after successful resource capture,
/// before optional work. It may construct numeric work and reserve a budget;
/// it must not call the executor, acquire its locks, wait, or publish work.
/// Returning true starts one optional phase. The provider must then call
/// `forecast_finished` exactly once after all optional work, including failures,
/// and, for shared capture, before releasing its capture guards. Both callbacks
/// are numeric only.
/// The optional budget must retain its original deadline across every access.
pub trait ExecutorPlanningCapture {
    fn resource_budget(&mut self) -> &mut dyn ResourcePlanningBudget;
    fn completion_ready(&mut self, resources: &ResourcePlanningView) -> bool;
    fn forecast_budget(&mut self) -> &mut dyn ResourcePlanningBudget;
    fn forecast_finished(&mut self);
}
