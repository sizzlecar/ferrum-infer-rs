//! Checkpoint successors share the exact resource/route capture used by model
//! waves. Advancing a numeric frontier never grants a restore or submit permit.
use super::*;
use crate::vnext::{
    DeviceDescriptor, DeviceRuntime, ExecutionPlan, NativeCheckpointTransferCostDomain,
    NativeCheckpointTransferHostWork, NativeCheckpointTransferKind, PlanRuntimeResources,
    ResourcePlanningAvailability, ResourcePlanningBudget, ResourcePlanningCheckpoint,
    SequenceCheckpoint, SequenceCheckpointCapability,
};

/// One explicit checkpoint transition within an existing captured rollout.
/// The checkpoint in Restore must descend from this rollout's Capture.
#[derive(Debug, Clone, Copy)]
pub enum FutureCheckpointCostQuery<'a> {
    Capture {
        source: usize,
        span_start: u64,
        boundary: u64,
        prompt_tokens: u64,
    },
    Restore {
        checkpoint: &'a ResourcePlanningCheckpoint,
        target: usize,
        prompt_tokens: u64,
    },
}

#[derive(Debug, Clone)]
pub struct FutureCheckpointCostProjection {
    pub cost_domain: NativeCheckpointTransferCostDomain,
    pub host_work: NativeCheckpointTransferHostWork,
    pub state: ExecutionCostRouteState,
    pub checkpoint: ResourcePlanningCheckpoint,
}

#[derive(Debug, Clone)]
pub struct FutureRetainedCheckpointBinding {
    pub state: ExecutionCostRouteState,
    pub checkpoint: ResourcePlanningCheckpoint,
}

impl ExecutionCostRouteState {
    pub fn retains_checkpoint(&self, checkpoint: &ResourcePlanningCheckpoint) -> bool {
        checkpoint.belongs_to(&self.resources)
    }
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    pub fn bind_future_retained_checkpoint(
        self: &Arc<Self>,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        source: usize,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureRetainedCheckpointBinding> {
        self.bind_future_owned_checkpoint(view, state, plan, checkpoint, Some(source), budget)
    }

    /// A retained cache allocation remains authoritative after its producer
    /// retires. No source scheduling row is invented for this binding.
    pub fn bind_future_ready_checkpoint(
        self: &Arc<Self>,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureRetainedCheckpointBinding> {
        self.bind_future_owned_checkpoint(view, state, plan, checkpoint, None, budget)
    }

    fn bind_future_owned_checkpoint(
        self: &Arc<Self>,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        source: Option<usize>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureRetainedCheckpointBinding> {
        availability((|| {
            validate(view, state, budget)?;
            if view.lane_id() != checkpoint.captured_evidence().identity().lane_id() {
                return Err(ExecutionCostRouteUnknown::StaleView);
            }
            let binding = match source {
                Some(source) => self.bind_retained_checkpoint(
                    &view.resources,
                    &state.resources,
                    plan,
                    checkpoint,
                    source,
                    budget,
                ),
                None => self.bind_ready_checkpoint(
                    &view.resources,
                    &state.resources,
                    plan,
                    checkpoint,
                    budget,
                ),
            };
            let bound = match binding {
                ResourcePlanningAvailability::Known(bound) => bound,
                ResourcePlanningAvailability::Unknown(reason) => {
                    return Err(ExecutionCostRouteUnknown::Resource(reason));
                }
            };
            let mut successor = state.clone();
            successor.resources = bound.state;
            validate(view, &successor, budget)?;
            Ok(FutureRetainedCheckpointBinding {
                state: successor,
                checkpoint: bound.checkpoint,
            })
        })())
    }

    pub fn project_future_checkpoint_capture(
        self: &Arc<Self>,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        plan: &ExecutionPlan,
        descriptor: &DeviceDescriptor,
        source: usize,
        span_start: u64,
        boundary: u64,
        prompt_tokens: u64,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureCheckpointCostProjection> {
        let result = (|| {
            validate(view, state, budget)?;
            if !self.checkpoint_descriptor_matches(descriptor) {
                return Err(ExecutionCostRouteUnknown::StaleView);
            }
            let completed = view
                .resources
                .participants()
                .get(source)
                .and_then(|participant| participant.completed_checkpoint_boundary())
                .is_some_and(|evidence| {
                    evidence.span_start() == span_start
                        && evidence.completed_tokens() == boundary
                        && evidence.prompt_tokens() == prompt_tokens
                });
            if state.frontiers.get(source).copied() != Some(boundary)
                || (state.initialized.get(source).copied() != Some(true) && !completed)
                || boundary >= prompt_tokens
            {
                return Err(ExecutionCostRouteUnknown::InvalidInput);
            }
            let SequenceCheckpointCapability::Enabled(layout) =
                plan.sequence_checkpoint_capability()
            else {
                return Err(ExecutionCostRouteUnknown::Unsupported);
            };
            if !layout.permits_capture_from(span_start, boundary, prompt_tokens) {
                return Err(ExecutionCostRouteUnknown::InvalidInput);
            }
            let projection = match self.project_checkpoint_capture(
                &view.resources,
                &state.resources,
                plan,
                source,
                boundary,
                budget,
            ) {
                ResourcePlanningAvailability::Known(value) => value,
                ResourcePlanningAvailability::Unknown(reason) => {
                    return Err(ExecutionCostRouteUnknown::Resource(reason));
                }
            };
            let host_work = NativeCheckpointTransferHostWork::from_lengths(boundary, prompt_tokens)
                .map_err(|_| ExecutionCostRouteUnknown::InvalidInput)?;
            let cost_domain = NativeCheckpointTransferCostDomain::from_projection(
                projection.checkpoint.byte_plan(),
                descriptor,
                NativeCheckpointTransferKind::Capture,
                projection.geometry,
            );
            let mut successor = state.clone();
            successor.resources = projection.state;
            validate(view, &successor, budget)?;
            Ok(FutureCheckpointCostProjection {
                cost_domain,
                host_work,
                state: successor,
                checkpoint: projection.checkpoint,
            })
        })();
        availability(result)
    }

    pub fn project_future_checkpoint_restore(
        self: &Arc<Self>,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        plan: &ExecutionPlan,
        descriptor: &DeviceDescriptor,
        checkpoint: &ResourcePlanningCheckpoint,
        target: usize,
        prompt_tokens: u64,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureCheckpointCostProjection> {
        let result = (|| {
            validate(view, state, budget)?;
            if !self.checkpoint_descriptor_matches(descriptor) {
                return Err(ExecutionCostRouteUnknown::StaleView);
            }
            let boundary = checkpoint.byte_plan().boundary();
            if state.frontiers.get(target).copied() != Some(0)
                || state.initialized.get(target).copied() != Some(false)
                || boundary >= prompt_tokens
            {
                return Err(ExecutionCostRouteUnknown::InvalidInput);
            }
            let SequenceCheckpointCapability::Enabled(layout) =
                plan.sequence_checkpoint_capability()
            else {
                return Err(ExecutionCostRouteUnknown::Unsupported);
            };
            if !layout.permits_suffix(boundary, prompt_tokens) {
                return Err(ExecutionCostRouteUnknown::InvalidInput);
            }
            let projection = match self.project_checkpoint_restore(
                &view.resources,
                &state.resources,
                plan,
                checkpoint,
                target,
                budget,
            ) {
                ResourcePlanningAvailability::Known(value) => value,
                ResourcePlanningAvailability::Unknown(reason) => {
                    return Err(ExecutionCostRouteUnknown::Resource(reason));
                }
            };
            let host_work = NativeCheckpointTransferHostWork::from_lengths(boundary, prompt_tokens)
                .map_err(|_| ExecutionCostRouteUnknown::InvalidInput)?;
            let cost_domain = NativeCheckpointTransferCostDomain::from_projection(
                checkpoint.byte_plan(),
                descriptor,
                NativeCheckpointTransferKind::Restore,
                projection.geometry,
            );
            let mut successor = state.clone();
            successor.resources = projection.state;
            successor.frontiers[target] = boundary;
            successor.initialized[target] = true;
            validate(view, &successor, budget)?;
            Ok(FutureCheckpointCostProjection {
                cost_domain,
                host_work,
                state: successor,
                checkpoint: checkpoint.clone(),
            })
        })();
        availability(result)
    }
}

fn validate(
    view: &ExecutionCostRouteView,
    state: &ExecutionCostRouteState,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ExecutionCostRouteUnknown> {
    if !budget.has_budget() {
        return Err(ExecutionCostRouteUnknown::BudgetExhausted);
    }
    if !Arc::ptr_eq(&view.fence, &state.fence)
        || state.frontiers.len() != view.participant_count()
        || state.initialized.len() != view.participant_count()
    {
        return Err(ExecutionCostRouteUnknown::StaleView);
    }
    Ok(())
}

fn availability<T>(
    result: Result<T, ExecutionCostRouteUnknown>,
) -> ExecutionCostRouteAvailability<T> {
    match result {
        Ok(value) => ExecutionCostRouteAvailability::Known(value),
        Err(reason) => ExecutionCostRouteAvailability::Unknown(reason),
    }
}
