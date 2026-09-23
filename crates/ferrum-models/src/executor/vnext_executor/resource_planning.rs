//! Registry-to-resource bridge. Temporary borrows/guards are released before
//! returning; the public result owns only bounded numeric evidence.
use super::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;
use ferrum_interfaces::vnext::{
    ResourcePlanningAvailability, ResourcePlanningBudget, ResourcePlanningLimits,
    ResourcePlanningUnknown, ResourcePlanningView,
};

#[cfg(test)]
mod tests;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    /// The immutable first-fit policy shared by actual Step admission and both
    /// numerical projections. The phase, rather than the token count, selects
    /// the workspace class; no matching bucket means the actual eager route.
    pub(super) fn reusable_bucket_for_shape(
        &self,
        kind: VNextExecutionWaveKind,
        sequences: u32,
        tokens: u64,
        pages: u64,
    ) -> Option<&ReusableExecutionBucketId> {
        self.resolved_plan
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .and_then(|plan| {
                plan.buckets().iter().find(|resolved| {
                    let bucket = resolved.bucket();
                    bucket.class_id().as_str() == kind.reusable_execution_class()
                        && bucket.capacity().covers(sequences, tokens, pages)
                })
            })
            .map(|resolved| resolved.bucket().bucket_id())
    }

    pub(super) fn capture_resource_planning_view(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningView> {
        match capture_registry_view(
            &self.sequences,
            &self.plan_resources,
            Some(&self.lane),
            requests,
            limits,
            budget,
        ) {
            Ok(view) => view,
            Err(reason) => ResourcePlanningAvailability::Unknown(reason),
        }
    }
}

fn capture_registry_view<R: DeviceRuntime>(
    sequences: &Mutex<VNextSequenceRegistry<R>>,
    plan_resources: &Arc<PlanRuntimeResources<R>>,
    lane: Option<&ExecutionLane<R>>,
    requests: &[ExecutorResourcePlanningRequest<'_>],
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> std::result::Result<ResourcePlanningAvailability<ResourcePlanningView>, ResourcePlanningUnknown>
{
    with_registry_sequences(
        sequences,
        requests,
        limits,
        budget,
        |_, sessions, budget| match lane {
            Some(lane) => {
                plan_resources.resource_planning_view_on_lane(sessions, lane, limits, budget)
            }
            None => plan_resources.resource_planning_view(sessions, limits, budget),
        },
    )
}

pub(super) fn with_registry_sequences<R: DeviceRuntime, T>(
    sequences: &Mutex<VNextSequenceRegistry<R>>,
    requests: &[ExecutorResourcePlanningRequest<'_>],
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
    capture: impl FnOnce(
        &[Arc<VNextSequence<R>>],
        &[&SequenceSession<R>],
        &mut dyn ResourcePlanningBudget,
    ) -> T,
) -> std::result::Result<T, ResourcePlanningUnknown> {
    use ResourcePlanningReadStage as Stage;
    use ResourcePlanningUnknown as U;
    if !limits.is_valid() || requests.is_empty() {
        return Err(U::InvalidInput);
    }
    if requests.len() > limits.maximum_participants {
        return Err(U::LimitExceeded);
    }
    if !budget.has_budget() {
        return Err(U::BudgetExhausted);
    }
    let registry = sequences
        .try_lock()
        .ok_or(U::ReadUnavailable(Stage::ModelRegistry))?;
    // Cache-free lookup scans current admitted entries. It must remain
    // bounded even if the caller asks for only one of many requests.
    if registry.total_len() > limits.maximum_participants {
        return Err(U::LimitExceeded);
    }
    let mut slots = Vec::with_capacity(requests.len());
    let mut sequences = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        if requests[..index]
            .iter()
            .any(|other| other.request_id == request.request_id)
        {
            return Err(U::InvalidInput);
        }
        let sequence = match request.cache_id {
            Some(cache_id) => {
                let sequence = registry.active.get(cache_id).ok_or(U::StaleIdentity)?;
                if sequence.request_id() != request.request_id {
                    return Err(U::StaleIdentity);
                }
                Arc::clone(sequence)
            }
            None => {
                let slot = registry
                    .prefills
                    .get(request.request_id)
                    .ok_or(U::BusyOrUnavailable)?;
                if slot.cancelled.load(Ordering::Acquire) {
                    return Err(U::BusyOrUnavailable);
                }
                slots.push(
                    slot.state
                        .try_lock()
                        .ok_or(U::ReadUnavailable(Stage::ModelRegistrySlot))?,
                );
                match &**slots.last().unwrap() {
                    VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
                    _ => return Err(U::BusyOrUnavailable),
                }
            }
        };
        if !sequence.active.load(Ordering::Acquire) {
            return Err(U::BusyOrUnavailable);
        }
        sequences.push(sequence);
    }
    let mut operations = Vec::with_capacity(sequences.len());
    for sequence in &sequences {
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        operations.push(
            sequence
                .operation
                .try_lock()
                .map_err(|_| U::ReadUnavailable(Stage::ModelSequenceOperation))?,
        );
        if !sequence.active.load(Ordering::Acquire) {
            return Err(U::BusyOrUnavailable);
        }
    }
    let sessions: Vec<_> = sequences
        .iter()
        .map(|sequence| sequence.session.as_ref())
        .collect();
    Ok(capture(&sequences, &sessions, budget))
}
