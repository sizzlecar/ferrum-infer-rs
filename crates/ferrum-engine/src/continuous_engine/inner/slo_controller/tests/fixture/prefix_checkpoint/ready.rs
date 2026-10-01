//! Ready index views pin a real checkpoint after its producer has retired.
use super::*;

struct ReadyLease {
    owner: std::sync::Weak<vnext::ExecutionLane<contract::TestRuntime>>,
    checkpoint: vnext::SequenceCheckpoint<contract::TestRuntime>,
}
impl std::fmt::Debug for ReadyLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuReadyPrefix")
            .field("boundary", &self.boundary())
            .finish()
    }
}
impl PrefixCaptureLease for ReadyLease {
    fn boundary(&self) -> usize {
        self.checkpoint.completed_tokens()
    }
    fn status(&self) -> PrefixCaptureStatus {
        PrefixCaptureStatus::Ready
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn busy<T>() -> vnext::ExecutionCostRouteAvailability<T> {
    vnext::ExecutionCostRouteAvailability::Unknown(vnext::ExecutionCostRouteUnknown::Resource(
        vnext::ResourcePlanningUnknown::BusyOrUnavailable,
    ))
}
fn expired<T>() -> vnext::ExecutionCostRouteAvailability<T> {
    vnext::ExecutionCostRouteAvailability::Unknown(
        vnext::ExecutionCostRouteUnknown::BudgetExhausted,
    )
}

impl ControlledExecutor {
    pub(in super::super) fn prefix_prompt_tail(
        &self,
        chunk: ferrum_interfaces::model_executor::PrefillChunk,
    ) -> Option<PrefixCapturePlan> {
        self.evidence.prefix.as_ref()?;
        let vnext::SequenceCheckpointCapability::Enabled(layout) = self
            .evidence
            .fixture
            .as_ref()?
            .plan
            .sequence_checkpoint_capability()
        else {
            return None;
        };
        let boundary = usize::try_from(layout.prompt_tail_boundary(
            chunk.tokens_processed() as u64,
            chunk.total_prompt_tokens() as u64,
        )?)
        .ok()?;
        let span = boundary.checked_sub(chunk.tokens_processed())?;
        (boundary <= chunk.end()
            && span > 0
            && layout.capture_span_constraint().permits(span as u64))
        .then_some(PrefixCapturePlan {
            boundary,
            span: layout.capture_span_constraint(),
        })
    }

    pub(super) fn prefix_checkpoint_from_lease(
        &self,
        lease: &dyn PrefixCaptureLease,
    ) -> Option<vnext::SequenceCheckpoint<contract::TestRuntime>> {
        if let Some(ready) = lease.as_any().downcast_ref::<ReadyLease>() {
            let owner = ready.owner.upgrade()?;
            return Arc::ptr_eq(&owner, self.evidence.lane.as_ref()?)
                .then(|| ready.checkpoint.clone());
        }
        let lease = lease.as_any().downcast_ref::<Lease>()?;
        if lease.status() != PrefixCaptureStatus::Ready {
            return None;
        }
        lease.checkpoint.try_lock()?.clone()
    }

    pub(in super::super) fn prefix_ready(
        &self,
        input: ferrum_interfaces::model_executor::PrefixReadyRestoreRequest<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<Option<Arc<dyn PrefixCaptureLease>>> {
        if !budget.has_budget() {
            return expired();
        }
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return unknown();
        };
        let Some(lane) = self.evidence.lane.as_ref() else {
            return unknown();
        };
        if self.prefix_session(input.request_id).is_none()
            || input.maximum_sequence_tokens < input.input_tokens.len()
        {
            return unknown();
        }
        let Some(entries) = prefix.entries.try_lock() else {
            return busy();
        };
        let mut selected: Option<vnext::SequenceCheckpoint<contract::TestRuntime>> = None;
        for entry in entries.iter() {
            if !budget.has_budget() {
                return expired();
            }
            if entry.boundary >= input.input_tokens.len()
                || entry.boundary == 0
                || selected
                    .as_ref()
                    .is_some_and(|old| old.completed_tokens() >= entry.boundary)
            {
                continue;
            }
            let mut matches = true;
            for (old, new) in entry.tokens[..entry.boundary]
                .iter()
                .zip(input.input_tokens)
            {
                if !budget.has_budget() {
                    return expired();
                }
                if old != new {
                    matches = false;
                    break;
                }
            }
            if !matches {
                continue;
            }
            let Some(checkpoint) = entry.checkpoint.try_lock() else {
                return busy();
            };
            // Publication installed this native checkpoint before its full ack.
            // The old waiting lease's expiry is not the ready index's lifetime.
            if let Some(checkpoint) = checkpoint.as_ref() {
                selected = Some(checkpoint.clone());
            }
        }
        drop(entries);
        if !budget.has_budget() {
            return expired();
        }
        vnext::ExecutionCostRouteAvailability::Known(selected.map(|checkpoint| {
            Arc::new(ReadyLease {
                owner: Arc::downgrade(lane),
                checkpoint,
            }) as Arc<dyn PrefixCaptureLease>
        }))
    }

    pub(in super::super) fn prefix_ready_bind(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        lease: &dyn PrefixCaptureLease,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureRetainedCheckpointBinding> {
        if !budget.has_budget() {
            return expired();
        }
        let Some(checkpoint) = self.prefix_checkpoint_from_lease(lease) else {
            return unknown();
        };
        let Some(fixture) = self.evidence.fixture.as_ref() else {
            return unknown();
        };
        if self
            .evidence
            .lane
            .as_ref()
            .is_none_or(|lane| lane.id() != view.lane_id())
        {
            return unknown();
        }
        fixture.plan_resources.bind_future_ready_checkpoint(
            view,
            state,
            &fixture.plan,
            &checkpoint,
            budget,
        )
    }
}
