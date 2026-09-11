use super::*;
use crate::vnext::{
    CheckpointBackingAllocationDecision, CheckpointPartitionNumerics, ExecutionPlan,
    SequenceCheckpointCapability, SequenceCheckpointLayout, SequenceStateTransferKind,
    SequenceStateTransferPreparation, TrustedPlanRuntimeBinding,
};

fn access_layout(
    plan: &ExecutionPlan,
) -> Result<&SequenceCheckpointLayout, CheckpointAccessSkipReason> {
    if plan.payload().memory().checkpoint_capacity().is_none() {
        return Err(CheckpointAccessSkipReason::Disabled);
    }
    let SequenceCheckpointCapability::Enabled(layout) = plan.sequence_checkpoint_capability()
    else {
        return Err(CheckpointAccessSkipReason::Unsupported);
    };
    if !layout.inputs().conditioning_inputs().is_empty() {
        return Err(CheckpointAccessSkipReason::MissingConditioningEvidence);
    }
    if layout.providers().iter().any(|provider| {
        provider.contract().partition_numerics() == CheckpointPartitionNumerics::SamePartitionOnly
    }) {
        return Err(CheckpointAccessSkipReason::MissingPartitionEvidence);
    }
    Ok(layout)
}

fn wrap_submission<R: DeviceRuntime>(
    submission: Result<StateTransferSubmission<R>, VNextError>,
    restore: Option<RestoreInput<R>>,
) -> NativeCheckpointStart<R> {
    match submission {
        Ok(StateTransferSubmission::Submitted(handle)) => {
            NativeCheckpointStart::Submitted(NativeCheckpointTransfer { handle, restore })
        }
        Ok(StateTransferSubmission::Indeterminate(handle)) => {
            NativeCheckpointStart::Indeterminate(NativeCheckpointTransfer { handle, restore })
        }
        Ok(StateTransferSubmission::ContractAfterSubmission { error, handle }) => {
            NativeCheckpointStart::ContractAfterSubmission {
                error,
                transfer: NativeCheckpointTransfer { handle, restore },
            }
        }
        Err(error) => NativeCheckpointStart::NotSubmitted(error),
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    /// Capture only the source's current proven complete boundary. Optional
    /// retention never grows backing or waits for foreground capacity here.
    pub fn try_capture_sequence_checkpoint(
        self: &Arc<Self>,
        plan: &ExecutionPlan,
        binding: &TrustedPlanRuntimeBinding<R>,
        source: Arc<SequenceSession<R>>,
        lane: Arc<ExecutionLane<R>>,
    ) -> Result<NativeCheckpointStart<R>, VNextError> {
        let layout = match access_layout(plan) {
            Ok(layout) => layout,
            Err(reason) => return Ok(NativeCheckpointStart::Skipped(reason)),
        };
        if plan.plan_hash() != binding.plan_hash()
            || plan.plan_hash() != source.resources().plan_evidence().plan_hash()
            || binding.coordinator_id() != source.resources().coordinator_id()
        {
            return Err(invalid_completion(
                "checkpoint capture names another admitted plan",
            ));
        }
        let generation = source.resources().backing_generation()?;
        let guard = match source
            .try_prepare_state_transfer(SequenceStateTransferKind::CaptureRead, generation)?
        {
            SequenceStateTransferPreparation::Prepared(guard) => guard,
            SequenceStateTransferPreparation::Busy => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::Busy,
                ));
            }
            SequenceStateTransferPreparation::StaleBacking => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::StaleBacking,
                ));
            }
        };
        let boundary = guard.completed_boundary()?;
        let boundary_tokens = u64::try_from(boundary.completed_tokens())
            .map_err(|_| invalid_completion("checkpoint boundary exceeds u64"))?;
        let source_start = u64::try_from(boundary.capture_span_start())
            .map_err(|_| invalid_completion("checkpoint span start exceeds u64"))?;
        let input_length = u64::try_from(boundary.full_input().len())
            .map_err(|_| invalid_completion("checkpoint input length exceeds u64"))?;
        if !layout.permits_capture_from(source_start, boundary_tokens, input_length) {
            return Ok(NativeCheckpointStart::Skipped(
                CheckpointAccessSkipReason::BoundaryNotPermitted,
            ));
        }
        let byte_plan = Arc::new(plan.checkpoint_byte_plan(boundary_tokens)?);
        let owner = match binding.try_allocate_checkpoint_backing(&byte_plan.backing_requests()?)? {
            CheckpointBackingAllocationDecision::Allocated(owner) => owner,
            CheckpointBackingAllocationDecision::Skipped(reason) => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::Retention(reason),
                ));
            }
            CheckpointBackingAllocationDecision::Deferred(reason) => {
                // Maintenance must never hold the source state gate. Its
                // authenticated requests grant no permission to copy this
                // boundary later; a fresh capture must validate it again.
                drop(guard);
                return Ok(NativeCheckpointStart::CapacityMaintenance {
                    reason: CheckpointAccessSkipReason::CapacityDeferred(reason),
                    maintenance: binding
                        .prepare_checkpoint_capacity_maintenance(plan, &byte_plan)?,
                });
            }
            CheckpointBackingAllocationDecision::BackingDeferred(reason) => {
                drop(guard);
                return Ok(NativeCheckpointStart::CapacityMaintenance {
                    reason: CheckpointAccessSkipReason::BackingDeferred(reason),
                    maintenance: binding
                        .prepare_checkpoint_capacity_maintenance(plan, &byte_plan)?,
                });
            }
            CheckpointBackingAllocationDecision::PermanentRejected(reason) => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::PermanentRejected(reason),
                ));
            }
        };
        let permit = owner.try_reserve_capture()?;
        Ok(wrap_submission(
            self.submit_capture(guard, permit, byte_plan, lane),
            None,
        ))
    }

    /// The target must already have sufficient admitted backing. This binds
    /// its exact incarnation and actual complete input before device encoding;
    /// neither can be replaced later when taking the restore result.
    pub fn try_restore_sequence_checkpoint(
        self: &Arc<Self>,
        plan: &ExecutionPlan,
        target: Arc<SequenceSession<R>>,
        checkpoint: &SequenceCheckpoint<R>,
        full_input: Arc<[u32]>,
        lane: Arc<ExecutionLane<R>>,
    ) -> Result<NativeCheckpointStart<R>, VNextError> {
        let layout = match access_layout(plan) {
            Ok(layout) => layout,
            Err(reason) => return Ok(NativeCheckpointStart::Skipped(reason)),
        };
        if plan.plan_hash() != target.resources().plan_evidence().plan_hash()
            || plan.plan_hash() != checkpoint.plan_hash()
            || full_input.get(..checkpoint.completed_tokens()) != Some(checkpoint.token_prefix())
        {
            return Err(invalid_completion(
                "checkpoint restore differs from the target plan or token prefix",
            ));
        }
        checkpoint
            .inner
            .validate_reuse_contract(layout, &full_input)?;
        let generation = target.resources().backing_generation()?;
        let guard = match target
            .try_prepare_state_transfer(SequenceStateTransferKind::RestoreWrite, generation)?
        {
            SequenceStateTransferPreparation::Prepared(guard) => guard,
            SequenceStateTransferPreparation::Busy => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::Busy,
                ));
            }
            SequenceStateTransferPreparation::StaleBacking => {
                return Ok(NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::StaleBacking,
                ));
            }
        };
        guard.validate_restore_input(Arc::clone(&full_input))?;
        let submission = self.submit_restore(guard, Arc::clone(&checkpoint.inner), layout, lane);
        Ok(wrap_submission(
            submission,
            Some(RestoreInput { target, full_input }),
        ))
    }
}
