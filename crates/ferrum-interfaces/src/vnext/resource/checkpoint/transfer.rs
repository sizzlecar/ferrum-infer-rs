use super::{invalid_resource, Arc, CheckpointBackingOwner, DeviceRuntime, VNextError};
use crate::vnext::{
    ExecutionLane, PreparedSequenceStateTransfer, SequenceCheckpointBytePlan,
    SequenceStateTransferKind,
};

impl<R: DeviceRuntime> CheckpointBackingOwner<R> {
    /// Checks the concrete owners before translating any state-copy ranges.
    /// Matching device names or plan hashes alone cannot establish ownership
    /// in the same live runtime and admission coordinator.
    pub(crate) fn validate_transfer_binding(
        self: &Arc<Self>,
        guard: &PreparedSequenceStateTransfer<R>,
        byte_plan: &SequenceCheckpointBytePlan,
        lane: &ExecutionLane<R>,
    ) -> Result<(), VNextError> {
        let sequence_plan = &guard.session().resources().request.plan;
        let _lifecycle = self.plan.read_lifecycle("bind sequence state transfer")?;
        if !Arc::ptr_eq(&self.plan, &sequence_plan.resources)
            || !Arc::ptr_eq(&self.plan.runtime, lane.runtime_arc())
            || self.authority().coordinator_id() != sequence_plan.coordinator_id()
            || byte_plan.plan_hash() != sequence_plan.plan_hash()
            || byte_plan.boundary() == 0
            || byte_plan.logical_bytes() != self.logical_bytes()
            || byte_plan.resources().len() != self.backing_slices.len()
        {
            return Err(invalid_resource(
                "state transfer owners do not belong to the same live plan and byte layout",
            ));
        }
        match guard.kind() {
            SequenceStateTransferKind::CaptureRead => {
                let boundary = guard.completed_boundary()?;
                if boundary.plan_hash() != byte_plan.plan_hash()
                    || u64::try_from(boundary.completed_tokens()).ok() != Some(byte_plan.boundary())
                {
                    return Err(invalid_resource(
                        "capture bytes differ from the exact completed sequence boundary",
                    ));
                }
            }
            SequenceStateTransferKind::RestoreWrite => guard.ensure_fresh_restore_target()?,
        }
        for (resource, owned) in byte_plan.resources().iter().zip(&self.backing_slices) {
            if resource.resource_id() != owned.resource_id() {
                return Err(invalid_resource(
                    "checkpoint storage does not own the complete state-copy resource set",
                ));
            }
            let checkpoint = self.view(resource.resource_id())?;
            if checkpoint.size_bytes() != resource.logical_bytes() {
                return Err(invalid_resource(
                    "checkpoint storage differs from its compact logical byte plan",
                ));
            }
            let sequence = self
                .plan
                .dynamic_pools
                .view_many(guard.backing().backing_slices_for(resource.resource_id()))?;
            for range in resource.ranges() {
                let compact_end = range
                    .checkpoint_offset()
                    .checked_add(range.length_bytes())
                    .ok_or_else(|| invalid_resource("checkpoint copy range overflows"))?;
                if range.source().end > sequence.size_bytes()
                    || compact_end > checkpoint.size_bytes()
                {
                    return Err(invalid_resource(
                        "state copy exceeds the concrete sequence or checkpoint backing",
                    ));
                }
            }
        }
        Ok(())
    }
}
