//! Optional checkpoint pool growth, separate from capture submission and claims.

use super::*;
use crate::vnext::{
    CheckpointCapacityPolicy, DeviceCapacityPressure, DynamicPoolGrowthBatchReceipt,
    DynamicPoolResidentPressure, ExecutionPlan, SequenceCheckpointBytePlan,
};

/// One plan-authenticated attempt to make compact checkpoint storage available.
/// This owner neither reserves a sequence slot nor authorizes a state copy.
/// After maintenance the caller must retry capture and its ordinary claims.
#[must_use = "checkpoint maintenance must be attempted or dropped"]
pub struct CheckpointCapacityMaintenance<R: DeviceRuntime> {
    binding: TrustedPlanRuntimeBinding<R>,
    requests: CheckpointBackingRequests,
    policy: Option<CheckpointCapacityPolicy>,
}

#[derive(Debug)]
pub enum CheckpointCapacityMaintenanceSkipReason {
    Retention(CheckpointRetentionSkipReason),
    DeviceCapacity(DeviceCapacityPressure),
    PoolResident(DynamicPoolResidentPressure),
}

#[derive(Debug)]
pub enum CheckpointCapacityMaintenanceOutcome {
    /// Space was already available or growth was published. A concurrent claim
    /// may consume it, so this receipt is not an allocation guarantee.
    Ready(DynamicPoolGrowthBatchReceipt),
    Skipped(CheckpointCapacityMaintenanceSkipReason),
}

impl<R: DeviceRuntime> TrustedPlanRuntimeBinding<R> {
    pub fn prepare_checkpoint_capacity_maintenance(
        &self,
        plan: &ExecutionPlan,
        byte_plan: &SequenceCheckpointBytePlan,
    ) -> Result<CheckpointCapacityMaintenance<R>, VNextError> {
        let _lifecycle = self
            .resources
            .read_lifecycle("prepare checkpoint maintenance")?;
        if plan.plan_hash() != self.plan_hash()
            || byte_plan.plan_hash() != self.plan_hash()
            || plan.checkpoint_byte_plan(byte_plan.boundary())? != *byte_plan
        {
            return Err(invalid_resource(
                "checkpoint maintenance requires this plan's certified byte layout",
            ));
        }
        let requests = byte_plan.backing_requests()?;
        self.evaluate_checkpoint_backing(&requests)?;
        Ok(CheckpointCapacityMaintenance {
            binding: TrustedPlanRuntimeBinding {
                resources: Arc::clone(&self.resources),
            },
            requests,
            policy: plan.payload().memory().checkpoint_capacity().copied(),
        })
    }
}

impl<R: DeviceRuntime> CheckpointCapacityMaintenance<R> {
    /// Recomputes free-space and contiguous packing requirements against the
    /// actual pools. Only presently available device capacity may be used:
    /// no cross-pool reclamation, waiting, or foreground admission mutation.
    pub fn try_maintain(self) -> Result<CheckpointCapacityMaintenanceOutcome, VNextError> {
        let _lifecycle = self
            .binding
            .resources
            .read_lifecycle("maintain checkpoint capacity")?;
        let Some(policy) = self.policy else {
            return Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
                CheckpointCapacityMaintenanceSkipReason::Retention(
                    CheckpointRetentionSkipReason::Disabled,
                ),
            ));
        };
        let evaluated = self.binding.evaluate_checkpoint_backing(&self.requests)?;
        let retained = self
            .binding
            .logical_admission()
            .checkpoint_retained_bytes()?;
        let maximum = policy.maximum_retained_bytes();
        let remaining = maximum.checked_sub(retained).ok_or_else(|| {
            invalid_resource("checkpoint retained bytes exceed the bound plan policy")
        })?;
        if evaluated.extent_bytes > remaining {
            return Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
                CheckpointCapacityMaintenanceSkipReason::Retention(
                    CheckpointRetentionSkipReason::Capacity {
                        requested_bytes: evaluated.extent_bytes,
                        retained_bytes: retained,
                        maximum_bytes: maximum,
                    },
                ),
            ));
        }
        match self
            .binding
            .dynamic_pools()
            .maintain_checkpoint_capacity(&evaluated.slices)
        {
            Ok(receipt) => Ok(CheckpointCapacityMaintenanceOutcome::Ready(receipt)),
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
                Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
                    CheckpointCapacityMaintenanceSkipReason::DeviceCapacity(pressure),
                ))
            }
            Err(VNextError::DynamicPoolResidentUnavailable(pressure)) => {
                Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
                    CheckpointCapacityMaintenanceSkipReason::PoolResident(pressure),
                ))
            }
            Err(error) => Err(error),
        }
    }
}
