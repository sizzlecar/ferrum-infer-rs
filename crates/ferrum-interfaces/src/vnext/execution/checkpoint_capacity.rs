//! Optional plan growth permission for authenticated Sequence checkpoints.
//!
//! These bounds neither reserve memory nor enable capture. A runtime must also
//! enforce the aggregate retained-byte limit and foreground admission priority
//! before using the permission; every allocation still uses the device budget.

use std::num::NonZeroU64;

use super::{
    invalid_plan, AllocationKind, AllocationLifetime, BTreeMap, BTreeSet, BufferUsage, Deserialize,
    DynamicBackingPoolId, DynamicResourceDescriptor, MemoryPlan, PlanNode, ResourceId,
    SequenceCheckpointLayout, Serialize, VNextError,
};

/// One aggregate cap for independently retained, allocator-aligned checkpoint
/// extents, including in-flight and index-evicted but still pinned owners.
/// Per-pool growth ceilings are alternatives within this cap, not additive
/// reservations. This policy does not add request or sequence slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCapacityPolicy {
    maximum_retained_bytes: NonZeroU64,
}

impl CheckpointCapacityPolicy {
    pub fn new(maximum_retained_bytes: u64) -> Result<Self, VNextError> {
        Ok(Self {
            maximum_retained_bytes: NonZeroU64::new(maximum_retained_bytes)
                .ok_or_else(|| invalid_plan("checkpoint capacity must be non-zero"))?,
        })
    }

    pub const fn maximum_retained_bytes(&self) -> u64 {
        self.maximum_retained_bytes.get()
    }

    fn pool_ceiling(&self, descriptor: &DynamicResourceDescriptor) -> u64 {
        let quantum = descriptor.physical_allocation_quantum_bytes();
        // Round down: a partial quantum cannot authorize a physical extent.
        self.maximum_retained_bytes() / quantum * quantum
    }
}

/// Layout authority is required here. A Sequence/State descriptor alone is
/// insufficient to authorize checkpoint storage, including for token-scaled
/// demands. Aliased states and multiple resources in one pool add no quota.
pub(super) fn derive_checkpoint_growth_ceilings(
    policy: Option<&CheckpointCapacityPolicy>,
    layout: Option<&SequenceCheckpointLayout>,
    descriptors: &[DynamicResourceDescriptor],
) -> Result<BTreeMap<DynamicBackingPoolId, u64>, VNextError> {
    let (Some(policy), Some(layout)) = (policy, layout) else {
        return Ok(BTreeMap::new());
    };
    let descriptors = descriptors
        .iter()
        .map(|descriptor| (descriptor.base_resource_id(), descriptor))
        .collect::<BTreeMap<_, _>>();
    let resources = layout
        .states()
        .iter()
        .map(|state| state.resource_id())
        .collect::<BTreeSet<_>>();
    let mut ceilings = BTreeMap::new();
    for resource in resources {
        let descriptor = descriptors
            .get(resource)
            .ok_or_else(|| invalid_plan("checkpoint state has no memory descriptor"))?;
        if descriptor.lifetime() != AllocationLifetime::Sequence
            || descriptor.usage() != BufferUsage::State
            || *descriptor.kind() != AllocationKind::Value
        {
            return Err(invalid_plan(
                "checkpoint growth requires certified Sequence state value storage",
            ));
        }
        let ceiling = policy.pool_ceiling(descriptor);
        if ceiling != 0 {
            if ceilings
                .insert(descriptor.pool_id().clone(), ceiling)
                .is_some_and(|previous| previous != ceiling)
            {
                return Err(invalid_plan(
                    "checkpoint resources disagree on their physical pool quantum",
                ));
            }
        }
    }
    Ok(ceilings)
}

impl MemoryPlan {
    /// Rebuild only physical pool bounds after the complete semantic layout
    /// has been authenticated. Normal descriptor demands and minima are kept.
    pub(super) fn with_checkpoint_capacity(
        mut self,
        policy: Option<CheckpointCapacityPolicy>,
        layout: Option<&SequenceCheckpointLayout>,
        nodes: &[PlanNode],
        retained_completion_resources: &BTreeSet<ResourceId>,
    ) -> Result<Self, VNextError> {
        if policy.is_none() && self.checkpoint_capacity.is_none() {
            return Ok(self);
        }
        let dynamic_capacity = self
            .usable_capacity_bytes
            .checked_sub(self.static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let ceilings =
            derive_checkpoint_growth_ceilings(policy.as_ref(), layout, &self.dynamic_descriptors)?;
        let reusable = self
            .reusable_execution
            .as_ref()
            .map(|plan| plan.pool_workspace_ceilings())
            .transpose()?
            .unwrap_or_default();
        self.dynamic_pools = Self::derive_dynamic_pools_with_checkpoint(
            &self.dynamic_descriptors,
            nodes,
            dynamic_capacity,
            &reusable,
            retained_completion_resources,
            &ceilings,
        )?;
        self.checkpoint_capacity = policy;
        self.validate()?;
        Ok(self)
    }
}

/// Wire-local bounds checking is deliberately weaker than authority. The
/// execution plan additionally rebuilds the exact allowed pool set from its
/// certified layout, and external wire data must survive semantic rebuilding.
pub(super) fn validate_checkpoint_pool_ceiling(
    policy: Option<&CheckpointCapacityPolicy>,
    members: &[&DynamicResourceDescriptor],
    ceiling: u64,
) -> Result<(), VNextError> {
    if ceiling == 0 {
        return Ok(());
    }
    let policy =
        policy.ok_or_else(|| invalid_plan("checkpoint pool growth has no capacity policy"))?;
    if !members.iter().any(|descriptor| {
        descriptor.lifetime() == AllocationLifetime::Sequence
            && descriptor.usage() == BufferUsage::State
            && *descriptor.kind() == AllocationKind::Value
            && policy.pool_ceiling(descriptor) == ceiling
    }) {
        return Err(invalid_plan(
            "checkpoint pool growth differs from its State allocation quantum or cap",
        ));
    }
    Ok(())
}
