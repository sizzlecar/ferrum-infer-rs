//! Full configured-shape sizing for explicit startup residency preparation.

use super::*;
use crate::{DynamicPoolGrowthRequest, DynamicStorageView};

/// Metadata-only startup plan. Its pool targets are initial allocations, never
/// admission permits or a residency floor. Original maintenance may reclaim them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PersistentPreallocationPlan {
    context_tokens: u64,
    active_sequences: u32,
    step_tokens: u64,
    checked_peak_bytes: u64,
    usable_capacity_bytes: u64,
    persistent_bytes: u64,
    pools: Vec<PersistentPreallocationPool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PersistentPreallocationPool {
    pool_id: DynamicBackingPoolId,
    target_bytes: u64,
}

impl PersistentPreallocationPool {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }
    pub const fn target_bytes(&self) -> u64 {
        self.target_bytes
    }
}

impl PersistentPreallocationPlan {
    pub const fn checked_peak_bytes(&self) -> u64 {
        self.checked_peak_bytes
    }
    pub const fn persistent_bytes(&self) -> u64 {
        self.persistent_bytes
    }
    pub fn pools(&self) -> &[PersistentPreallocationPool] {
        &self.pools
    }

    /// For a newly provisioned, empty pool set only. These are additional bytes;
    /// callers must not replay this plan against an already populated runtime.
    pub fn initial_growth_requests(&self) -> Result<Vec<DynamicPoolGrowthRequest>, VNextError> {
        self.pools
            .iter()
            .map(|pool| DynamicPoolGrowthRequest::new(pool.pool_id.clone(), pool.target_bytes))
            .collect()
    }
}

impl MemoryPlan {
    /// Validate the entire full-context concurrent shape, including static and
    /// transient workspace, before selecting persistent-only pool allocations.
    /// Sequence-only paged token demand and uniform fixed contiguous cells are
    /// supported. Other geometries require an explicit allocation proof.
    pub fn persistent_preallocation_plan(
        &self,
        context_tokens: u64,
        active_sequences: u32,
        step_tokens: u64,
    ) -> Result<PersistentPreallocationPlan, VNextError> {
        let peak = self.startup_peak_bytes(context_tokens, active_sequences, step_tokens)?;
        if peak > self.usable_capacity_bytes {
            return Err(VNextError::PlanCapacityExceeded {
                required_bytes: peak,
                usable_capacity_bytes: self.usable_capacity_bytes,
            });
        }
        let shape = DynamicResourceShape::from_validated(1, context_tokens, 0);
        let descriptors = self
            .dynamic_descriptors
            .iter()
            .map(|d| (d.base_resource_id(), d))
            .collect::<BTreeMap<_, _>>();
        let mut pools = Vec::new();
        let mut persistent_bytes = 0_u64;
        for pool in &self.dynamic_pools {
            let members = pool
                .resource_ids
                .iter()
                .map(|id| {
                    descriptors.get(id).copied().ok_or_else(|| {
                        invalid_plan("persistent startup pool has a missing descriptor")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !members.iter().any(|d| {
                matches!(
                    d.lifetime(),
                    AllocationLifetime::Request | AllocationLifetime::Sequence
                )
            }) {
                continue;
            }
            if members
                .iter()
                .any(|d| d.lifetime() != AllocationLifetime::Sequence)
            {
                return Err(invalid_plan(
                    "configured persistent envelope requires Sequence-only persistent pools",
                ));
            }
            let paged = matches!(
                pool.compatibility().profile().view(),
                DynamicStorageView::PagedRegions { .. }
            );
            let mut fixed_cell = None;
            let mut target = 0_u64;
            for descriptor in members {
                if matches!(
                    descriptor.demand(),
                    DynamicResourceDemand::Pages { .. }
                        | DynamicResourceDemand::BoundedShapeBuckets { .. }
                ) {
                    return Err(invalid_plan(
                        "configured persistent envelope requires explicit token-derived storage",
                    ));
                }
                let per_sequence = descriptor.evaluate_request_bytes_for_shape(shape)?;
                if !paged {
                    if !matches!(descriptor.demand(), DynamicResourceDemand::Fixed { .. })
                        || fixed_cell.is_some_and(|cell| cell != per_sequence)
                    {
                        return Err(invalid_plan("configured persistent envelope requires uniform fixed contiguous cells"));
                    }
                    fixed_cell = Some(per_sequence);
                }
                target = per_sequence
                    .checked_mul(u64::from(active_sequences))
                    .and_then(|amount| target.checked_add(amount))
                    .ok_or_else(|| invalid_plan("configured persistent pool bytes overflow u64"))?;
            }
            target = target.max(pool.provisioning().minimum_resident_bytes());
            if target > pool.provisioning().maximum_resident_bytes() {
                return Err(invalid_plan(
                    "configured persistent pool exceeds its original resident ceiling",
                ));
            }
            persistent_bytes = persistent_bytes
                .checked_add(target)
                .ok_or_else(|| invalid_plan("configured persistent bytes overflow u64"))?;
            pools.push(PersistentPreallocationPool {
                pool_id: pool.pool_id().clone(),
                target_bytes: target,
            });
        }
        if pools.is_empty() {
            return Err(invalid_plan(
                "configured persistent envelope has no supported persistent pools",
            ));
        }
        Ok(PersistentPreallocationPlan {
            context_tokens,
            active_sequences,
            step_tokens,
            checked_peak_bytes: peak,
            usable_capacity_bytes: self.usable_capacity_bytes,
            persistent_bytes,
            pools,
        })
    }
}
