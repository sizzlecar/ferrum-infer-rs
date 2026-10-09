//! One node's short, canonically ordered pool read permit. No runtime or
//! provider callback runs while the guards exist. On every exit, guards drop
//! before the newly retained output can release allocation/claim owners.
use super::{
    invalid_resource, BufferUsage, DeviceBufferRetention, DeviceRuntime, DynamicPoolSet,
    DynamicStorageProfile, ElementType, LogicalBackingSegmentBinding, LogicalBackingSliceAuthority,
    VNextError,
};
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

pub(crate) struct SteadyBackingSummary {
    pub logical_size_bytes: u64,
    pub capacity_size_bytes: u64,
    pub alignment_bytes: u64,
    pub usage: BufferUsage,
    pub element_type: ElementType,
    pub storage_profile: DynamicStorageProfile,
    pub bindings: Range<usize>,
}

pub(crate) struct SteadyBackingBatch<B> {
    pub resources: Vec<SteadyBackingSummary>,
    pub bindings: Vec<LogicalBackingSegmentBinding<B>>,
}

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    pub(super) fn steady_backing_batch(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
    ) -> Result<SteadyBackingBatch<R::Buffer>, VNextError> {
        // Allocation and all eventual owner cleanup are outside the guard
        // lifetimes. In particular a failed later group cannot release an
        // earlier group's last segment lease while its pool lock is held.
        let mut result = SteadyBackingBatch {
            resources: Vec::with_capacity(groups.len()),
            bindings: Vec::new(),
        };
        let mut ids = BTreeSet::new();
        for group in groups {
            let first = group
                .first()
                .ok_or_else(|| invalid_resource("steady resource has no backing authority"))?;
            ids.insert(first.evidence.pool_id.clone());
        }
        let pools = ids
            .into_iter()
            .map(|id| {
                self.pools
                    .get(&id)
                    .map(|pool| (id, pool))
                    .ok_or_else(|| invalid_resource("steady authority has no dynamic pool"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let guards = pools
            .iter()
            .map(|(_, pool)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let checked = (|| {
            for group in groups {
                let first = &group[0];
                let index = pools
                    .binary_search_by(|(id, _)| id.cmp(&first.evidence.pool_id))
                    .map_err(|_| invalid_resource("steady pool permit is incomplete"))?;
                let pool = pools[index].1;
                let state = &guards[index];
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let start = result.bindings.len();
                let mut logical_size_bytes = 0u64;
                let mut capacity_size_bytes = 0u64;
                for (index, authority) in group.iter().enumerate() {
                    if authority.evidence.pool_id != first.evidence.pool_id
                        || authority.evidence.resource_id != first.evidence.resource_id
                        || authority.evidence.storage_profile != first.evidence.storage_profile
                        || authority.evidence.alignment_bytes != first.evidence.alignment_bytes
                        || authority.evidence.usage != first.evidence.usage
                        || authority.evidence.element_type != first.evidence.element_type
                        || authority.evidence.initialization != first.evidence.initialization
                    {
                        return Err(invalid_resource(
                            "logical backing authorities have incompatible resource metadata",
                        ));
                    }
                    Self::validate_authority(pool, authority)?;
                    if index + 1 < group.len()
                        && authority.evidence.logical_size_bytes
                            != authority.evidence.capacity_size_bytes
                    {
                        return Err(invalid_resource(
                            "multi-extent logical backing cannot contain interior capacity slack",
                        ));
                    }
                    logical_size_bytes = logical_size_bytes
                        .checked_add(authority.evidence.logical_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("logical backing view size overflows u64")
                        })?;
                    capacity_size_bytes = capacity_size_bytes
                        .checked_add(authority.evidence.capacity_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("logical backing capacity overflows u64")
                        })?;
                    for segment in &authority.evidence.segments {
                        let chunk =
                            state.chunks.get(&segment.chunk_ordinal()).ok_or_else(|| {
                                invalid_resource("logical backing references a missing chunk")
                            })?;
                        if segment.pool_id() != &authority.evidence.pool_id
                            || chunk.backing.identity != *segment.chunk()
                            || segment
                                .offset_bytes()
                                .checked_add(segment.length_bytes())
                                .is_none_or(|end| end > chunk.backing.descriptor.size_bytes)
                        {
                            return Err(invalid_resource(
                                "logical backing references a stale or out-of-bounds chunk region",
                            ));
                        }
                        let retention = match authority.reusable_lane {
                            Some(lane) => DeviceBufferRetention::lane_pair(
                                lane,
                                Arc::clone(&authority.segment_lease),
                                Arc::clone(&chunk.backing),
                            ),
                            None => DeviceBufferRetention::pair(
                                Arc::clone(&authority.segment_lease),
                                Arc::clone(&chunk.backing),
                            ),
                        };
                        result.bindings.push(LogicalBackingSegmentBinding {
                            segment: segment.clone(),
                            chunk: Arc::clone(&chunk.backing),
                            retention,
                        });
                    }
                }
                result.resources.push(SteadyBackingSummary {
                    logical_size_bytes,
                    capacity_size_bytes,
                    alignment_bytes: first.evidence.alignment_bytes,
                    usage: first.evidence.usage,
                    element_type: first.evidence.element_type,
                    storage_profile: first.evidence.storage_profile,
                    bindings: start..result.bindings.len(),
                });
            }
            Ok(())
        })();
        drop(guards);
        checked?;
        Ok(result)
    }
}
