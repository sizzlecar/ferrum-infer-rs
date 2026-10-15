//! A physical comparison template is not an allocation or submission permit.
//! Each hit checks all current authorities under the ordinary pool locks.
use super::*;
use crate::vnext::resource::dynamic_pool::ResidentChunkBacking;

struct ResidentPhysicalBinding<B> {
    segment: crate::vnext::BackingSegment,
    chunk: Arc<ResidentChunkBacking<B>>,
}

struct ResidentResource<B> {
    resource_id: ResourceId,
    capacity_bytes: u64,
    expected: SegmentBackingExpectation,
    first_binding: usize,
    bindings: Vec<ResidentPhysicalBinding<B>>,
}

/// Holds physical allocations only: never a segment, slot, or frame lease.
pub(crate) struct ResidentBackingTemplate<B> {
    resources: Vec<ResidentResource<B>>,
    binding_count: usize,
}

impl<B> SegmentBackingBatch<B> {
    pub(crate) fn resident_template(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
    ) -> Result<ResidentBackingTemplate<B>, VNextError> {
        if groups.len() != self.resources.len() || expected.len() != groups.len() {
            return Err(invalid_resource(
                "resident template resource coverage differs",
            ));
        }
        let resources = self
            .resources
            .iter()
            .zip(groups)
            .zip(expected)
            .map(|((resource, group), expected)| {
                let first = group
                    .first()
                    .ok_or_else(|| invalid_resource("resident template authority is absent"))?;
                Ok(ResidentResource {
                    resource_id: first.evidence.resource_id.clone(),
                    capacity_bytes: resource.capacity_size_bytes,
                    expected: *expected,
                    first_binding: resource.bindings.start,
                    bindings: self.bindings[resource.bindings.clone()]
                        .iter()
                        .map(|binding| ResidentPhysicalBinding {
                            segment: binding.segment.clone(),
                            chunk: Arc::clone(&binding.chunk),
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        Ok(ResidentBackingTemplate {
            resources,
            binding_count: self.bindings.len(),
        })
    }
}

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    /// Err means invalid current authority. None is a valid current allocation
    /// layout which differs from the cold template. Check every resource before
    /// reporting a miss so poison/invalid claims are never hidden by a mismatch.
    pub(in crate::vnext::resource) fn authorize_resident_backing(
        &self,
        template: &ResidentBackingTemplate<R::Buffer>,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
    ) -> Result<Option<Vec<DeviceBufferRetention>>, VNextError> {
        if groups.is_empty() || groups.len() != expected.len() {
            return Err(invalid_resource(
                "resident resources require matching nonempty expectations",
            ));
        }
        let mut ids = BTreeSet::new();
        for group in groups {
            let first = group
                .first()
                .ok_or_else(|| invalid_resource("resident resource has no authority"))?;
            ids.insert(first.evidence.pool_id.clone());
        }
        let pools = ids
            .into_iter()
            .map(|id| {
                self.pools
                    .get(&id)
                    .map(|pool| (id, pool))
                    .ok_or_else(|| invalid_resource("resident authority has no pool"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Results precede guards: errors must unlock before dropping leases.
        let mut retained = vec![None::<DeviceBufferRetention>; template.binding_count];
        let guards = pools
            .iter()
            .map(|(_, pool)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut matches = groups.len() == template.resources.len();
        for (resource_index, (group, requirement)) in groups.iter().zip(expected).enumerate() {
            let first = &group[0].evidence;
            let pool_index = pools
                .binary_search_by(|(id, _)| id.cmp(&first.pool_id))
                .map_err(|_| invalid_resource("resident pool permit is incomplete"))?;
            let pool = pools[pool_index].1;
            let state = &guards[pool_index];
            if state.poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            let cold = template.resources.get(resource_index);
            let mut logical = 0u64;
            let mut capacity = 0u64;
            let mut binding_index = 0usize;
            for (index, authority) in group.iter().enumerate() {
                let evidence = &authority.evidence;
                if evidence.pool_id != first.pool_id
                    || evidence.resource_id != first.resource_id
                    || evidence.storage_profile != first.storage_profile
                    || evidence.alignment_bytes != first.alignment_bytes
                    || evidence.usage != first.usage
                    || evidence.element_type != first.element_type
                    || evidence.initialization != first.initialization
                {
                    return Err(invalid_resource(
                        "resident authorities have incompatible resource metadata",
                    ));
                }
                Self::validate_authority(pool, authority)?;
                if index + 1 < group.len()
                    && evidence.logical_size_bytes != evidence.capacity_size_bytes
                {
                    return Err(invalid_resource(
                        "resident backing has interior capacity slack",
                    ));
                }
                logical = logical
                    .checked_add(evidence.logical_size_bytes)
                    .ok_or_else(|| invalid_resource("resident logical size overflows"))?;
                capacity = capacity
                    .checked_add(evidence.capacity_size_bytes)
                    .ok_or_else(|| invalid_resource("resident capacity overflows"))?;
                for segment in &evidence.segments {
                    let chunk = state.chunks.get(&segment.chunk_ordinal()).ok_or_else(|| {
                        invalid_resource("resident backing references a missing chunk")
                    })?;
                    if segment.pool_id() != &evidence.pool_id
                        || chunk.backing.identity != *segment.chunk()
                        || segment
                            .offset_bytes()
                            .checked_add(segment.length_bytes())
                            .is_none_or(|end| end > chunk.backing.descriptor.size_bytes)
                    {
                        return Err(invalid_resource(
                            "resident backing references a stale or out-of-bounds chunk",
                        ));
                    }
                    matches &= cold
                        .and_then(|cold| cold.bindings.get(binding_index))
                        .is_some_and(|old| {
                            old.segment == *segment
                                && Arc::ptr_eq(&old.chunk, &chunk.backing)
                                && old.chunk.descriptor == chunk.backing.descriptor
                        });
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
                    if let Some(slot) =
                        cold.and_then(|cold| retained.get_mut(cold.first_binding + binding_index))
                    {
                        if let Some(prior) = slot {
                            matches &= prior.same_owners(&retention);
                        } else {
                            *slot = Some(retention);
                        }
                    }
                    binding_index += 1;
                }
            }
            let size_matches = match requirement.logical_size_rule {
                SegmentLogicalSizeRule::Exact => logical == requirement.logical_bytes,
                SegmentLogicalSizeRule::AtLeast => logical >= requirement.logical_bytes,
            };
            if requirement.logical_bytes == 0
                || !size_matches
                || capacity < logical
                || first.usage != requirement.usage
                || first.storage_profile != requirement.storage_profile
                || first.element_type != requirement.element_type
                || first.alignment_bytes != requirement.alignment_bytes
            {
                return Err(invalid_resource(
                    "resident backing differs from its current-wave requirement",
                ));
            }
            matches &= cold.is_some_and(|old| {
                old.resource_id == first.resource_id
                    && old.expected.logical_size_rule == requirement.logical_size_rule
                    && old.expected.usage == requirement.usage
                    && old.expected.storage_profile == requirement.storage_profile
                    && old.expected.element_type == requirement.element_type
                    && old.expected.alignment_bytes == requirement.alignment_bytes
                    && old.capacity_bytes == capacity
                    && old.bindings.len() == binding_index
            });
        }
        // The same linearization as segment_backing_batch: later pool writers
        // are permitted; these exact current owners survive through the fence.
        drop(guards);
        if !matches {
            return Ok(None);
        }
        retained
            .into_iter()
            .map(|owner| {
                owner.ok_or_else(|| {
                    invalid_resource("resident physical owner coverage is incomplete")
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}
