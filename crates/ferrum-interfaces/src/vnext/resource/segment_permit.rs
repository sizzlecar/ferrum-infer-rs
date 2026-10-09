//! One fresh resource observation for an entire compiled binding segment.
//!
//! The operation layer supplies already-bound current-wave requirements. This
//! module resolves their captured authorities once, locks the unique pools in
//! canonical order, and consumes every declared window before unlocking. No
//! provider, runtime getter, or backend callback executes under these locks.
use super::{
    invalid_resource, AllocationLifetime, Arc, BufferDescriptor, BufferUsage,
    DeviceBufferRetention, DeviceRuntime, DynamicPoolSet, DynamicStorageProfile, ElementType,
    LogicalBackingSegmentBinding, LogicalBackingSliceAuthority, ResourceId, VNextError,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub(crate) struct SegmentPlanResourceRequest<'a> {
    pub slot_index: usize,
    pub allocation: &'a super::ResourceAllocation,
}

/// A fresh current lease observation. Its retention owns Plan resources only;
/// the borrow cannot be retargeted to a later wave or another admitted Plan.
pub(crate) struct SegmentPlanResourceView<'a, B> {
    pub leased: super::LeasedBufferView<'a, B>,
    pub retention: DeviceBufferRetention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentLogicalSizeRule {
    Exact,
    AtLeast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBackingExpectation {
    pub logical_bytes: u64,
    pub logical_size_rule: SegmentLogicalSizeRule,
    pub usage: BufferUsage,
    pub storage_profile: DynamicStorageProfile,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// A compiled unique-resource index resolved against this wave, never a node
/// invocation or a previously captured physical authority.
#[derive(Clone, Debug)]
pub struct SegmentBackingRequest {
    pub participant_index: usize,
    pub resource_id: ResourceId,
    pub lifetime: AllocationLifetime,
    pub expected: SegmentBackingExpectation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBackingWindow {
    pub resource_index: usize,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// Only the permit can construct these summaries. The effective view is the
/// requested current-wave size, not all captured capacity or committed slack.
pub struct SegmentBackingResource {
    logical_size_bytes: u64,
    capacity_size_bytes: u64,
    view_size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    storage_profile: DynamicStorageProfile,
    bindings: Range<usize>,
}

impl SegmentBackingResource {
    pub const fn logical_size_bytes(&self) -> u64 {
        self.logical_size_bytes
    }
    pub const fn capacity_size_bytes(&self) -> u64 {
        self.capacity_size_bytes
    }
    pub const fn view_size_bytes(&self) -> u64 {
        self.view_size_bytes
    }
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }
}

struct WindowRegion {
    binding_index: usize,
    logical_offset_bytes: u64,
    physical_offset_bytes: u64,
    length_bytes: u64,
}

/// Owns real allocation/lease retention, but no pool lock or whole wave owner.
/// A later mutation is ordered after this observation, not prohibited until
/// submit. The existing submission/fence ownership must retain these regions.
pub struct SegmentBackingBatch<B> {
    resources: Vec<SegmentBackingResource>,
    bindings: Vec<LogicalBackingSegmentBinding<B>>,
    windows: Vec<Range<usize>>,
    regions: Vec<WindowRegion>,
}

impl<B> SegmentBackingBatch<B> {
    pub fn resources(&self) -> &[SegmentBackingResource] {
        &self.resources
    }
    pub fn resource(&self, index: usize) -> Option<&SegmentBackingResource> {
        self.resources.get(index)
    }
    pub fn window(&self, index: usize) -> Option<SegmentBackingWindowView<'_, B>> {
        self.windows
            .get(index)
            .map(|range| SegmentBackingWindowView {
                batch: self,
                regions: &self.regions[range.clone()],
            })
    }
    /// The operation layer uses each actual buffer once for its explicit
    /// immutable-metadata capability. This never calls a runtime under a lock.
    pub(crate) fn bindings(&self) -> &[LogicalBackingSegmentBinding<B>] {
        &self.bindings
    }
}

pub struct SegmentBackingWindowView<'a, B> {
    batch: &'a SegmentBackingBatch<B>,
    regions: &'a [WindowRegion],
}

impl<'a, B> SegmentBackingWindowView<'a, B> {
    pub fn physical_regions(
        &self,
    ) -> impl ExactSizeIterator<Item = SegmentPhysicalRegion<'a, B>> + 'a {
        let batch = self.batch;
        self.regions
            .iter()
            .map(move |region| SegmentPhysicalRegion {
                binding: &batch.bindings[region.binding_index],
                region,
            })
    }
}

pub struct SegmentPhysicalRegion<'a, B> {
    binding: &'a LogicalBackingSegmentBinding<B>,
    region: &'a WindowRegion,
}

impl<'a, B> SegmentPhysicalRegion<'a, B> {
    pub const fn logical_offset_bytes(&self) -> u64 {
        self.region.logical_offset_bytes
    }
    pub const fn length_bytes(&self) -> u64 {
        self.region.length_bytes
    }
    pub fn descriptor(&self) -> &BufferDescriptor {
        self.binding.descriptor()
    }
    pub fn buffer_and_physical_range(&self) -> (&'a B, Range<u64>, DeviceBufferRetention) {
        (
            self.binding.buffer(),
            self.region.physical_offset_bytes
                ..self.region.physical_offset_bytes + self.region.length_bytes,
            self.binding.retention(),
        )
    }
}

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    pub(super) fn segment_backing_batch(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
        windows: &[SegmentBackingWindow],
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        if groups.len() != expected.len() || groups.is_empty() {
            return Err(invalid_resource(
                "segment resources require matching nonempty expectations",
            ));
        }
        // Allocate before locking. Partial retained results are also declared
        // before guards, so panic/error unwinding unlocks before owner Drop.
        let mut result = SegmentBackingBatch {
            resources: Vec::new(),
            bindings: Vec::new(),
            windows: Vec::new(),
            regions: Vec::new(),
        };
        let mut ids = BTreeSet::new();
        let mut unique = BTreeMap::new();
        let mut group_indices = Vec::new();
        let mut unique_groups = Vec::new();
        let mut segment_count = 0usize;
        for group in groups {
            let first = group
                .first()
                .ok_or_else(|| invalid_resource("segment resource has no authority"))?;
            ids.insert(first.evidence.pool_id.clone());
            // Pointer identity is only a dedup key. Every unique slice is still
            // fully validated below against its exact pool and segment lease.
            let key = (group.as_ptr(), group.len());
            let index = match unique.get(&key) {
                Some(index) => *index,
                None => {
                    let index = unique_groups.len();
                    unique_groups.push(*group);
                    unique.insert(key, index);
                    for authority in *group {
                        segment_count = segment_count
                            .checked_add(authority.evidence.segments.len())
                            .ok_or_else(|| {
                                invalid_resource("segment binding count overflows usize")
                            })?;
                    }
                    index
                }
            };
            group_indices.push(index);
        }
        let region_capacity = windows.iter().try_fold(0usize, |total, window| {
            let group = groups
                .get(window.resource_index)
                .ok_or_else(|| invalid_resource("segment window references an unknown resource"))?;
            group.iter().try_fold(total, |total, authority| {
                total
                    .checked_add(authority.evidence.segments.len())
                    .ok_or_else(|| invalid_resource("segment window capacity overflows usize"))
            })
        })?;
        result
            .resources
            .try_reserve_exact(groups.len())
            .map_err(|_| invalid_resource("segment resource allocation failed"))?;
        result
            .bindings
            .try_reserve_exact(segment_count)
            .map_err(|_| invalid_resource("segment binding allocation failed"))?;
        result
            .windows
            .try_reserve_exact(windows.len())
            .map_err(|_| invalid_resource("segment window allocation failed"))?;
        result
            .regions
            .try_reserve_exact(region_capacity)
            .map_err(|_| invalid_resource("segment region allocation failed"))?;
        let mut summaries = Vec::new();
        summaries
            .try_reserve_exact(unique_groups.len())
            .map_err(|_| invalid_resource("segment summary allocation failed"))?;
        let pools = ids
            .into_iter()
            .map(|id| {
                self.pools
                    .get(&id)
                    .map(|pool| (id, pool))
                    .ok_or_else(|| invalid_resource("segment authority has no dynamic pool"))
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
            for group in &unique_groups {
                let first = &group[0];
                let pool_index = pools
                    .binary_search_by(|(id, _)| id.cmp(&first.evidence.pool_id))
                    .map_err(|_| invalid_resource("segment pool permit is incomplete"))?;
                let pool = pools[pool_index].1;
                let state = &guards[pool_index];
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let start = result.bindings.len();
                let mut logical = 0u64;
                let mut capacity = 0u64;
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
                    logical = logical
                        .checked_add(authority.evidence.logical_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("logical backing view size overflows u64")
                        })?;
                    capacity = capacity
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
                summaries.push((logical, capacity, start..result.bindings.len()));
            }
            for (index, requirement) in expected.iter().enumerate() {
                let first = &groups[index][0].evidence;
                let (logical, capacity, bindings) = &summaries[group_indices[index]];
                let size_matches = match requirement.logical_size_rule {
                    SegmentLogicalSizeRule::Exact => *logical == requirement.logical_bytes,
                    SegmentLogicalSizeRule::AtLeast => *logical >= requirement.logical_bytes,
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
                        "segment backing differs from its current-wave resource requirement",
                    ));
                }
                result.resources.push(SegmentBackingResource {
                    logical_size_bytes: *logical,
                    capacity_size_bytes: *capacity,
                    view_size_bytes: requirement.logical_bytes,
                    alignment_bytes: first.alignment_bytes,
                    usage: first.usage,
                    element_type: first.element_type,
                    storage_profile: first.storage_profile,
                    bindings: bindings.clone(),
                });
            }
            for window in windows {
                let resource = result.resources.get(window.resource_index).ok_or_else(|| {
                    invalid_resource("segment window references an unknown resource")
                })?;
                let end = window
                    .offset_bytes
                    .checked_add(window.length_bytes)
                    .ok_or_else(|| invalid_resource("segment window range overflows u64"))?;
                if window.length_bytes == 0
                    || end > resource.view_size_bytes
                    || window.element_type != resource.element_type
                    || window.alignment_bytes == 0
                    || !window.alignment_bytes.is_power_of_two()
                    || resource.alignment_bytes < window.alignment_bytes
                    || !window.offset_bytes.is_multiple_of(window.alignment_bytes)
                {
                    return Err(invalid_resource(
                        "segment window is outside its typed current-wave view",
                    ));
                }
                let start = result.regions.len();
                let mut logical_origin = 0u64;
                let mut covered = 0u64;
                for binding_index in resource.bindings.clone() {
                    let binding = &result.bindings[binding_index];
                    let segment_end = logical_origin
                        .checked_add(binding.segment.length_bytes())
                        .ok_or_else(|| invalid_resource("segment logical range overflows u64"))?;
                    let begin = logical_origin.max(window.offset_bytes);
                    let finish = segment_end.min(end);
                    if begin < finish {
                        let physical = binding
                            .segment
                            .offset_bytes()
                            .checked_add(begin - logical_origin)
                            .ok_or_else(|| {
                                invalid_resource("segment physical offset overflows u64")
                            })?;
                        let length = finish - begin;
                        if !physical.is_multiple_of(window.alignment_bytes)
                            || physical
                                .checked_add(length)
                                .is_none_or(|limit| limit > binding.descriptor().size_bytes)
                        {
                            return Err(invalid_resource(
                                "segment physical window exceeds its actual allocation",
                            ));
                        }
                        covered = covered
                            .checked_add(length)
                            .ok_or_else(|| invalid_resource("segment covered bytes overflow"))?;
                        result.regions.push(WindowRegion {
                            binding_index,
                            logical_offset_bytes: begin - window.offset_bytes,
                            physical_offset_bytes: physical,
                            length_bytes: length,
                        });
                    }
                    logical_origin = segment_end;
                    if logical_origin >= end {
                        break;
                    }
                }
                if covered != window.length_bytes {
                    return Err(invalid_resource(
                        "segment physical backing does not cover its window",
                    ));
                }
                result.windows.push(start..result.regions.len());
            }
            Ok(())
        })();
        drop(guards);
        checked?;
        Ok(result)
    }
}
