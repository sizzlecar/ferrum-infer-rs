//! One current-wave row plan, current-chunk observation and retained result.
//! Borrowed chunk slots are local to the held guards; they are not authorities.
use super::super::{DynamicBackingPoolState, ResidentChunkBacking};
use super::*;

pub(crate) struct SegmentBackingRow<'a> {
    pub(crate) authorities: &'a [LogicalBackingSliceAuthority],
    pub(crate) expected: SegmentBackingExpectation,
}

struct PlannedRow {
    first_row: usize,
    pool_slot: usize,
    segment_count: usize,
}

struct CurrentChunkSlot<'guard, B> {
    ordinal: u32,
    backing: &'guard Arc<ResidentChunkBacking<B>>,
    first_binding: Option<usize>,
}

struct CurrentPoolChunks<'guard, B> {
    slots: Vec<CurrentChunkSlot<'guard, B>>,
    observed: bool,
}

impl<'guard, B> CurrentPoolChunks<'guard, B> {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            observed: false,
        }
    }

    fn observe(&mut self, state: &'guard DynamicBackingPoolState<B>) -> Result<(), VNextError> {
        if !self.observed {
            self.slots
                .try_reserve_exact(state.chunks.len())
                .map_err(|_| {
                    invalid_resource("segment current chunk directory allocation failed")
                })?;
            // BTree iteration supplies the ordinal order without a segment
            // pre-scan, an N-entry routing table, sorting or an extra Arc pin.
            self.slots.extend(
                state
                    .chunks
                    .iter()
                    .map(|(&ordinal, chunk)| CurrentChunkSlot {
                        ordinal,
                        backing: &chunk.backing,
                        first_binding: None,
                    }),
            );
            self.observed = true;
        }
        Ok(())
    }

    fn get(&mut self, ordinal: u32) -> Result<&mut CurrentChunkSlot<'guard, B>, VNextError> {
        let index = self
            .slots
            .binary_search_by_key(&ordinal, |slot| slot.ordinal)
            .map_err(|_| invalid_resource("logical backing references a missing chunk"))?;
        Ok(&mut self.slots[index])
    }
}

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    /// Consume rows already resolved from this exact prepared wave. General
    /// parallel-input callers retain the independent existing implementation.
    pub(crate) fn segment_backing_batch_with_rows<S: SubmissionWaveDispatchTimingSink>(
        &self,
        rows: &[SegmentBackingRow<'_>],
        windows: SegmentBackingWindowInput<'_>,
        timing_sink: &S,
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        let mut timings = SegmentBackingTimings::new(timing_sink);
        if rows.is_empty() {
            return Err(invalid_resource(
                "segment resources require matching nonempty expectations",
            ));
        }
        // All ownership and scalar containers precede the guards. Ordinary
        // error and panic cleanup unlocks before any partial retention Drop.
        let mut result = SegmentBackingBatch {
            resources: Vec::new(),
            bindings: Vec::new(),
            windows: Vec::new(),
            regions: Vec::new(),
            metadata_binding_indices: Some(Vec::new()),
        };
        let mut planned: Vec<PlannedRow> = Vec::new();
        let mut requirement_error = None;
        planned
            .try_reserve_exact(rows.len())
            .map_err(|_| invalid_resource("segment row plan allocation failed"))?;
        let mut unique = BTreeMap::new();
        let mut pool_ids = BTreeMap::new();
        let mut segment_count = 0usize;
        for (index, row) in rows.iter().enumerate() {
            let first = row
                .authorities
                .first()
                .ok_or_else(|| invalid_resource("segment resource has no authority"))?;
            let key = (row.authorities.as_ptr(), row.authorities.len());
            let plan = if let Some(&first_row) = unique.get(&key) {
                let previous: &PlannedRow = &planned[first_row];
                PlannedRow {
                    first_row,
                    pool_slot: previous.pool_slot,
                    segment_count: previous.segment_count,
                }
            } else {
                let next_pool_slot = pool_ids.len();
                let pool_slot = *pool_ids
                    .entry(&first.evidence.pool_id)
                    .or_insert(next_pool_slot);
                let count = row
                    .authorities
                    .iter()
                    .try_fold(0usize, |total, authority| {
                        total
                            .checked_add(authority.evidence.segments.len())
                            .ok_or_else(|| {
                                invalid_resource("segment binding count overflows usize")
                            })
                    })?;
                segment_count = segment_count
                    .checked_add(count)
                    .ok_or_else(|| invalid_resource("segment binding count overflows usize"))?;
                unique.insert(key, index);
                PlannedRow {
                    first_row: index,
                    pool_slot,
                    segment_count: count,
                }
            };
            planned.push(plan);
        }
        let region_capacity = windows.iter().try_fold(0usize, |total, window| {
            let window = window?;
            let row = planned
                .get(window.resource_index)
                .ok_or_else(|| invalid_resource("segment window references an unknown resource"))?;
            total
                .checked_add(row.segment_count)
                .ok_or_else(|| invalid_resource("segment window capacity overflows usize"))
        })?;
        result
            .resources
            .try_reserve_exact(rows.len())
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
        result
            .metadata_binding_indices
            .as_mut()
            .unwrap()
            .try_reserve_exact(segment_count)
            .map_err(|_| invalid_resource("segment metadata index allocation failed"))?;

        // Discovery slots avoid another pool-ID binary search for each row;
        // guards still acquire the actual pools in canonical pool-ID order.
        let mut guard_slots = Vec::new();
        guard_slots
            .try_reserve_exact(pool_ids.len())
            .map_err(|_| invalid_resource("segment pool slot allocation failed"))?;
        guard_slots.resize(pool_ids.len(), 0usize);
        let pools = pool_ids
            .into_iter()
            .enumerate()
            .map(|(guard_index, (id, pool_slot))| {
                guard_slots[pool_slot] = guard_index;
                self.pools
                    .get(id)
                    .ok_or_else(|| invalid_resource("segment authority has no dynamic pool"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut directories = Vec::new();
        directories
            .try_reserve_exact(pools.len())
            .map_err(|_| invalid_resource("segment current pool directory allocation failed"))?;
        directories.resize_with(pools.len(), CurrentPoolChunks::new);
        timings.next();
        let guards = pools
            .iter()
            .map(|pool| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        timings.next();
        let checked = (|| {
            for (index, row) in rows.iter().enumerate() {
                let plan = &planned[index];
                let first = &row.authorities[0];
                let (logical, capacity, bindings) = if plan.first_row == index {
                    let pool_index = guard_slots[plan.pool_slot];
                    let pool = pools[pool_index];
                    let state = &guards[pool_index];
                    if state.poisoned {
                        return Err(invalid_resource("dynamic backing pool is fail-closed"));
                    }
                    directories[pool_index].observe(state)?;
                    let start = result.bindings.len();
                    let mut logical = 0u64;
                    let mut capacity = 0u64;
                    for (authority_index, authority) in row.authorities.iter().enumerate() {
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
                        if authority_index + 1 < row.authorities.len()
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
                            let slot = directories[pool_index].get(segment.chunk_ordinal())?;
                            let backing = slot.backing;
                            // Directory identity is never a generation or range
                            // proof: every segment keeps the complete checks.
                            if segment.pool_id() != &authority.evidence.pool_id
                                || backing.identity != *segment.chunk()
                                || segment
                                    .offset_bytes()
                                    .checked_add(segment.length_bytes())
                                    .is_none_or(|end| end > backing.descriptor.size_bytes)
                            {
                                return Err(invalid_resource(
                                    "logical backing references a stale or out-of-bounds chunk region",
                                ));
                            }
                            let retention = match authority.reusable_lane {
                                Some(lane) => DeviceBufferRetention::lane_pair(
                                    lane,
                                    Arc::clone(&authority.segment_lease),
                                    Arc::clone(backing),
                                ),
                                None => DeviceBufferRetention::pair(
                                    Arc::clone(&authority.segment_lease),
                                    Arc::clone(backing),
                                ),
                            };
                            let binding_index = result.bindings.len();
                            result.bindings.push(LogicalBackingSegmentBinding {
                                segment: segment.clone(),
                                chunk: Arc::clone(backing),
                                retention,
                            });
                            // A valid current identity admits only its own
                            // pool/ordinal slot. Reusing that slot never merges
                            // the distinct leases/lanes retained above.
                            if slot.first_binding.is_none() {
                                slot.first_binding = Some(binding_index);
                                result
                                    .metadata_binding_indices
                                    .as_mut()
                                    .unwrap()
                                    .push(binding_index);
                            }
                        }
                    }
                    (logical, capacity, start..result.bindings.len())
                } else {
                    let previous = &result.resources[plan.first_row];
                    (
                        previous.logical_size_bytes,
                        previous.capacity_size_bytes,
                        previous.bindings.clone(),
                    )
                };
                // The retained resource result is also the canonical row's
                // summary. Exact aliases get their own required view size.
                let requirement = row.expected;
                let size_matches = match requirement.logical_size_rule {
                    SegmentLogicalSizeRule::Exact => logical == requirement.logical_bytes,
                    SegmentLogicalSizeRule::AtLeast => logical >= requirement.logical_bytes,
                };
                if requirement.logical_bytes == 0
                    || !size_matches
                    || capacity < logical
                    || first.evidence.usage != requirement.usage
                    || first.evidence.storage_profile != requirement.storage_profile
                    || first.evidence.element_type != requirement.element_type
                    || first.evidence.alignment_bytes != requirement.alignment_bytes
                {
                    // General permits finish every unique authority before
                    // returning the first current-wave requirement failure.
                    // Preserve that error precedence without another row walk.
                    if requirement_error.is_none() {
                        requirement_error = Some(invalid_resource(
                            "segment backing differs from its current-wave resource requirement",
                        ));
                    }
                }
                result.resources.push(SegmentBackingResource {
                    logical_size_bytes: logical,
                    capacity_size_bytes: capacity,
                    view_size_bytes: requirement.logical_bytes,
                    alignment_bytes: first.evidence.alignment_bytes,
                    usage: first.evidence.usage,
                    element_type: first.evidence.element_type,
                    storage_profile: first.evidence.storage_profile,
                    bindings,
                });
            }
            if let Some(error) = requirement_error.take() {
                return Err(error);
            }
            timings.next();
            result.build_windows(windows)?;
            Ok(())
        })();
        timings.finish();
        // The directory borrows current chunks; only original result bindings
        // retain them past this observation. No borrowed view leaves the guard.
        drop(directories);
        drop(guards);
        checked?;
        Ok(result)
    }
}
