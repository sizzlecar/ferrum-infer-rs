//! Sequence-owned physical facts. Logical views and wave authorization are
//! deliberately absent: callers must obtain a fresh physical segment wave.
use super::*;
use std::any::Any;
use std::ops::Range;

struct PhysicalTableRow {
    ordinal: usize,
    resource_id: ResourceId,
    batch_index: usize,
    metadata_identities: Vec<usize>,
}

struct PhysicalTablePool {
    pool_id: DynamicBackingPoolId,
    instance_id: u64,
    validated_epoch: AtomicU64,
}

/// Immutable physical ranges and independent leases of one Sequence catalog.
/// Backend payloads must not retain an Arc back to this table.
pub struct SequencePhysicalBindingTable<B> {
    backing: SegmentBackingBatch<B>,
    rows: Vec<PhysicalTableRow>,
    ordinal_rows: Vec<Option<usize>>,
    pools: Vec<PhysicalTablePool>,
    runtime_owner: usize,
    backend: OnceLock<Arc<dyn Any + Send + Sync>>,
    // Physical ranges must release before capacity and the logical parent.
    _lifetime: Arc<dyn Send + Sync>,
}

impl<B> SequencePhysicalBindingTable<B> {
    /// Structural append proof only. The caller must separately validate the
    /// current pool epochs/poison and issue fresh participant/window authority.
    /// Both tables retain their real allocations, so pointer equality cannot ABA.
    pub(crate) fn is_append_of(&self, previous: &Self) -> bool {
        if self.runtime_owner != previous.runtime_owner
            || self.rows.len() != previous.rows.len()
            || self.ordinal_rows != previous.ordinal_rows
        {
            return false;
        }
        for (current, old) in self.rows().zip(previous.rows()) {
            let a = current.descriptor();
            let b = old.descriptor();
            if current.resource_ordinal() != old.resource_ordinal()
                || current.resource_id() != old.resource_id()
                || a.logical_size_bytes() < b.logical_size_bytes()
                || a.capacity_size_bytes() < b.capacity_size_bytes()
                || a.view_size_bytes() < b.view_size_bytes()
                || a.alignment_bytes() != b.alignment_bytes()
                || a.usage() != b.usage()
                || a.element_type() != b.element_type()
                || a.storage_profile() != b.storage_profile()
            {
                return false;
            }
            let mut current_regions = current.physical_regions();
            let mut end = 0;
            for old_region in old.physical_regions() {
                let Some(region) = current_regions.next() else {
                    return false;
                };
                let (current_buffer, current_range, _) =
                    region.borrowed_buffer_and_physical_range();
                let (old_buffer, old_range, _) = old_region.borrowed_buffer_and_physical_range();
                let current_binding = &self.backing.bindings()[region.region.binding_index()];
                let old_binding = &previous.backing.bindings()[old_region.region.binding_index()];
                if region.logical_offset_bytes() != end
                    || old_region.logical_offset_bytes() != end
                    || region.length_bytes() != old_region.length_bytes()
                    || region.descriptor() != old_region.descriptor()
                    || !std::ptr::eq(current_buffer, old_buffer)
                    || current_range != old_range
                    || !Arc::ptr_eq(&current_binding.chunk, &old_binding.chunk)
                    || current_binding.segment() != old_binding.segment()
                {
                    return false;
                }
                let Some(next) = end.checked_add(region.length_bytes()) else {
                    return false;
                };
                end = next;
            }
            for region in current_regions {
                if region.logical_offset_bytes() != end || region.length_bytes() == 0 {
                    return false;
                }
                let Some(next) = end.checked_add(region.length_bytes()) else {
                    return false;
                };
                end = next;
            }
            if end != a.view_size_bytes() {
                return false;
            }
        }
        true
    }

    pub fn rows(&self) -> impl ExactSizeIterator<Item = SequencePhysicalBindingRow<'_, B>> {
        self.rows
            .iter()
            .map(|row| SequencePhysicalBindingRow { table: self, row })
    }

    pub fn resource(&self, ordinal: usize) -> Option<SequencePhysicalBindingRow<'_, B>> {
        self.ordinal_rows
            .get(ordinal)
            .copied()
            .flatten()
            .map(|index| SequencePhysicalBindingRow {
                table: self,
                row: &self.rows[index],
            })
    }

    pub fn backend_bindings(&self) -> Option<&Arc<dyn Any + Send + Sync>> {
        self.backend.get()
    }

    /// Independent capacity/logical owner keeper. A compiled backend payload
    /// must retain this until all commands using its physical ranges complete.
    /// It has no reference back to this table or its Sequence catalog.
    pub fn lifetime_keeper(&self) -> &Arc<dyn Send + Sync> {
        &self._lifetime
    }

    pub(super) fn current_batch<R: DeviceRuntime<Buffer = B>>(
        tables: &[Arc<Self>],
        pools: &DynamicPoolSet<R>,
        runtime: &R,
    ) -> Result<bool, VNextError> {
        // Only pool cardinality is mapped here. Stable epochs avoid every
        // resource/chunk loop and each pool is locked once for this wave.
        let mut by_pool = BTreeMap::<_, Vec<_>>::new();
        for table in tables {
            if table.runtime_owner != runtime as *const R as usize {
                return Err(invalid_resource(
                    "physical binding table belongs to another runtime",
                ));
            }
            for proof in &table.pools {
                by_pool
                    .entry(&proof.pool_id)
                    .or_default()
                    .push((table, proof));
            }
        }
        let mut groups = Vec::with_capacity(by_pool.len());
        for (pool_id, entries) in by_pool {
            let pool = pools
                .pools
                .get(pool_id)
                .ok_or_else(|| invalid_resource("physical table pool is absent"))?;
            if entries
                .iter()
                .any(|(_, proof)| proof.instance_id != pool.instance_id)
            {
                return Err(invalid_resource("physical table pool owner changed"));
            }
            groups.push((pool_id, entries, pool));
        }
        // Preserve the ordinary permit's common observation point. No callback,
        // owner destruction or backend operation runs while these guards live.
        let guards = groups
            .iter()
            .map(|(_, _, pool)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("physical table pool is poisoned"))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        for ((pool_id, entries, _), state) in groups.iter().zip(&guards) {
            if state.poisoned {
                return Err(invalid_resource("physical table pool is fail-closed"));
            }
            let epoch = state.chunks.epoch();
            if epoch == 0 {
                return Ok(false);
            }
            for (table, proof) in entries {
                if proof.validated_epoch.load(Ordering::Relaxed) == epoch {
                    continue;
                }
                for binding in table
                    .backing
                    .bindings()
                    .iter()
                    .filter(|binding| binding.segment().pool_id() == *pool_id)
                {
                    let Some(current) = state.chunks.get(&binding.segment().chunk_ordinal()) else {
                        return Ok(false);
                    };
                    if !Arc::ptr_eq(&current.backing, &binding.chunk)
                        || current.backing.identity != *binding.segment().chunk()
                        || binding
                            .segment()
                            .offset_bytes()
                            .checked_add(binding.segment().length_bytes())
                            .is_none_or(|end| end > current.backing.descriptor.size_bytes)
                    {
                        return Ok(false);
                    }
                }
                proof.validated_epoch.store(epoch, Ordering::Relaxed);
            }
        }
        Ok(true)
    }

    pub(super) fn current<R: DeviceRuntime<Buffer = B>>(
        &self,
        pools: &DynamicPoolSet<R>,
        runtime: &R,
    ) -> Result<bool, VNextError> {
        if self.runtime_owner != runtime as *const R as usize {
            return Err(invalid_resource(
                "physical binding table belongs to another runtime",
            ));
        }
        for proof in &self.pools {
            let pool = pools
                .pools
                .get(&proof.pool_id)
                .ok_or_else(|| invalid_resource("physical table pool is absent"))?;
            if pool.instance_id != proof.instance_id {
                return Err(invalid_resource("physical table pool owner changed"));
            }
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("physical table pool is poisoned"))?;
            if state.poisoned {
                return Err(invalid_resource("physical table pool is fail-closed"));
            }
            let epoch = state.chunks.epoch();
            if epoch == 0 {
                return Ok(false);
            }
            if proof.validated_epoch.load(Ordering::Relaxed) == epoch {
                continue;
            }
            for binding in self
                .backing
                .bindings()
                .iter()
                .filter(|binding| binding.segment().pool_id() == &proof.pool_id)
            {
                let Some(current) = state.chunks.get(&binding.segment().chunk_ordinal()) else {
                    return Ok(false);
                };
                if !Arc::ptr_eq(&current.backing, &binding.chunk)
                    || current.backing.identity != *binding.segment().chunk()
                    || binding
                        .segment()
                        .offset_bytes()
                        .checked_add(binding.segment().length_bytes())
                        .is_none_or(|end| end > current.backing.descriptor.size_bytes)
                {
                    return Ok(false);
                }
            }
            proof.validated_epoch.store(epoch, Ordering::Relaxed);
        }
        Ok(true)
    }

    pub(super) fn build<R, S>(
        pools: &DynamicPoolSet<R>,
        runtime: &R,
        sources: Vec<(
            usize,
            ResourceId,
            super::segment_permit::SegmentBackingGroup<'_, B>,
        )>,
        ordinal_count: usize,
        lifetime: Arc<dyn Send + Sync>,
        timing: &S,
    ) -> Result<Option<Arc<Self>>, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
        S: crate::vnext::SubmissionWaveDispatchTimingSink,
    {
        if sources.is_empty() {
            return Ok(None);
        }
        let mut expected = Vec::with_capacity(sources.len());
        let mut windows = Vec::with_capacity(sources.len());
        let mut rows = Vec::with_capacity(sources.len());
        let mut ordinal_rows = vec![None; ordinal_count];
        let mut pool_ids = BTreeSet::new();
        for (index, (ordinal, resource_id, group)) in sources.iter().enumerate() {
            let first = group
                .authorities
                .first()
                .ok_or_else(|| invalid_resource("physical table row is empty"))?;
            let logical_bytes = group
                .authorities
                .iter()
                .try_fold(0u64, |total, authority| {
                    total
                        .checked_add(authority.evidence.logical_size_bytes)
                        .ok_or_else(|| invalid_resource("physical table logical size overflows"))
                })?;
            let requirement = SegmentBackingExpectation {
                logical_bytes,
                logical_size_rule: SegmentLogicalSizeRule::Exact,
                usage: first.evidence.usage,
                storage_profile: first.evidence.storage_profile,
                element_type: first.evidence.element_type,
                alignment_bytes: first.evidence.alignment_bytes,
            };
            expected.push(requirement);
            windows.push(SegmentBackingWindow {
                resource_index: index,
                offset_bytes: 0,
                length_bytes: logical_bytes,
                element_type: requirement.element_type,
                alignment_bytes: requirement.alignment_bytes,
            });
            rows.push(PhysicalTableRow {
                ordinal: *ordinal,
                resource_id: resource_id.clone(),
                batch_index: index,
                metadata_identities: Vec::new(),
            });
            *ordinal_rows
                .get_mut(*ordinal)
                .ok_or_else(|| invalid_resource("physical table ordinal is invalid"))? =
                Some(index);
            pool_ids.insert(first.evidence.pool_id.clone());
        }
        let groups = sources
            .into_iter()
            .map(|(_, _, group)| group)
            .collect::<Vec<_>>();
        let backing = pools.segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&windows),
            timing,
        )?;
        for (index, requirement) in expected.iter().enumerate() {
            if matches!(
                requirement.storage_profile.view(),
                DynamicStorageView::Contiguous
            ) && backing
                .window(index)
                .is_none_or(|window| window.physical_regions().len() != 1)
            {
                return Err(invalid_resource(
                    "physical table contiguous resource has multiple regions",
                ));
            }
        }
        let proofs = pool_ids
            .into_iter()
            .map(|pool_id| {
                let pool = pools
                    .pools
                    .get(&pool_id)
                    .ok_or_else(|| invalid_resource("physical table pool is absent"))?;
                Ok(PhysicalTablePool {
                    pool_id,
                    instance_id: pool.instance_id,
                    validated_epoch: AtomicU64::new(0),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let mut metadata = BTreeMap::<usize, &BufferDescriptor>::new();
        for binding in backing.bindings().iter() {
            let key = binding.buffer() as *const B as usize;
            if let Some(previous) = metadata.get(&key) {
                if **previous != *binding.descriptor() {
                    return Err(invalid_resource(
                        "physical table buffer has conflicting metadata",
                    ));
                }
                continue;
            }
            let Some(capability) = runtime.immutable_buffer_metadata(binding.buffer()) else {
                return Ok(None);
            };
            if capability.for_owner(runtime, binding.buffer()) != Some(binding.descriptor())
                || runtime.buffer_descriptor(binding.buffer()) != *binding.descriptor()
            {
                return Err(invalid_resource(
                    "physical table buffer metadata differs from its actual owner",
                ));
            }
            metadata.insert(key, binding.descriptor());
        }
        drop(metadata);
        // A recipe can consume only a subset of the Plan's Sequence rows.
        // Prepare exact row identity sets once, without rebuilding owner views.
        for row in &mut rows {
            row.metadata_identities = backing
                .window(row.batch_index)
                .expect("sealed physical row retains its full window")
                .physical_regions()
                .map(|region| region.borrowed_buffer_and_physical_range().0 as *const B as usize)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
        let table = Arc::new(Self {
            backing,
            rows,
            ordinal_rows,
            pools: proofs,
            runtime_owner: runtime as *const R as usize,
            backend: OnceLock::new(),
            _lifetime: lifetime,
        });
        if !table.current(pools, runtime)? {
            return Ok(None);
        }
        let Some(backend) =
            runtime
                .compile_sequence_physical_bindings(&table)
                .map_err(|error| {
                    invalid_resource(format!(
                        "physical table backend compilation failed: {error}"
                    ))
                })?
        else {
            return Ok(None);
        };
        let _ = table.backend.set(backend);
        Ok(Some(table))
    }
}

pub struct SequencePhysicalBindingRow<'a, B> {
    table: &'a SequencePhysicalBindingTable<B>,
    row: &'a PhysicalTableRow,
}

impl<'a, B> SequencePhysicalBindingRow<'a, B> {
    pub(crate) fn metadata_identities(&self) -> &'a [usize] {
        &self.row.metadata_identities
    }
    pub fn resource_ordinal(&self) -> usize {
        self.row.ordinal
    }
    pub fn resource_id(&self) -> &'a ResourceId {
        &self.row.resource_id
    }
    pub fn descriptor(&self) -> &'a SegmentBackingResource {
        &self.table.backing.resources()[self.row.batch_index]
    }
    pub fn storage_profile(&self) -> DynamicStorageProfile {
        self.descriptor().storage_profile()
    }
    pub fn physical_regions(
        &self,
    ) -> impl ExactSizeIterator<Item = SequencePhysicalBindingRegion<'a, B>> + 'a {
        self.table
            .backing
            .window(self.row.batch_index)
            .expect("sealed physical row retains its full window")
            .physical_regions()
            .map(|region| SequencePhysicalBindingRegion { region })
    }
}

pub struct SequencePhysicalBindingRegion<'a, B> {
    region: SegmentPhysicalRegion<'a, B>,
}

impl<'a, B> SequencePhysicalBindingRegion<'a, B> {
    pub fn logical_offset_bytes(&self) -> u64 {
        self.region.logical_offset_bytes()
    }
    pub fn length_bytes(&self) -> u64 {
        self.region.length_bytes()
    }
    pub fn descriptor(&self) -> &BufferDescriptor {
        self.region.descriptor()
    }
    pub fn borrowed_buffer_and_physical_range(
        &self,
    ) -> (&'a B, Range<u64>, &'a DeviceBufferRetention) {
        self.region.borrowed_buffer_and_physical_range()
    }
}
