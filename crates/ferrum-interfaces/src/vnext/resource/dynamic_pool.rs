use super::{
    invalid_resource, validate_runtime_descriptor_for_admission, AdmissionDeferred, AllocationSeal,
    Arc, AtomicU64, BTreeMap, BTreeSet, BufferDescriptor, BufferRequest, BufferUsage,
    CapacityDomainId, CapacityEpochs, CapacityUnits, DeviceAllocationPermit, DeviceCapacityBudget,
    DeviceCapacityGrant, DeviceCapacityReservation, DeviceRuntime, DynamicBackingPoolId,
    DynamicBackingPoolSpec, DynamicResourceDescriptor, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ElementType, InvocationLivenessMode,
    LogicalAdmissionCoordinator, Mutex, Ordering, PlanNode, ResourceId, ResourceReservation,
    ResourceRetentionPolicy, ResourceTransactionIdentity, RunId, Serialize,
    StaticProvisioningBinding, StepResourceSlotKind, TransactionId, VNextError,
};
use crate::vnext::{
    CapacityShortfallKind, DeferredAction, DeviceCapacityPressure, DeviceCapacityPressureScope,
};

static NEXT_DYNAMIC_POOL_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn align_up_resource(value: u64, alignment: u64) -> Result<u64, VNextError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_resource(
            "dynamic pool alignment is not a non-zero power of two",
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| invalid_resource("dynamic pool aligned bytes overflow u64"))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct DynamicPoolDomainSpec {
    pub(super) domain_id: CapacityDomainId,
    pub(super) pool: DynamicBackingPoolSpec,
    pub(super) descriptors: Vec<DynamicResourceDescriptor>,
}

impl DynamicPoolDomainSpec {
    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub(super) fn pool_id(&self) -> &DynamicBackingPoolId {
        self.pool.pool_id()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BackingChunkIdentity {
    pool_id: DynamicBackingPoolId,
    ordinal: u32,
    generation: u64,
}

impl BackingChunkIdentity {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackingSegment {
    chunk: BackingChunkIdentity,
    offset_bytes: u64,
    length_bytes: u64,
}

impl BackingSegment {
    pub(crate) fn new(
        chunk: BackingChunkIdentity,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        Self::from_chunk(
            &chunk.pool_id,
            chunk.ordinal,
            chunk.generation,
            offset_bytes,
            length_bytes,
        )
    }

    pub(super) fn from_chunk(
        pool_id: &DynamicBackingPoolId,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
        {
            return Err(invalid_resource(
                "backing segment has invalid chunk identity or physical range",
            ));
        }
        Ok(Self {
            chunk: BackingChunkIdentity {
                pool_id: pool_id.clone(),
                ordinal: chunk_ordinal,
                generation: chunk_generation,
            },
            offset_bytes,
            length_bytes,
        })
    }

    pub fn chunk(&self) -> &BackingChunkIdentity {
        &self.chunk
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        self.chunk.pool_id()
    }

    pub const fn chunk_ordinal(&self) -> u32 {
        self.chunk.ordinal()
    }

    pub const fn chunk_generation(&self) -> u64 {
        self.chunk.generation()
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }
}

pub(super) fn backing_segment_range(
    segments: &[BackingSegment],
    physical_offset_bytes: u64,
    size_bytes: u64,
) -> Result<Vec<BackingSegment>, VNextError> {
    let physical_end = physical_offset_bytes
        .checked_add(size_bytes)
        .ok_or_else(|| invalid_resource("logical backing projection range overflows u64"))?;
    if size_bytes == 0 {
        return Err(invalid_resource(
            "logical backing projection must have non-zero size",
        ));
    }
    let mut physical_cursor = 0_u64;
    let mut covered = 0_u64;
    let mut projection = Vec::new();
    for segment in segments {
        if physical_cursor >= physical_end {
            break;
        }
        let segment_end = physical_cursor
            .checked_add(segment.length_bytes())
            .ok_or_else(|| invalid_resource("physical backing extent range overflows u64"))?;
        let overlap_start = physical_cursor.max(physical_offset_bytes);
        let overlap_end = segment_end.min(physical_end);
        if overlap_start < overlap_end {
            let within_segment = overlap_start - physical_cursor;
            let translated_offset = segment
                .offset_bytes()
                .checked_add(within_segment)
                .ok_or_else(|| invalid_resource("backing projection offset overflows u64"))?;
            let length = overlap_end - overlap_start;
            projection.push(BackingSegment::from_chunk(
                segment.pool_id(),
                segment.chunk_ordinal(),
                segment.chunk_generation(),
                translated_offset,
                length,
            )?);
            covered = covered.checked_add(length).ok_or_else(|| {
                invalid_resource("logical backing projection coverage overflows u64")
            })?;
        }
        physical_cursor = segment_end;
    }
    if covered != size_bytes {
        return Err(invalid_resource(
            "logical backing projection exceeds its physical extent",
        ));
    }
    Ok(projection)
}

pub(super) struct ResidentChunkBacking<B> {
    // Buffer must drop before its physical capacity grant is returned.
    pub(super) buffer: B,
    _grant: DeviceCapacityGrant,
    identity: BackingChunkIdentity,
    pub(super) descriptor: BufferDescriptor,
}

pub(super) struct ResidentChunkState<B> {
    pub(super) backing: Arc<ResidentChunkBacking<B>>,
    pub(super) live_segments: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FreeExtent {
    pub(super) chunk_generation: u64,
    pub(super) length_bytes: u64,
}

#[derive(Debug, Default)]
pub(super) struct FreeExtentIndex {
    pub(super) by_offset: BTreeMap<(u32, u64), FreeExtent>,
    pub(super) by_size: BTreeSet<(u64, u32, u64, u64)>,
    pub(super) free_bytes: u64,
    pub(super) search_probes: u64,
}

impl FreeExtentIndex {
    fn rollback_segments(&mut self, segments: &[BackingSegment]) -> Result<(), VNextError> {
        for segment in segments.iter().rev() {
            self.release(segment)?;
        }
        Ok(())
    }

    fn with_rollback_context(
        &mut self,
        segments: &[BackingSegment],
        error: VNextError,
    ) -> VNextError {
        match self.rollback_segments(segments) {
            Ok(()) => error,
            Err(rollback) => invalid_resource(format!(
                "dynamic allocator failed and its journal rollback also failed: {error}; rollback: {rollback}"
            )),
        }
    }

    pub(super) fn insert_extent(
        &mut self,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<(), VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
            || self.by_offset.contains_key(&(chunk_ordinal, offset_bytes))
        {
            return Err(invalid_resource("free extent identity or range is invalid"));
        }
        let end = offset_bytes + length_bytes;
        if self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
            .is_some_and(|(&(ordinal, previous_offset), previous)| {
                ordinal == chunk_ordinal && previous_offset + previous.length_bytes > offset_bytes
            })
            || self
                .by_offset
                .range((chunk_ordinal, offset_bytes)..)
                .next()
                .is_some_and(|(&(ordinal, next_offset), _)| {
                    ordinal == chunk_ordinal && next_offset < end
                })
        {
            return Err(invalid_resource("free extent overlaps an existing extent"));
        }
        let next_free = self
            .free_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes overflow u64"))?;
        self.by_offset.insert(
            (chunk_ordinal, offset_bytes),
            FreeExtent {
                chunk_generation,
                length_bytes,
            },
        );
        assert!(self
            .by_size
            .insert((length_bytes, chunk_ordinal, chunk_generation, offset_bytes,)));
        self.free_bytes = next_free;
        Ok(())
    }

    fn remove_extent(
        &mut self,
        chunk_ordinal: u32,
        offset_bytes: u64,
    ) -> Result<FreeExtent, VNextError> {
        let extent = *self
            .by_offset
            .get(&(chunk_ordinal, offset_bytes))
            .ok_or_else(|| invalid_resource("free extent journal references a missing range"))?;
        let size_key = (
            extent.length_bytes,
            chunk_ordinal,
            extent.chunk_generation,
            offset_bytes,
        );
        if !self.by_size.contains(&size_key) {
            return Err(invalid_resource("free extent indexes diverged"));
        }
        let next_free_bytes = self
            .free_bytes
            .checked_sub(extent.length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes underflowed"))?;
        self.by_offset.remove(&(chunk_ordinal, offset_bytes));
        assert!(self.by_size.remove(&size_key));
        self.free_bytes = next_free_bytes;
        Ok(extent)
    }

    pub(super) fn allocate_contiguous(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
    ) -> Result<Option<BackingSegment>, VNextError> {
        self.search_probes = self.search_probes.saturating_add(1);
        let selected = self.by_size.range((size_bytes, 0, 0, 0)..).next().copied();
        let Some((length_bytes, chunk_ordinal, chunk_generation, offset_bytes)) = selected else {
            return Ok(None);
        };
        let segment = BackingSegment::from_chunk(
            pool_id,
            chunk_ordinal,
            chunk_generation,
            offset_bytes,
            size_bytes,
        )?;
        let removed = self.remove_extent(chunk_ordinal, offset_bytes)?;
        debug_assert_eq!(removed.length_bytes, length_bytes);
        debug_assert_eq!(removed.chunk_generation, chunk_generation);
        if size_bytes < length_bytes {
            if let Err(error) = self.insert_extent(
                chunk_ordinal,
                chunk_generation,
                offset_bytes + size_bytes,
                length_bytes - size_bytes,
            ) {
                let restore =
                    self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes);
                return Err(match restore {
                    Ok(()) => error,
                    Err(rollback) => invalid_resource(format!(
                        "contiguous allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                    )),
                });
            }
        }
        Ok(Some(segment))
    }

    pub(super) fn allocate_paged(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
        block_bytes: u64,
    ) -> Result<Option<Vec<BackingSegment>>, VNextError> {
        if size_bytes == 0 || size_bytes % block_bytes != 0 {
            return Err(invalid_resource(
                "paged backing reservation is not block aligned",
            ));
        }
        if self.free_bytes < size_bytes {
            return Ok(None);
        }
        let mut remaining = size_bytes;
        let mut segments = Vec::new();
        while remaining != 0 {
            self.search_probes = self.search_probes.saturating_add(1);
            let Some((&(chunk_ordinal, offset_bytes), &extent)) = self.by_offset.first_key_value()
            else {
                self.rollback_segments(&segments)?;
                return Ok(None);
            };
            if extent.length_bytes % block_bytes != 0 {
                let error = invalid_resource("paged free extent lost fixed-block alignment");
                return Err(self.with_rollback_context(&segments, error));
            }
            let take = extent.length_bytes.min(remaining);
            let segment = match BackingSegment::from_chunk(
                pool_id,
                chunk_ordinal,
                extent.chunk_generation,
                offset_bytes,
                take,
            ) {
                Ok(segment) => segment,
                Err(error) => return Err(self.with_rollback_context(&segments, error)),
            };
            if let Err(error) = self.remove_extent(chunk_ordinal, offset_bytes) {
                return Err(self.with_rollback_context(&segments, error));
            }
            if take < extent.length_bytes {
                if let Err(error) = self.insert_extent(
                    chunk_ordinal,
                    extent.chunk_generation,
                    offset_bytes + take,
                    extent.length_bytes - take,
                ) {
                    let restore = self.insert_extent(
                        chunk_ordinal,
                        extent.chunk_generation,
                        offset_bytes,
                        extent.length_bytes,
                    );
                    let error = match restore {
                        Ok(()) => error,
                        Err(rollback) => invalid_resource(format!(
                            "paged allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                        )),
                    };
                    return Err(self.with_rollback_context(&segments, error));
                }
            }
            segments.push(segment);
            remaining -= take;
        }
        Ok(Some(segments))
    }

    fn release(&mut self, segment: &BackingSegment) -> Result<(), VNextError> {
        let chunk_ordinal = segment.chunk_ordinal();
        let chunk_generation = segment.chunk_generation();
        let mut offset_bytes = segment.offset_bytes();
        let mut length_bytes = segment.length_bytes();
        if let Some((&(ordinal, previous_offset), &previous)) = self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
        {
            if ordinal == chunk_ordinal
                && previous.chunk_generation == chunk_generation
                && previous_offset + previous.length_bytes == offset_bytes
            {
                self.remove_extent(ordinal, previous_offset)?;
                offset_bytes = previous_offset;
                length_bytes = length_bytes
                    .checked_add(previous.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        if let Some((&(ordinal, next_offset), &next)) =
            self.by_offset.range((chunk_ordinal, offset_bytes)..).next()
        {
            if ordinal == chunk_ordinal
                && next.chunk_generation == chunk_generation
                && offset_bytes + length_bytes == next_offset
            {
                self.remove_extent(ordinal, next_offset)?;
                length_bytes = length_bytes
                    .checked_add(next.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes)
    }

    pub(super) fn largest_contiguous_bytes(&self) -> u64 {
        self.by_size
            .last()
            .map_or(0, |(length_bytes, _, _, _)| *length_bytes)
    }
}

fn rollback_free_extent_journal<B>(
    states: &mut [std::sync::MutexGuard<'_, DynamicBackingPoolState<B>>],
    journals: &[Vec<Vec<BackingSegment>>],
) -> Result<(), VNextError> {
    for group_index in (0..journals.len()).rev() {
        for segments in journals[group_index].iter().rev() {
            for segment in segments.iter().rev() {
                if let Err(error) = states[group_index].allocator.release(segment) {
                    states[group_index].poisoned = true;
                    return Err(invalid_resource(format!(
                        "dynamic backing rollback failed and poisoned its pool: {error}"
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) struct DynamicBackingPoolState<B> {
    pub(super) resident_bytes: u64,
    pub(super) pending_growth_bytes: u64,
    pub(super) next_chunk_ordinal: u32,
    pub(super) next_chunk_generation: u64,
    pub(super) chunks: BTreeMap<u32, ResidentChunkState<B>>,
    pub(super) allocator: FreeExtentIndex,
    pub(super) quarantined: Vec<QuarantinedDynamicChunk<B>>,
    pub(super) poisoned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicChunkQuarantineReason {
    DescriptorMismatch,
    PublicationRejected,
}

pub(super) struct QuarantinedDynamicChunk<B> {
    pub(super) backing: Arc<ResidentChunkBacking<B>>,
    pub(super) reason: DynamicChunkQuarantineReason,
}

pub(super) struct DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    instance_id: u64,
    pub(super) domain: DynamicPoolDomainSpec,
    logical_admission: LogicalAdmissionCoordinator,
    maintenance: Mutex<()>,
    next_extent_generation: AtomicU64,
    pub(super) state: Mutex<DynamicBackingPoolState<R::Buffer>>,
}

struct PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    bytes: u64,
    armed: bool,
}

impl<R> PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<R> Drop for PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.armed {
            self.pool.cancel_pending_growth(self.bytes);
        }
    }
}

trait BackingExtentOwner: Send + Sync {
    fn instance_id(&self) -> u64;
    fn release_segments(&self, segments: &[BackingSegment]);
}

pub(super) struct BackingSegmentLease {
    owner: Arc<dyn BackingExtentOwner>,
    owner_instance_id: u64,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) segment_generation: u64,
    segments: Vec<BackingSegment>,
    pub(super) size_bytes: u64,
    released: bool,
}

impl Drop for BackingSegmentLease {
    fn drop(&mut self) {
        if !self.released {
            self.owner.release_segments(&self.segments);
            self.released = true;
        }
    }
}

impl<R> BackingExtentOwner for DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    fn release_segments(&self, segments: &[BackingSegment]) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                return;
            }
        };
        if state.poisoned {
            return;
        }
        for segment in segments {
            if segment.pool_id() != self.domain.pool_id() {
                state.poisoned = true;
                return;
            }
            let Some(chunk) = state.chunks.get_mut(&segment.chunk.ordinal) else {
                state.poisoned = true;
                return;
            };
            if chunk.backing.identity != segment.chunk || chunk.live_segments == 0 {
                state.poisoned = true;
                return;
            }
        }
        for segment in segments {
            if state.allocator.release(segment).is_err() {
                state.poisoned = true;
                return;
            }
            let chunk = state
                .chunks
                .get_mut(&segment.chunk_ordinal())
                .expect("validated released chunk remains installed");
            chunk.live_segments -= 1;
        }
        drop(state);
        if self
            .logical_admission
            .notify_domain_availability_changed(self.domain.domain_id)
            .is_err()
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.poisoned = true;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthReceipt {
    pool_id: DynamicBackingPoolId,
    chunk: BackingChunkIdentity,
    chunk_bytes: u64,
    published_capacity_bytes: u64,
    capacity_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolStatus {
    pool_id: DynamicBackingPoolId,
    domain_id: CapacityDomainId,
    storage_profile: DynamicStorageProfile,
    resident_bytes: u64,
    pending_growth_bytes: u64,
    free_bytes: u64,
    largest_contiguous_bytes: u64,
    resident_chunks: usize,
    live_segments: u64,
    quarantined_chunks: usize,
    quarantined_bytes: u64,
    descriptor_mismatch_chunks: usize,
    publication_rejected_chunks: usize,
    poisoned: bool,
}

impl DynamicPoolStatus {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn pending_growth_bytes(&self) -> u64 {
        self.pending_growth_bytes
    }

    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub const fn largest_contiguous_bytes(&self) -> u64 {
        self.largest_contiguous_bytes
    }

    pub const fn resident_chunks(&self) -> usize {
        self.resident_chunks
    }

    pub const fn live_segments(&self) -> u64 {
        self.live_segments
    }

    pub const fn quarantined_chunks(&self) -> usize {
        self.quarantined_chunks
    }

    pub const fn quarantined_bytes(&self) -> u64 {
        self.quarantined_bytes
    }

    pub const fn descriptor_mismatch_chunks(&self) -> usize {
        self.descriptor_mismatch_chunks
    }

    pub const fn publication_rejected_chunks(&self) -> usize {
        self.publication_rejected_chunks
    }

    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceStatus {
    epochs: CapacityEpochs,
    device_capacity_bytes: u64,
    effective_device_usable_ceiling_bytes: u64,
    process_claimed_bytes: u64,
    budget_device_wide_usable_ceiling_bytes: u64,
    budget_claimed_bytes: u64,
    pools: Vec<DynamicPoolStatus>,
}

impl DynamicPoolMaintenanceStatus {
    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn effective_device_usable_ceiling_bytes(&self) -> u64 {
        self.effective_device_usable_ceiling_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn budget_device_wide_usable_ceiling_bytes(&self) -> u64 {
        self.budget_device_wide_usable_ceiling_bytes
    }

    pub const fn budget_claimed_bytes(&self) -> u64 {
        self.budget_claimed_bytes
    }

    pub fn pools(&self) -> &[DynamicPoolStatus] {
        &self.pools
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DynamicDeferredMaintenanceOutcome {
    RetryWithoutGrowth {
        current_epochs: CapacityEpochs,
    },
    WaitForRelease {
        current_epochs: CapacityEpochs,
        pressure: DeviceCapacityPressure,
    },
    Maintained(DynamicPoolGrowthBatchReceipt),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineRelease {
    pool_id: DynamicBackingPoolId,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineRelease {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineReleaseReceipt {
    pools: Vec<DynamicPoolQuarantineRelease>,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineReleaseReceipt {
    pub fn pools(&self) -> &[DynamicPoolQuarantineRelease] {
        &self.pools
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

/// Plan-owner capability for changing physical dynamic-pool residency. It is
/// created once during provisioning, is intentionally not `Clone`, and cannot
/// be derived from any request, sequence, step, invocation, or static lease.
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{AdmittedRequestResources, DeviceRuntime};
/// fn request_cannot_mint<R: DeviceRuntime>(request: &AdmittedRequestResources<R>) {
///     let _ = request.dynamic_pool_maintenance_controller();
/// }
/// ```
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningLease};
/// fn lease_cannot_mint<R: DeviceRuntime>(lease: &StaticProvisioningLease<R>) {
///     let _ = lease.dynamic_pool_maintenance_controller();
/// }
/// ```
#[must_use = "dynamic pool maintenance controller must be retained by the plan owner"]
pub struct DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    pools: Arc<DynamicPoolSet<R>>,
}

impl<R> DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(pools: Arc<DynamicPoolSet<R>>) -> Self {
        Self { pools }
    }

    pub fn pool_ids(&self) -> impl ExactSizeIterator<Item = &DynamicBackingPoolId> {
        self.pools.pools.keys()
    }

    /// Returns one exact domain-to-pool and physical-capacity snapshot. The
    /// process-wide usable ceiling is the conservative minimum across all live
    /// plan budgets; plan claims remain separately visible.
    pub fn status(&self) -> Result<DynamicPoolMaintenanceStatus, VNextError> {
        let mut pools = Vec::with_capacity(self.pools.pools.len());
        for pool in self.pools.pools.values() {
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            let live_segments = state.chunks.values().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.live_segments)
                    .ok_or_else(|| invalid_resource("dynamic live segment count overflows u64"))
            })?;
            let quarantined_bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.backing._grant.bytes())
                    .ok_or_else(|| invalid_resource("dynamic quarantine bytes overflow u64"))
            })?;
            pools.push(DynamicPoolStatus {
                pool_id: pool.domain.pool_id().clone(),
                domain_id: pool.domain.domain_id,
                storage_profile: pool.domain.pool.compatibility().profile(),
                resident_bytes: state.resident_bytes,
                pending_growth_bytes: state.pending_growth_bytes,
                free_bytes: state.allocator.free_bytes,
                largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                resident_chunks: state.chunks.len(),
                live_segments,
                quarantined_chunks: state.quarantined.len(),
                quarantined_bytes,
                descriptor_mismatch_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::DescriptorMismatch
                    })
                    .count(),
                publication_rejected_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::PublicationRejected
                    })
                    .count(),
                poisoned: state.poisoned,
            });
        }
        let account = &self.pools.budget.account;
        let state = account
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        let effective_device_usable_ceiling_bytes = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .ok_or_else(|| invalid_resource("device capacity account has no live budget"))?;
        let budget_claimed_bytes = state
            .budgets
            .get(&self.pools.budget.budget_id)
            .ok_or_else(|| invalid_resource("dynamic pool plan budget is stale"))?
            .claimed_bytes;
        Ok(DynamicPoolMaintenanceStatus {
            epochs: self.pools.logical_admission.epochs()?,
            device_capacity_bytes: account.device_capacity_bytes,
            effective_device_usable_ceiling_bytes,
            process_claimed_bytes: state.claimed_bytes,
            budget_device_wide_usable_ceiling_bytes: self
                .pools
                .budget
                .device_wide_usable_ceiling_bytes,
            budget_claimed_bytes,
            pools,
        })
    }

    /// Makes the core-derived runnable minimum resident. A second call is a
    /// no-op and returns `None`; it never creates a duplicate initial chunk.
    pub fn initialize_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
    ) -> Result<Option<DynamicPoolGrowthReceipt>, VNextError> {
        let mut receipt = self
            .pools
            .maintain_pools(vec![DynamicPoolGrowthIntent::Minimum(pool_id.clone())])?;
        Ok(receipt.growths.pop())
    }

    /// Initializes several pools in one canonical all-or-nothing publication.
    pub fn initialize_pools(
        &self,
        pool_ids: &[DynamicBackingPoolId],
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            pool_ids
                .iter()
                .cloned()
                .map(DynamicPoolGrowthIntent::Minimum)
                .collect(),
        )
    }

    /// Adds one explicitly-sized resident chunk. The physical allocation is
    /// globally reserved before the runtime is called and published only after
    /// allocation, descriptor validation, and installation all succeed.
    pub fn grow_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
        requested_bytes: u64,
    ) -> Result<DynamicPoolGrowthReceipt, VNextError> {
        let request = DynamicPoolGrowthRequest::new(pool_id.clone(), requested_bytes)?;
        let mut receipt = self.grow_pools(vec![request])?;
        receipt
            .growths
            .pop()
            .ok_or_else(|| invalid_resource("single-pool growth produced no receipt"))
    }

    /// Grows distinct pools under one global reservation and one capacity
    /// epoch publication. Input order is ignored; duplicate pools are rejected.
    pub fn grow_pools(
        &self,
        requests: Vec<DynamicPoolGrowthRequest>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            requests
                .into_iter()
                .map(DynamicPoolGrowthIntent::Additional)
                .collect(),
        )
    }

    /// Resolves a still-current physical deferral by pool identity. If any
    /// release or capacity event happened since observation, no allocation is
    /// performed and the caller must retry admission against the new state.
    pub fn maintain_for_deferred(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let current_epochs = self.pools.logical_admission.epochs()?;
        if current_epochs.coordinator_id() != deferred.epochs().coordinator_id() {
            return Err(invalid_resource(
                "dynamic backing deferral belongs to another admission coordinator",
            ));
        }
        if current_epochs != deferred.epochs() {
            return Ok(DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth { current_epochs });
        }
        if deferred.blockers().is_empty() {
            return Err(invalid_resource(
                "dynamic backing deferral contains no blocking pool",
            ));
        }
        let mut requested_by_pool = BTreeMap::<DynamicBackingPoolId, u64>::new();
        for blocker in deferred.blockers() {
            if !self.pools.pools.contains_key(blocker.pool_id()) {
                return Err(invalid_resource(
                    "dynamic backing deferral belongs to another plan owner",
                ));
            }
            requested_by_pool
                .entry(blocker.pool_id().clone())
                .and_modify(|bytes| *bytes = (*bytes).max(blocker.requested_bytes()))
                .or_insert(blocker.requested_bytes());
        }
        let growth = self.grow_pools(
            requested_by_pool
                .into_iter()
                .map(|(pool_id, bytes)| DynamicPoolGrowthRequest::new(pool_id, bytes))
                .collect::<Result<Vec<_>, _>>()?,
        );
        match growth {
            Ok(receipt) => Ok(DynamicDeferredMaintenanceOutcome::Maintained(receipt)),
            Err(VNextError::DeviceCapacityUnavailable(pressure))
                if pressure.scope() == &DeviceCapacityPressureScope::PlanBudget =>
            {
                let after_pressure = self.pools.logical_admission.epochs()?;
                if after_pressure != current_epochs {
                    Ok(DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth {
                        current_epochs: after_pressure,
                    })
                } else {
                    Ok(DynamicDeferredMaintenanceOutcome::WaitForRelease {
                        current_epochs: after_pressure,
                        pressure,
                    })
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Materializes fit capacity that logical admission identified as
    /// growable but that immediate backing preparation does not yet claim.
    pub fn maintain_for_admission_deferred(
        &self,
        deferred: &AdmissionDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let current_epochs = self.pools.logical_admission.epochs()?;
        if current_epochs.coordinator_id() != deferred.epochs().coordinator_id() {
            return Err(invalid_resource(
                "logical admission deferral belongs to another coordinator",
            ));
        }
        if deferred.action() != DeferredAction::AwaitBackingGrowth {
            return Err(invalid_resource(
                "logical admission deferral does not request backing growth",
            ));
        }
        let pools_by_domain = self
            .pools
            .pools
            .values()
            .map(|pool| (pool.domain.domain_id, pool.domain.pool_id().clone()))
            .collect::<BTreeMap<_, _>>();
        let current = self.pools.logical_admission.snapshot()?;
        let mut requested_by_pool = BTreeMap::<DynamicBackingPoolId, u64>::new();
        for blocker in deferred
            .blockers()
            .iter()
            .filter(|blocker| blocker.kind() == CapacityShortfallKind::BackingGrowthRequired)
        {
            let domain = blocker.domain().ok_or_else(|| {
                invalid_resource("backing-growth blocker contains no capacity domain")
            })?;
            let pool_id = pools_by_domain.get(&domain).ok_or_else(|| {
                invalid_resource("backing-growth blocker references a non-pool domain")
            })?;
            let current_total = current
                .domains()
                .iter()
                .find(|snapshot| snapshot.domain() == domain)
                .ok_or_else(|| {
                    invalid_resource("backing-growth blocker references an unknown domain")
                })?
                .total()
                .get();
            let missing = blocker.requested().get().saturating_sub(current_total);
            if missing == 0 {
                continue;
            }
            requested_by_pool
                .entry(pool_id.clone())
                .and_modify(|bytes| *bytes = (*bytes).max(missing))
                .or_insert(missing);
        }
        if requested_by_pool.is_empty() {
            return Ok(DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth { current_epochs });
        }
        let growth = self.grow_pools(
            requested_by_pool
                .into_iter()
                .map(|(pool_id, bytes)| DynamicPoolGrowthRequest::new(pool_id, bytes))
                .collect::<Result<Vec<_>, _>>()?,
        );
        match growth {
            Ok(receipt) => Ok(DynamicDeferredMaintenanceOutcome::Maintained(receipt)),
            Err(VNextError::DeviceCapacityUnavailable(pressure))
                if pressure.scope() == &DeviceCapacityPressureScope::PlanBudget =>
            {
                let after_pressure = self.pools.logical_admission.epochs()?;
                if after_pressure != current_epochs {
                    Ok(DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth {
                        current_epochs: after_pressure,
                    })
                } else {
                    Ok(DynamicDeferredMaintenanceOutcome::WaitForRelease {
                        current_epochs: after_pressure,
                        pressure,
                    })
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Explicitly releases only unclaimable quarantined chunks. Resident
    /// chunks and logical totals are unchanged; the returned grants are
    /// dropped after all pool locks have been released.
    pub fn release_quarantined_chunks(
        &self,
    ) -> Result<DynamicPoolQuarantineReleaseReceipt, VNextError> {
        let pools = self.pools.pools.values().cloned().collect::<Vec<_>>();
        let _maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = pools
            .iter()
            .map(|pool| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pool_totals = states
            .iter()
            .map(|state| {
                let bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                    total
                        .checked_add(chunk.backing._grant.bytes())
                        .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
                })?;
                Ok((state.quarantined.len(), bytes))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let released_chunks = pool_totals.iter().try_fold(0_usize, |total, (count, _)| {
            total
                .checked_add(*count)
                .ok_or_else(|| invalid_resource("released quarantine count overflows usize"))
        })?;
        let released_bytes = pool_totals.iter().try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
        })?;
        let mut released = Vec::with_capacity(pools.len());
        let mut receipts = Vec::new();
        for ((pool, state), (pool_chunks, pool_bytes)) in
            pools.iter().zip(states.iter_mut()).zip(pool_totals)
        {
            if pool_chunks == 0 {
                continue;
            }
            let chunks = std::mem::take(&mut state.quarantined);
            debug_assert_eq!(chunks.len(), pool_chunks);
            receipts.push(DynamicPoolQuarantineRelease {
                pool_id: pool.domain.pool_id().clone(),
                released_chunks: pool_chunks,
                released_bytes: pool_bytes,
            });
            released.push(chunks);
        }
        drop(states);
        drop(released);
        Ok(DynamicPoolQuarantineReleaseReceipt {
            pools: receipts,
            released_chunks,
            released_bytes,
        })
    }
}

impl DynamicPoolGrowthReceipt {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn chunk(&self) -> &BackingChunkIdentity {
        &self.chunk
    }

    pub const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub const fn published_capacity_bytes(&self) -> u64 {
        self.published_capacity_bytes
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthRequest {
    pool_id: DynamicBackingPoolId,
    requested_bytes: u64,
}

impl DynamicPoolGrowthRequest {
    pub fn new(pool_id: DynamicBackingPoolId, requested_bytes: u64) -> Result<Self, VNextError> {
        if requested_bytes == 0 {
            return Err(invalid_resource(
                "dynamic pool growth must request non-zero bytes",
            ));
        }
        Ok(Self {
            pool_id,
            requested_bytes,
        })
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthBatchReceipt {
    growths: Vec<DynamicPoolGrowthReceipt>,
    capacity_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicBackingDeferralReason {
    GrowthRequired,
    FragmentedContiguous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingBlocker {
    pool_id: DynamicBackingPoolId,
    reason: DynamicBackingDeferralReason,
    requested_bytes: u64,
    free_bytes: u64,
    largest_contiguous_bytes: u64,
}

impl DynamicBackingBlocker {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn reason(&self) -> DynamicBackingDeferralReason {
        self.reason
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub const fn largest_contiguous_bytes(&self) -> u64 {
        self.largest_contiguous_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingDeferred {
    blockers: Vec<DynamicBackingBlocker>,
    epochs: CapacityEpochs,
}

impl DynamicBackingDeferred {
    pub fn blockers(&self) -> &[DynamicBackingBlocker] {
        &self.blockers
    }

    pub const fn release_epoch(&self) -> u64 {
        self.epochs.release_epoch()
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.epochs.capacity_epoch()
    }

    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }
}

impl DynamicPoolGrowthBatchReceipt {
    pub fn growths(&self) -> &[DynamicPoolGrowthReceipt] {
        &self.growths
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }
}

enum DynamicPoolGrowthIntent {
    Additional(DynamicPoolGrowthRequest),
    Minimum(DynamicBackingPoolId),
}

impl DynamicPoolGrowthIntent {
    fn pool_id(&self) -> &DynamicBackingPoolId {
        match self {
            Self::Additional(request) => request.pool_id(),
            Self::Minimum(pool_id) => pool_id,
        }
    }
}

struct PlannedDynamicGrowth<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    chunk: BackingChunkIdentity,
    expected_resource_id: ResourceId,
    chunk_bytes: u64,
}

struct AllocatedDynamicGrowth<B> {
    backing: Arc<ResidentChunkBacking<B>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct PhysicalBackingClaimIdentity {
    pool_id: DynamicBackingPoolId,
    resource_ids: Vec<ResourceId>,
}

impl PhysicalBackingClaimIdentity {
    pub(super) fn new(
        pool_id: DynamicBackingPoolId,
        mut resource_ids: Vec<ResourceId>,
    ) -> Result<Self, VNextError> {
        resource_ids.sort();
        if resource_ids.is_empty() || resource_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid_resource(
                "physical backing claim identity requires unique logical resources",
            ));
        }
        Ok(Self {
            pool_id,
            resource_ids,
        })
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }

    pub const fn is_shared(&self) -> bool {
        self.resource_ids.len() > 1
    }
}

pub(super) struct EvaluatedBackingProjection<'a> {
    pub(super) descriptor: &'a DynamicResourceDescriptor,
    pub(super) physical_offset_bytes: u64,
    pub(super) size_bytes: u64,
}

pub(super) struct EvaluatedBackingRequest<'a> {
    pub(super) domain: &'a DynamicPoolDomainSpec,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) size_bytes: u64,
    pub(super) projections: Vec<EvaluatedBackingProjection<'a>>,
}

struct PreparedBackingExtent<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    claim_identity: PhysicalBackingClaimIdentity,
    segment_generation: u64,
    segments: Vec<BackingSegment>,
    size_bytes: u64,
    projections: Vec<LogicalBackingSliceEvidence>,
}

pub(super) struct PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    extents: Vec<PreparedBackingExtent<R>>,
    committed: bool,
}

impl<R> PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    fn empty() -> Self {
        Self {
            extents: Vec::new(),
            committed: false,
        }
    }

    pub(super) fn commit(mut self) -> Vec<LogicalBackingSliceAuthority> {
        let mut slices = Vec::new();
        for extent in std::mem::take(&mut self.extents) {
            let owner: Arc<dyn BackingExtentOwner> = extent.pool;
            let segment_lease = Arc::new(BackingSegmentLease {
                owner_instance_id: owner.instance_id(),
                owner,
                claim_identity: extent.claim_identity,
                segment_generation: extent.segment_generation,
                segments: extent.segments,
                size_bytes: extent.size_bytes,
                released: false,
            });
            slices.extend(extent.projections.into_iter().map(|evidence| {
                LogicalBackingSliceAuthority {
                    evidence,
                    segment_lease: Arc::clone(&segment_lease),
                }
            }));
        }
        slices.sort_by(|left, right| left.resource_id().cmp(right.resource_id()));
        self.committed = true;
        slices
    }
}

impl<R> Drop for PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for extent in self.extents.iter().rev() {
            extent.pool.rollback_prepared(&extent.segments);
        }
    }
}

pub(super) enum BackingPrepareDecision<R>
where
    R: DeviceRuntime,
{
    Prepared(PreparedBackingClaim<R>),
    Deferred(DynamicBackingDeferred),
}

pub(super) struct DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pub(super) pools: BTreeMap<DynamicBackingPoolId, Arc<DynamicBackingPool<R>>>,
    pub(super) domains: Vec<DynamicPoolDomainSpec>,
    pub(super) nodes: Arc<[PlanNode]>,
    pub(super) logical_admission: LogicalAdmissionCoordinator,
    pub(super) budget: Arc<DeviceCapacityBudget>,
    binding: StaticProvisioningBinding,
    // Backend context must outlive every resident/quarantined buffer above.
    runtime: Arc<R>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogicalBackingSliceEvidence {
    pub(super) domain_id: CapacityDomainId,
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) resource_id: ResourceId,
    pub(super) pool_instance_id: u64,
    pub(super) physical_claim_identity: PhysicalBackingClaimIdentity,
    pub(super) segment_generation: u64,
    pub(super) segments: Vec<BackingSegment>,
    pub(super) physical_offset_bytes: u64,
    pub(super) size_bytes: u64,
    pub(super) physical_size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) storage_profile: DynamicStorageProfile,
}

impl LogicalBackingSliceEvidence {
    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn pool_instance_id(&self) -> u64 {
        self.pool_instance_id
    }

    pub const fn segment_generation(&self) -> u64 {
        self.segment_generation
    }

    pub fn physical_claim_identity(&self) -> &PhysicalBackingClaimIdentity {
        &self.physical_claim_identity
    }

    pub fn segments(&self) -> &[BackingSegment] {
        &self.segments
    }

    pub const fn physical_offset_bytes(&self) -> u64 {
        self.physical_offset_bytes
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub const fn physical_size_bytes(&self) -> u64 {
        self.physical_size_bytes
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

#[must_use = "a logical backing authority owns its physical arena extents"]
pub struct LogicalBackingSliceAuthority {
    pub(super) evidence: LogicalBackingSliceEvidence,
    pub(super) segment_lease: Arc<BackingSegmentLease>,
}

impl LogicalBackingSliceAuthority {
    pub fn evidence(&self) -> &LogicalBackingSliceEvidence {
        &self.evidence
    }

    pub(super) fn retained(&self) -> Self {
        Self {
            evidence: self.evidence.clone(),
            segment_lease: Arc::clone(&self.segment_lease),
        }
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.evidence.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.evidence.resource_id
    }

    pub const fn size_bytes(&self) -> u64 {
        self.evidence.size_bytes
    }
}

pub struct LogicalBackingBufferView<'a, B> {
    pub(super) bindings: Vec<LogicalBackingSegmentBinding<B>>,
    authorities: &'a [LogicalBackingSliceAuthority],
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    storage_profile: DynamicStorageProfile,
}

pub(crate) struct LogicalBackingSegmentBinding<B> {
    pub(super) segment: BackingSegment,
    pub(super) chunk: Arc<ResidentChunkBacking<B>>,
}

impl<B> LogicalBackingSegmentBinding<B> {
    pub(crate) fn segment(&self) -> &BackingSegment {
        &self.segment
    }

    pub(crate) fn chunk(&self) -> &BackingChunkIdentity {
        self.segment.chunk()
    }

    pub(crate) fn buffer(&self) -> &B {
        &self.chunk.buffer
    }

    pub(crate) fn descriptor(&self) -> &BufferDescriptor {
        &self.chunk.descriptor
    }
}

impl<'a, B> LogicalBackingBufferView<'a, B> {
    pub(crate) fn segment_bindings(&self) -> &[LogicalBackingSegmentBinding<B>] {
        &self.bindings
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
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

    pub fn committed_evidence_segments(&self) -> impl Iterator<Item = &BackingSegment> {
        self.authorities
            .iter()
            .flat_map(|authority| authority.evidence.segments())
    }

    /// Compatibility accessor for callers that construct a single-slice view.
    /// Multi-extent callers must use the aggregate metadata and segment iterator.
    pub fn slice(&self) -> &'a LogicalBackingSliceEvidence {
        &self
            .authorities
            .first()
            .expect("logical backing views always contain an authority")
            .evidence
    }
}

impl<R> DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(
        runtime: Arc<R>,
        binding: StaticProvisioningBinding,
        budget: Arc<DeviceCapacityBudget>,
        logical_admission: LogicalAdmissionCoordinator,
        domains: Vec<DynamicPoolDomainSpec>,
        nodes: Arc<[PlanNode]>,
    ) -> Result<Self, VNextError> {
        let mut pools = BTreeMap::new();
        for domain in &domains {
            let instance_id = NEXT_DYNAMIC_POOL_INSTANCE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_add(1)
                })
                .map_err(|_| invalid_resource("dynamic pool instance id space is exhausted"))?;
            let pool = Arc::new(DynamicBackingPool {
                instance_id,
                domain: domain.clone(),
                logical_admission: logical_admission.clone(),
                maintenance: Mutex::new(()),
                next_extent_generation: AtomicU64::new(1),
                state: Mutex::new(DynamicBackingPoolState {
                    resident_bytes: 0,
                    pending_growth_bytes: 0,
                    next_chunk_ordinal: 1,
                    next_chunk_generation: 1,
                    chunks: BTreeMap::new(),
                    allocator: FreeExtentIndex::default(),
                    quarantined: Vec::new(),
                    poisoned: false,
                }),
            });
            if pools.insert(domain.pool_id().clone(), pool).is_some() {
                return Err(invalid_resource(
                    "dynamic pool set contains a duplicate pool",
                ));
            }
        }
        Ok(Self {
            runtime,
            binding,
            budget,
            logical_admission,
            domains,
            pools,
            nodes,
        })
    }

    fn maintain_pools(
        &self,
        mut intents: Vec<DynamicPoolGrowthIntent>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        intents.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
        if intents
            .windows(2)
            .any(|pair| pair[0].pool_id() == pair[1].pool_id())
        {
            return Err(invalid_resource(
                "dynamic maintenance batch contains a duplicate pool",
            ));
        }
        let pools = intents
            .iter()
            .map(|intent| {
                self.pools.get(intent.pool_id()).cloned().ok_or_else(|| {
                    invalid_resource("dynamic maintenance references an unknown pool")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut planned = Vec::with_capacity(intents.len());
        let mut pending = Vec::with_capacity(intents.len());
        for (intent, pool) in intents.iter().zip(&pools) {
            let requested_bytes = {
                let state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                match intent {
                    DynamicPoolGrowthIntent::Additional(request) => request.requested_bytes(),
                    DynamicPoolGrowthIntent::Minimum(_) => {
                        let current = state
                            .resident_bytes
                            .checked_add(state.pending_growth_bytes)
                            .ok_or_else(|| {
                                invalid_resource("dynamic pool residency overflows u64")
                            })?;
                        let minimum = pool.domain.pool.provisioning().minimum_resident_bytes();
                        if current >= minimum {
                            continue;
                        }
                        minimum - current
                    }
                }
            };
            let chunk_bytes = align_up_resource(requested_bytes, pool.allocation_quantum())?;
            let chunk = {
                let mut state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let next_residency = state
                    .resident_bytes
                    .checked_add(state.pending_growth_bytes)
                    .and_then(|bytes| bytes.checked_add(chunk_bytes))
                    .ok_or_else(|| invalid_resource("dynamic pool resident bytes overflow u64"))?;
                if next_residency > pool.domain.pool.provisioning().maximum_resident_bytes() {
                    return Err(invalid_resource(
                        "dynamic pool growth exceeds its core-derived resident maximum",
                    ));
                }
                let ordinal = state.next_chunk_ordinal;
                let generation = state.next_chunk_generation;
                let next_ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid_resource("dynamic chunk ordinal space is exhausted"))?;
                let next_generation = generation.checked_add(1).ok_or_else(|| {
                    invalid_resource("dynamic chunk generation space is exhausted")
                })?;
                let next_pending = state
                    .pending_growth_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| invalid_resource("pending dynamic growth bytes overflow u64"))?;
                state.next_chunk_ordinal = next_ordinal;
                state.next_chunk_generation = next_generation;
                state.pending_growth_bytes = next_pending;
                BackingChunkIdentity {
                    pool_id: pool.domain.pool_id().clone(),
                    ordinal,
                    generation,
                }
            };
            pending.push(PendingGrowthGuard {
                pool: Arc::clone(pool),
                bytes: chunk_bytes,
                armed: true,
            });
            let expected_resource_id = ResourceId::new(format!(
                "{}/chunk/{}/{}",
                pool.domain.pool_id().as_str(),
                chunk.ordinal,
                chunk.generation
            ))?;
            planned.push(PlannedDynamicGrowth {
                pool: Arc::clone(pool),
                chunk,
                expected_resource_id,
                chunk_bytes,
            });
        }
        if planned.is_empty() {
            return Ok(DynamicPoolGrowthBatchReceipt {
                growths: Vec::new(),
                capacity_epoch: self.logical_admission.epochs()?.capacity_epoch(),
            });
        }

        let total_bytes = planned.iter().try_fold(0_u64, |total, growth| {
            total
                .checked_add(growth.chunk_bytes)
                .ok_or_else(|| invalid_resource("dynamic maintenance batch bytes overflow u64"))
        })?;
        let reservation = DeviceCapacityReservation::reserve(&self.budget, total_bytes)?;
        let grants = reservation.commit_split(
            &planned
                .iter()
                .map(|growth| growth.chunk_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut allocated = Vec::with_capacity(planned.len());
        for (growth, grant) in planned.iter().zip(grants) {
            let transaction_identity = ResourceTransactionIdentity {
                pool_id: self.binding.pool_id(),
                run_id: RunId::new(format!("dynamic-grow-{}", growth.chunk.generation))?,
                transaction_id: TransactionId::new(format!(
                    "dynamic-grow-{}-{}",
                    growth.chunk.ordinal, growth.chunk.generation
                ))?,
                request_id: self.binding.request_id().clone(),
            };
            let reservation_evidence = ResourceReservation {
                resource_id: growth.expected_resource_id.clone(),
                request_id: self.binding.request_id().clone(),
                owner_node_id: None,
                size_bytes: growth.chunk_bytes,
                alignment_bytes: growth.pool.domain.pool.compatibility().alignment_bytes(),
                usage: growth.pool.domain.pool.compatibility().usage(),
                element_type: growth.pool.domain.pool.compatibility().element_type(),
                retention_policy: ResourceRetentionPolicy::Plan,
                backing_domain_id: Some(growth.pool.domain.domain_id),
                generation: growth.chunk.generation,
            };
            let request = BufferRequest::new(
                growth.expected_resource_id.clone(),
                growth.chunk_bytes,
                reservation_evidence.alignment_bytes,
                reservation_evidence.usage,
                reservation_evidence.element_type,
            )?;
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth preflight",
            )?;
            let buffer = self
                .runtime
                .allocate(DeviceAllocationPermit {
                    identity: &transaction_identity,
                    binding: &self.binding,
                    reservation: &reservation_evidence,
                    request: &request,
                    seal: AllocationSeal,
                })
                .map_err(|error| {
                    invalid_resource(format!("dynamic pool device allocation failed: {error}"))
                })?;
            let actual_descriptor = self.runtime.buffer_descriptor(&buffer);
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth completion",
            )?;
            if grant.bytes() != growth.chunk_bytes {
                return Err(invalid_resource(
                    "dynamic pool capacity grant differs from its chunk",
                ));
            }
            let backing = Arc::new(ResidentChunkBacking {
                buffer,
                _grant: grant,
                identity: growth.chunk.clone(),
                descriptor: actual_descriptor.clone(),
            });
            if !reservation_evidence.matches_descriptor(&actual_descriptor) {
                let mut state = growth
                    .pool
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.quarantined.push(QuarantinedDynamicChunk {
                    backing,
                    reason: DynamicChunkQuarantineReason::DescriptorMismatch,
                });
                return Err(invalid_resource(
                    "dynamic chunk descriptor mismatch was quarantined without capacity publication",
                ));
            }
            allocated.push(AllocatedDynamicGrowth { backing });
        }

        let mut states = planned
            .iter()
            .map(|growth| {
                growth.pool.state.lock().map_err(|_| {
                    invalid_resource("dynamic backing pool is poisoned after allocation")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut published_totals = Vec::with_capacity(planned.len());
        for ((growth, allocation), state) in planned.iter().zip(&allocated).zip(&states) {
            if state.poisoned
                || state.pending_growth_bytes < growth.chunk_bytes
                || state.chunks.contains_key(&growth.chunk.ordinal)
                || state
                    .allocator
                    .by_offset
                    .range((growth.chunk.ordinal, 0)..=(growth.chunk.ordinal, u64::MAX))
                    .next()
                    .is_some()
                || state
                    .allocator
                    .free_bytes
                    .checked_add(growth.chunk_bytes)
                    .is_none()
                || allocation.backing.identity != growth.chunk
            {
                return Err(invalid_resource(
                    "dynamic batch installation preconditions changed before publication",
                ));
            }
            published_totals.push(
                state
                    .resident_bytes
                    .checked_add(growth.chunk_bytes)
                    .ok_or_else(|| invalid_resource("published dynamic capacity overflows u64"))?,
            );
        }
        for index in 0..planned.len() {
            let growth = &planned[index];
            let state = &mut states[index];
            state.pending_growth_bytes -= growth.chunk_bytes;
            pending[index].disarm();
            state
                .allocator
                .insert_extent(
                    growth.chunk.ordinal,
                    growth.chunk.generation,
                    0,
                    growth.chunk_bytes,
                )
                .expect("validated new chunk has one disjoint free extent");
            state.chunks.insert(
                growth.chunk.ordinal,
                ResidentChunkState {
                    backing: Arc::clone(&allocated[index].backing),
                    live_segments: 0,
                },
            );
        }
        let updates = planned
            .iter()
            .zip(&published_totals)
            .map(|(growth, &total)| (growth.pool.domain.domain_id, CapacityUnits::new(total)))
            .collect::<Vec<_>>();
        let epochs = match self.logical_admission.set_domain_totals(&updates) {
            Ok(epochs) => epochs,
            Err(error) => {
                for index in 0..planned.len() {
                    states[index]
                        .allocator
                        .remove_extent(planned[index].chunk.ordinal, 0)
                        .expect("unpublished dynamic chunk free extent remains installed");
                    let removed = states[index]
                        .chunks
                        .remove(&planned[index].chunk.ordinal)
                        .expect("unpublished dynamic chunk remains installed");
                    states[index].quarantined.push(QuarantinedDynamicChunk {
                        backing: removed.backing,
                        reason: DynamicChunkQuarantineReason::PublicationRejected,
                    });
                }
                return Err(error);
            }
        };
        for (state, &published_total) in states.iter_mut().zip(&published_totals) {
            state.resident_bytes = published_total;
        }
        Ok(DynamicPoolGrowthBatchReceipt {
            growths: planned
                .iter()
                .zip(published_totals)
                .map(
                    |(growth, published_capacity_bytes)| DynamicPoolGrowthReceipt {
                        pool_id: growth.pool.domain.pool_id().clone(),
                        chunk: growth.chunk.clone(),
                        chunk_bytes: growth.chunk_bytes,
                        published_capacity_bytes,
                        capacity_epoch: epochs.capacity_epoch(),
                    },
                )
                .collect(),
            capacity_epoch: epochs.capacity_epoch(),
        })
    }

    pub(super) fn prepare_claim(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        if requests.is_empty() {
            return Ok(BackingPrepareDecision::Prepared(
                PreparedBackingClaim::empty(),
            ));
        }
        let mut grouped =
            BTreeMap::<DynamicBackingPoolId, Vec<&EvaluatedBackingRequest<'_>>>::new();
        for request in requests {
            grouped
                .entry(request.domain.pool_id().clone())
                .or_default()
                .push(request);
        }
        let mut groups = Vec::with_capacity(grouped.len());
        for (pool_id, mut requests) in grouped {
            requests.sort_by(|left, right| left.claim_identity.cmp(&right.claim_identity));
            if requests
                .windows(2)
                .any(|pair| pair[0].claim_identity == pair[1].claim_identity)
            {
                return Err(invalid_resource(
                    "dynamic backing reservation contains a duplicate physical claim",
                ));
            }
            let pool = self.pools.get(&pool_id).cloned().ok_or_else(|| {
                invalid_resource("dynamic backing reservation references an unknown pool")
            })?;
            groups.push((pool, requests));
        }
        let mut states = groups
            .iter()
            .map(|(pool, _)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (group_index, (pool, pool_requests)) in groups.iter().enumerate() {
            if states[group_index].poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            let quantum = pool.allocation_quantum();
            for request in pool_requests {
                let projection_ids = request
                    .projections
                    .iter()
                    .map(|projection| projection.descriptor.base_resource_id().clone())
                    .collect::<Vec<_>>();
                let single_projection = request.projections.len() == 1
                    && request.projections[0].physical_offset_bytes == 0
                    && request.projections[0].size_bytes == request.size_bytes;
                let shared_step_slot = request.projections.len() > 1
                    && request
                        .projections
                        .iter()
                        .all(|projection| projection.physical_offset_bytes == 0)
                    && request
                        .projections
                        .iter()
                        .map(|projection| projection.size_bytes)
                        .max()
                        == Some(request.size_bytes)
                    && pool.domain.pool.step_resource_slots().iter().any(|slot| {
                        slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave
                            && slot.resource_ids() == request.claim_identity.resource_ids()
                    });
                let invocation_wave = self.validate_invocation_wave_projection(pool, request)?;
                if request.domain.pool_id() != pool.domain.pool_id()
                    || request.claim_identity.pool_id() != pool.domain.pool_id()
                    || request.claim_identity.resource_ids() != projection_ids
                    || request.projections.is_empty()
                    || request.projections.windows(2).any(|pair| {
                        pair[0].descriptor.base_resource_id()
                            >= pair[1].descriptor.base_resource_id()
                    })
                    || request.projections.iter().any(|projection| {
                        projection.descriptor.pool_id() != pool.domain.pool_id()
                            || projection.size_bytes == 0
                            || projection.size_bytes % quantum != 0
                            || projection.physical_offset_bytes % quantum != 0
                            || projection
                                .physical_offset_bytes
                                .checked_add(projection.size_bytes)
                                .is_none_or(|end| end > request.size_bytes)
                            || !request
                                .domain
                                .descriptors
                                .iter()
                                .any(|descriptor| descriptor == projection.descriptor)
                    })
                    || request.size_bytes == 0
                    || request.size_bytes % quantum != 0
                    || !(single_projection || shared_step_slot || invocation_wave)
                {
                    return Err(invalid_resource(
                        "dynamic backing request violates its physical claim, projection, pool, or allocation quantum",
                    ));
                }
            }
        }
        let blockers = groups
            .iter()
            .enumerate()
            .map(|(group_index, (pool, pool_requests))| {
                let requested_bytes = pool_requests.iter().try_fold(0_u64, |total, request| {
                    total
                        .checked_add(request.size_bytes)
                        .ok_or_else(|| invalid_resource("dynamic backing batch bytes overflow u64"))
                })?;
                let state = &states[group_index];
                Ok(
                    (state.allocator.free_bytes < requested_bytes).then(|| DynamicBackingBlocker {
                        pool_id: pool.domain.pool_id().clone(),
                        reason: DynamicBackingDeferralReason::GrowthRequired,
                        requested_bytes: requested_bytes - state.allocator.free_bytes,
                        free_bytes: state.allocator.free_bytes,
                        largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                    }),
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !blockers.is_empty() {
            drop(states);
            return Ok(BackingPrepareDecision::Deferred(DynamicBackingDeferred {
                blockers,
                epochs: self.logical_admission.epochs()?,
            }));
        }
        let segment_generations = groups
            .iter()
            .map(|(pool, requests)| {
                (0..requests.len())
                    .map(|_| {
                        pool.next_extent_generation
                            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                                current.checked_add(1)
                            })
                            .map_err(|_| {
                                invalid_resource("dynamic extent generation space is exhausted")
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut selections = groups
            .iter()
            .map(|_| Vec::<(&EvaluatedBackingRequest<'_>, u64, Vec<BackingSegment>)>::new())
            .collect::<Vec<_>>();
        let mut journals = groups
            .iter()
            .map(|_| Vec::<Vec<BackingSegment>>::new())
            .collect::<Vec<_>>();
        for group_index in 0..groups.len() {
            let (pool, pool_requests) = &groups[group_index];
            let profile = pool.domain.pool.compatibility().profile();
            for request in pool_requests {
                let reserved = match match profile.view() {
                    DynamicStorageView::Contiguous => states[group_index]
                        .allocator
                        .allocate_contiguous(pool.domain.pool_id(), request.size_bytes)
                        .map(|segment| segment.map(|segment| vec![segment])),
                    DynamicStorageView::PagedRegions { block_bytes } => states[group_index]
                        .allocator
                        .allocate_paged(pool.domain.pool_id(), request.size_bytes, block_bytes),
                } {
                    Ok(reserved) => reserved,
                    Err(error) => {
                        states[group_index].poisoned = true;
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(error);
                    }
                };
                let Some(segments) = reserved else {
                    let blocker = DynamicBackingBlocker {
                        pool_id: pool.domain.pool_id().clone(),
                        reason: if states[group_index].allocator.free_bytes < request.size_bytes {
                            DynamicBackingDeferralReason::GrowthRequired
                        } else {
                            DynamicBackingDeferralReason::FragmentedContiguous
                        },
                        requested_bytes: request.size_bytes,
                        free_bytes: states[group_index].allocator.free_bytes,
                        largest_contiguous_bytes: states[group_index]
                            .allocator
                            .largest_contiguous_bytes(),
                    };
                    rollback_free_extent_journal(&mut states, &journals)?;
                    drop(states);
                    return Ok(BackingPrepareDecision::Deferred(DynamicBackingDeferred {
                        blockers: vec![blocker],
                        epochs: self.logical_admission.epochs()?,
                    }));
                };
                journals[group_index].push(segments.clone());
                let extent_bytes = match segments.iter().try_fold(0_u64, |total, segment| {
                    total.checked_add(segment.length_bytes()).ok_or_else(|| {
                        invalid_resource("dynamic backing extent bytes overflow u64")
                    })
                }) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(error);
                    }
                };
                if extent_bytes != request.size_bytes {
                    rollback_free_extent_journal(&mut states, &journals)?;
                    return Err(invalid_resource(
                        "dynamic backing extents differ from their exact logical claim",
                    ));
                }
                let generation = segment_generations[group_index][selections[group_index].len()];
                selections[group_index].push((request, generation, segments));
            }
        }

        for group_selections in &selections {
            for (request, _, segments) in group_selections {
                for projection in &request.projections {
                    if let Err(error) = backing_segment_range(
                        segments,
                        projection.physical_offset_bytes,
                        projection.size_bytes,
                    ) {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(error);
                    }
                }
            }
        }

        let increments = match (0..groups.len())
            .map(|group_index| {
                let mut increments = BTreeMap::<u32, u64>::new();
                for (_, _, segments) in &selections[group_index] {
                    for segment in segments {
                        let count = increments.entry(segment.chunk_ordinal()).or_default();
                        *count = count.checked_add(1).ok_or_else(|| {
                            invalid_resource("dynamic chunk live extent increment overflows u64")
                        })?;
                    }
                }
                for (&ordinal, &increment) in &increments {
                    states[group_index]
                        .chunks
                        .get(&ordinal)
                        .ok_or_else(|| invalid_resource("reserved dynamic chunk disappeared"))?
                        .live_segments
                        .checked_add(increment)
                        .ok_or_else(|| {
                            invalid_resource("dynamic chunk live extent count overflowed")
                        })?;
                }
                Ok(increments)
            })
            .collect::<Result<Vec<_>, VNextError>>()
        {
            Ok(increments) => increments,
            Err(error) => {
                rollback_free_extent_journal(&mut states, &journals)?;
                return Err(error);
            }
        };
        for (group_index, increments) in increments.into_iter().enumerate() {
            for (ordinal, increment) in increments {
                states[group_index]
                    .chunks
                    .get_mut(&ordinal)
                    .expect("validated reserved dynamic chunk remains installed")
                    .live_segments += increment;
            }
        }
        drop(states);
        let mut extents = Vec::new();
        for ((pool, _), selections) in groups.into_iter().zip(selections) {
            for (request, segment_generation, segments) in selections {
                let projections = request
                    .projections
                    .iter()
                    .map(|projection| {
                        Ok(LogicalBackingSliceEvidence {
                            domain_id: pool.domain.domain_id,
                            pool_id: pool.domain.pool_id().clone(),
                            resource_id: projection.descriptor.base_resource_id().clone(),
                            pool_instance_id: pool.instance_id,
                            physical_claim_identity: request.claim_identity.clone(),
                            segment_generation,
                            segments: backing_segment_range(
                                &segments,
                                projection.physical_offset_bytes,
                                projection.size_bytes,
                            )?,
                            physical_offset_bytes: projection.physical_offset_bytes,
                            size_bytes: projection.size_bytes,
                            physical_size_bytes: request.size_bytes,
                            alignment_bytes: projection.descriptor.alignment_bytes(),
                            usage: projection.descriptor.usage(),
                            element_type: projection.descriptor.element_type(),
                            storage_profile: pool.domain.pool.compatibility().profile(),
                        })
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?;
                extents.push(PreparedBackingExtent {
                    pool: Arc::clone(&pool),
                    claim_identity: request.claim_identity.clone(),
                    segment_generation,
                    segments,
                    size_bytes: request.size_bytes,
                    projections,
                });
            }
        }
        Ok(BackingPrepareDecision::Prepared(PreparedBackingClaim {
            extents,
            committed: false,
        }))
    }

    fn validate_invocation_wave_projection(
        &self,
        pool: &DynamicBackingPool<R>,
        request: &EvaluatedBackingRequest<'_>,
    ) -> Result<bool, VNextError> {
        let mode = pool.domain.pool.invocation_liveness_mode();
        if mode == InvocationLivenessMode::NoInvocationResources
            || request.projections.len() < 2
            || request.projections.iter().any(|projection| {
                projection.descriptor.lifetime() != super::AllocationLifetime::Invocation
            })
        {
            return Ok(false);
        }
        let mut expected_resources = pool
            .domain
            .pool
            .invocation_liveness()
            .iter()
            .flat_map(|row| row.resource_ids().iter().cloned())
            .collect::<Vec<_>>();
        expected_resources.sort();
        if expected_resources.windows(2).any(|pair| pair[0] == pair[1])
            || expected_resources != request.claim_identity.resource_ids()
        {
            return Ok(false);
        }
        let projections = request
            .projections
            .iter()
            .map(|projection| (projection.descriptor.base_resource_id().clone(), projection))
            .collect::<BTreeMap<_, _>>();
        if projections.len() != request.projections.len() {
            return Ok(false);
        }
        let rows_by_node = pool
            .domain
            .pool
            .invocation_liveness()
            .iter()
            .map(|row| (row.node_id(), row))
            .collect::<BTreeMap<_, _>>();
        let rows = self
            .nodes
            .iter()
            .filter_map(|node| rows_by_node.get(node.id()).copied())
            .collect::<Vec<_>>();
        if rows.len() != rows_by_node.len() {
            return Ok(false);
        }

        let mut concurrent_cursor = 0_u64;
        let mut peak = 0_u64;
        for row in rows {
            let row_base = match mode {
                InvocationLivenessMode::TotalOrderReuse => 0,
                InvocationLivenessMode::ConservativeConcurrent => concurrent_cursor,
                InvocationLivenessMode::NoInvocationResources => unreachable!(),
            };
            let mut row_cursor = 0_u64;
            for resource_id in row.resource_ids() {
                let Some(projection) = projections.get(resource_id) else {
                    return Ok(false);
                };
                let expected_offset = row_base.checked_add(row_cursor).ok_or_else(|| {
                    invalid_resource("invocation wave projection offset overflows u64")
                })?;
                if projection.physical_offset_bytes != expected_offset {
                    return Ok(false);
                }
                row_cursor = row_cursor
                    .checked_add(projection.size_bytes)
                    .ok_or_else(|| invalid_resource("invocation wave row size overflows u64"))?;
            }
            peak = peak.max(row_cursor);
            if mode == InvocationLivenessMode::ConservativeConcurrent {
                concurrent_cursor = concurrent_cursor.checked_add(row_cursor).ok_or_else(|| {
                    invalid_resource("concurrent invocation wave size overflows u64")
                })?;
            }
        }
        Ok(request.size_bytes
            == match mode {
                InvocationLivenessMode::TotalOrderReuse => peak,
                InvocationLivenessMode::ConservativeConcurrent => concurrent_cursor,
                InvocationLivenessMode::NoInvocationResources => 0,
            })
    }

    pub(super) fn view<'lease>(
        &'lease self,
        authority: &'lease LogicalBackingSliceAuthority,
    ) -> Result<LogicalBackingBufferView<'lease, R::Buffer>, VNextError> {
        self.view_many(std::slice::from_ref(authority))
    }

    pub(super) fn view_many<'lease>(
        &'lease self,
        authorities: &'lease [LogicalBackingSliceAuthority],
    ) -> Result<LogicalBackingBufferView<'lease, R::Buffer>, VNextError> {
        let first = authorities
            .first()
            .ok_or_else(|| invalid_resource("logical backing view requires an authority"))?;
        let pool = self
            .pools
            .get(&first.evidence.pool_id)
            .ok_or_else(|| invalid_resource("logical backing authority has no dynamic pool"))?;
        let mut size_bytes = 0_u64;
        let mut segment_count = 0_usize;
        for authority in authorities {
            if authority.evidence.pool_id != first.evidence.pool_id
                || authority.evidence.resource_id != first.evidence.resource_id
                || authority.evidence.storage_profile != first.evidence.storage_profile
                || authority.evidence.alignment_bytes != first.evidence.alignment_bytes
                || authority.evidence.usage != first.evidence.usage
                || authority.evidence.element_type != first.evidence.element_type
            {
                return Err(invalid_resource(
                    "logical backing authorities have incompatible resource metadata",
                ));
            }
            Self::validate_authority(pool, authority)?;
            size_bytes = size_bytes
                .checked_add(authority.evidence.size_bytes)
                .ok_or_else(|| invalid_resource("logical backing view size overflows u64"))?;
            segment_count = segment_count
                .checked_add(authority.evidence.segments.len())
                .ok_or_else(|| invalid_resource("logical backing segment count overflows usize"))?;
        }
        let state = pool
            .state
            .lock()
            .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("dynamic backing pool is fail-closed"));
        }
        let mut bindings = Vec::with_capacity(segment_count);
        for authority in authorities {
            for segment in &authority.evidence.segments {
                let chunk = state.chunks.get(&segment.chunk_ordinal()).ok_or_else(|| {
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
                bindings.push(LogicalBackingSegmentBinding {
                    segment: segment.clone(),
                    chunk: Arc::clone(&chunk.backing),
                });
            }
        }
        drop(state);
        Ok(LogicalBackingBufferView {
            bindings,
            authorities,
            size_bytes,
            alignment_bytes: first.evidence.alignment_bytes,
            usage: first.evidence.usage,
            element_type: first.evidence.element_type,
            storage_profile: first.evidence.storage_profile,
        })
    }

    fn validate_authority(
        pool: &DynamicBackingPool<R>,
        authority: &LogicalBackingSliceAuthority,
    ) -> Result<(), VNextError> {
        if pool.instance_id != authority.evidence.pool_instance_id
            || authority.segment_lease.owner_instance_id != pool.instance_id
            || authority.segment_lease.owner.instance_id() != pool.instance_id
            || authority.segment_lease.claim_identity != authority.evidence.physical_claim_identity
            || authority.segment_lease.segment_generation != authority.evidence.segment_generation
            || authority.segment_lease.size_bytes != authority.evidence.physical_size_bytes
            || authority.evidence.domain_id != pool.domain.domain_id
            || authority.evidence.physical_claim_identity.pool_id() != pool.domain.pool_id()
            || authority
                .evidence
                .physical_claim_identity
                .resource_ids()
                .binary_search(&authority.evidence.resource_id)
                .is_err()
            || authority
                .evidence
                .physical_offset_bytes
                .checked_add(authority.evidence.size_bytes)
                .is_none_or(|end| end > authority.evidence.physical_size_bytes)
            || authority.evidence.storage_profile != pool.domain.pool.compatibility().profile()
        {
            return Err(invalid_resource(
                "logical backing authority belongs to another dynamic pool instance",
            ));
        }
        let expected_projection = backing_segment_range(
            &authority.segment_lease.segments,
            authority.evidence.physical_offset_bytes,
            authority.evidence.size_bytes,
        )?;
        if expected_projection != authority.evidence.segments {
            return Err(invalid_resource(
                "logical backing projection differs from its shared physical extent",
            ));
        }
        Ok(())
    }
}

impl<R> DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    fn allocation_quantum(&self) -> u64 {
        match self.domain.pool.compatibility().profile().allocator() {
            DynamicStorageAllocator::LinearArena => {
                self.domain.pool.compatibility().alignment_bytes()
            }
            DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
                block_bytes.max(self.domain.pool.compatibility().alignment_bytes())
            }
        }
    }

    fn cancel_pending_growth(&self, bytes: u64) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.pending_growth_bytes < bytes {
            state.poisoned = true;
            return;
        }
        state.pending_growth_bytes -= bytes;
    }

    fn rollback_prepared(&self, segments: &[BackingSegment]) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                return;
            }
        };
        if state.poisoned {
            return;
        }
        for segment in segments.iter().rev() {
            let valid = state
                .chunks
                .get(&segment.chunk_ordinal())
                .is_some_and(|chunk| {
                    chunk.backing.identity == *segment.chunk() && chunk.live_segments != 0
                });
            if !valid || state.allocator.release(segment).is_err() {
                state.poisoned = true;
                return;
            }
            state
                .chunks
                .get_mut(&segment.chunk_ordinal())
                .expect("validated prepared chunk remains installed")
                .live_segments -= 1;
        }
        drop(state);
        if self
            .logical_admission
            .notify_domain_availability_changed(self.domain.domain_id)
            .is_err()
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.poisoned = true;
        }
    }
}
