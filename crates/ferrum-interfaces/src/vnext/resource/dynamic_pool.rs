use super::{
    invalid_resource, lane_stable_layout_fingerprint, AllocationKind, AllocationLifetime, Arc,
    AtomicU64, AtomicU8, BTreeMap, BackingChunkIdentity, BackingSegment, BufferDescriptor,
    BufferUsage, CapacityDomainId, CapacityEntry, CapacityEpochs, CapacityUnits, CapacityVector,
    CapacityWaitCondition, DeviceBufferRetention, DeviceCapacityAvailabilitySnapshot,
    DeviceCapacityGrant, DeviceRuntime, DynamicBackingPoolId, DynamicBackingPoolSpec,
    DynamicResourceDescriptor, DynamicResourceShape, DynamicStorageAllocator,
    DynamicStorageProfile, ElementType, ExecutionLane, ExecutionLaneId, FreeExtentIndex,
    InvocationLivenessMode, LaneStableArenaSlotIdentity, LogicalAdmissionCoordinator, Mutex,
    Ordering, PlanNode, ResourceId, Serialize, StateInitialization, VNextError, Weak,
};
use crate::vnext::{
    DeviceCapacityPressure, DeviceReusableAddressScope, DynamicPoolProvisioningPolicy,
    DynamicResourceDemand, PoolCompatibilityKey, ReusableExecutionBucketId,
    ReusableExecutionMemoryPlan,
};
use sha2::{Digest, Sha256};

pub(super) static NEXT_DYNAMIC_POOL_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn align_up_resource(value: u64, alignment: u64) -> Result<u64, VNextError> {
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

pub(super) fn free_extent_layout_fingerprint(allocator: &FreeExtentIndex) -> String {
    let mut hasher = Sha256::new();
    for (&(chunk_ordinal, offset_bytes), extent) in &allocator.by_offset {
        hasher.update(chunk_ordinal.to_be_bytes());
        hasher.update(extent.chunk_generation.to_be_bytes());
        hasher.update(offset_bytes.to_be_bytes());
        hasher.update(extent.length_bytes.to_be_bytes());
    }
    format!("sha256/{:x}", hasher.finalize())
}

fn unused_simulation_chunk_ordinal(allocator: &FreeExtentIndex) -> Result<u32, VNextError> {
    let mut candidate = u32::MAX;
    loop {
        if !allocator
            .by_offset
            .keys()
            .any(|(chunk_ordinal, _)| *chunk_ordinal == candidate)
        {
            return Ok(candidate);
        }
        candidate = candidate.checked_sub(1).ok_or_else(|| {
            invalid_resource("contiguous packing simulation exhausted chunk identities")
        })?;
    }
}

/// Returns one additional chunk size that makes the canonical
/// best-fit-decreasing transaction packable. This only runs after physical
/// deferral; successful hot-path claims still touch the allocator once.
pub(super) fn contiguous_packing_growth_bytes(
    allocator: &FreeExtentIndex,
    pool_id: &DynamicBackingPoolId,
    claim_bytes_descending: &[u64],
) -> Result<u64, VNextError> {
    if claim_bytes_descending.is_empty()
        || claim_bytes_descending.iter().any(|bytes| *bytes == 0)
        || claim_bytes_descending
            .windows(2)
            .any(|pair| pair[0] < pair[1])
    {
        return Err(invalid_resource(
            "contiguous packing demand is empty, zero-sized, or non-canonical",
        ));
    }
    let maximum_growth = claim_bytes_descending
        .iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
        .ok_or_else(|| invalid_resource("contiguous packing demand overflows u64"))?;
    let synthetic_chunk = unused_simulation_chunk_ordinal(allocator)?;
    let mut growth_bytes = 0_u64;
    loop {
        let mut simulation = allocator.clone();
        if growth_bytes != 0 {
            simulation.insert_extent(synthetic_chunk, u64::MAX, 0, growth_bytes)?;
        }
        let mut failed_claim = None;
        for &claim_bytes in claim_bytes_descending {
            if simulation
                .allocate_contiguous(pool_id, claim_bytes)?
                .is_none()
            {
                failed_claim = Some(claim_bytes);
                break;
            }
        }
        let Some(failed_claim) = failed_claim else {
            return Ok(growth_bytes);
        };
        growth_bytes = growth_bytes
            .checked_add(failed_claim)
            .ok_or_else(|| invalid_resource("contiguous packing growth overflows u64"))?;
        if growth_bytes > maximum_growth {
            return Err(invalid_resource(
                "contiguous packing planner exceeded its guaranteed growth bound",
            ));
        }
    }
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

#[derive(Debug)]
pub(super) struct SubmissionWaveProjectionLayout {
    pub(super) descriptor_index: usize,
    pub(super) projection_index: usize,
}

#[derive(Debug)]
pub(super) struct SubmissionWaveRowLayout {
    pub(super) projections: Vec<SubmissionWaveProjectionLayout>,
}

#[derive(Debug)]
pub(super) struct SubmissionWaveDomainLayout {
    pub(super) rows: Vec<SubmissionWaveRowLayout>,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) projection_count: usize,
}

#[derive(Debug)]
pub(super) struct SubmissionWaveProjectionCapacity {
    pub(super) physical_offset_bytes: u64,
    pub(super) capacity_size_bytes: u64,
}

#[derive(Debug)]
pub(super) struct SubmissionWaveDomainCapacityLayout {
    pub(super) physical_size_bytes: u64,
    pub(super) projections: Vec<SubmissionWaveProjectionCapacity>,
}

pub(super) fn compile_submission_wave_domain_layout(
    domain: &DynamicPoolDomainSpec,
    nodes: &[PlanNode],
) -> Result<Option<SubmissionWaveDomainLayout>, VNextError> {
    if domain.pool.invocation_liveness_mode() == InvocationLivenessMode::NoInvocationResources {
        return Ok(None);
    }

    let canonical_projections = domain
        .descriptors
        .iter()
        .enumerate()
        .filter(|(_, descriptor)| descriptor.lifetime() == AllocationLifetime::Invocation)
        .collect::<Vec<_>>();
    let projection_by_resource = canonical_projections
        .iter()
        .enumerate()
        .map(|(projection_index, (descriptor_index, descriptor))| {
            (
                descriptor.base_resource_id(),
                (*descriptor_index, projection_index),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if canonical_projections.is_empty()
        || projection_by_resource.len() != canonical_projections.len()
    {
        return Err(invalid_resource(
            "submission wave layout requires unique invocation descriptors",
        ));
    }

    let liveness = domain.pool.invocation_liveness();
    let mut covered_projections = std::collections::BTreeSet::new();
    let mut rows = Vec::with_capacity(liveness.len());
    for node in nodes {
        let Ok(row_index) = liveness.binary_search_by(|row| row.node_id().cmp(node.id())) else {
            continue;
        };
        let row = &liveness[row_index];
        let projections = row
            .resource_ids()
            .iter()
            .map(|resource_id| {
                let &(descriptor_index, projection_index) =
                    projection_by_resource.get(resource_id).ok_or_else(|| {
                        invalid_resource(
                            "submission wave liveness references a descriptor outside its pool",
                        )
                    })?;
                if !covered_projections.insert(projection_index) {
                    return Err(invalid_resource(
                        "submission wave liveness repeats one invocation descriptor",
                    ));
                }
                Ok(SubmissionWaveProjectionLayout {
                    descriptor_index,
                    projection_index,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        rows.push(SubmissionWaveRowLayout { projections });
    }
    if rows.len() != liveness.len()
        || covered_projections.len() != canonical_projections.len()
        || covered_projections
            .iter()
            .copied()
            .ne(0..canonical_projections.len())
    {
        return Err(invalid_resource(
            "submission wave layout does not cover immutable plan invocation resources exactly",
        ));
    }

    Ok(Some(SubmissionWaveDomainLayout {
        rows,
        claim_identity: PhysicalBackingClaimIdentity::new(
            domain.pool_id().clone(),
            canonical_projections
                .iter()
                .map(|(_, descriptor)| descriptor.base_resource_id().clone())
                .collect(),
        )?,
        projection_count: canonical_projections.len(),
    }))
}

pub(super) fn compile_submission_wave_reusable_capacity_layouts(
    domains: &[DynamicPoolDomainSpec],
    layouts: &[Option<SubmissionWaveDomainLayout>],
    reusable_execution: Option<&ReusableExecutionMemoryPlan>,
) -> Result<
    BTreeMap<ReusableExecutionBucketId, Vec<Option<SubmissionWaveDomainCapacityLayout>>>,
    VNextError,
> {
    let Some(reusable_execution) = reusable_execution else {
        return Ok(BTreeMap::new());
    };
    if domains.len() != layouts.len() {
        return Err(invalid_resource(
            "submission wave reusable capacity layout count differs from dynamic pool domains",
        ));
    }

    reusable_execution
        .buckets()
        .iter()
        .map(|resolved| {
            let bucket = resolved.bucket();
            let capacity = bucket.capacity();
            let capacity_shape = DynamicResourceShape::from_validated(
                capacity.maximum_sequences(),
                capacity.maximum_tokens(),
                capacity.maximum_pages(),
            );
            let compiled = domains
                .iter()
                .zip(layouts)
                .map(|(domain, layout)| {
                    let Some(layout) = layout else {
                        return Ok(None);
                    };
                    let mode = domain.pool.invocation_liveness_mode();
                    let mut projections = (0..layout.projection_count)
                        .map(|_| None)
                        .collect::<Vec<_>>();
                    let mut physical_size_bytes = 0_u64;
                    for row in &layout.rows {
                        let row_base = match mode {
                            InvocationLivenessMode::TotalOrderReuse => 0,
                            InvocationLivenessMode::ConservativeConcurrent => physical_size_bytes,
                            InvocationLivenessMode::NoInvocationResources => unreachable!(),
                        };
                        let mut row_bytes = 0_u64;
                        for projection_layout in &row.projections {
                            let descriptor = domain
                                .descriptors
                                .get(projection_layout.descriptor_index)
                                .ok_or_else(|| {
                                    invalid_resource(
                                        "reusable submission layout references a descriptor outside its pool",
                                    )
                                })?;
                            if descriptor.lifetime() != AllocationLifetime::Invocation {
                                return Err(invalid_resource(
                                    "reusable submission layout references a non-Invocation descriptor",
                                ));
                            }
                            let capacity_size_bytes =
                                descriptor.evaluate_request_bytes_for_shape(capacity_shape)?;
                            let physical_offset_bytes =
                                row_base.checked_add(row_bytes).ok_or_else(|| {
                                    invalid_resource(
                                        "reusable submission projection offset overflows u64",
                                    )
                                })?;
                            row_bytes =
                                row_bytes.checked_add(capacity_size_bytes).ok_or_else(|| {
                                    invalid_resource(
                                        "reusable submission row capacity overflows u64",
                                    )
                                })?;
                            if projections[projection_layout.projection_index]
                                .replace(SubmissionWaveProjectionCapacity {
                                    physical_offset_bytes,
                                    capacity_size_bytes,
                                })
                                .is_some()
                            {
                                return Err(invalid_resource(
                                    "reusable submission layout repeats one canonical projection",
                                ));
                            }
                        }
                        physical_size_bytes = match mode {
                            InvocationLivenessMode::TotalOrderReuse => {
                                physical_size_bytes.max(row_bytes)
                            }
                            InvocationLivenessMode::ConservativeConcurrent => physical_size_bytes
                                .checked_add(row_bytes)
                                .ok_or_else(|| {
                                    invalid_resource(
                                        "reusable submission pool capacity overflows u64",
                                    )
                                })?,
                            InvocationLivenessMode::NoInvocationResources => unreachable!(),
                        };
                    }
                    let projections =
                        projections
                            .into_iter()
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| {
                                invalid_resource(
                                    "reusable submission layout left a projection uncompiled",
                                )
                            })?;
                    if projections.is_empty() || physical_size_bytes == 0 {
                        return Err(invalid_resource(
                            "reusable submission layout compiled empty capacity",
                        ));
                    }
                    Ok(Some(SubmissionWaveDomainCapacityLayout {
                        physical_size_bytes,
                        projections,
                    }))
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            Ok((bucket.bucket_id().clone(), compiled))
        })
        .collect()
}

pub(super) struct ResidentChunkBacking<B> {
    // Buffer must drop before its physical capacity grant is returned.
    pub(super) buffer: B,
    pub(super) _grant: DeviceCapacityGrant,
    pub(super) identity: BackingChunkIdentity,
    pub(super) descriptor: BufferDescriptor,
}

pub(super) struct ResidentChunkState<B> {
    pub(super) backing: Arc<ResidentChunkBacking<B>>,
    pub(super) live_segments: u64,
}

pub(super) fn rollback_free_extent_journal<B>(
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
    pub(super) instance_id: u64,
    pub(super) domain: DynamicPoolDomainSpec,
    pub(super) logical_admission: LogicalAdmissionCoordinator,
    pub(super) maintenance: Mutex<()>,
    pub(super) next_extent_generation: AtomicU64,
    pub(super) state: Mutex<DynamicBackingPoolState<R::Buffer>>,
}

pub(super) struct PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    pub(super) pool: Arc<DynamicBackingPool<R>>,
    pub(super) bytes: u64,
    pub(super) armed: bool,
}

impl<R> PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    pub(super) fn disarm(&mut self) {
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

pub(super) trait BackingExtentOwner: Send + Sync {
    fn instance_id(&self) -> u64;
    fn release_segments(&self, segments: &[BackingSegment]);
}

pub(super) struct BackingSegmentLease {
    pub(super) owner: Arc<dyn BackingExtentOwner>,
    pub(super) owner_instance_id: u64,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) segment_generation: u64,
    pub(super) segments: Vec<BackingSegment>,
    pub(super) size_bytes: u64,
    pub(super) initialization: Option<Arc<BackingInitializationCell>>,
    pub(super) released: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackingInitializationStatus {
    Pending,
    Prepared,
    InFlight,
    Initialized,
    Poisoned,
}

const BACKING_INITIALIZATION_PENDING: u8 = 0;
const BACKING_INITIALIZATION_PREPARED: u8 = 1;
const BACKING_INITIALIZATION_IN_FLIGHT: u8 = 2;
const BACKING_INITIALIZATION_INITIALIZED: u8 = 3;
const BACKING_INITIALIZATION_POISONED: u8 = 4;

#[derive(Debug)]
enum BackingInitializationState {
    Pending,
    Prepared { wave_fingerprint: String },
    InFlight { wave_fingerprint: String },
    Initialized,
    Poisoned,
}

#[derive(Debug)]
pub(super) struct BackingInitializationCell {
    target_fingerprint: String,
    status: AtomicU8,
    state: Mutex<BackingInitializationState>,
}

impl BackingInitializationCell {
    fn new(target_fingerprint: String) -> Self {
        Self {
            target_fingerprint,
            status: AtomicU8::new(BACKING_INITIALIZATION_PENDING),
            state: Mutex::new(BackingInitializationState::Pending),
        }
    }

    pub(super) fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }

    pub(super) fn status(&self) -> Result<BackingInitializationStatus, VNextError> {
        match self.status.load(Ordering::Acquire) {
            BACKING_INITIALIZATION_PENDING => Ok(BackingInitializationStatus::Pending),
            BACKING_INITIALIZATION_PREPARED => Ok(BackingInitializationStatus::Prepared),
            BACKING_INITIALIZATION_IN_FLIGHT => Ok(BackingInitializationStatus::InFlight),
            BACKING_INITIALIZATION_INITIALIZED => Ok(BackingInitializationStatus::Initialized),
            BACKING_INITIALIZATION_POISONED => Ok(BackingInitializationStatus::Poisoned),
            _ => Err(invalid_resource(
                "backing initialization status contains an invalid value",
            )),
        }
    }

    pub(super) fn prepare(&self, wave_fingerprint: &str) -> Result<bool, VNextError> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
                return Err(invalid_resource("backing initialization state is poisoned"));
            }
        };
        match &*state {
            BackingInitializationState::Pending => {
                *state = BackingInitializationState::Prepared {
                    wave_fingerprint: wave_fingerprint.to_owned(),
                };
                self.status
                    .store(BACKING_INITIALIZATION_PREPARED, Ordering::Release);
                Ok(true)
            }
            BackingInitializationState::Initialized => Ok(false),
            BackingInitializationState::Prepared {
                wave_fingerprint: current,
            } if current == wave_fingerprint => Ok(true),
            BackingInitializationState::Prepared { .. }
            | BackingInitializationState::InFlight { .. } => Err(invalid_resource(
                "backing initialization is owned by another submission wave",
            )),
            BackingInitializationState::Poisoned => Err(invalid_resource(
                "backing initialization authority is fail-closed",
            )),
        }
    }

    pub(super) fn mark_in_flight(&self, wave_fingerprint: &str) -> Result<(), VNextError> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
                return Err(invalid_resource("backing initialization state is poisoned"));
            }
        };
        match &*state {
            BackingInitializationState::Prepared {
                wave_fingerprint: current,
            } if current == wave_fingerprint => {
                *state = BackingInitializationState::InFlight {
                    wave_fingerprint: wave_fingerprint.to_owned(),
                };
                self.status
                    .store(BACKING_INITIALIZATION_IN_FLIGHT, Ordering::Release);
                Ok(())
            }
            _ => {
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
                Err(invalid_resource(
                    "backing initialization fence was installed from an invalid state",
                ))
            }
        }
    }

    pub(super) fn finish(&self, wave_fingerprint: &str, succeeded: bool) -> Result<(), VNextError> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
                return Err(invalid_resource("backing initialization state is poisoned"));
            }
        };
        match &*state {
            BackingInitializationState::InFlight {
                wave_fingerprint: current,
            } if current == wave_fingerprint => {
                *state = if succeeded {
                    BackingInitializationState::Initialized
                } else {
                    BackingInitializationState::Poisoned
                };
                self.status.store(
                    if succeeded {
                        BACKING_INITIALIZATION_INITIALIZED
                    } else {
                        BACKING_INITIALIZATION_POISONED
                    },
                    Ordering::Release,
                );
                Ok(())
            }
            _ => {
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
                Err(invalid_resource(
                    "backing initialization completed from an invalid state",
                ))
            }
        }
    }

    pub(super) fn rollback_prepared(&self, wave_fingerprint: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            BackingInitializationState::Prepared {
                wave_fingerprint: current,
            } if current == wave_fingerprint => {
                *state = BackingInitializationState::Pending;
                self.status
                    .store(BACKING_INITIALIZATION_PENDING, Ordering::Release);
            }
            BackingInitializationState::Initialized | BackingInitializationState::Pending => {}
            _ => {
                *state = BackingInitializationState::Poisoned;
                self.status
                    .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
            }
        }
    }

    pub(super) fn mark_indeterminate(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = BackingInitializationState::Poisoned;
        self.status
            .store(BACKING_INITIALIZATION_POISONED, Ordering::Release);
    }
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
            let Some(chunk) = state.chunks.get_mut(&segment.chunk_ordinal()) else {
                state.poisoned = true;
                return;
            };
            if chunk.backing.identity != *segment.chunk() || chunk.live_segments == 0 {
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
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) chunk: BackingChunkIdentity,
    pub(super) chunk_bytes: u64,
    pub(super) published_capacity_bytes: u64,
    pub(super) capacity_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolResourceContract {
    pub(super) resource_id: ResourceId,
    pub(super) demand: DynamicResourceDemand,
    pub(super) lifetime: AllocationLifetime,
    pub(super) kind: AllocationKind,
    pub(super) physical_allocation_quantum_bytes: u64,
    pub(super) initialization: StateInitialization,
}

impl DynamicPoolResourceContract {
    fn from_descriptor(descriptor: &DynamicResourceDescriptor) -> Self {
        Self {
            resource_id: descriptor.base_resource_id().clone(),
            demand: descriptor.demand().clone(),
            lifetime: descriptor.lifetime(),
            kind: descriptor.kind().clone(),
            physical_allocation_quantum_bytes: descriptor.physical_allocation_quantum_bytes(),
            initialization: descriptor.initialization(),
        }
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub fn demand(&self) -> &DynamicResourceDemand {
        &self.demand
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn kind(&self) -> &AllocationKind {
        &self.kind
    }

    pub const fn physical_allocation_quantum_bytes(&self) -> u64 {
        self.physical_allocation_quantum_bytes
    }

    pub const fn initialization(&self) -> StateInitialization {
        self.initialization
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolContractStatus {
    pub(super) compatibility: PoolCompatibilityKey,
    pub(super) resources: Vec<DynamicPoolResourceContract>,
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) reusable_workspace_ceiling_bytes: u64,
    pub(super) provisioning: DynamicPoolProvisioningPolicy,
    pub(super) invocation_liveness_mode: InvocationLivenessMode,
}

impl DynamicPoolContractStatus {
    pub(super) fn from_domain(domain: &DynamicPoolDomainSpec) -> Self {
        Self {
            compatibility: domain.pool.compatibility().clone(),
            resources: domain
                .descriptors
                .iter()
                .map(DynamicPoolResourceContract::from_descriptor)
                .collect(),
            minimum_request_bytes: domain.pool.minimum_request_bytes(),
            minimum_sequence_bytes: domain.pool.minimum_sequence_bytes(),
            minimum_step_bytes: domain.pool.minimum_step_bytes(),
            minimum_invocation_peak_bytes: domain.pool.minimum_invocation_peak_bytes(),
            reusable_workspace_ceiling_bytes: domain.pool.reusable_workspace_ceiling_bytes(),
            provisioning: domain.pool.provisioning().clone(),
            invocation_liveness_mode: domain.pool.invocation_liveness_mode(),
        }
    }

    pub fn compatibility(&self) -> &PoolCompatibilityKey {
        &self.compatibility
    }

    pub fn resources(&self) -> &[DynamicPoolResourceContract] {
        &self.resources
    }

    pub const fn minimum_request_bytes(&self) -> u64 {
        self.minimum_request_bytes
    }

    pub const fn minimum_sequence_bytes(&self) -> u64 {
        self.minimum_sequence_bytes
    }

    pub const fn minimum_step_bytes(&self) -> u64 {
        self.minimum_step_bytes
    }

    pub const fn minimum_invocation_peak_bytes(&self) -> u64 {
        self.minimum_invocation_peak_bytes
    }

    pub const fn reusable_workspace_ceiling_bytes(&self) -> u64 {
        self.reusable_workspace_ceiling_bytes
    }

    pub fn provisioning(&self) -> &DynamicPoolProvisioningPolicy {
        &self.provisioning
    }

    pub const fn invocation_liveness_mode(&self) -> InvocationLivenessMode {
        self.invocation_liveness_mode
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolStatus {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) domain_id: CapacityDomainId,
    pub(super) contract: DynamicPoolContractStatus,
    pub(super) storage_profile: DynamicStorageProfile,
    pub(super) resident_bytes: u64,
    pub(super) pending_growth_bytes: u64,
    pub(super) free_bytes: u64,
    pub(super) largest_contiguous_bytes: u64,
    pub(super) resident_chunks: usize,
    pub(super) live_segments: u64,
    pub(super) quarantined_chunks: usize,
    pub(super) quarantined_bytes: u64,
    pub(super) descriptor_mismatch_chunks: usize,
    pub(super) publication_rejected_chunks: usize,
    pub(super) poisoned: bool,
}

impl DynamicPoolStatus {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub fn contract(&self) -> &DynamicPoolContractStatus {
        &self.contract
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
pub struct DynamicPoolIdleReclaim {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) chunks: Vec<BackingChunkIdentity>,
    pub(super) reclaimed_bytes: u64,
    pub(super) published_capacity_bytes: u64,
}

impl DynamicPoolIdleReclaim {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn chunks(&self) -> &[BackingChunkIdentity] {
        &self.chunks
    }

    pub const fn reclaimed_bytes(&self) -> u64 {
        self.reclaimed_bytes
    }

    pub const fn published_capacity_bytes(&self) -> u64 {
        self.published_capacity_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolRebalanceReceipt {
    pub(super) pools: Vec<DynamicPoolIdleReclaim>,
    pub(super) reclaimed_chunks: usize,
    pub(super) reclaimed_bytes: u64,
    pub(super) logical_capacity_epoch: u64,
    pub(super) plan_device_capacity_epoch: u64,
    pub(super) process_device_capacity_epoch: u64,
}

impl DynamicPoolRebalanceReceipt {
    pub fn pools(&self) -> &[DynamicPoolIdleReclaim] {
        &self.pools
    }

    pub const fn reclaimed_chunks(&self) -> usize {
        self.reclaimed_chunks
    }

    pub const fn reclaimed_bytes(&self) -> u64 {
        self.reclaimed_bytes
    }

    pub const fn logical_capacity_epoch(&self) -> u64 {
        self.logical_capacity_epoch
    }

    pub const fn plan_device_capacity_epoch(&self) -> u64 {
        self.plan_device_capacity_epoch
    }

    pub const fn process_device_capacity_epoch(&self) -> u64 {
        self.process_device_capacity_epoch
    }
}

pub(super) struct DynamicDeviceCapacityBlocked {
    pub(super) pressure: DeviceCapacityPressure,
    pub(super) availability: DeviceCapacityAvailabilitySnapshot,
    pub(super) planned_domains: Vec<CapacityDomainId>,
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
    pub(super) growths: Vec<DynamicPoolGrowthReceipt>,
    pub(super) capacity_epoch: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rebalance: Option<DynamicPoolRebalanceReceipt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicBackingDeferralReason {
    GrowthRequired,
    FragmentedContiguous,
}

/// Semantic ownership boundary for one atomic physical backing attempt.
/// `InitialSequenceBundle` is the only scope allowed to combine Request and
/// Sequence descriptors; it publishes neither lifetime unless both can commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicBackingClaimScope {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
    InitialSequenceBundle,
}

impl DynamicBackingClaimScope {
    pub(super) const fn accepts(self, lifetime: AllocationLifetime) -> bool {
        match self {
            Self::Plan => matches!(lifetime, AllocationLifetime::Plan),
            Self::Request => matches!(lifetime, AllocationLifetime::Request),
            Self::Sequence => matches!(lifetime, AllocationLifetime::Sequence),
            Self::Step => matches!(lifetime, AllocationLifetime::Step),
            Self::Invocation => matches!(lifetime, AllocationLifetime::Invocation),
            Self::InitialSequenceBundle => matches!(
                lifetime,
                AllocationLifetime::Request | AllocationLifetime::Sequence
            ),
        }
    }

    pub const fn lifetime(self) -> Option<AllocationLifetime> {
        match self {
            Self::Plan => Some(AllocationLifetime::Plan),
            Self::Request => Some(AllocationLifetime::Request),
            Self::Sequence => Some(AllocationLifetime::Sequence),
            Self::Step => Some(AllocationLifetime::Step),
            Self::Invocation => Some(AllocationLifetime::Invocation),
            Self::InitialSequenceBundle => None,
        }
    }
}

impl From<AllocationLifetime> for DynamicBackingClaimScope {
    fn from(lifetime: AllocationLifetime) -> Self {
        match lifetime {
            AllocationLifetime::Plan => Self::Plan,
            AllocationLifetime::Request => Self::Request,
            AllocationLifetime::Sequence => Self::Sequence,
            AllocationLifetime::Step => Self::Step,
            AllocationLifetime::Invocation => Self::Invocation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingBlocker {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) domain_id: CapacityDomainId,
    pub(super) reason: DynamicBackingDeferralReason,
    pub(super) requested_bytes: u64,
    pub(super) free_bytes: u64,
    pub(super) largest_contiguous_bytes: u64,
    pub(super) free_extent_layout_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) contiguous_claim_bytes_descending: Option<Vec<u64>>,
}

impl DynamicBackingBlocker {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
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

    pub fn free_extent_layout_fingerprint(&self) -> &str {
        &self.free_extent_layout_fingerprint
    }

    pub fn contiguous_claim_bytes_descending(&self) -> Option<&[u64]> {
        self.contiguous_claim_bytes_descending.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingDeferred {
    pub(super) blockers: Vec<DynamicBackingBlocker>,
    pub(super) epochs: CapacityEpochs,
    pub(super) wait_condition: CapacityWaitCondition,
    pub(super) scope: DynamicBackingClaimScope,
    pub(super) protected_immediate: CapacityVector,
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

    pub fn wait_condition(&self) -> &CapacityWaitCondition {
        &self.wait_condition
    }

    pub const fn scope(&self) -> DynamicBackingClaimScope {
        self.scope
    }

    pub const fn lifetime(&self) -> Option<AllocationLifetime> {
        self.scope.lifetime()
    }

    /// Exact uncommitted physical demand that must remain simultaneously
    /// runnable while maintenance rebalances other pools.
    pub fn protected_immediate(&self) -> &CapacityVector {
        &self.protected_immediate
    }
}

impl DynamicPoolGrowthBatchReceipt {
    pub fn growths(&self) -> &[DynamicPoolGrowthReceipt] {
        &self.growths
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }

    pub const fn rebalance(&self) -> Option<&DynamicPoolRebalanceReceipt> {
        self.rebalance.as_ref()
    }
}

#[derive(Clone)]
pub(super) enum DynamicPoolGrowthIntent {
    Additional(DynamicPoolGrowthRequest),
    Minimum(DynamicBackingPoolId),
    RevalidatedDeferral(DynamicBackingBlocker),
}

impl DynamicPoolGrowthIntent {
    pub(super) fn pool_id(&self) -> &DynamicBackingPoolId {
        match self {
            Self::Additional(request) => request.pool_id(),
            Self::Minimum(pool_id) => pool_id,
            Self::RevalidatedDeferral(blocker) => blocker.pool_id(),
        }
    }
}

pub(super) struct PlannedDynamicGrowth<R>
where
    R: DeviceRuntime,
{
    pub(super) pool: Arc<DynamicBackingPool<R>>,
    pub(super) chunk: BackingChunkIdentity,
    pub(super) expected_resource_id: ResourceId,
    pub(super) chunk_bytes: u64,
}

pub(super) struct AllocatedDynamicGrowth<B> {
    pub(super) backing: Arc<ResidentChunkBacking<B>>,
}

#[derive(Clone)]
pub(super) struct IdleChunkReclaimCandidate {
    pub(super) pool_index: usize,
    pub(super) chunk: BackingChunkIdentity,
    pub(super) chunk_bytes: u64,
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

#[derive(Clone)]
pub(super) struct EvaluatedBackingProjection<'a> {
    pub(super) descriptor: &'a DynamicResourceDescriptor,
    pub(super) physical_offset_bytes: u64,
    pub(super) logical_size_bytes: u64,
    pub(super) capacity_size_bytes: u64,
}

#[derive(Clone)]
pub(super) struct EvaluatedBackingRequest<'a> {
    pub(super) domain: &'a DynamicPoolDomainSpec,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) capacity_size_bytes: u64,
    pub(super) reusable_execution_bucket_id: Option<ReusableExecutionBucketId>,
    pub(super) projections: Vec<EvaluatedBackingProjection<'a>>,
}

pub(super) struct PreparedBackingExtent<R>
where
    R: DeviceRuntime,
{
    pub(super) pool: Arc<DynamicBackingPool<R>>,
    pub(super) claim_identity: PhysicalBackingClaimIdentity,
    pub(super) segment_generation: u64,
    pub(super) segments: Vec<BackingSegment>,
    pub(super) capacity_size_bytes: u64,
    pub(super) projections: Vec<LogicalBackingSliceEvidence>,
}

pub(super) struct PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    pub(super) extents: Vec<PreparedBackingExtent<R>>,
    pub(super) committed: bool,
}

impl<R> PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    pub(super) fn empty() -> Self {
        Self {
            extents: Vec::new(),
            committed: false,
        }
    }

    pub(super) fn commit(mut self) -> Vec<LogicalBackingSliceAuthority> {
        let mut slices = Vec::new();
        for extent in std::mem::take(&mut self.extents) {
            let initialization = extent
                .projections
                .iter()
                .any(|projection| projection.initialization == StateInitialization::Zero)
                .then(|| {
                    Arc::new(BackingInitializationCell::new(
                        backing_initialization_target_fingerprint(&extent),
                    ))
                });
            let owner: Arc<dyn BackingExtentOwner> = extent.pool;
            let segment_lease = Arc::new(BackingSegmentLease {
                owner_instance_id: owner.instance_id(),
                owner,
                claim_identity: extent.claim_identity,
                segment_generation: extent.segment_generation,
                segments: extent.segments,
                size_bytes: extent.capacity_size_bytes,
                initialization,
                released: false,
            });
            slices.extend(extent.projections.into_iter().map(|evidence| {
                LogicalBackingSliceAuthority {
                    evidence,
                    segment_lease: Arc::clone(&segment_lease),
                    reusable_lane: None,
                }
            }));
        }
        slices.sort_by(|left, right| left.resource_id().cmp(right.resource_id()));
        self.committed = true;
        slices
    }
}

fn backing_initialization_target_fingerprint<R>(extent: &PreparedBackingExtent<R>) -> String
where
    R: DeviceRuntime,
{
    let mut hasher = Sha256::new();
    hasher.update(b"ferrum.runtime-vnext.backing-initialization-target.v1\0");
    hasher.update(extent.pool.instance_id.to_be_bytes());
    hasher.update(extent.segment_generation.to_be_bytes());
    hasher.update(extent.claim_identity.pool_id().as_str().as_bytes());
    for resource_id in extent.claim_identity.resource_ids() {
        hasher.update([0]);
        hasher.update(resource_id.as_str().as_bytes());
    }
    for segment in &extent.segments {
        hasher.update(segment.chunk_ordinal().to_be_bytes());
        hasher.update(segment.chunk_generation().to_be_bytes());
        hasher.update(segment.offset_bytes().to_be_bytes());
        hasher.update(segment.length_bytes().to_be_bytes());
    }
    for projection in extent
        .projections
        .iter()
        .filter(|projection| projection.initialization == StateInitialization::Zero)
    {
        hasher.update([1]);
        hasher.update(projection.resource_id.as_str().as_bytes());
        hasher.update(projection.physical_offset_bytes.to_be_bytes());
        hasher.update(projection.capacity_size_bytes.to_be_bytes());
    }
    format!("sha256/{:x}", hasher.finalize())
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

/// Cold physical-layout proof for one committed backing claim.
///
/// Lane-stable arenas build this once when a slot is published. Reusing the
/// slot only binds the current logical projection sizes, so hot admission does
/// not need to reconstruct physical-claim maps or re-encode allocation
/// evidence on every wave.
#[derive(Debug)]
pub(super) struct BackingClaimCertificate {
    allocations: Box<[Arc<LogicalBackingSliceAllocationEvidence>]>,
    physical_capacity: CapacityVector,
    reusable_execution_bucket_id: Option<ReusableExecutionBucketId>,
    physical_claim_count: usize,
    has_shared_physical_claims: bool,
    fingerprint: String,
}

#[derive(Debug)]
pub(super) struct BoundBackingClaimCertificate {
    fingerprint: String,
    physical_claim_count: usize,
    has_shared_physical_claims: bool,
}

impl BoundBackingClaimCertificate {
    pub(super) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(super) const fn physical_claim_count(&self) -> usize {
        self.physical_claim_count
    }

    pub(super) const fn has_shared_physical_claims(&self) -> bool {
        self.has_shared_physical_claims
    }
}

impl BackingClaimCertificate {
    pub(super) fn from_slices(
        backing_slices: &[LogicalBackingSliceAuthority],
    ) -> Result<Self, VNextError> {
        if backing_slices
            .windows(2)
            .any(|pair| pair[0].resource_id() >= pair[1].resource_id())
        {
            return Err(invalid_resource(
                "backing claim certificate requires canonical unique logical projections",
            ));
        }
        let reusable_execution_bucket_id = backing_slices
            .first()
            .and_then(|slice| slice.evidence().reusable_execution_bucket_id())
            .cloned();
        if backing_slices.iter().any(|slice| {
            slice.evidence().reusable_execution_bucket_id() != reusable_execution_bucket_id.as_ref()
        }) {
            return Err(invalid_resource(
                "one backing certificate cannot mix reusable execution buckets",
            ));
        }

        let mut backing_by_domain = BTreeMap::<CapacityDomainId, u64>::new();
        let mut physical_claims = BTreeMap::<
            PhysicalBackingClaimIdentity,
            (Arc<BackingSegmentLease>, CapacityDomainId, u64),
        >::new();
        let mut has_shared_physical_claims = false;
        for slice in backing_slices {
            let evidence = slice.evidence();
            let claim_identity = evidence.physical_claim_identity();
            if claim_identity.pool_id() != evidence.pool_id()
                || claim_identity
                    .resource_ids()
                    .binary_search(evidence.resource_id())
                    .is_err()
                || slice.segment_lease.claim_identity != *claim_identity
                || slice.segment_lease.segment_generation != evidence.segment_generation()
                || slice.segment_lease.size_bytes != evidence.physical_size_bytes()
                || evidence.size_bytes() == 0
                || evidence.size_bytes() > evidence.capacity_size_bytes()
                || evidence
                    .physical_offset_bytes()
                    .checked_add(evidence.capacity_size_bytes())
                    .is_none_or(|end| end > evidence.physical_size_bytes())
            {
                return Err(invalid_resource(
                    "logical backing projection differs from its physical claim authority",
                ));
            }
            has_shared_physical_claims |= claim_identity.is_shared();
            match physical_claims.entry(claim_identity.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let total = backing_by_domain.entry(slice.domain_id()).or_default();
                    *total = total
                        .checked_add(evidence.physical_size_bytes())
                        .ok_or_else(|| {
                            invalid_resource("certified backing domain bytes overflow u64")
                        })?;
                    entry.insert((
                        Arc::clone(&slice.segment_lease),
                        slice.domain_id(),
                        evidence.physical_size_bytes(),
                    ));
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    let (lease, domain_id, size_bytes) = entry.get();
                    if !Arc::ptr_eq(lease, &slice.segment_lease)
                        || *domain_id != slice.domain_id()
                        || *size_bytes != evidence.physical_size_bytes()
                    {
                        return Err(invalid_resource(
                            "shared logical projections do not retain one physical claim",
                        ));
                    }
                }
            }
        }
        let physical_capacity = if backing_by_domain.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(
                backing_by_domain
                    .into_iter()
                    .map(|(domain, bytes)| CapacityEntry::new(domain, CapacityUnits::new(bytes)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?
        };
        let allocations = backing_slices
            .iter()
            .map(|slice| Arc::clone(&slice.evidence.allocation))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut hasher = Sha256::new();
        hasher.update(b"ferrum.runtime-vnext.backing-claim-certificate.v1\0");
        for allocation in &allocations {
            let fingerprint = allocation.fingerprint.as_bytes();
            hasher.update(
                u64::try_from(fingerprint.len())
                    .map_err(|_| {
                        invalid_resource(
                            "backing allocation fingerprint length exceeds portable range",
                        )
                    })?
                    .to_be_bytes(),
            );
            hasher.update(fingerprint);
        }
        Ok(Self {
            allocations,
            physical_capacity,
            reusable_execution_bucket_id,
            physical_claim_count: physical_claims.len(),
            has_shared_physical_claims,
            fingerprint: format!("{:x}", hasher.finalize()),
        })
    }

    pub(super) fn bind(
        &self,
        backing_slices: &[LogicalBackingSliceAuthority],
        demand: &super::AdmissionDemand,
    ) -> Result<BoundBackingClaimCertificate, VNextError> {
        if backing_slices.len() != self.allocations.len() {
            return Err(invalid_resource(
                "bound backing projection count differs from its physical certificate",
            ));
        }
        let mut hasher = Sha256::new();
        hasher.update(b"ferrum.runtime-vnext.bound-backing-claim.v1\0");
        hasher.update(self.fingerprint.as_bytes());
        for (slice, allocation) in backing_slices.iter().zip(&self.allocations) {
            if !Arc::ptr_eq(&slice.evidence.allocation, allocation)
                || slice.evidence.size_bytes() == 0
                || slice.evidence.size_bytes() > allocation.capacity_size_bytes
            {
                return Err(invalid_resource(
                    "bound logical projection differs from its certified allocation",
                ));
            }
            hasher.update(slice.evidence.size_bytes().to_be_bytes());
        }
        let physical_covers_logical = self.physical_capacity.entries().len()
            == demand.immediate_claim().entries().len()
            && self.physical_capacity.entries().iter().all(|physical| {
                demand
                    .immediate_claim()
                    .units_for(physical.domain())
                    .is_some_and(|logical| physical.units().get() >= logical.get())
            });
        let claim_matches = if self.reusable_execution_bucket_id.is_some() {
            physical_covers_logical
        } else {
            self.physical_capacity == *demand.immediate_claim()
        };
        if !claim_matches {
            return Err(invalid_resource(
                "certified physical backing does not cover the evaluated logical demand",
            ));
        }
        Ok(BoundBackingClaimCertificate {
            fingerprint: format!("{:x}", hasher.finalize()),
            physical_claim_count: self.physical_claim_count,
            has_shared_physical_claims: self.has_shared_physical_claims,
        })
    }
}

pub(super) struct CommittedLaneBackingClaim {
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    certificate: Arc<BackingClaimCertificate>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl CommittedLaneBackingClaim {
    #[cfg(test)]
    pub(super) fn certificate(&self) -> &Arc<BackingClaimCertificate> {
        &self.certificate
    }

    #[cfg(test)]
    pub(super) fn into_parts(
        self,
    ) -> (
        Vec<LogicalBackingSliceAuthority>,
        Option<LaneStableArenaSlotLease>,
    ) {
        (self.backing_slices, self.slot_lease)
    }

    pub(super) fn into_certified_parts(
        self,
    ) -> (
        Vec<LogicalBackingSliceAuthority>,
        Arc<BackingClaimCertificate>,
        Option<LaneStableArenaSlotLease>,
    ) {
        (self.backing_slices, self.certificate, self.slot_lease)
    }
}

pub(super) struct PreparedLaneBackingClaim {
    stable: Vec<LogicalBackingSliceAuthority>,
    certificate: Arc<BackingClaimCertificate>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl PreparedLaneBackingClaim {
    pub(super) fn new(
        stable: Vec<LogicalBackingSliceAuthority>,
        slot_lease: Option<LaneStableArenaSlotLease>,
    ) -> Result<Self, VNextError> {
        let certificate = Arc::new(BackingClaimCertificate::from_slices(&stable)?);
        Ok(Self {
            stable,
            certificate,
            slot_lease,
        })
    }

    pub(super) fn certified(
        stable: Vec<LogicalBackingSliceAuthority>,
        certificate: Arc<BackingClaimCertificate>,
        slot_lease: LaneStableArenaSlotLease,
    ) -> Self {
        Self {
            stable,
            certificate,
            slot_lease: Some(slot_lease),
        }
    }

    pub(super) fn commit(self) -> CommittedLaneBackingClaim {
        CommittedLaneBackingClaim {
            backing_slices: self.stable,
            certificate: self.certificate,
            slot_lease: self.slot_lease,
        }
    }
}

pub(super) enum LaneBackingPrepareDecision {
    Prepared(PreparedLaneBackingClaim),
    Deferred(DynamicBackingDeferred),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LaneStableArenaKey {
    pub(super) lane_id: ExecutionLaneId,
    pub(super) lifetime: AllocationLifetime,
    pub(super) reusable_execution_bucket_id: ReusableExecutionBucketId,
    pub(super) layout_fingerprint: String,
}

pub(super) struct LaneStableProjectionBinding {
    pub(super) request_index: usize,
    pub(super) projection_index: usize,
}

pub(super) struct LaneStableArenaSlot {
    pub(super) slot_id: u64,
    pub(super) authorities: Vec<LogicalBackingSliceAuthority>,
    pub(super) certificate: Arc<BackingClaimCertificate>,
    pub(super) projection_bindings: Vec<LaneStableProjectionBinding>,
    pub(super) availability_domains: Vec<CapacityDomainId>,
    pub(super) in_use: bool,
    pub(super) last_used: u64,
}

impl LaneStableArenaSlot {
    pub(super) fn has_external_address_pins(&self) -> bool {
        self.authorities
            .iter()
            .enumerate()
            .filter(|(index, authority)| {
                !self.authorities[..*index]
                    .iter()
                    .any(|prior| Arc::ptr_eq(&prior.segment_lease, &authority.segment_lease))
            })
            .any(|(_, authority)| {
                let retained_by_slot = self
                    .authorities
                    .iter()
                    .filter(|candidate| {
                        Arc::ptr_eq(&candidate.segment_lease, &authority.segment_lease)
                    })
                    .count();
                Arc::strong_count(&authority.segment_lease) > retained_by_slot
            })
    }
}

pub(super) trait LaneStableArenaLane: Send + Sync {
    fn try_trim_reusable_executables(&self) -> Result<bool, VNextError>;
}

impl<R> LaneStableArenaLane for ExecutionLane<R>
where
    R: DeviceRuntime,
{
    fn try_trim_reusable_executables(&self) -> Result<bool, VNextError> {
        self.trim_reusable_executables_if_quiescent()
    }
}

pub(super) struct LaneStableArenaEntry {
    pub(super) lane: Weak<dyn LaneStableArenaLane>,
    pub(super) slots: BTreeMap<u64, LaneStableArenaSlot>,
}

pub(super) struct LaneStableArenaEvictionCandidate {
    pub(super) key: LaneStableArenaKey,
    pub(super) slot_id: u64,
    pub(super) last_used: u64,
    pub(super) lane: Arc<dyn LaneStableArenaLane>,
}

fn lane_stable_projection_matches(
    authority: &LogicalBackingSliceAuthority,
    request: &EvaluatedBackingRequest<'_>,
    projection: &EvaluatedBackingProjection<'_>,
) -> bool {
    request.claim_identity == *authority.evidence.physical_claim_identity()
        && projection.descriptor.base_resource_id() == authority.evidence.resource_id()
        && request.capacity_size_bytes == authority.evidence.physical_size_bytes
        && projection.physical_offset_bytes == authority.evidence.physical_offset_bytes
        && projection.capacity_size_bytes == authority.evidence.capacity_size_bytes
        && projection.logical_size_bytes != 0
        && projection.logical_size_bytes <= projection.capacity_size_bytes
}

pub(super) fn bind_lane_stable_slot_projections(
    authorities: &[LogicalBackingSliceAuthority],
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<Vec<LaneStableProjectionBinding>, VNextError> {
    authorities
        .iter()
        .map(|authority| {
            let request_index = requests
                .binary_search_by(|request| {
                    request
                        .claim_identity
                        .cmp(authority.evidence.physical_claim_identity())
                })
                .map_err(|_| {
                    invalid_resource("lane-stable arena request lost its physical claim projection")
                })?;
            let request = requests[request_index];
            let projection_index = request
                .projections
                .binary_search_by(|projection| {
                    projection
                        .descriptor
                        .base_resource_id()
                        .cmp(authority.evidence.resource_id())
                })
                .map_err(|_| {
                    invalid_resource(
                        "lane-stable arena request lost its logical resource projection",
                    )
                })?;
            if !lane_stable_projection_matches(
                authority,
                request,
                &request.projections[projection_index],
            ) {
                return Err(invalid_resource(
                    "lane-stable arena request differs from its retained capacity layout",
                ));
            }
            Ok(LaneStableProjectionBinding {
                request_index,
                projection_index,
            })
        })
        .collect()
}

impl LaneStableArenaEntry {
    pub(super) fn claim_idle_slot(
        &mut self,
        lane_id: ExecutionLaneId,
        now: u64,
        requests: &[&EvaluatedBackingRequest<'_>],
    ) -> Result<
        Option<(
            u64,
            Vec<LogicalBackingSliceAuthority>,
            Arc<BackingClaimCertificate>,
            Vec<CapacityDomainId>,
        )>,
        VNextError,
    > {
        let Some(slot) = self.slots.values_mut().find(|slot| !slot.in_use) else {
            return Ok(None);
        };
        if slot.authorities.len() != slot.projection_bindings.len() {
            return Err(invalid_resource(
                "lane-stable arena slot lost its projection bindings",
            ));
        }
        let stable = slot
            .authorities
            .iter()
            .zip(&slot.projection_bindings)
            .map(|(authority, binding)| {
                let request = requests
                    .get(binding.request_index)
                    .copied()
                    .ok_or_else(|| {
                        invalid_resource(
                            "lane-stable arena request lost its physical claim projection",
                        )
                    })?;
                let projection = request
                    .projections
                    .get(binding.projection_index)
                    .ok_or_else(|| {
                        invalid_resource(
                            "lane-stable arena request lost its logical resource projection",
                        )
                    })?;
                if !lane_stable_projection_matches(authority, request, projection) {
                    return Err(invalid_resource(
                        "lane-stable arena request differs from its retained capacity layout",
                    ));
                }
                let mut retained = authority.retained_for_lane(lane_id);
                retained.evidence.logical_size_bytes = projection.logical_size_bytes;
                Ok(retained)
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        slot.in_use = true;
        slot.last_used = now;
        Ok(Some((
            slot.slot_id,
            stable,
            Arc::clone(&slot.certificate),
            slot.availability_domains.clone(),
        )))
    }
}

pub(super) struct LaneStableArenaState {
    pub(super) clock: u64,
    pub(super) next_slot_id: u64,
    pub(super) poisoned: bool,
    pub(super) entries: BTreeMap<LaneStableArenaKey, LaneStableArenaEntry>,
}

impl Default for LaneStableArenaState {
    fn default() -> Self {
        Self {
            clock: 0,
            next_slot_id: 1,
            poisoned: false,
            entries: BTreeMap::new(),
        }
    }
}

impl LaneStableArenaState {
    pub(super) fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    pub(super) fn issue_slot_id(&mut self) -> Result<u64, VNextError> {
        let slot_id = self.next_slot_id;
        self.next_slot_id = self.next_slot_id.checked_add(1).ok_or_else(|| {
            invalid_resource("lane-stable arena slot identity space is exhausted")
        })?;
        Ok(slot_id)
    }

    pub(super) fn take_expired_lanes(&mut self) -> Result<Vec<LaneStableArenaEntry>, VNextError> {
        let expired = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.lane.upgrade().is_none())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if expired.iter().any(|key| {
            self.entries
                .get(key)
                .is_some_and(|entry| entry.slots.values().any(|slot| slot.in_use))
        }) {
            self.poisoned = true;
            return Err(invalid_resource(
                "lane-stable arena retained a busy slot after its execution lane expired",
            ));
        }
        Ok(expired
            .into_iter()
            .filter_map(|key| self.entries.remove(&key))
            .collect())
    }
}

pub(super) struct LaneStableArenaSlotLease {
    pub(super) arenas: Arc<Mutex<LaneStableArenaState>>,
    pub(super) logical_admission: LogicalAdmissionCoordinator,
    pub(super) availability_domains: Vec<CapacityDomainId>,
    pub(super) key: LaneStableArenaKey,
    pub(super) slot_id: u64,
}

impl LaneStableArenaSlotLease {
    pub(super) fn identity(&self) -> LaneStableArenaSlotIdentity {
        LaneStableArenaSlotIdentity::new(
            self.key.lane_id,
            self.key.lifetime,
            self.key.reusable_execution_bucket_id.clone(),
            self.key.layout_fingerprint.clone(),
            self.slot_id,
        )
    }
}

impl Drop for LaneStableArenaSlotLease {
    fn drop(&mut self) {
        let mut arenas = match self.arenas.lock() {
            Ok(arenas) => arenas,
            Err(poisoned) => {
                poisoned.into_inner().poisoned = true;
                return;
            }
        };
        if arenas.poisoned {
            return;
        }
        let now = arenas.tick();
        let released = arenas
            .entries
            .get_mut(&self.key)
            .and_then(|entry| entry.slots.get_mut(&self.slot_id))
            .is_some_and(|slot| {
                if !slot.in_use {
                    return false;
                }
                slot.in_use = false;
                slot.last_used = now;
                true
            });
        if !released {
            arenas.poisoned = true;
            return;
        }
        drop(arenas);

        let mut notification_failed = false;
        for domain in &self.availability_domains {
            notification_failed |= self
                .logical_admission
                .notify_domain_availability_changed(*domain)
                .is_err();
        }
        if notification_failed {
            let mut arenas = self
                .arenas
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            arenas.poisoned = true;
        }
    }
}

#[doc(hidden)]
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct LogicalBackingSliceAllocationEvidence {
    pub(in crate::vnext::resource) domain_id: CapacityDomainId,
    pub(in crate::vnext::resource) pool_id: DynamicBackingPoolId,
    pub(in crate::vnext::resource) resource_id: ResourceId,
    pub(in crate::vnext::resource) pool_instance_id: u64,
    pub(in crate::vnext::resource) physical_claim_identity: PhysicalBackingClaimIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::vnext::resource) reusable_execution_bucket_id: Option<ReusableExecutionBucketId>,
    pub(in crate::vnext::resource) segment_generation: u64,
    pub(in crate::vnext::resource) segments: Vec<BackingSegment>,
    pub(in crate::vnext::resource) physical_offset_bytes: u64,
    pub(in crate::vnext::resource) capacity_size_bytes: u64,
    pub(in crate::vnext::resource) physical_size_bytes: u64,
    pub(in crate::vnext::resource) alignment_bytes: u64,
    pub(in crate::vnext::resource) usage: BufferUsage,
    pub(in crate::vnext::resource) element_type: ElementType,
    pub(in crate::vnext::resource) storage_profile: DynamicStorageProfile,
    pub(in crate::vnext::resource) initialization: StateInitialization,
    #[serde(skip)]
    pub(in crate::vnext::resource) fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalBackingSliceEvidence {
    pub(super) allocation: Arc<LogicalBackingSliceAllocationEvidence>,
    pub(in crate::vnext::resource) logical_size_bytes: u64,
}

impl std::ops::Deref for LogicalBackingSliceEvidence {
    type Target = LogicalBackingSliceAllocationEvidence;

    fn deref(&self) -> &Self::Target {
        self.allocation.as_ref()
    }
}

impl Serialize for LogicalBackingSliceEvidence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Wire<'a> {
            domain_id: CapacityDomainId,
            pool_id: &'a DynamicBackingPoolId,
            resource_id: &'a ResourceId,
            pool_instance_id: u64,
            physical_claim_identity: &'a PhysicalBackingClaimIdentity,
            #[serde(skip_serializing_if = "Option::is_none")]
            reusable_execution_bucket_id: Option<&'a ReusableExecutionBucketId>,
            segment_generation: u64,
            segments: &'a [BackingSegment],
            physical_offset_bytes: u64,
            #[serde(rename = "size_bytes")]
            logical_size_bytes: u64,
            capacity_size_bytes: u64,
            physical_size_bytes: u64,
            alignment_bytes: u64,
            usage: BufferUsage,
            element_type: ElementType,
            storage_profile: DynamicStorageProfile,
            initialization: StateInitialization,
        }

        Wire {
            domain_id: self.domain_id,
            pool_id: &self.pool_id,
            resource_id: &self.resource_id,
            pool_instance_id: self.pool_instance_id,
            physical_claim_identity: &self.physical_claim_identity,
            reusable_execution_bucket_id: self.reusable_execution_bucket_id.as_ref(),
            segment_generation: self.segment_generation,
            segments: &self.segments,
            physical_offset_bytes: self.physical_offset_bytes,
            logical_size_bytes: self.logical_size_bytes,
            capacity_size_bytes: self.capacity_size_bytes,
            physical_size_bytes: self.physical_size_bytes,
            alignment_bytes: self.alignment_bytes,
            usage: self.usage,
            element_type: self.element_type,
            storage_profile: self.storage_profile,
            initialization: self.initialization,
        }
        .serialize(serializer)
    }
}

impl LogicalBackingSliceEvidence {
    pub fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn pool_instance_id(&self) -> u64 {
        self.pool_instance_id
    }

    pub fn segment_generation(&self) -> u64 {
        self.segment_generation
    }

    pub fn physical_claim_identity(&self) -> &PhysicalBackingClaimIdentity {
        &self.physical_claim_identity
    }

    pub fn reusable_execution_bucket_id(&self) -> Option<&ReusableExecutionBucketId> {
        self.reusable_execution_bucket_id.as_ref()
    }

    pub fn segments(&self) -> &[BackingSegment] {
        &self.segments
    }

    pub fn physical_offset_bytes(&self) -> u64 {
        self.physical_offset_bytes
    }

    pub const fn size_bytes(&self) -> u64 {
        self.logical_size_bytes
    }

    pub fn capacity_size_bytes(&self) -> u64 {
        self.capacity_size_bytes
    }

    pub fn physical_size_bytes(&self) -> u64 {
        self.physical_size_bytes
    }

    pub fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }

    pub fn initialization(&self) -> StateInitialization {
        self.initialization
    }
}

#[must_use = "a logical backing authority owns its physical arena extents"]
pub struct LogicalBackingSliceAuthority {
    pub(in crate::vnext::resource) evidence: LogicalBackingSliceEvidence,
    pub(in crate::vnext::resource) segment_lease: Arc<BackingSegmentLease>,
    pub(super) reusable_lane: Option<ExecutionLaneId>,
}

impl LogicalBackingSliceAuthority {
    pub fn evidence(&self) -> &LogicalBackingSliceEvidence {
        &self.evidence
    }

    pub(in crate::vnext::resource) fn retained(&self) -> Self {
        Self {
            evidence: self.evidence.clone(),
            segment_lease: Arc::clone(&self.segment_lease),
            reusable_lane: self.reusable_lane,
        }
    }

    pub(in crate::vnext::resource) fn retained_for_lane(&self, lane_id: ExecutionLaneId) -> Self {
        Self {
            evidence: self.evidence.clone(),
            segment_lease: Arc::clone(&self.segment_lease),
            reusable_lane: Some(lane_id),
        }
    }

    pub(crate) const fn reusable_address_scope(&self) -> Option<DeviceReusableAddressScope> {
        match self.reusable_lane {
            Some(lane_id) => Some(DeviceReusableAddressScope::ExecutionLane(lane_id)),
            None => None,
        }
    }

    pub fn domain_id(&self) -> CapacityDomainId {
        self.evidence.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.evidence.resource_id
    }

    pub const fn size_bytes(&self) -> u64 {
        self.evidence.logical_size_bytes
    }

    pub fn capacity_size_bytes(&self) -> u64 {
        self.evidence.capacity_size_bytes
    }

    pub fn initialization_status(&self) -> Result<Option<BackingInitializationStatus>, VNextError> {
        self.segment_lease
            .initialization
            .as_ref()
            .map(|cell| cell.status())
            .transpose()
    }

    pub(in crate::vnext::resource) fn initialization_cell(
        &self,
    ) -> Option<&Arc<BackingInitializationCell>> {
        self.segment_lease.initialization.as_ref()
    }
}

pub struct LogicalBackingBufferView<'a, B> {
    pub(in crate::vnext::resource) bindings: Vec<LogicalBackingSegmentBinding<B>>,
    pub(super) authorities: &'a [LogicalBackingSliceAuthority],
    pub(super) logical_size_bytes: u64,
    pub(super) capacity_size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) storage_profile: DynamicStorageProfile,
}

pub(crate) struct LogicalBackingSegmentBinding<B> {
    pub(in crate::vnext::resource) segment: BackingSegment,
    pub(in crate::vnext::resource) chunk: Arc<ResidentChunkBacking<B>>,
    pub(in crate::vnext::resource) retention: DeviceBufferRetention,
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

    pub(crate) fn retention(&self) -> DeviceBufferRetention {
        self.retention.clone()
    }
}

impl<'a, B> LogicalBackingBufferView<'a, B> {
    pub(crate) fn segment_bindings(&self) -> &[LogicalBackingSegmentBinding<B>] {
        &self.bindings
    }

    pub const fn size_bytes(&self) -> u64 {
        self.logical_size_bytes
    }

    pub const fn capacity_size_bytes(&self) -> u64 {
        self.capacity_size_bytes
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

pub(super) fn lane_stable_layout_key(
    lane_id: ExecutionLaneId,
    lifetime: AllocationLifetime,
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<LaneStableArenaKey, VNextError> {
    if requests.is_empty()
        || requests
            .windows(2)
            .any(|pair| pair[0].claim_identity >= pair[1].claim_identity)
        || requests.iter().any(|request| {
            request.projections.is_empty()
                || request
                    .projections
                    .iter()
                    .any(|projection| projection.descriptor.lifetime() != lifetime)
        })
    {
        return Err(invalid_resource(
            "lane-stable arena layout is empty, non-canonical, or mixes lifetimes",
        ));
    }
    let bucket_id = requests[0]
        .reusable_execution_bucket_id
        .as_ref()
        .ok_or_else(|| {
            invalid_resource(
                "lane-stable arena requires an immutable-plan reusable execution bucket",
            )
        })?;
    if requests
        .iter()
        .any(|request| request.reusable_execution_bucket_id.as_ref() != Some(bucket_id))
    {
        return Err(invalid_resource(
            "lane-stable arena layout mixes reusable execution buckets",
        ));
    }
    Ok(LaneStableArenaKey {
        lane_id,
        lifetime,
        reusable_execution_bucket_id: bucket_id.clone(),
        layout_fingerprint: lane_stable_layout_fingerprint(lifetime, bucket_id, requests)?,
    })
}

impl<R> DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    pub(super) fn allocation_quantum(&self) -> u64 {
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
