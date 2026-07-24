use super::{
    backing_segment_range, compile_program_binding_layouts, invalid_resource,
    lane_stable_layout_fingerprint, validate_runtime_descriptor_for_admission, AllocationLifetime,
    AllocationSeal, Arc, AtomicU64, AtomicU8, BTreeMap, BackingChunkIdentity, BackingSegment,
    BufferDescriptor, BufferRequest, BufferUsage, CapacityAvailabilityEpoch, CapacityDomainId,
    CapacityEntry, CapacityEpochs, CapacityUnits, CapacityVector, CapacityWaitCondition,
    DeviceAllocationPermit, DeviceBufferRetention, DeviceCapacityAvailabilitySnapshot,
    DeviceCapacityBudget, DeviceCapacityGrant, DeviceCapacityReservation, DeviceRuntime,
    DynamicBackingPoolId, DynamicBackingPoolSpec, DynamicResourceDescriptor, DynamicResourceShape,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView, ElementType, ExecutionLane,
    ExecutionLaneId, FreeExtentIndex, InvocationLivenessMode, LaneStableArenaSlotIdentity,
    LogicalAdmissionCoordinator, Mutex, Ordering, PlanNode, ProgramBindingLayout, ResourceId,
    ResourceReservation, ResourceRetentionPolicy, ResourceTransactionIdentity, RunId, Serialize,
    StateInitialization, StaticProvisioningBinding, StepResourceSlotKind, TransactionId,
    VNextError, Weak,
};
use crate::vnext::{
    DeviceCapacityPressure, DynamicPoolResidentPressure, ReusableExecutionBucketId,
    ReusableExecutionMemoryPlan,
};
use sha2::{Digest, Sha256};

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

fn free_extent_layout_fingerprint(allocator: &FreeExtentIndex) -> String {
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
fn contiguous_packing_growth_bytes(
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
    pub(super) maintenance: Mutex<()>,
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
    initialization: Option<Arc<BackingInitializationCell>>,
    released: bool,
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
    pool_id: DynamicBackingPoolId,
    chunk: BackingChunkIdentity,
    chunk_bytes: u64,
    published_capacity_bytes: u64,
    capacity_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolStatus {
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) domain_id: CapacityDomainId,
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
    pool_id: DynamicBackingPoolId,
    chunks: Vec<BackingChunkIdentity>,
    reclaimed_bytes: u64,
    published_capacity_bytes: u64,
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
    pools: Vec<DynamicPoolIdleReclaim>,
    reclaimed_chunks: usize,
    reclaimed_bytes: u64,
    logical_capacity_epoch: u64,
    plan_device_capacity_epoch: u64,
    process_device_capacity_epoch: u64,
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
    capacity_epoch: u64,
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
    const fn accepts(self, lifetime: AllocationLifetime) -> bool {
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
    pool_id: DynamicBackingPoolId,
    domain_id: CapacityDomainId,
    reason: DynamicBackingDeferralReason,
    requested_bytes: u64,
    free_bytes: u64,
    largest_contiguous_bytes: u64,
    free_extent_layout_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    contiguous_claim_bytes_descending: Option<Vec<u64>>,
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
    blockers: Vec<DynamicBackingBlocker>,
    epochs: CapacityEpochs,
    wait_condition: CapacityWaitCondition,
    scope: DynamicBackingClaimScope,
    protected_immediate: CapacityVector,
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

#[derive(Clone)]
struct IdleChunkReclaimCandidate {
    pool_index: usize,
    chunk: BackingChunkIdentity,
    chunk_bytes: u64,
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

struct PreparedBackingExtent<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    claim_identity: PhysicalBackingClaimIdentity,
    segment_generation: u64,
    segments: Vec<BackingSegment>,
    capacity_size_bytes: u64,
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

pub(super) struct CommittedLaneBackingClaim {
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl CommittedLaneBackingClaim {
    pub(super) fn into_parts(
        self,
    ) -> (
        Vec<LogicalBackingSliceAuthority>,
        Option<LaneStableArenaSlotLease>,
    ) {
        (self.backing_slices, self.slot_lease)
    }
}

pub(super) struct PreparedLaneBackingClaim {
    stable: Vec<LogicalBackingSliceAuthority>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl PreparedLaneBackingClaim {
    pub(super) fn commit(self) -> CommittedLaneBackingClaim {
        CommittedLaneBackingClaim {
            backing_slices: self.stable,
            slot_lease: self.slot_lease,
        }
    }
}

pub(super) enum LaneBackingPrepareDecision {
    Prepared(PreparedLaneBackingClaim),
    Deferred(DynamicBackingDeferred),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LaneStableArenaKey {
    lane_id: ExecutionLaneId,
    lifetime: AllocationLifetime,
    reusable_execution_bucket_id: ReusableExecutionBucketId,
    layout_fingerprint: String,
}

struct LaneStableProjectionBinding {
    request_index: usize,
    projection_index: usize,
}

struct LaneStableArenaSlot {
    slot_id: u64,
    authorities: Vec<LogicalBackingSliceAuthority>,
    projection_bindings: Vec<LaneStableProjectionBinding>,
    availability_domains: Vec<CapacityDomainId>,
    in_use: bool,
    last_used: u64,
}

impl LaneStableArenaSlot {
    fn has_external_address_pins(&self) -> bool {
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

trait LaneStableArenaLane: Send + Sync {
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

struct LaneStableArenaEntry {
    lane: Weak<dyn LaneStableArenaLane>,
    slots: BTreeMap<u64, LaneStableArenaSlot>,
}

struct LaneStableArenaEvictionCandidate {
    key: LaneStableArenaKey,
    slot_id: u64,
    last_used: u64,
    lane: Arc<dyn LaneStableArenaLane>,
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

fn bind_lane_stable_slot_projections(
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
    fn claim_idle_slot(
        &mut self,
        lane_id: ExecutionLaneId,
        now: u64,
        requests: &[&EvaluatedBackingRequest<'_>],
    ) -> Result<
        Option<(
            u64,
            Vec<LogicalBackingSliceAuthority>,
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
            slot.availability_domains.clone(),
        )))
    }
}

struct LaneStableArenaState {
    clock: u64,
    next_slot_id: u64,
    poisoned: bool,
    entries: BTreeMap<LaneStableArenaKey, LaneStableArenaEntry>,
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
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    fn issue_slot_id(&mut self) -> Result<u64, VNextError> {
        let slot_id = self.next_slot_id;
        self.next_slot_id = self.next_slot_id.checked_add(1).ok_or_else(|| {
            invalid_resource("lane-stable arena slot identity space is exhausted")
        })?;
        Ok(slot_id)
    }

    fn take_expired_lanes(&mut self) -> Result<Vec<LaneStableArenaEntry>, VNextError> {
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
    arenas: Arc<Mutex<LaneStableArenaState>>,
    logical_admission: LogicalAdmissionCoordinator,
    availability_domains: Vec<CapacityDomainId>,
    key: LaneStableArenaKey,
    slot_id: u64,
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

fn lane_stable_layout_key(
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

pub(super) struct DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pub(super) pools: BTreeMap<DynamicBackingPoolId, Arc<DynamicBackingPool<R>>>,
    pub(super) domains: Vec<DynamicPoolDomainSpec>,
    pub(super) nodes: Arc<[PlanNode]>,
    pub(super) submission_wave_layouts: Vec<Option<SubmissionWaveDomainLayout>>,
    pub(super) submission_wave_reusable_capacity_layouts:
        BTreeMap<ReusableExecutionBucketId, Vec<Option<SubmissionWaveDomainCapacityLayout>>>,
    pub(super) program_binding_layouts:
        BTreeMap<ReusableExecutionBucketId, Arc<ProgramBindingLayout>>,
    pub(super) reusable_execution: Option<ReusableExecutionMemoryPlan>,
    pub(super) logical_admission: LogicalAdmissionCoordinator,
    pub(super) budget: Arc<DeviceCapacityBudget>,
    lane_stable_arenas: Arc<Mutex<LaneStableArenaState>>,
    binding: StaticProvisioningBinding,
    // Backend context must outlive every resident/quarantined buffer above.
    runtime: Arc<R>,
}

#[doc(hidden)]
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct LogicalBackingSliceAllocationEvidence {
    pub(super) domain_id: CapacityDomainId,
    pub(super) pool_id: DynamicBackingPoolId,
    pub(super) resource_id: ResourceId,
    pub(super) pool_instance_id: u64,
    pub(super) physical_claim_identity: PhysicalBackingClaimIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reusable_execution_bucket_id: Option<ReusableExecutionBucketId>,
    pub(super) segment_generation: u64,
    pub(super) segments: Vec<BackingSegment>,
    pub(super) physical_offset_bytes: u64,
    pub(super) capacity_size_bytes: u64,
    pub(super) physical_size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) storage_profile: DynamicStorageProfile,
    pub(super) initialization: StateInitialization,
    #[serde(skip)]
    pub(super) fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalBackingSliceEvidence {
    allocation: Arc<LogicalBackingSliceAllocationEvidence>,
    pub(super) logical_size_bytes: u64,
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

    pub(super) fn allocation_fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[must_use = "a logical backing authority owns its physical arena extents"]
pub struct LogicalBackingSliceAuthority {
    pub(super) evidence: LogicalBackingSliceEvidence,
    pub(super) segment_lease: Arc<BackingSegmentLease>,
    reusable_lane: Option<ExecutionLaneId>,
}

impl LogicalBackingSliceAuthority {
    pub fn evidence(&self) -> &LogicalBackingSliceEvidence {
        &self.evidence
    }

    pub(super) fn retained(&self) -> Self {
        Self {
            evidence: self.evidence.clone(),
            segment_lease: Arc::clone(&self.segment_lease),
            reusable_lane: self.reusable_lane,
        }
    }

    pub(super) fn retained_for_lane(&self, lane_id: ExecutionLaneId) -> Self {
        Self {
            evidence: self.evidence.clone(),
            segment_lease: Arc::clone(&self.segment_lease),
            reusable_lane: Some(lane_id),
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

    pub(super) fn initialization_cell(&self) -> Option<&Arc<BackingInitializationCell>> {
        self.segment_lease.initialization.as_ref()
    }
}

pub struct LogicalBackingBufferView<'a, B> {
    pub(super) bindings: Vec<LogicalBackingSegmentBinding<B>>,
    authorities: &'a [LogicalBackingSliceAuthority],
    logical_size_bytes: u64,
    capacity_size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    storage_profile: DynamicStorageProfile,
}

pub(crate) struct LogicalBackingSegmentBinding<B> {
    pub(super) segment: BackingSegment,
    pub(super) chunk: Arc<ResidentChunkBacking<B>>,
    pub(super) retention: DeviceBufferRetention,
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
        reusable_execution: Option<ReusableExecutionMemoryPlan>,
    ) -> Result<Self, VNextError> {
        let submission_wave_layouts = domains
            .iter()
            .map(|domain| compile_submission_wave_domain_layout(domain, &nodes))
            .collect::<Result<Vec<_>, _>>()?;
        let submission_wave_reusable_capacity_layouts =
            compile_submission_wave_reusable_capacity_layouts(
                &domains,
                &submission_wave_layouts,
                reusable_execution.as_ref(),
            )?;
        let program_binding_layouts = compile_program_binding_layouts(
            &domains,
            &nodes,
            &submission_wave_layouts,
            &submission_wave_reusable_capacity_layouts,
        )?
        .into_iter()
        .map(|(bucket_id, layout)| (bucket_id, Arc::new(layout)))
        .collect();
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
            submission_wave_layouts,
            submission_wave_reusable_capacity_layouts,
            program_binding_layouts,
            reusable_execution,
            lane_stable_arenas: Arc::new(Mutex::new(LaneStableArenaState::default())),
        })
    }

    pub(super) fn program_binding_layout(
        &self,
        bucket_id: &ReusableExecutionBucketId,
    ) -> Option<&Arc<ProgramBindingLayout>> {
        self.program_binding_layouts.get(bucket_id)
    }

    pub(super) fn write_capacity_availability(
        &self,
        out: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<CapacityEpochs, VNextError> {
        let epochs = self.logical_admission.write_availability_epochs(out)?;
        self.budget.write_availability_epochs(out)?;
        debug_assert!(out
            .windows(2)
            .all(|pair| pair[0].source() < pair[1].source()));
        Ok(epochs)
    }

    /// Rebalances only whole, unreferenced chunks from non-target pools. The
    /// batch is selected before mutation, logical totals publish atomically,
    /// and physical grants are returned only after every pool lock is dropped.
    pub(super) fn reclaim_idle_chunks_for_pressure(
        &self,
        pressure: &DeviceCapacityPressure,
        excluded_domains: &[CapacityDomainId],
        protected_immediate: &CapacityVector,
    ) -> Result<Option<DynamicPoolRebalanceReceipt>, VNextError> {
        if pressure.device_id() != self.runtime.descriptor().id.to_string() {
            return Err(invalid_resource(
                "dynamic pool rebalance received pressure for another device",
            ));
        }
        let deficit = pressure
            .requested_bytes()
            .checked_sub(pressure.available_bytes())
            .ok_or_else(|| invalid_resource("dynamic pool pressure has no reclaimable deficit"))?;
        if deficit == 0 {
            return Err(invalid_resource(
                "dynamic pool pressure has an empty reclaimable deficit",
            ));
        }

        let excluded_domains = excluded_domains
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let pools = self.pools.values().cloned().collect::<Vec<_>>();
        let maintenance = pools
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
        let logical = self.logical_admission.snapshot()?;
        let used_by_domain = logical
            .domains()
            .iter()
            .map(|domain| (domain.domain(), domain.used().get()))
            .collect::<BTreeMap<_, _>>();
        let protected_by_domain = protected_immediate
            .entries()
            .iter()
            .map(|entry| (entry.domain(), entry.units().get()))
            .collect::<BTreeMap<_, _>>();

        let mut candidates = Vec::new();
        let mut reclaimable_by_pool = vec![0_u64; pools.len()];
        for (pool_index, (pool, state)) in pools.iter().zip(states.iter()).enumerate() {
            if state.poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            if excluded_domains.contains(&pool.domain.domain_id) || state.pending_growth_bytes != 0
            {
                continue;
            }
            let used = used_by_domain
                .get(&pool.domain.domain_id)
                .copied()
                .ok_or_else(|| invalid_resource("dynamic pool domain is absent from admission"))?;
            let protected = protected_by_domain
                .get(&pool.domain.domain_id)
                .copied()
                .unwrap_or(0);
            let coherent_runnable_floor = used.checked_add(protected).ok_or_else(|| {
                invalid_resource("dynamic pool protected runnable floor overflows u64")
            })?;
            let resident_floor = pool
                .domain
                .pool
                .provisioning()
                .minimum_resident_bytes()
                .max(coherent_runnable_floor);
            let reclaimable = state.resident_bytes.saturating_sub(resident_floor);
            reclaimable_by_pool[pool_index] = reclaimable;
            if reclaimable == 0 {
                continue;
            }
            for (&ordinal, chunk) in &state.chunks {
                let chunk_bytes = chunk.backing._grant.bytes();
                let full_extent = state.allocator.by_offset.get(&(ordinal, 0));
                if chunk.live_segments != 0
                    || Arc::strong_count(&chunk.backing) != 1
                    || chunk_bytes > reclaimable
                    || chunk.backing.descriptor.size_bytes != chunk_bytes
                    || full_extent.is_none_or(|extent| {
                        extent.chunk_generation != chunk.backing.identity.generation()
                            || extent.length_bytes != chunk_bytes
                    })
                {
                    continue;
                }
                candidates.push(IdleChunkReclaimCandidate {
                    pool_index,
                    chunk: chunk.backing.identity.clone(),
                    chunk_bytes,
                });
            }
        }

        let mut selected = Vec::<IdleChunkReclaimCandidate>::new();
        let mut selected_by_pool = vec![0_u64; pools.len()];
        let mut reclaimed_bytes = 0_u64;
        let best_single = candidates
            .iter()
            .filter(|candidate| candidate.chunk_bytes >= deficit)
            .min_by(|left, right| {
                left.chunk_bytes
                    .cmp(&right.chunk_bytes)
                    .then_with(|| left.pool_index.cmp(&right.pool_index))
                    .then_with(|| right.chunk.ordinal().cmp(&left.chunk.ordinal()))
            })
            .cloned();
        if let Some(candidate) = best_single {
            selected_by_pool[candidate.pool_index] = candidate.chunk_bytes;
            reclaimed_bytes = candidate.chunk_bytes;
            selected.push(candidate);
        } else {
            candidates.sort_by(|left, right| {
                right
                    .chunk_bytes
                    .cmp(&left.chunk_bytes)
                    .then_with(|| left.pool_index.cmp(&right.pool_index))
                    .then_with(|| right.chunk.ordinal().cmp(&left.chunk.ordinal()))
            });
            for candidate in candidates {
                let next_pool_total = selected_by_pool[candidate.pool_index]
                    .checked_add(candidate.chunk_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim bytes overflow u64"))?;
                if next_pool_total > reclaimable_by_pool[candidate.pool_index] {
                    continue;
                }
                selected_by_pool[candidate.pool_index] = next_pool_total;
                reclaimed_bytes = reclaimed_bytes
                    .checked_add(candidate.chunk_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim bytes overflow u64"))?;
                selected.push(candidate);
                if reclaimed_bytes >= deficit {
                    break;
                }
            }
        }
        if reclaimed_bytes < deficit {
            return Ok(None);
        }
        selected.sort_by(|left, right| {
            left.pool_index
                .cmp(&right.pool_index)
                .then_with(|| left.chunk.ordinal().cmp(&right.chunk.ordinal()))
        });

        for candidate in &selected {
            let state = &states[candidate.pool_index];
            let chunk = state
                .chunks
                .get(&candidate.chunk.ordinal())
                .ok_or_else(|| invalid_resource("selected dynamic reclaim chunk disappeared"))?;
            let extent = state
                .allocator
                .by_offset
                .get(&(candidate.chunk.ordinal(), 0))
                .ok_or_else(|| invalid_resource("selected dynamic reclaim extent disappeared"))?;
            if chunk.live_segments != 0
                || Arc::strong_count(&chunk.backing) != 1
                || chunk.backing.identity != candidate.chunk
                || extent.chunk_generation != candidate.chunk.generation()
                || extent.length_bytes != candidate.chunk_bytes
            {
                return Err(invalid_resource(
                    "selected dynamic reclaim chunk changed before publication",
                ));
            }
        }

        let published_totals = states
            .iter()
            .zip(&selected_by_pool)
            .map(|(state, &selected_bytes)| {
                state
                    .resident_bytes
                    .checked_sub(selected_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim resident bytes underflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut removed = Vec::with_capacity(selected.len());
        for candidate in &selected {
            let state = &mut states[candidate.pool_index];
            let extent = state
                .allocator
                .remove_extent(candidate.chunk.ordinal(), 0)
                .expect("validated idle chunk retains its exact full extent");
            debug_assert_eq!(extent.chunk_generation, candidate.chunk.generation());
            debug_assert_eq!(extent.length_bytes, candidate.chunk_bytes);
            let chunk = state
                .chunks
                .remove(&candidate.chunk.ordinal())
                .expect("validated idle chunk remains resident");
            removed.push((candidate.clone(), chunk));
        }
        let updates = selected_by_pool
            .iter()
            .enumerate()
            .filter(|(_, selected_bytes)| **selected_bytes != 0)
            .map(|(pool_index, _)| {
                (
                    pools[pool_index].domain.domain_id,
                    CapacityUnits::new(published_totals[pool_index]),
                )
            })
            .collect::<Vec<_>>();
        let epochs = match self.logical_admission.set_domain_totals(&updates) {
            Ok(epochs) => epochs,
            Err(error) => {
                for (candidate, chunk) in removed.drain(..).rev() {
                    let state = &mut states[candidate.pool_index];
                    state
                        .allocator
                        .insert_extent(
                            candidate.chunk.ordinal(),
                            candidate.chunk.generation(),
                            0,
                            candidate.chunk_bytes,
                        )
                        .expect("unpublished idle chunk extent can be restored");
                    assert!(state
                        .chunks
                        .insert(candidate.chunk.ordinal(), chunk)
                        .is_none());
                }
                return Err(error);
            }
        };
        for (state, &published_total) in states.iter_mut().zip(&published_totals) {
            state.resident_bytes = published_total;
        }

        let mut pool_receipts = Vec::new();
        for (pool_index, &pool_reclaimed_bytes) in selected_by_pool.iter().enumerate() {
            if pool_reclaimed_bytes == 0 {
                continue;
            }
            pool_receipts.push(DynamicPoolIdleReclaim {
                pool_id: pools[pool_index].domain.pool_id().clone(),
                chunks: selected
                    .iter()
                    .filter(|candidate| candidate.pool_index == pool_index)
                    .map(|candidate| candidate.chunk.clone())
                    .collect(),
                reclaimed_bytes: pool_reclaimed_bytes,
                published_capacity_bytes: published_totals[pool_index],
            });
        }
        let reclaimed_chunks = selected.len();
        drop(states);
        drop(maintenance);
        drop(removed);
        let availability = self.budget.availability_snapshot()?;
        Ok(Some(DynamicPoolRebalanceReceipt {
            pools: pool_receipts,
            reclaimed_chunks,
            reclaimed_bytes,
            logical_capacity_epoch: epochs.capacity_epoch(),
            plan_device_capacity_epoch: availability.plan_epoch(),
            process_device_capacity_epoch: availability.process_epoch(),
        }))
    }

    pub(super) fn maintain_pools(
        &self,
        intents: Vec<DynamicPoolGrowthIntent>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        let mut ignored_capacity_block = None;
        self.maintain_pools_observed(intents, &mut ignored_capacity_block)
    }

    pub(super) fn maintain_pools_observed(
        &self,
        mut intents: Vec<DynamicPoolGrowthIntent>,
        capacity_blocked: &mut Option<DynamicDeviceCapacityBlocked>,
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
                    DynamicPoolGrowthIntent::RevalidatedDeferral(blocker) => {
                        if let Some(claim_bytes) = blocker.contiguous_claim_bytes_descending() {
                            let required_growth = contiguous_packing_growth_bytes(
                                &state.allocator,
                                pool.domain.pool_id(),
                                claim_bytes,
                            )?;
                            if required_growth == 0 {
                                continue;
                            }
                            required_growth
                        } else {
                            match blocker.reason() {
                                DynamicBackingDeferralReason::GrowthRequired => {
                                    let required_free = blocker
                                        .free_bytes()
                                        .checked_add(blocker.requested_bytes())
                                        .ok_or_else(|| {
                                            invalid_resource(
                                            "dynamic backing deferred requirement overflows u64",
                                        )
                                        })?;
                                    if state.allocator.free_bytes >= required_free {
                                        continue;
                                    }
                                    required_free - state.allocator.free_bytes
                                }
                                DynamicBackingDeferralReason::FragmentedContiguous => {
                                    return Err(invalid_resource(
                                        "fragmented contiguous blocker lost its transaction demand",
                                    ));
                                }
                            }
                        }
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
                BackingChunkIdentity::from_parts(
                    pool.domain.pool_id().clone(),
                    ordinal,
                    generation,
                )?
            };
            pending.push(PendingGrowthGuard {
                pool: Arc::clone(pool),
                bytes: chunk_bytes,
                armed: true,
            });
            let expected_resource_id = ResourceId::new(format!(
                "{}/chunk/{}/{}",
                pool.domain.pool_id().as_str(),
                chunk.ordinal(),
                chunk.generation()
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
                rebalance: None,
            });
        }

        let total_bytes = planned.iter().try_fold(0_u64, |total, growth| {
            total
                .checked_add(growth.chunk_bytes)
                .ok_or_else(|| invalid_resource("dynamic maintenance batch bytes overflow u64"))
        })?;
        let capacity_availability = self.budget.availability_snapshot()?;
        let reservation = match DeviceCapacityReservation::reserve(&self.budget, total_bytes) {
            Ok(reservation) => reservation,
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
                *capacity_blocked = Some(DynamicDeviceCapacityBlocked {
                    pressure: pressure.clone(),
                    availability: capacity_availability,
                    planned_domains: planned
                        .iter()
                        .map(|growth| growth.pool.domain.domain_id)
                        .collect(),
                });
                return Err(VNextError::DeviceCapacityUnavailable(pressure));
            }
            Err(error) => return Err(error),
        };
        // Device-budget saturation is recoverable pressure even when the same
        // growth also crosses a pool's device-derived resident ceiling. Only a
        // growth that the authoritative device budget accepted can prove that
        // the remaining pool ceiling is a terminal theoretical-plan violation.
        for growth in &planned {
            let state = growth
                .pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            let next_residency = state
                .resident_bytes
                .checked_add(state.pending_growth_bytes)
                .ok_or_else(|| invalid_resource("dynamic pool resident bytes overflow u64"))?;
            if next_residency
                > growth
                    .pool
                    .domain
                    .pool
                    .provisioning()
                    .maximum_resident_bytes()
            {
                return Err(VNextError::DynamicPoolResidentUnavailable(
                    DynamicPoolResidentPressure::new(
                        growth.pool.domain.pool_id().clone(),
                        growth.chunk_bytes,
                        state.resident_bytes,
                        growth
                            .pool
                            .domain
                            .pool
                            .provisioning()
                            .maximum_resident_bytes(),
                    )?,
                ));
            }
        }
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
                run_id: RunId::new(format!("dynamic-grow-{}", growth.chunk.generation()))?,
                transaction_id: TransactionId::new(format!(
                    "dynamic-grow-{}-{}",
                    growth.chunk.ordinal(),
                    growth.chunk.generation()
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
                generation: growth.chunk.generation(),
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
                || state.chunks.contains_key(&growth.chunk.ordinal())
                || state
                    .allocator
                    .by_offset
                    .range((growth.chunk.ordinal(), 0)..=(growth.chunk.ordinal(), u64::MAX))
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
                    growth.chunk.ordinal(),
                    growth.chunk.generation(),
                    0,
                    growth.chunk_bytes,
                )
                .expect("validated new chunk has one disjoint free extent");
            state.chunks.insert(
                growth.chunk.ordinal(),
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
                        .remove_extent(planned[index].chunk.ordinal(), 0)
                        .expect("unpublished dynamic chunk free extent remains installed");
                    let removed = states[index]
                        .chunks
                        .remove(&planned[index].chunk.ordinal())
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
            rebalance: None,
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
        let lifetime = requests
            .first()
            .and_then(|request| request.projections.first())
            .map(|projection| projection.descriptor.lifetime())
            .ok_or_else(|| invalid_resource("dynamic backing request has no projection"))?;
        self.prepare_claim_scoped(requests, DynamicBackingClaimScope::from(lifetime))
    }

    fn reusable_capacity_shape_for_requests(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<Option<DynamicResourceShape>, VNextError> {
        let reusable_execution_bucket_id = requests
            .first()
            .and_then(|request| request.reusable_execution_bucket_id.as_ref());
        if requests.iter().any(|request| {
            request.reusable_execution_bucket_id.as_ref() != reusable_execution_bucket_id
        }) {
            return Err(invalid_resource(
                "one dynamic backing claim cannot mix reusable execution buckets",
            ));
        }
        reusable_execution_bucket_id
            .map(|bucket_id| {
                let bucket = self
                    .reusable_execution
                    .as_ref()
                    .and_then(|plan| plan.bucket(bucket_id))
                    .map(|resolved| resolved.bucket())
                    .ok_or_else(|| {
                        invalid_resource(
                            "dynamic backing claim references a reusable bucket outside its immutable plan",
                        )
                    })?;
                Ok(DynamicResourceShape::from_validated(
                    bucket.capacity().maximum_sequences(),
                    bucket.capacity().maximum_tokens(),
                    bucket.capacity().maximum_pages(),
                ))
            })
            .transpose()
    }

    pub(super) fn prepare_lane_stable_claim(
        self: &Arc<Self>,
        lane: &Arc<ExecutionLane<R>>,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<LaneBackingPrepareDecision, VNextError> {
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc())
            || lane.descriptor() != self.runtime.descriptor()
            || !lane.is_reusable()
        {
            return Err(invalid_resource(
                "lane-stable backing requires the reusable execution lane bound to this plan runtime",
            ));
        }
        if requests.is_empty() {
            return Ok(LaneBackingPrepareDecision::Prepared(
                PreparedLaneBackingClaim {
                    stable: Vec::new(),
                    slot_lease: None,
                },
            ));
        }
        let reusable_capacity_shape = self.reusable_capacity_shape_for_requests(requests)?;
        if reusable_capacity_shape.is_none() {
            return self.prepare_claim(requests).map(|decision| match decision {
                BackingPrepareDecision::Prepared(prepared) => {
                    LaneBackingPrepareDecision::Prepared(PreparedLaneBackingClaim {
                        stable: prepared.commit(),
                        slot_lease: None,
                    })
                }
                BackingPrepareDecision::Deferred(deferred) => {
                    LaneBackingPrepareDecision::Deferred(deferred)
                }
            });
        }
        let lifetime = requests
            .first()
            .and_then(|request| request.projections.first())
            .map(|projection| projection.descriptor.lifetime())
            .ok_or_else(|| invalid_resource("dynamic backing request has no projection"))?;
        if !matches!(
            lifetime,
            AllocationLifetime::Step | AllocationLifetime::Invocation
        ) || requests.iter().any(|request| {
            request.projections.is_empty()
                || request.projections.iter().any(|projection| {
                    projection.descriptor.lifetime() != lifetime
                        || projection.descriptor.initialization() != StateInitialization::None
                })
        }) {
            return Err(invalid_resource(
                "lane-stable backing accepts only non-initialized Step or Invocation resources",
            ));
        }

        let mut canonical_requests = requests.iter().collect::<Vec<_>>();
        canonical_requests
            .sort_unstable_by(|left, right| left.claim_identity.cmp(&right.claim_identity));
        let key = lane_stable_layout_key(lane.id(), lifetime, &canonical_requests)?;
        let lane_owner: Arc<dyn LaneStableArenaLane> =
            Arc::clone(lane) as Arc<dyn LaneStableArenaLane>;

        loop {
            {
                let mut arenas = self
                    .lane_stable_arenas
                    .lock()
                    .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                if arenas.poisoned {
                    return Err(invalid_resource(
                        "lane-stable arena registry is fail-closed",
                    ));
                }
                let now = arenas.tick();
                if let Some(entry) = arenas.entries.get_mut(&key) {
                    let owner = entry.lane.upgrade().ok_or_else(|| {
                        invalid_resource("lane-stable arena retained an expired execution lane")
                    })?;
                    if !Arc::ptr_eq(&owner, &lane_owner) {
                        return Err(invalid_resource(
                            "lane-stable arena identity aliases another execution lane",
                        ));
                    }
                    if let Some((slot_id, stable, slot_domains)) =
                        entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                    {
                        return Ok(LaneBackingPrepareDecision::Prepared(
                            PreparedLaneBackingClaim {
                                stable,
                                slot_lease: Some(LaneStableArenaSlotLease {
                                    arenas: Arc::clone(&self.lane_stable_arenas),
                                    logical_admission: self.logical_admission.clone(),
                                    availability_domains: slot_domains,
                                    key: key.clone(),
                                    slot_id,
                                }),
                            },
                        ));
                    }
                }
            }

            match self.prepare_claim(requests)? {
                BackingPrepareDecision::Prepared(prepared) => {
                    let mut arenas = self
                        .lane_stable_arenas
                        .lock()
                        .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                    if arenas.poisoned {
                        return Err(invalid_resource(
                            "lane-stable arena registry is fail-closed",
                        ));
                    }
                    let now = arenas.tick();
                    if let Some(entry) = arenas.entries.get_mut(&key) {
                        let owner = entry.lane.upgrade().ok_or_else(|| {
                            invalid_resource("lane-stable arena retained an expired execution lane")
                        })?;
                        if !Arc::ptr_eq(&owner, &lane_owner) {
                            return Err(invalid_resource(
                                "lane-stable arena identity aliases another execution lane",
                            ));
                        }
                        if let Some((slot_id, stable, slot_domains)) =
                            entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                        {
                            drop(arenas);
                            drop(prepared);
                            return Ok(LaneBackingPrepareDecision::Prepared(
                                PreparedLaneBackingClaim {
                                    stable,
                                    slot_lease: Some(LaneStableArenaSlotLease {
                                        arenas: Arc::clone(&self.lane_stable_arenas),
                                        logical_admission: self.logical_admission.clone(),
                                        availability_domains: slot_domains,
                                        key: key.clone(),
                                        slot_id,
                                    }),
                                },
                            ));
                        }
                    }
                    let authorities = prepared.commit();
                    let projection_bindings =
                        bind_lane_stable_slot_projections(&authorities, &canonical_requests)?;
                    let stable = authorities
                        .iter()
                        .map(|authority| authority.retained_for_lane(lane.id()))
                        .collect();
                    let availability_domains = requests
                        .iter()
                        .map(|request| request.domain.domain_id())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    let slot_id = arenas.issue_slot_id()?;
                    let entry =
                        arenas
                            .entries
                            .entry(key.clone())
                            .or_insert_with(|| LaneStableArenaEntry {
                                lane: Arc::downgrade(&lane_owner),
                                slots: BTreeMap::new(),
                            });
                    if entry
                        .slots
                        .insert(
                            slot_id,
                            LaneStableArenaSlot {
                                slot_id,
                                authorities,
                                projection_bindings,
                                availability_domains: availability_domains.clone(),
                                in_use: true,
                                last_used: now,
                            },
                        )
                        .is_some()
                    {
                        arenas.poisoned = true;
                        return Err(invalid_resource(
                            "lane-stable arena slot publication replaced an existing slot",
                        ));
                    }
                    return Ok(LaneBackingPrepareDecision::Prepared(
                        PreparedLaneBackingClaim {
                            stable,
                            slot_lease: Some(LaneStableArenaSlotLease {
                                arenas: Arc::clone(&self.lane_stable_arenas),
                                logical_admission: self.logical_admission.clone(),
                                availability_domains: availability_domains.clone(),
                                key: key.clone(),
                                slot_id,
                            }),
                        },
                    ));
                }
                BackingPrepareDecision::Deferred(deferred) => {
                    let mut arenas = self
                        .lane_stable_arenas
                        .lock()
                        .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                    if arenas.poisoned {
                        return Err(invalid_resource(
                            "lane-stable arena registry is fail-closed",
                        ));
                    }
                    let now = arenas.tick();
                    if let Some(entry) = arenas.entries.get_mut(&key) {
                        let owner = entry.lane.upgrade().ok_or_else(|| {
                            invalid_resource("lane-stable arena retained an expired execution lane")
                        })?;
                        if !Arc::ptr_eq(&owner, &lane_owner) {
                            return Err(invalid_resource(
                                "lane-stable arena identity aliases another execution lane",
                            ));
                        }
                        if let Some((slot_id, stable, slot_domains)) =
                            entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                        {
                            drop(arenas);
                            drop(deferred);
                            return Ok(LaneBackingPrepareDecision::Prepared(
                                PreparedLaneBackingClaim {
                                    stable,
                                    slot_lease: Some(LaneStableArenaSlotLease {
                                        arenas: Arc::clone(&self.lane_stable_arenas),
                                        logical_admission: self.logical_admission.clone(),
                                        availability_domains: slot_domains,
                                        key: key.clone(),
                                        slot_id,
                                    }),
                                },
                            ));
                        }
                    }
                    return Ok(LaneBackingPrepareDecision::Deferred(deferred));
                }
            }
        }
    }

    pub(super) fn try_reclaim_expired_lane_slots(&self) -> Result<bool, VNextError> {
        let expired_entries = {
            let mut arenas = self
                .lane_stable_arenas
                .lock()
                .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
            if arenas.poisoned {
                return Err(invalid_resource(
                    "lane-stable arena registry is fail-closed",
                ));
            }
            arenas.take_expired_lanes()?
        };
        let reclaimed = !expired_entries.is_empty();
        // Releasing backing owners can enter backend/pool destruction paths.
        // Keep that work outside the arena registry's hot mutex.
        drop(expired_entries);
        Ok(reclaimed)
    }

    pub(super) fn try_reclaim_one_idle_lane_slot(&self) -> Result<bool, VNextError> {
        if self.try_reclaim_expired_lane_slots()? {
            return Ok(true);
        }
        let mut candidates = {
            let arenas = self
                .lane_stable_arenas
                .lock()
                .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
            if arenas.poisoned {
                return Err(invalid_resource(
                    "lane-stable arena registry is fail-closed",
                ));
            }
            arenas
                .entries
                .iter()
                .filter_map(|(key, entry)| entry.lane.upgrade().map(|lane| (key, entry, lane)))
                .flat_map(|(key, entry, lane)| {
                    entry
                        .slots
                        .values()
                        .filter(|slot| !slot.in_use)
                        .map(move |slot| LaneStableArenaEvictionCandidate {
                            key: key.clone(),
                            slot_id: slot.slot_id,
                            last_used: slot.last_used,
                            lane: Arc::clone(&lane),
                        })
                })
                .collect::<Vec<_>>()
        };
        candidates.sort_by_key(|candidate| candidate.last_used);

        for candidate in candidates {
            if !candidate.lane.try_trim_reusable_executables()? {
                continue;
            }
            let victim = {
                let mut arenas = self
                    .lane_stable_arenas
                    .lock()
                    .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                if arenas.poisoned {
                    return Err(invalid_resource(
                        "lane-stable arena registry is fail-closed",
                    ));
                }
                let removable = arenas
                    .entries
                    .get(&candidate.key)
                    .and_then(|entry| entry.slots.get(&candidate.slot_id))
                    .is_some_and(|slot| !slot.in_use && !slot.has_external_address_pins());
                if !removable {
                    None
                } else {
                    let (victim, remove_entry) = {
                        let entry = arenas.entries.get_mut(&candidate.key).ok_or_else(|| {
                            invalid_resource("lane-stable arena eviction lost its entry")
                        })?;
                        let victim = entry.slots.remove(&candidate.slot_id).ok_or_else(|| {
                            invalid_resource("lane-stable arena eviction lost its idle slot")
                        })?;
                        (victim, entry.slots.is_empty())
                    };
                    if remove_entry {
                        arenas.entries.remove(&candidate.key);
                    }
                    Some(victim)
                }
            };
            if let Some(victim) = victim {
                drop(victim);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn prepare_initial_sequence_claim(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        self.prepare_claim_scoped(requests, DynamicBackingClaimScope::InitialSequenceBundle)
    }

    fn prepare_claim_scoped(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
        scope: DynamicBackingClaimScope,
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        if requests.is_empty() {
            return Ok(BackingPrepareDecision::Prepared(
                PreparedBackingClaim::empty(),
            ));
        }
        let reusable_capacity_shape = self.reusable_capacity_shape_for_requests(requests)?;
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
        let protected_immediate = CapacityVector::new(
            groups
                .iter()
                .map(|(pool, requests)| {
                    let bytes = requests.iter().try_fold(0_u64, |total, request| {
                        total
                            .checked_add(request.capacity_size_bytes)
                            .ok_or_else(|| {
                                invalid_resource("dynamic backing protection bytes overflow u64")
                            })
                    })?;
                    CapacityEntry::new(pool.domain.domain_id, CapacityUnits::new(bytes))
                })
                .collect::<Result<Vec<_>, VNextError>>()?,
        )?;
        'prepare: loop {
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
                        && request.projections[0].capacity_size_bytes
                            == request.capacity_size_bytes;
                    let shared_step_slot = request.projections.len() > 1
                        && request
                            .projections
                            .iter()
                            .all(|projection| projection.physical_offset_bytes == 0)
                        && request
                            .projections
                            .iter()
                            .map(|projection| projection.capacity_size_bytes)
                            .max()
                            == Some(request.capacity_size_bytes)
                        && pool.domain.pool.step_resource_slots().iter().any(|slot| {
                            slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave
                                && slot.resource_ids() == request.claim_identity.resource_ids()
                        });
                    let invocation_wave =
                        self.validate_invocation_wave_projection(pool, request)?;
                    if request.domain.pool_id() != pool.domain.pool_id()
                        || request.claim_identity.pool_id() != pool.domain.pool_id()
                        || request.claim_identity.resource_ids() != projection_ids
                        || request.projections.is_empty()
                        || request.projections.windows(2).any(|pair| {
                            pair[0].descriptor.base_resource_id()
                                >= pair[1].descriptor.base_resource_id()
                        })
                        || request.projections.iter().any(|projection| {
                            let capacity_matches_plan = match reusable_capacity_shape {
                                Some(shape) => {
                                    matches!(
                                        projection.descriptor.lifetime(),
                                        AllocationLifetime::Step | AllocationLifetime::Invocation
                                    ) && projection
                                        .descriptor
                                        .evaluate_request_bytes_for_shape(shape)
                                        .is_ok_and(|bytes| bytes == projection.capacity_size_bytes)
                                }
                                None => {
                                    projection.logical_size_bytes == projection.capacity_size_bytes
                                }
                            };
                            projection.descriptor.pool_id() != pool.domain.pool_id()
                                || !scope.accepts(projection.descriptor.lifetime())
                                || !capacity_matches_plan
                                || projection.logical_size_bytes == 0
                                || projection.logical_size_bytes > projection.capacity_size_bytes
                                || projection.capacity_size_bytes == 0
                                || projection.capacity_size_bytes % quantum != 0
                                || projection.physical_offset_bytes % quantum != 0
                                || projection
                                    .physical_offset_bytes
                                    .checked_add(projection.capacity_size_bytes)
                                    .is_none_or(|end| end > request.capacity_size_bytes)
                                || !request
                                    .domain
                                    .descriptors
                                    .iter()
                                    .any(|descriptor| descriptor == projection.descriptor)
                        })
                        || request.capacity_size_bytes == 0
                        || request.capacity_size_bytes % quantum != 0
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
                    let requested_group_bytes =
                        pool_requests.iter().try_fold(0_u64, |total, request| {
                            total
                                .checked_add(request.capacity_size_bytes)
                                .ok_or_else(|| {
                                    invalid_resource("dynamic backing batch bytes overflow u64")
                                })
                        })?;
                    let state = &states[group_index];
                    if state.allocator.free_bytes >= requested_group_bytes {
                        return Ok(None);
                    }
                    let (reason, requested_bytes, contiguous_claim_bytes_descending) =
                        match pool.domain.pool.compatibility().profile().view() {
                            DynamicStorageView::Contiguous => {
                                let mut claim_bytes = pool_requests
                                    .iter()
                                    .map(|request| request.capacity_size_bytes)
                                    .collect::<Vec<_>>();
                                claim_bytes.sort_unstable_by(|left, right| right.cmp(left));
                                let growth = contiguous_packing_growth_bytes(
                                    &state.allocator,
                                    pool.domain.pool_id(),
                                    &claim_bytes,
                                )?;
                                if growth == 0 {
                                    return Err(invalid_resource(
                                    "insufficient contiguous capacity produced zero packing growth",
                                ));
                                }
                                (
                                    DynamicBackingDeferralReason::GrowthRequired,
                                    growth,
                                    Some(claim_bytes),
                                )
                            }
                            DynamicStorageView::PagedRegions { .. } => (
                                DynamicBackingDeferralReason::GrowthRequired,
                                requested_group_bytes - state.allocator.free_bytes,
                                None,
                            ),
                        };
                    Ok(Some(DynamicBackingBlocker {
                        pool_id: pool.domain.pool_id().clone(),
                        domain_id: pool.domain.domain_id,
                        reason,
                        requested_bytes,
                        free_bytes: state.allocator.free_bytes,
                        largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                        free_extent_layout_fingerprint: free_extent_layout_fingerprint(
                            &state.allocator,
                        ),
                        contiguous_claim_bytes_descending,
                    }))
                })
                .collect::<Result<Vec<_>, VNextError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            if !blockers.is_empty() {
                drop(states);
                if let Some(deferred) =
                    self.confirm_backing_deferral(blockers, scope, protected_immediate.clone())?
                {
                    return Ok(BackingPrepareDecision::Deferred(deferred));
                }
                continue 'prepare;
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
                let mut allocation_requests = pool_requests.clone();
                allocation_requests.sort_by(|left, right| {
                    right
                        .capacity_size_bytes
                        .cmp(&left.capacity_size_bytes)
                        .then_with(|| left.claim_identity.cmp(&right.claim_identity))
                });
                for request in allocation_requests {
                    let reserved = match match profile.view() {
                        DynamicStorageView::Contiguous => states[group_index]
                            .allocator
                            .allocate_contiguous(pool.domain.pool_id(), request.capacity_size_bytes)
                            .map(|segment| segment.map(|segment| vec![segment])),
                        DynamicStorageView::PagedRegions { block_bytes } => {
                            states[group_index].allocator.allocate_paged(
                                pool.domain.pool_id(),
                                request.capacity_size_bytes,
                                block_bytes,
                            )
                        }
                    } {
                        Ok(reserved) => reserved,
                        Err(error) => {
                            states[group_index].poisoned = true;
                            rollback_free_extent_journal(&mut states, &journals)?;
                            return Err(error);
                        }
                    };
                    let Some(segments) = reserved else {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        if !matches!(profile.view(), DynamicStorageView::Contiguous) {
                            states[group_index].poisoned = true;
                            return Err(invalid_resource(
                                "paged backing allocation failed after its aggregate fit check",
                            ));
                        }
                        let free_bytes = states[group_index].allocator.free_bytes;
                        let reason = DynamicBackingDeferralReason::FragmentedContiguous;
                        let mut claim_bytes_descending = pool_requests
                            .iter()
                            .map(|request| request.capacity_size_bytes)
                            .collect::<Vec<_>>();
                        claim_bytes_descending.sort_unstable_by(|left, right| right.cmp(left));
                        let requested_bytes = contiguous_packing_growth_bytes(
                            &states[group_index].allocator,
                            pool.domain.pool_id(),
                            &claim_bytes_descending,
                        )?;
                        if requested_bytes == 0 {
                            return Err(invalid_resource(
                                "contiguous packing failed without a progress-producing growth",
                            ));
                        }
                        let blocker = DynamicBackingBlocker {
                            pool_id: pool.domain.pool_id().clone(),
                            domain_id: pool.domain.domain_id,
                            reason,
                            requested_bytes,
                            free_bytes,
                            largest_contiguous_bytes: states[group_index]
                                .allocator
                                .largest_contiguous_bytes(),
                            free_extent_layout_fingerprint: free_extent_layout_fingerprint(
                                &states[group_index].allocator,
                            ),
                            contiguous_claim_bytes_descending: Some(claim_bytes_descending),
                        };
                        drop(states);
                        if let Some(deferred) = self.confirm_backing_deferral(
                            vec![blocker],
                            scope,
                            protected_immediate.clone(),
                        )? {
                            return Ok(BackingPrepareDecision::Deferred(deferred));
                        }
                        continue 'prepare;
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
                    if extent_bytes != request.capacity_size_bytes {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(invalid_resource(
                            "dynamic backing extents differ from their physical capacity claim",
                        ));
                    }
                    let generation =
                        segment_generations[group_index][selections[group_index].len()];
                    selections[group_index].push((request, generation, segments));
                }
            }

            for group_selections in &selections {
                for (request, _, segments) in group_selections {
                    for projection in &request.projections {
                        if let Err(error) = backing_segment_range(
                            segments,
                            projection.physical_offset_bytes,
                            projection.capacity_size_bytes,
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
                                invalid_resource(
                                    "dynamic chunk live extent increment overflows u64",
                                )
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
                            let mut allocation = LogicalBackingSliceAllocationEvidence {
                                domain_id: pool.domain.domain_id,
                                pool_id: pool.domain.pool_id().clone(),
                                resource_id: projection.descriptor.base_resource_id().clone(),
                                pool_instance_id: pool.instance_id,
                                physical_claim_identity: request.claim_identity.clone(),
                                reusable_execution_bucket_id: request
                                    .reusable_execution_bucket_id
                                    .clone(),
                                segment_generation,
                                segments: backing_segment_range(
                                    &segments,
                                    projection.physical_offset_bytes,
                                    projection.capacity_size_bytes,
                                )?,
                                physical_offset_bytes: projection.physical_offset_bytes,
                                capacity_size_bytes: projection.capacity_size_bytes,
                                physical_size_bytes: request.capacity_size_bytes,
                                alignment_bytes: projection.descriptor.alignment_bytes(),
                                usage: projection.descriptor.usage(),
                                element_type: projection.descriptor.element_type(),
                                storage_profile: pool.domain.pool.compatibility().profile(),
                                initialization: projection.descriptor.initialization(),
                                fingerprint: String::new(),
                            };
                            let bytes = serde_json::to_vec(&allocation).map_err(|error| {
                                invalid_resource(format!(
                                    "logical backing allocation evidence encode failed: {error}"
                                ))
                            })?;
                            allocation.fingerprint = format!("sha256/{:x}", Sha256::digest(bytes));
                            Ok(LogicalBackingSliceEvidence {
                                allocation: Arc::new(allocation),
                                logical_size_bytes: projection.logical_size_bytes,
                            })
                        })
                        .collect::<Result<Vec<_>, VNextError>>()?;
                    extents.push(PreparedBackingExtent {
                        pool: Arc::clone(&pool),
                        claim_identity: request.claim_identity.clone(),
                        segment_generation,
                        segments,
                        capacity_size_bytes: request.capacity_size_bytes,
                        projections,
                    });
                }
            }
            return Ok(BackingPrepareDecision::Prepared(PreparedBackingClaim {
                extents,
                committed: false,
            }));
        }
    }

    /// Publishes a physical deferral with the event-subscription ordering
    /// required to avoid a lost release:
    ///
    /// 1. observe the exact coordinator generations after the failed check;
    /// 2. recheck the physical allocator observations;
    /// 3. publish only if those observations are still current.
    ///
    /// A release before step 1 is visible to step 2. A release after step 1
    /// advances the returned predicate (or is visible to step 2 before its
    /// coordinator notification), so the scheduler cannot sleep forever on a
    /// stale blocker.
    fn confirm_backing_deferral(
        &self,
        blockers: Vec<DynamicBackingBlocker>,
        scope: DynamicBackingClaimScope,
        protected_immediate: CapacityVector,
    ) -> Result<Option<DynamicBackingDeferred>, VNextError> {
        let wait_snapshot = self
            .logical_admission
            .wait_snapshot_for_domains(blockers.iter().map(|blocker| blocker.domain_id))?;
        for blocker in &blockers {
            let pool = self.pools.get(blocker.pool_id()).ok_or_else(|| {
                invalid_resource("dynamic backing blocker references an unknown pool")
            })?;
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            if state.poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            if state.allocator.free_bytes != blocker.free_bytes
                || state.allocator.largest_contiguous_bytes() != blocker.largest_contiguous_bytes
                || free_extent_layout_fingerprint(&state.allocator)
                    != blocker.free_extent_layout_fingerprint
            {
                return Ok(None);
            }
        }
        Ok(Some(DynamicBackingDeferred {
            blockers,
            epochs: wait_snapshot.epochs(),
            wait_condition: wait_snapshot.wait_condition().clone(),
            scope,
            protected_immediate,
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
                    .checked_add(projection.capacity_size_bytes)
                    .ok_or_else(|| invalid_resource("invocation wave row size overflows u64"))?;
            }
            peak = peak.max(row_cursor);
            if mode == InvocationLivenessMode::ConservativeConcurrent {
                concurrent_cursor = concurrent_cursor.checked_add(row_cursor).ok_or_else(|| {
                    invalid_resource("concurrent invocation wave size overflows u64")
                })?;
            }
        }
        Ok(request.capacity_size_bytes
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
        let mut logical_size_bytes = 0_u64;
        let mut capacity_size_bytes = 0_u64;
        let mut segment_count = 0_usize;
        for (index, authority) in authorities.iter().enumerate() {
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
            if index + 1 < authorities.len()
                && authority.evidence.logical_size_bytes != authority.evidence.capacity_size_bytes
            {
                return Err(invalid_resource(
                    "multi-extent logical backing cannot contain interior capacity slack",
                ));
            }
            logical_size_bytes = logical_size_bytes
                .checked_add(authority.evidence.logical_size_bytes)
                .ok_or_else(|| invalid_resource("logical backing view size overflows u64"))?;
            capacity_size_bytes = capacity_size_bytes
                .checked_add(authority.evidence.capacity_size_bytes)
                .ok_or_else(|| invalid_resource("logical backing capacity overflows u64"))?;
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
                let retention = match authority.reusable_lane {
                    Some(lane_id) => DeviceBufferRetention::lane_pair(
                        lane_id,
                        Arc::clone(&authority.segment_lease),
                        Arc::clone(&chunk.backing),
                    ),
                    None => DeviceBufferRetention::pair(
                        Arc::clone(&authority.segment_lease),
                        Arc::clone(&chunk.backing),
                    ),
                };
                bindings.push(LogicalBackingSegmentBinding {
                    segment: segment.clone(),
                    chunk: Arc::clone(&chunk.backing),
                    retention,
                });
            }
        }
        drop(state);
        Ok(LogicalBackingBufferView {
            bindings,
            authorities,
            logical_size_bytes,
            capacity_size_bytes,
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
            || authority.evidence.logical_size_bytes == 0
            || authority.evidence.logical_size_bytes > authority.evidence.capacity_size_bytes
            || authority
                .evidence
                .physical_offset_bytes
                .checked_add(authority.evidence.capacity_size_bytes)
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
            authority.evidence.capacity_size_bytes,
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
